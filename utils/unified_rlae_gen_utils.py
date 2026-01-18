import inspect
import warnings
from typing import Callable, List, Optional, Union

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers.generation import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import *
from transformers.generation.utils import GenerateOutput, GreedySearchOutput

from .logger import setup_custom_logger

logger = setup_custom_logger("TSP")


def generate_ensemnble_response_unified(
    model_actors_list,
    model_name_list,
    tokenizers,
    vocab_union,
    mapping_matrices,
    index_to_vocab,
    special_prefix_tokens_dict,
    byte_mappings_list,
    primary_index,
    threshold,
    until,
    span_mode=False,
    span_config=None,
    weight_update_queue=None,
    **kwargs,
):
    """Unified ensemble generation function supporting token-level and span-level modes."""

    if span_mode:
        # Span-level mode
        return generate_ensemnble_response_span(
            model_actors_list=model_actors_list,
            model_name_list=model_name_list,
            tokenizers=tokenizers,
            vocab_union=vocab_union,
            mapping_matrices=mapping_matrices,
            index_to_vocab=index_to_vocab,
            special_prefix_tokens_dict=special_prefix_tokens_dict,
            byte_mappings_list=byte_mappings_list,
            primary_index=primary_index,
            threshold=threshold,
            until=until,
            span_config=span_config,
            weight_update_queue=weight_update_queue,
            **kwargs,
        )
    else:
        # Token-level mode (original logic)
        return generate_ensemnble_response(
            model_actors_list=model_actors_list,
            model_name_list=model_name_list,
            tokenizers=tokenizers,
            vocab_union=vocab_union,
            mapping_matrices=mapping_matrices,
            index_to_vocab=index_to_vocab,
            special_prefix_tokens_dict=special_prefix_tokens_dict,
            byte_mappings_list=byte_mappings_list,
            primary_index=primary_index,
            threshold=threshold,
            until=until,
            weight_update_queue=weight_update_queue,
            **kwargs,
        )


def generate_ensemnble_response_span(
    model_actors_list,
    model_name_list,
    tokenizers,
    vocab_union,
    mapping_matrices,
    index_to_vocab,
    special_prefix_tokens_dict,
    byte_mappings_list,
    primary_index,
    threshold,
    until,
    span_config=None,
    weight_update_queue=None,
    **kwargs,
):
    """Span-level ensemble generation function."""

    # Initialize span state
    span_state = {
        "current_span_position": 0,
        "tokens_in_span": 0,
        "span_length": 8,  # Default span length
        "current_weights": None,
        "weight_update_queue": weight_update_queue,
        "span_config": span_config or {},
    }

    # Parse span configuration
    if span_state["span_config"]:
        if "span_length" in span_state["span_config"]:
            span_state["span_length"] = span_state["span_config"]["span_length"]
        if "span_position" in span_state["span_config"]:
            span_state["current_span_position"] = span_state["span_config"][
                "span_position"
            ]
    elif weight_update_queue and not weight_update_queue.empty():
        span_config = weight_update_queue.get()
        if isinstance(span_config, dict) and "span_length" in span_config:
            span_state["span_length"] = span_config["span_length"]
            if "span_position" in span_config:
                span_state["current_span_position"] = span_config["span_position"]

    # Initialize model actors
    refs = []
    ensemble_weight_list = []
    for model_actor in model_actors_list:
        refs.append(model_actor.generate_prepare.remote(**kwargs))
        ensemble_weight_list.append(model_actor.get_ensemble_weight.remote())
    ray.get(refs)
    ensemble_weight_list = ray.get(ensemble_weight_list)

    # Set initial weights
    span_state["current_weights"] = ensemble_weight_list

    cached_output_ids = [
        [] for _ in ray.get(model_actors_list[0].get_input_ids.remote())
    ]

    while True:
        # Check whether span weights need to be updated
        if span_state["tokens_in_span"] >= span_state["span_length"]:
            span_state["current_span_position"] += 1
            span_state["tokens_in_span"] = 0

            # Get new weights from queue (if any)
            if (
                span_state["weight_update_queue"]
                and not span_state["weight_update_queue"].empty()
            ):
                new_span_config = span_state["weight_update_queue"].get()
                if isinstance(new_span_config, dict) and "weights" in new_span_config:
                    span_state["current_weights"] = new_span_config["weights"]
                    logger.info(
                        f"Updated span weights at position {span_state['current_span_position']}: {span_state['current_weights']}"
                    )

        # Apply current span weights (consistent within a span)
        weight_update_refs = [
            model_actor.update_ensemble_weight.remote(weight)
            for model_actor, weight in zip(
                model_actors_list, span_state["current_weights"]
            )
        ]
        ray.get(weight_update_refs)

        # Get next token (using current span weights)
        tmp_outputs_refs = [
            model_actor.get_one_token.remote() for model_actor in model_actors_list
        ]

        tmp_outputs, tmp_outputs_times, need_ensemble = check_threshold_ensemble(
            tmp_outputs_refs, primary_index, threshold
        )

        # Process model outputs
        process_and_log_model_outputs(
            tokenizers, model_name_list, tmp_outputs, span_state["current_weights"]
        )

        # Merge token probabilities
        merged_token_ids = merge_and_convert_tokens(
            tmp_outputs,
            tokenizers,
            mapping_matrices,
            vocab_union,
            index_to_vocab,
            special_prefix_tokens_dict,
            byte_mappings_list,
            primary_index,
            threshold,
            need_ensemble,
            tmp_outputs_times,
        )

        # Check early stopping
        cached_output_ids, merged_token_ids = check_until(
            until, cached_output_ids, tokenizers, merged_token_ids
        )

        # Update model states
        refs = []
        for i, model_actor in enumerate(model_actors_list):
            ref = model_actor.update_input_ids_and_model_kwargs.remote(
                next_tokens_list=merged_token_ids[i]
            )
            refs.append(ref)
        ray.get(refs)

        # Synchronize unfinished sequence states
        unfinished_sequences_list = [
            ray.get(model_actor.get_unfinished_sequences.remote())
            for model_actor in model_actors_list
        ]

        synced_unfinished_sequences = synchronize_unfinished_sequences(
            unfinished_sequences_list
        )

        update_refs = [
            model_actor.update_unfinished_sequences.remote(synced_unfinished_sequences)
            for model_actor in model_actors_list
        ]
        ray.get(update_refs)

        # Check whether generation is finished
        finish_refs = [
            model_actor.check_if_stop.remote() for model_actor in model_actors_list
        ]
        finish = any(ray.get(finish_refs))

        if finish:
            break

        # Important: update span token count
        span_state["tokens_in_span"] += 1

    return ray.get(model_actors_list[0].get_input_ids.remote())


# Keep original helper functions unchanged (imported from the original module)
from .rlae_gen_utils import (
    process_and_log_model_outputs,
    synchronize_unfinished_sequences,
    merge_and_convert_tokens,
    check_until,
    check_threshold_ensemble,
    extract_generated_texts,
    update_input_ids_and_model_kwargs,
    get_one_token,
    generate_prepare,
)
