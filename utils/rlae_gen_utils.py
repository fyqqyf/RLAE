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


def generate_ensemnble_response(
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
    weight_update_queue=[0.5, 0.5],
    **kwargs,
):
    # Initiate asynchronous preparation for text generation across multiple model actors.
    # This includes setting up variables like stopping_criteria, etc.
    refs = []
    ensemble_weight_list = []
    for model_actor in model_actors_list:
        refs.append(model_actor.generate_prepare.remote(**kwargs))
        ensemble_weight_list.append(model_actor.get_ensemble_weight.remote())
    ray.get(refs)
    ensemble_weight_list = ray.get(ensemble_weight_list)

    cached_output_ids = [
        [] for _ in ray.get(model_actors_list[0].get_input_ids.remote())
    ]
    while True:
        if not weight_update_queue.empty():
            new_weights = weight_update_queue.get()
            print(new_weights)
            weight_update_refs = [
                model_actor.update_ensemble_weight.remote(new_weight)
                for model_actor, new_weight in zip(model_actors_list, new_weights)
            ]
            # Wait for all weight updates to complete
            ray.get(weight_update_refs)
            # Retrieve updated weights
            ensemble_weight_list = ray.get(
                [
                    model_actor.get_ensemble_weight.remote()
                    for model_actor in model_actors_list
                ]
            )
        # Request each model in the list to asynchronously predict the probability distribution of the next token.
        tmp_outputs_refs = [
            model_actor.get_one_token.remote() for model_actor in model_actors_list
        ]

        tmp_outputs, tmp_outputs_times, need_ensemble = check_threshold_ensemble(
            tmp_outputs_refs, primary_index, threshold
        )

        # This function extracts and logs the token with the highest probability from each model's output.
        process_and_log_model_outputs(
            tokenizers, model_name_list, tmp_outputs, ensemble_weight_list
        )

        # Merge probability distributions from different models to identify a unified token,
        # then map this token to corresponding IDs across models using tokenizer and vocabulary mappings.
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

        # check whether should early stopping
        cached_output_ids, merged_token_ids = check_until(
            until, cached_output_ids, tokenizers, merged_token_ids
        )

        # Update the state required for text generation in each model, such as attention masks,
        # input IDs, and past key-value pairs. This prepares each model for the next step of generation.
        refs = []
        for i, model_actor in enumerate(model_actors_list):
            ref = model_actor.update_input_ids_and_model_kwargs.remote(
                next_tokens_list=merged_token_ids[i]
            )
            refs.append(ref)
        ray.get(refs)

        # Retrieve the list of unfinished sequences from each model to determine if any sentence has finished.
        unfinished_sequences_list = [
            ray.get(model_actor.get_unfinished_sequences.remote())
            for model_actor in model_actors_list
        ]

        # Synchronize the status of unfinished sequences across all models, ensuring consistency in tracking which sentences are still being generated.
        synced_unfinished_sequences = synchronize_unfinished_sequences(
            unfinished_sequences_list
        )

        # Update each model with the synchronized status of unfinished sequences.
        update_refs = [
            model_actor.update_unfinished_sequences.remote(synced_unfinished_sequences)
            for model_actor in model_actors_list
        ]
        ray.get(update_refs)

        # Check across all models to determine if the text generation should stop, i.e., if any model has finished generating its sentence.
        finish_refs = [
            model_actor.check_if_stop.remote() for model_actor in model_actors_list
        ]
        finish = any(
            ray.get(finish_refs)
        )  # Determine if any model signals to stop generation.

        # If any model has completed its sentence, break out of the loop to stop the generation process.
        if finish:
            break

    return ray.get(model_actors_list[0].get_input_ids.remote())


def process_and_log_model_outputs(
    tokenizers, model_name_list, model_outputs, ensemble_weight_list
):

    for output, tokenizer, model_name, ensemble_weight in zip(
        model_outputs, tokenizers, model_name_list, ensemble_weight_list
    ):
        if output is None:
            logger.info(f"Token from Model {model_name}: N/A")
            continue
        # Extract the highest scoring token and its score for each model's output
        max_scores, max_indices = torch.max(output, dim=-1)
        decoded_tokens = [
            tokenizer.decode([idx], skip_special_tokens=False)
            for idx in max_indices.tolist()
        ]
        max_scores_list = [
            round(score.item() / ensemble_weight, 4) for score in max_scores
        ]

        # Log the decoded token, its ID, and confidence score
        logger.info(
            f"Token from Model {model_name}: {decoded_tokens} (token id {max_indices.tolist()}) with Conf {max_scores_list}"
        )


def synchronize_unfinished_sequences(unfinished_sequences_list):

    device = unfinished_sequences_list[0].device

    # Check if the shape of unfinished_sequences is consistent across all states
    first_shape = unfinished_sequences_list[0].shape
    for unfinished_sequences in unfinished_sequences_list:
        if unfinished_sequences.shape != first_shape:
            raise ValueError(
                "All 'unfinished_sequences' tensors must have the same shape."
            )

    # Initialize a tensor filled with 1s, with the same size as unfinished_sequences
    sync_tensor = torch.ones_like(unfinished_sequences_list[0]).to(device)

    # Iterate through all unfinished_sequences to identify which positions need to be set to 1
    for unfinished_sequences in unfinished_sequences_list:
        sync_tensor = torch.logical_and(sync_tensor, unfinished_sequences.to(device))

    # Convert True/False values in sync_tensor to 1/0
    sync_tensor = sync_tensor.long()  # Use .long() to convert True/False to 1/0

    return sync_tensor


def update_input_ids_and_model_kwargs(model, state):

    outputs = state["outputs"]
    input_ids = state["input_ids"]
    next_tokens = state["next_tokens_list"]
    model_kwargs = state["model_kwargs"]
    unfinished_sequences = state["unfinished_sequences"]
    pad_token_id = state["pad_token_id"]
    eos_token_id_tensor = state["eos_token_id_tensor"]

    # Check if pad_token_id is provided
    if pad_token_id is None:
        raise ValueError("pad_token_id must be defined.")

    # Replace next_tokens with pad_token_id where sequences are finished
    next_tokens = [
        tokens if unfinished else [pad_token_id] * len(tokens)
        for tokens, unfinished in zip(next_tokens, unfinished_sequences)
    ]

    # Determine the device of input_ids
    device = input_ids.device

    # Calculate the maximum length after adding next_tokens
    max_length = max([input_ids.shape[1] + len(tokens) for tokens in next_tokens])

    # Pad input_ids and next_tokens to the same length
    padded_input_ids = []
    attention_masks = []  # To store the updated attention masks
    for i, tokens in enumerate(next_tokens):
        # Calculate padding size for input_ids
        input_padding_size = max_length - input_ids.shape[1] - len(tokens)

        # Pad input_ids
        padded_input = torch.cat(
            [
                torch.full(
                    (1, input_padding_size),
                    pad_token_id,
                    dtype=torch.long,
                    device=device,
                ),
                input_ids[i].unsqueeze(0),
                torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
            ],
            dim=1,
        )
        padded_input_ids.append(padded_input)

        # Update the attention mask
        if "attention_mask" in model_kwargs:
            original_attention_mask = model_kwargs["attention_mask"][i]
            updated_attention_mask = torch.cat(
                [
                    torch.zeros(input_padding_size, dtype=torch.long, device=device),
                    original_attention_mask,
                    torch.ones(len(tokens), dtype=torch.long, device=device),
                ]
            )
            attention_masks.append(updated_attention_mask)

    # Convert the list of padded input_ids to a tensor
    padded_input_ids_tensor = torch.cat(padded_input_ids, dim=0)
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    )

    # Update the attention masks in model_kwargs
    if attention_masks:
        model_kwargs["attention_mask"] = torch.stack(attention_masks)
        model_kwargs["cache_position"] = torch.tensor(
            [model_kwargs["attention_mask"].shape[1] - 1],
            dtype=torch.int64,
            device=model_kwargs["attention_mask"].device,
        )

    # Update model_kwargs, set past_key_values to None if any sequence has more than one token to add
    if any(len(tokens) > 1 for tokens in next_tokens):
        model_kwargs["past_key_values"] = None

        # Find the index of the first non-pad token for each sequence
        first_non_pad_indices = [
            (
                input_id.ne(pad_token_id).nonzero(as_tuple=True)[0][0].item()
                if pad_token_id in input_id
                else 0
            )
            for input_id in padded_input_ids_tensor
        ]

        # Calculate the maximum number of leading pads that can be removed (minimum index of the first non-pad token)
        max_pads_to_remove = min(first_non_pad_indices)

        # Remove the unnecessary leading pads
        if max_pads_to_remove > 0:

            padded_input_ids_tensor = padded_input_ids_tensor[:, max_pads_to_remove:]
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"][
                    :, max_pads_to_remove:
                ]

    # Update unfinished_sequences based on eos_token_id
    if eos_token_id_tensor is not None:
        for i, tokens in enumerate(next_tokens):
            for token in tokens:
                unfinished_sequences[i] = unfinished_sequences[i] & (
                    token != eos_token_id_tensor
                )

    return padded_input_ids_tensor, model_kwargs, unfinished_sequences


def check_byte_mappings(tokenizer):

    vocab = tokenizer.get_vocab()
    g_prefix_count = sum(token.startswith("Ġ") for token in vocab)
    u_prefix_count = sum(token.startswith("▁") for token in vocab)

    byte_mapping = {}

    # For BBPE, handle bytes from 0x00 to 0x7F
    if g_prefix_count > u_prefix_count:
        for byte_val in range(128):  # Limit to 0x00 to 0x7F
            byte_char = chr(byte_val)
            token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(byte_char))[0]
            hex_token = f"<0x{byte_val:02X}>"
            byte_mapping[hex_token] = token_id
    else:
        # For non-BBPE, attempt to find a direct mapping in vocab
        for byte_val in range(256):
            hex_token = f"<0x{byte_val:02X}>"
            # For cases like "\t" being replaced in vocab
            if hex_token == "<0x09>" and hex_token not in vocab:
                continue
            if hex_token not in vocab:
                raise ValueError(
                    f"Token {hex_token} not found in tokenizer's vocabulary."
                )
            byte_mapping[hex_token] = vocab[hex_token]

    return byte_mapping


def get_vocab_union_and_mapping(tokenizers):

    # Initialize a set to store all tokens
    vocab_union = set()
    # Initialize a list to store the mappings for each tokenizer
    tokenizers_mapping = []
    byte_mappings_list = []

    # First, add '<0x00>' to '<0xFF>'
    for byte_val in range(256):
        vocab_union.add(f"<0x{byte_val:02X}>")

    # Process each tokenizer separately
    for tokenizer in tokenizers:
        vocab = tokenizer.get_vocab()
        token_set = set()
        mapping = {}

        # Check and record each tokenizer's mapping for '<0x00>' to '<0xFF>'
        byte_mapping = check_byte_mappings(tokenizer)
        byte_mappings_list.append(byte_mapping)

        if len(byte_mapping) == 128:
            logger.warning(
                "BBPE detected. Please be cautious in usage as currently it only supports applications such as multiple-choice questions eg.(A)"
            )

        # Remove the existing mappings for '<0x00>' to '<0xFF>'
        for hex_token, token_id in byte_mapping.items():
            # Remove tokens from the vocabulary whose token IDs appear in the byte_mapping
            actual_tokens = [token for token, id in vocab.items() if id == token_id]

            if len(actual_tokens) != 1:
                # Raise an error if more than one matching token is found
                raise ValueError(
                    f"Multiple tokens/ Zero token found for token ID {token_id} in tokenizer's vocabulary."
                )
            del vocab[actual_tokens[0]]

        # Detect usage of 'Ġ' and '▁'
        g_prefix_count = sum(token.startswith("Ġ") for token in vocab)
        u_prefix_count = sum(token.startswith("▁") for token in vocab)

        # Process tokens based on prefix type
        if g_prefix_count > u_prefix_count:
            # Handle tokens starting with 'Ġ'
            for token, token_id in vocab.items():
                processed_token = token.replace("Ġ", " ").replace("Ċ", "\n")
                token_set.add(processed_token)
                mapping[token_id] = processed_token
        else:
            # Handle tokens starting with '▁'
            for token, token_id in vocab.items():
                if token.startswith("▁"):
                    processed_token = token.replace("▁", " ")
                else:
                    # For tokens without '▁', use the decode method
                    processed_token = token  # tokenizer.decode([token_id])
                token_set.add(processed_token)
                mapping[token_id] = processed_token

        # Merge into the total vocab_union
        vocab_union = vocab_union.union(token_set)
        # Append the mapping for this tokenizer to the list
        tokenizers_mapping.append(mapping)

    # Generate a mapping for each token in the union to a unique index
    vocab_to_index = {token: i for i, token in enumerate(vocab_union)}

    # Convert vocab_to_index to index_to_vocab
    index_to_vocab = {index: token for token, index in vocab_to_index.items()}

    for tokenizer, byte_mapping, mapping in zip(
        tokenizers, byte_mappings_list, tokenizers_mapping
    ):
        # Update the mappings for each tokenizer to map to the index in the unified vocab
        for token_id, token in mapping.items():
            mapping[token_id] = vocab_to_index[token]

        # Define the extended mapping dictionary
        bbpe_mapping = {
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x30, 0x3A)
            },  # mapping '0' to '9'
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x41, 0x5B)
            },  # mapping 'A' to 'Z'
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x61, 0x7B)
            },  # mapping 'a' to 'z'
        }

        # Add the '<0x00>' to '<0xFF>' mappings for each tokenizer
        for hex_token, original_token_id in byte_mapping.items():
            # First, check the original conditions
            if (
                not all(len(bm) == 128 for bm in byte_mappings_list)
                and len(byte_mapping) == 128
            ):
                # Apply special handling to the specified characters
                if hex_token in bbpe_mapping:
                    logger.warning(
                        f"We force-mapped the BBPE {hex_token} to {bbpe_mapping[hex_token]} in union vocab"
                    )
                    mapping[original_token_id] = vocab_to_index[bbpe_mapping[hex_token]]
                    continue
            mapping[original_token_id] = vocab_to_index[hex_token]

    return vocab_union, tokenizers_mapping, index_to_vocab, byte_mappings_list


def create_mapping_matrix(mapping, union_vocab_size, model_vocab_size):

    if model_vocab_size == 151646:
        logger.warning(
            "The qwen series has been detected, where the length of tokenizer.get_vocab() and the vocab_size in the model config are inconsistent. We have forcefully set it to the latter. https://github.com/QwenLM/Qwen3/issues/29"
        )
        model_vocab_size = 151936

    indices = []  # Store the coordinates of non-zero elements
    values = []  # Non-zero values, typically 1 for a mapping matrix

    for model_token_id, unified_token_index in mapping.items():
        indices.append([model_token_id, unified_token_index])  # (rows, cols)
        values.append(1.0)

    # Convert to a tensor suitable for COO format
    indices = torch.tensor(
        indices, dtype=torch.long
    ).t()  # Transpose to meet (rows, cols)
    values = torch.tensor(values, dtype=torch.float)

    # Create a sparse tensor
    size = torch.Size([model_vocab_size, union_vocab_size])
    mapping_matrix = torch.sparse_coo_tensor(indices, values, size, device="cuda")

    return mapping_matrix


def check_until(until, cached_batch_output_ids, tokenizers, merged_token_ids):

    if len(cached_batch_output_ids) != len(merged_token_ids[0]):
        raise ValueError(
            f"len(cached_batch_output_ids):{len(cached_batch_output_ids)} != len(merged_token_ids[0]): {len(merged_token_ids[0])}"
        )
    for i, _ in enumerate(cached_batch_output_ids):
        cached_batch_output_ids[i] = cached_batch_output_ids[i] + merged_token_ids[0][i]
        tmp_text = tokenizers[0].decode(cached_batch_output_ids[i])

        if until:
            for stop_txt in until:
                if stop_txt in tmp_text:
                    for j, tokenizer in enumerate(tokenizers):
                        merged_token_ids[j][i] = merged_token_ids[j][i] + [
                            tokenizer.eos_token_id
                        ]
                    break
    return cached_batch_output_ids, merged_token_ids


def check_threshold_ensemble(tmp_outputs_refs, primary_index, threshold):

    if primary_index == -1:
        tmp = ray.get(tmp_outputs_refs)
        outputs = [t[0] for t in tmp]
        outputs_times = [t[1] for t in tmp]
        need_ensemble = True
    else:
        primary_model_outputs, primary_model_outputs_times = ray.get(
            tmp_outputs_refs[primary_index]
        )
        if primary_model_outputs.shape[0] != 1:
            raise ValueError(
                "For thresholded ensemble, we only support batch size is 1."
            )
        max_probs, _ = torch.max(primary_model_outputs, dim=1)  # Get max value

        if max_probs.item() > threshold:
            for i, ref in enumerate(tmp_outputs_refs):
                if i != primary_index:
                    ray.cancel(ref)
            outputs = [None] * len(tmp_outputs_refs)
            outputs[primary_index] = primary_model_outputs
            outputs_times = [primary_model_outputs_times] * len(tmp_outputs_refs)
            need_ensemble = False
        else:
            tmp = ray.get(tmp_outputs_refs)
            outputs = [t[0] for t in tmp]
            outputs_times = [t[1] for t in tmp]
            need_ensemble = True

    return outputs, outputs_times, need_ensemble


def merge_and_convert_tokens(
    outputs,
    tokenizers,
    mapping_matrices,
    vocab_union,
    index_to_vocab,
    special_prefix_token,
    byte_mappings_list,
    primary_index,
    threshold,
    need_ensemble,
    tmp_outputs_times,
):

    eos_token_list = [tokenizer.eos_token for tokenizer in tokenizers]
    eos_token_list.extend(["<|end_of_text|>", "<|endoftext|>", "<|im_end|>", "<|end|>"])

    for i, output in enumerate(outputs):
        if need_ensemble:
            if output is None:
                raise ValueError(
                    "We detect a probability vector of None, which need to excute ensemble!"
                )
        else:
            if output is not None and i != primary_index:
                raise ValueError(
                    "We detect a probability vector from non-primary model, but no ensemble excuted!"
                )

    # Initialize the merged probability vector and store it on the GPU
    if primary_index == -1:
        merged_probs = torch.zeros(
            (outputs[0].size(0), len(vocab_union)), device="cuda"
        )
    else:
        # Now we only support batch size = 1 for thresholded ensemble
        merged_probs = torch.zeros(
            (outputs[primary_index].size(0), len(vocab_union)), device="cuda"
        )

    if need_ensemble:
        for output, mapping_matrix in zip(outputs, mapping_matrices):
            # Evert outputs of all models will be mapped
            transformed_probs = torch.sparse.mm(output, mapping_matrix)
            merged_probs += transformed_probs
    else:
        # Only process the output at the primary_index
        transformed_probs = torch.sparse.mm(
            outputs[primary_index], mapping_matrices[primary_index]
        )
        merged_probs += transformed_probs
        logger.info("RLAE do not ensemble in this step.")

    max_token_indices = torch.argmax(merged_probs, dim=1)
    max_tokens = [index_to_vocab[index.item()] for index in max_token_indices]
    logger.info(f"Token chosen by RLAE: {str(max_tokens)}\n")

    # Convert to token IDs for each tokenizer
    batch_token_ids = [
        [] for _ in range(len(tokenizers))
    ]  # Initialize list for each model
    for i, tokenizer in enumerate(tokenizers):
        for token in max_tokens:
            if token in eos_token_list:
                token_id = [tokenizer.eos_token_id]
            else:
                # Convert token to corresponding tokenizer's token IDs using special_prefix_token
                token_id = get_token_ids(
                    tokenizer,
                    token,
                    special_prefix_token[tokenizer],
                    byte_mappings_list[i],
                )

            batch_token_ids[i].append(token_id)  # Append token IDs for each batch

    return batch_token_ids


def get_token_ids(tokenizer, token, special_prefix_token, byte_mapping):

    # Check if the token is a standard byte representation and return its token ID if found
    if token in byte_mapping:
        return [byte_mapping[token]]

    if byte_mapping != 128:
        prefix_tokens = [special_prefix_token, ";"]

        for prefix_token in prefix_tokens:
            # Tokenize individually
            token_id_list1 = tokenizer.encode(prefix_token, add_special_tokens=False)

            # Tokenize doubled token
            token_id_list2 = tokenizer.encode(
                prefix_token + token, add_special_tokens=False
            )

            # Check if the start of token_id_list2 matches token_id_list1
            if token_id_list2[: len(token_id_list1)] == token_id_list1:
                result = token_id_list2[len(token_id_list1) :]
                if result:
                    return result

        # If tokenization doesn't match as expected with any prefix token, return the token IDs for 'token'
        logger.warning(f"Warning: Token '{token}' may not be tokenized as expected.")
    return tokenizer.encode(token, add_special_tokens=False)


def find_special_underscore_token(tokenizer):

    # get tokenizer vocab
    vocab = tokenizer.get_vocab()

    # Count tokens that start with 'Ġ' and '▁'
    count_prefix_G = sum(1 for token in vocab if token.startswith("Ġ"))
    count_prefix_underscore = sum(1 for token in vocab if token.startswith("▁"))

    # Return an empty string if 'Ġ' tokens are more frequent
    if count_prefix_G > count_prefix_underscore:
        return ""

    # Filter tokens that start with '▁'
    underscore_tokens = [
        token for token in vocab if token.startswith("▁") and token != "▁"
    ]

    # Filter tokens that meet the criteria
    special_tokens = []
    for token in tqdm(underscore_tokens, desc="Analyzing tokens"):
        cleaned_token = token[1:]  # remove '▁'

        # Ensure the token is not part of another token, contains no additional tokens besides the first '▁',
        # has no multiple '▁', and is not a space after removing '▁'
        if (
            not any(
                token in other_token
                for other_token in underscore_tokens
                if other_token != token
            )
            and token.count("▁") == 1
            and cleaned_token.strip() != ""
        ):
            special_tokens.append(cleaned_token)

    # Raise an error if no token meets the criteria
    if not special_tokens:
        raise ValueError("No special underscore token found that meets the criteria.")

    # Return the shortest token to ensure consistency
    return min(special_tokens, key=lambda x: (len(x), x))


def get_special_prefix_tokens_for_all(tokenizers):

    # Initialize an empty dictionary to store the results
    special_prefix_tokens = {}

    # Iterate through the list of tokenizers
    for tokenizer in tokenizers:
        if tokenizer.vocab_size == 256000:
            logger.info("gemma-it detected, use '¢' as special_prefix_token")
            special_prefix_tokens[tokenizer] = "¢"
            continue
        # Get the special prefix token for each tokenizer
        token = find_special_underscore_token(tokenizer)
        # Store the tokenizer and its special prefix token in the dictionary
        special_prefix_tokens[tokenizer] = token
    return special_prefix_tokens


def greedy_search(
    model,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else model.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else model.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    output_scores = (
        output_scores
        if output_scores is not None
        else model.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else model.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else model.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else model.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    # decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    # cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    # decoder_hidden_states = (
    #     () if (return_dict_in_generate and output_hidden_states) else None
    # )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    # if return_dict_in_generate and model.config.is_encoder_decoder:
    #     encoder_attentions = (
    #         model_kwargs["encoder_outputs"].get("attentions")
    #         if output_attentions
    #         else None
    #     )
    #     encoder_hidden_states = (
    #         model_kwargs["encoder_outputs"].get("hidden_states")
    #         if output_hidden_states
    #         else None
    #     )

    model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
    if model.config.is_encoder_decoder:
        raise Exception("We only support decorder arch!")

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    this_peer_finished = False  # used by synced_gpus only

    return {
        "input_ids": input_ids,
        "model_kwargs": model_kwargs,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "stopping_criteria": stopping_criteria,
        "logits_processor": logits_processor,
        "scores": scores,
        "pad_token_id": pad_token_id,
        "eos_token_id_tensor": eos_token_id_tensor,
        "unfinished_sequences": unfinished_sequences,
        "this_peer_finished": this_peer_finished,
    }


def get_one_token(model, state):

    input_ids = state["input_ids"]
    model_kwargs = state["model_kwargs"]
    output_attentions = state["output_attentions"]
    output_hidden_states = state["output_hidden_states"]
    logits_processor = state["logits_processor"]

    # prepare model inputs
    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # disable kv cache for speed testing
    # model_inputs['use_cache'] = False
    # model_inputs['past_key_values'] = None

    with torch.no_grad():
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    next_token_logits = outputs.logits[:, -1, :]

    # pre-process distribution
    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    # Apply softmax to the scores
    next_tokens_scores = F.softmax(next_tokens_scores, dim=-1)

    return next_tokens_scores, outputs


def generate_prepare(
    model,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    model._validate_model_class()
    tokenizer = kwargs.pop(
        "tokenizer", None
    )  # Pull this out first, we only use it for stopping criteria
    generation_config, model_kwargs = model._prepare_generation_config(
        generation_config, **kwargs
    )
    model._validate_model_kwargs(model_kwargs.copy())
    model._validate_assistant(assistant_model)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(model.forward).parameters.keys()
    )
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    model._prepare_special_tokens(
        generation_config, kwargs_has_attention_mask, device=device
    )

    # decoder-only models must use left-padding for batched generation.
    if not model.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor)
            > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    if (
        not kwargs_has_attention_mask
        and requires_attention_mask
        and accepts_attention_mask
    ):
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor,
            generation_config._pad_token_tensor,
            generation_config._eos_token_tensor,
        )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = (
            inputs_tensor
            if model_input_name == "input_ids"
            else model_kwargs.pop("input_ids")
        )

    if generation_config.token_healing:
        input_ids = model.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    has_default_min_length = (
        kwargs.get("min_length") is None and generation_config.min_length is not None
    )
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    use_dynamic_cache_by_default = False
    if "mamba" in model.__class__.__name__.lower():
        cache_name = "cache_params"
    else:
        cache_name = "past_key_values"

    # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
    # which is only supported in dynamic caches atm
    if (
        assistant_model is not None
        and generation_config.cache_implementation is not None
        and model._supports_default_dynamic_cache()
    ):
        logger.warning_once(
            "An assistant model is provided, using a dynamic cache instead of a cache of type="
            f"'{generation_config.cache_implementation}'."
        )
        generation_config.cache_implementation = None

    if (model_kwargs.get(cache_name) is not None) and is_torchdynamo_compiling():
        raise ValueError(
            "Passing `past_key_values` is not supported when compiling `model.generate` with torch.compile -- you "
            "may get incorrect outputs. Please compile `model.forward` only or use the `cache_implementation` "
            "input argument."
        )

    if generation_config.cache_implementation is not None and (
        model_kwargs.get(cache_name) is not None
    ):
        raise ValueError(
            f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
            "Cache object) is unsupported. Please use only one of the two."
        )
    elif generation_config.cache_implementation is not None:
        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if (
                generation_config.cache_implementation == "static"
                and not model._supports_static_cache
            ):
                raise ValueError(
                    "This model does not support `cache_implementation='static'`. Please check the following "
                    "issue: https://github.com/huggingface/transformers/issues/28981"
                )
            model_kwargs[cache_name] = model._get_cache(
                cache_implementation=generation_config.cache_implementation,
                max_batch_size=generation_config.num_beams
                * generation_config.num_return_sequences
                * batch_size,
                max_cache_len=generation_config.max_length,
                device=device,
                model_kwargs=model_kwargs,
            )
        elif generation_config.cache_implementation == "quantized":
            if not model._supports_quantized_cache:
                raise ValueError(
                    "This model does not support the quantized cache. If you want your model to support quantized "
                    "cache, please open an issue."
                )

            cache_config = (
                generation_config.cache_config
                if generation_config.cache_config is not None
                else QuantizedCacheConfig()
            )
            cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

            if cache_config.backend == "quanto" and not is_quanto_available():
                raise ImportError(
                    "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                    "Please install it via  with `pip install quanto`"
                )
            elif cache_config.backend == "HQQ" and not is_hqq_available():
                raise ImportError(
                    "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                    "Please install it via  with `pip install hqq`"
                )

            model_kwargs[cache_name] = cache_class(cache_config)
        elif generation_config.cache_implementation == "offloaded":
            model_kwargs[cache_name] = OffloadedCache()
    # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
    # keeps copying the cache thus using much more memory
    elif (
        generation_config.cache_implementation is None
        and model._supports_default_dynamic_cache()
    ):
        past = model_kwargs.get(cache_name, None)
        requires_cross_attention_cache = (
            model.config.is_encoder_decoder
            or model_kwargs.get("encoder_outputs") is not None
        )
        if past is None:
            model_kwargs[cache_name] = (
                DynamicCache()
                if not requires_cross_attention_cache
                else EncoderDecoderCache(DynamicCache(), DynamicCache())
            )
            use_dynamic_cache_by_default = True
        elif isinstance(past, tuple):
            model_kwargs[cache_name] = (
                DynamicCache.from_legacy_cache(past)
                if not requires_cross_attention_cache
                else EncoderDecoderCache.from_legacy_cache(past)
            )
            use_dynamic_cache_by_default = True

    model._validate_generated_length(
        generation_config, input_ids_length, has_default_max_length
    )

    # 7. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        tokenizer=tokenizer,
        **kwargs,
    )

    # 11. run greedy search
    return greedy_search(
        model,
        input_ids,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        output_scores=generation_config.output_scores,
        return_dict_in_generate=generation_config.return_dict_in_generate,
        synced_gpus=synced_gpus,
        streamer=streamer,
        **model_kwargs,
    )
