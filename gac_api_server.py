import argparse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils.rlae_gen_call import *
from utils.rlae_gen_utils import *


@asynccontextmanager
async def lifespan(app: FastAPI):

    config_api_server, norm_type_api_server, threshold_api_server = load_yaml_config(
        args.config_path
    )

    global model_actors_list, tokenizers, vocab_union, mapping_matrices, index_to_vocab, special_prefix_tokens_dict, byte_mappings_list, min_max_position_embeddings, model_name_list, primary_index, threshold

    (
        model_actors_list,
        tokenizers,
        vocab_union,
        mapping_matrices,
        index_to_vocab,
        special_prefix_tokens_dict,
        byte_mappings_list,
        min_max_position_embeddings,
        model_name_list,
        primary_index,
        threshold,
    ) = setup_model_actors_and_data(
        config_api_server, norm_type_api_server, threshold_api_server
    )

    yield


parser = argparse.ArgumentParser(
    description="A script that uses a config file for RLAE ensemble."
)
parser.add_argument(
    "--config-path",
    type=str,
    default="example_configs/ppo_span_config.yaml",
    help="Path to the configuration file.",
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="The host address to bind to. Default is 0.0.0.0",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind to. Default is 8000"
)
args = parser.parse_args()

app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    messages_list: List[
        List[Dict]
    ]  # List of messages(conversations) for batch processing
    max_length: Optional[int] = Field(default=None)  # Optional maximum length
    max_new_tokens: Optional[int] = Field(default=50)  # Specifying maximum new tokens
    apply_chat_template: Optional[bool] = Field(default=False)
    # For early stopping
    until: Optional[List[str]] = Field(default=None)
    new_weights: Optional[List[float]] = Field(default=[0.5, 0.5])


def normalize_and_validate_weights(new_weights: Optional[List[float]]) -> List[float]:
    if new_weights is None:
        raise HTTPException(status_code=422, detail="'new_weights' is required.")
    if len(new_weights) != len(model_actors_list):
        raise HTTPException(
            status_code=422,
            detail=f"'new_weights' length must match model count ({len(model_actors_list)}).",
        )
    if any((not isinstance(w, (int, float))) for w in new_weights):
        raise HTTPException(status_code=422, detail="'new_weights' must be numeric.")
    if any(w < 0 for w in new_weights):
        raise HTTPException(
            status_code=422, detail="'new_weights' values must be non-negative."
        )
    total = float(sum(new_weights))
    if total <= 0:
        raise HTTPException(
            status_code=422, detail="Sum of 'new_weights' must be > 0."
        )
    return [float(w) / total for w in new_weights]


@app.get("/status")
async def get_status():
    return {"status": "ready"}


@app.post("/api/generate/")
async def api_generate(request: GenerateRequest):
    chat_list = request.messages_list
    max_length = request.max_length
    max_new_tokens = request.max_new_tokens
    apply_chat_template = request.apply_chat_template
    until = request.until

    length_param = (
        {"max_length": max_length}
        if max_length is not None
        else {"max_new_tokens": max_new_tokens}
    )

    prepare_inputs = [
        model_actor.prepare_inputs_for_model.remote(
            chat_list, min_max_position_embeddings, apply_chat_template
        )
        for model_actor in model_actors_list
    ]
    models_inputs = ray.get(prepare_inputs)
    input_ids_0 = models_inputs[0]

    output = generate_ensemnble_response(
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
        **length_param,
    )

    generated_texts = extract_generated_texts(tokenizers[0], input_ids_0, output)

    logger.info(f"Generated text:{generated_texts}")

    return {"response": generated_texts}


import ray
from ray.util.queue import Queue


@app.post("/api/rl-train/")
async def api_rl_train(request: GenerateRequest):
    chat_list = request.messages_list
    max_length = request.max_length
    max_new_tokens = request.max_new_tokens
    apply_chat_template = request.apply_chat_template
    until = request.until
    new_weights = normalize_and_validate_weights(request.new_weights)

    weight_update_queue = Queue()

    length_param = (
        {"max_length": max_length}
        if max_length is not None
        else {"max_new_tokens": max_new_tokens}
    )

    prepare_inputs = [
        model_actor.prepare_inputs_for_model.remote(
            chat_list, min_max_position_embeddings, apply_chat_template
        )
        for model_actor in model_actors_list
    ]
    models_inputs = ray.get(prepare_inputs)
    input_ids_0 = models_inputs[0]

    weight_update_queue.put(new_weights)

    output = generate_ensemnble_response(
        model_actors_list=model_actors_list,
        weight_update_queue=weight_update_queue,
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
        **length_param,
    )

    generated_texts = extract_generated_texts(tokenizers[0], input_ids_0, output)

    logger.info(f"Generated text:{generated_texts}")

    return {"response": generated_texts}


@app.post("/api/rl-train-span/")
async def api_rl_train_span(request: GenerateRequest):
    # Backward-compatible alias used by span training scripts.
    return await api_rl_train(request)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
