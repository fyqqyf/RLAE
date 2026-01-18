import argparse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils.rlae_gen_call import *
from utils.rlae_gen_utils import *
from utils.span_config_manager import SpanConfigManager, load_span_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global model_actors_list, tokenizers, vocab_union, mapping_matrices, index_to_vocab, special_prefix_tokens_dict, byte_mappings_list, min_max_position_embeddings, model_name_list, primary_index, threshold, span_config_manager

    logger.info("🚀 Starting Span-level Ensemble API Server")

    # Load configuration
    try:
        if args.span_config:
            logger.info(f"📋 Loading span configuration: {args.span_config}")
            span_config = load_span_config(args.span_config)
            config_api_server = span_config.get("CONFIG_API_SERVER", [])
            norm_type_api_server = span_config.get("NORM_TYPE_API_SERVER", "average")
            threshold_api_server = span_config.get("THRESHOLD_API_SERVER", 1.0)

            # Get span-specific configuration
            span_level_config = span_config.get("SPAN_LEVEL_CONFIG", {})
            app.state.span_level_config = span_level_config

            logger.info(
                f"✅ Span-level config loaded: enabled={span_level_config.get('enabled', False)}"
            )
            logger.info(
                f"📏 Default span length: {span_level_config.get('default_span_length', 8)}"
            )

        else:
            # Use standard configuration
            logger.info("📋 Loading standard configuration")
            config_api_server, norm_type_api_server, threshold_api_server = (
                load_yaml_config(args.config_path)
            )
            app.state.span_level_config = {"enabled": args.span_mode}

        # Set up models and data
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

        # Create configuration manager
        span_config_manager = SpanConfigManager()

        logger.info("✅ Model setup completed")
        logger.info(f"🔢 Total models: {len(model_actors_list)}")
        logger.info(f"🎯 Primary model index: {primary_index}")
        logger.info(f"📊 Threshold: {threshold}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        raise

    yield

    logger.info("🛑 Shutting down Span-level Ensemble API Server")


# Argument parsing
parser = argparse.ArgumentParser(
    description="Span-level Ensemble RLAE API Server with Enhanced Configuration Support"
)
parser.add_argument(
    "--config-path",
    type=str,
    default="example_configs/example_thresholded_ensemble.yaml",
    help="Path to the standard configuration file.",
)
parser.add_argument(
    "--span-config",
    type=str,
    help="Path to span-level configuration file (e.g., span_ensemble_config.yaml)",
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
parser.add_argument(
    "--span-mode", action="store_true", help="Enable span-level mode by default"
)
args = parser.parse_args()

app = FastAPI(
    title="Span-level Ensemble RLAE API",
    description="Enhanced API server supporting both token-level and span-level ensemble",
    version="2.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    """Enhanced request model with span-level configuration support."""

    messages_list: List[
        List[Dict]
    ]  # List of messages(conversations) for batch processing
    max_length: Optional[int] = Field(default=None)  # Optional maximum length
    max_new_tokens: Optional[int] = Field(default=50)  # Specifying maximum new tokens
    apply_chat_template: Optional[bool] = Field(default=False)
    # For early stopping
    until: Optional[List[str]] = Field(default=None)
    new_weights: Optional[List[float]] = Field(default=[0.5, 0.5])
    # Span-level ensemble parameters
    span_position: Optional[int] = Field(default=0)  # Current span position
    span_length: Optional[int] = Field(default=8)  # Length of each span
    span_mode: Optional[bool] = Field(default=None)  # Override server span mode
    # Advanced span configuration
    span_config_override: Optional[Dict] = Field(default=None)  # Override span settings


@app.get("/status")
async def get_status():
    """Get server status."""
    return {
        "status": "ready",
        "span_mode": getattr(app.state, "span_level_config", {}).get(
            "enabled", args.span_mode
        ),
        "span_config": getattr(app.state, "span_level_config", {}),
        "models_loaded": (
            len(model_actors_list) if "model_actors_list" in globals() else 0
        ),
    }


@app.get("/config")
async def get_config():
    """Get current configuration."""
    try:
        config_info = {
            "span_mode": getattr(app.state, "span_level_config", {}).get(
                "enabled", args.span_mode
            ),
            "span_config": getattr(app.state, "span_level_config", {}),
            "models": model_name_list if "model_name_list" in globals() else [],
            "primary_model_index": (
                primary_index if "primary_index" in globals() else -1
            ),
            "threshold": threshold if "threshold" in globals() else 1.0,
        }
        return config_info
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {e}")


@app.post("/api/generate/")
async def api_generate(request: GenerateRequest):
    """General generation endpoint with span-level support."""
    try:
        chat_list = request.messages_list
        max_length = request.max_length
        max_new_tokens = request.max_new_tokens
        apply_chat_template = request.apply_chat_template
        until = request.until

        # Determine whether to use span mode
        use_span_mode = (
            request.span_mode
            if request.span_mode is not None
            else getattr(app.state, "span_level_config", {}).get(
                "enabled", args.span_mode
            )
        )

        logger.info(f"📝 Generate request - span_mode: {use_span_mode}")

        length_param = (
            {"max_length": max_length}
            if max_length is not None
            else {"max_new_tokens": max_new_tokens}
        )

        # Prepare model inputs
        prepare_inputs = [
            model_actor.prepare_inputs_for_model.remote(
                chat_list, min_max_position_embeddings, apply_chat_template
            )
            for model_actor in model_actors_list
        ]
        models_inputs = ray.get(prepare_inputs)
        input_ids_0 = models_inputs[0]

        if use_span_mode:
            # Span-level generation
            logger.info("🔧 Using span-level ensemble generation")
            from utils.unified_rlae_gen_utils import generate_ensemnble_response_unified

            # Prepare span configuration
            span_config = {
                "span_length": request.span_length,
                "span_position": request.span_position,
            }
            if request.span_config_override:
                span_config.update(request.span_config_override)

            output = generate_ensemnble_response_unified(
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
                span_mode=True,
                span_config=span_config,
                **length_param,
            )
        else:
            # Token-level generation (original logic)
            logger.info("🔧 Using token-level ensemble generation")
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
        logger.info(f"✅ Generated text: {generated_texts}")

        return {"response": generated_texts}

    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")


import ray
from ray.util.queue import Queue


@app.post("/api/rl-train/")
async def api_rl_train(request: GenerateRequest):
    """RL training endpoint with span-level support."""
    try:
        chat_list = request.messages_list
        max_length = request.max_length
        max_new_tokens = request.max_new_tokens
        apply_chat_template = request.apply_chat_template
        until = request.until
        new_weights = request.new_weights

        # Determine whether to use span mode
        use_span_mode = (
            request.span_mode
            if request.span_mode is not None
            else getattr(app.state, "span_level_config", {}).get(
                "enabled", args.span_mode
            )
        )

        logger.info(
            f"🎯 RL Train request - span_mode: {use_span_mode}, span_position: {request.span_position}"
        )

        weight_update_queue = Queue()

        length_param = (
            {"max_length": max_length}
            if max_length is not None
            else {"max_new_tokens": max_new_tokens}
        )

        # Prepare model inputs
        prepare_inputs = [
            model_actor.prepare_inputs_for_model.remote(
                chat_list, min_max_position_embeddings, apply_chat_template
            )
            for model_actor in model_actors_list
        ]
        models_inputs = ray.get(prepare_inputs)
        input_ids_0 = models_inputs[0]

        if use_span_mode:
            # Span-level training mode
            logger.info("🔧 Using span-level RL training")
            from utils.unified_rlae_gen_utils import generate_ensemnble_response_unified

            # Prepare span weight configuration
            span_weights = {
                "weights": new_weights,
                "span_position": request.span_position,
                "span_length": request.span_length,
            }

            if request.span_config_override:
                span_weights.update(request.span_config_override)

            weight_update_queue.put(span_weights)

            output = generate_ensemnble_response_unified(
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
                span_mode=True,
                span_config=span_weights,
                **length_param,
            )
        else:
            # Token-level training mode (original logic)
            logger.info("🔧 Using token-level RL training")
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
        logger.info(f"✅ RL Training response: {generated_texts}")

        return {"response": generated_texts}

    except Exception as e:
        logger.error(f"❌ RL Training error: {e}")
        raise HTTPException(status_code=500, detail=f"RL Training error: {e}")


@app.post("/api/validate-config/")
async def validate_config_endpoint(config_data: Dict):
    """Validate configuration data."""
    try:
        from utils.span_config_manager import SpanConfigValidator

        validator = SpanConfigValidator()
        is_valid = validator.validate_config(config_data)

        if is_valid:
            return {"valid": True, "message": "Configuration is valid"}
        else:
            return {"valid": False, "message": "Configuration validation failed"}

    except Exception as e:
        logger.error(f"❌ Config validation error: {e}")
        return {"valid": False, "message": f"Validation error: {e}"}


@app.get("/api/span-presets/")
async def get_span_presets():
    """Get preset configurations for span lengths."""
    presets = {
        "short_text": {
            "description": "Short text generation (< 50 tokens)",
            "span_length": 4,
            "recommended_models": ["small", "medium"],
        },
        "medium_text": {
            "description": "Medium text generation (50-200 tokens)",
            "span_length": 8,
            "recommended_models": ["medium", "large"],
        },
        "long_text": {
            "description": "Long text generation (> 200 tokens)",
            "span_length": 16,
            "recommended_models": ["large"],
        },
        "very_long_text": {
            "description": "Very long text generation (> 500 tokens)",
            "span_length": 32,
            "recommended_models": ["large", "xl"],
        },
    }

    return {"presets": presets}


if __name__ == "__main__":
    logger.info(f"🚀 Starting server on {args.host}:{args.port}")
    logger.info(
        f"📋 Configuration: span_mode={args.span_mode}, span_config={args.span_config}"
    )

    uvicorn.run(app, host=args.host, port=args.port)
