import inspect
import os
import time

import ray
import torch
from accelerate import dispatch_model, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .rlae_gen_utils import *


def get_remote_model_generator_class(num_gpus):
    # Dynamically register the ModelGenerator class as a Ray remote class, specifying the required number of GPUs
    return ray.remote(num_gpus=num_gpus)(UnifiedModelGenerator)


class UnifiedModelGenerator:
    """Model generator that supports both token-level and span-level modes."""

    def __init__(
        self,
        model_path,
        model_name,
        max_memory={0: "80GiB"},
        model_ensemble_weight=1,
        use_cache=True,
        quantization="none",
        span_mode=False,
        span_length=8,
    ):

        quantization_options = {
            "8bit": BitsAndBytesConfig(load_in_8bit=True),
            "4bit": BitsAndBytesConfig(load_in_4bit=True),
            "none": None,
        }

        # Retrieve the appropriate quantization_config
        quantization_config = quantization_options.get(quantization)

        # Raise an error if an invalid quantization option is provided
        if quantization_config is None and quantization != "none":
            raise ValueError(
                f"Invalid quantization value '{quantization}'. Allowed values are: 'none', '8bit', '4bit'."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=model._get_no_split_modules("auto"),
        )

        # https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/modeling_utils.py#L3773
        device_map_kwargs = {"device_map": device_map}
        if "skip_keys" in inspect.signature(dispatch_model).parameters:
            device_map_kwargs["skip_keys"] = model._skip_keys_device_placement

        self.model_name = model_name
        self.model_ensemble_weight = model_ensemble_weight
        self.use_cache = use_cache
        self.span_mode = span_mode
        self.span_length = span_length

        # Load model to GPU
        self.model = dispatch_model(model, **device_map_kwargs)
        if self.model_name in ["Yi-34B-Chat", "Yi-6B-Chat"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", use_fast=False, trust_remote_code=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", trust_remote_code=True
            )

        # Make sure use greedy search
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0

        if (
            isinstance(self.model.generation_config.eos_token_id, list)
            and len(self.model.generation_config.eos_token_id) > 1
        ):
            logger.warning(
                f"For model {self.model_name}, the eos_token_id in generation_config more than one, we only take first one."
            )
            self.model.generation_config.eos_token_id = (
                self.model.generation_config.eos_token_id[0]
            )

        if self.model.generation_config.eos_token_id and (
            self.model.generation_config.eos_token_id != self.tokenizer.eos_token_id
        ):
            logger.warning(
                f"For model {self.model_name}, the eos_token_id is inconsistent between the generation config and the tokenizer ({self.model.generation_config.eos_token_id} and {self.tokenizer.eos_token_id}). We will forcefully set the tokenizer to be consistent with the generation config ({self.model.generation_config.eos_token_id})."
            )
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        if (
            self.model_name == "Starling-LM-7B-alpha"
            and len(self.tokenizer) > self.model.vocab_size
        ):
            logger.warning(
                f"Model {self.model_name} used! You need remove sep_token from tokenizer_config.json because it cause vocab size +1!"
            )

        # Span-mode state tracking
        if self.span_mode:
            self.span_state = {
                "current_span_position": 0,
                "tokens_in_span": 0,
                "span_length": span_length,
                "span_weights_history": [],
            }

    def get_vocab_size(self):
        if len(self.tokenizer.get_vocab()) != self.model.config.vocab_size:
            logger.warning(
                f"For model {self.model_name}, the vocab_size of the tokenizer and model config are not equal! We will create the mapping matrix base on the model config."
            )
        return self.model.config.vocab_size

    def get_ensemble_weight(self):
        return self.model_ensemble_weight

    def update_ensemble_weight(self, new_weight):
        """Update ensemble weight."""
        self.model_ensemble_weight = new_weight

        # In span mode, record weight history
        if self.span_mode and hasattr(self, "span_state"):
            if isinstance(new_weight, (list, np.ndarray)):
                weight_str = f"[{', '.join([f'{w:.3f}' for w in new_weight])}]"
            else:
                weight_str = f"{new_weight:.3f}"

            self.span_state["span_weights_history"].append(
                {
                    "span_position": self.span_state["current_span_position"],
                    "tokens_in_span": self.span_state["tokens_in_span"],
                    "weight": weight_str,
                    "timestamp": time.time(),
                }
            )

            logger.info(
                f"Model {self.model_name} updated weight to {weight_str} "
                f"at span {self.span_state['current_span_position']}"
            )

        return True

    def update_span_state(self, span_position=None, span_length=None):
        """Update span state (only effective in span mode)."""
        if not self.span_mode:
            return

        if span_position is not None:
            self.span_state["current_span_position"] = span_position
        if span_length is not None:
            self.span_state["span_length"] = span_length

    def get_span_info(self):
        """Get current span information (only effective in span mode)."""
        if not self.span_mode:
            return None

        return {
            "current_span_position": self.span_state["current_span_position"],
            "tokens_in_span": self.span_state["tokens_in_span"],
            "span_length": self.span_state["span_length"],
            "current_weight": self.model_ensemble_weight,
        }

    def get_input_ids(self):
        return self.state["input_ids"]

    def check_if_stop(self):
        if self.state["unfinished_sequences"].max() == 0:
            self.state["this_peer_finished"] = True

        # stop if we exceed the maximum length
        if torch.all(
            self.state["stopping_criteria"](
                self.state["input_ids"], self.state["scores"]
            )
        ):
            self.state["this_peer_finished"] = True

        return self.state["this_peer_finished"]

    def update_unfinished_sequences(self, unfinished_sequences):
        self.state["unfinished_sequences"] = unfinished_sequences.to(
            self.state["unfinished_sequences"].device
        )

    def get_unfinished_sequences(self):
        return self.state["unfinished_sequences"]

    def update_input_ids_and_model_kwargs(self, next_tokens_list):
        """Update input IDs and model kwargs, and also update span state."""
        self.state["next_tokens_list"] = next_tokens_list

        # In span mode, update span state
        if self.span_mode and hasattr(self, "span_state"):
            self.span_state["tokens_in_span"] += 1

            # If span boundary is reached, reset counter
            if self.span_state["tokens_in_span"] >= self.span_state["span_length"]:
                self.span_state["tokens_in_span"] = 0
                self.span_state["current_span_position"] += 1
                logger.info(
                    f"Model {self.model_name} moved to span {self.span_state['current_span_position']}"
                )

        (
            self.state["input_ids"],
            self.state["model_kwargs"],
            self.state["unfinished_sequences"],
        ) = update_input_ids_and_model_kwargs(self.model, self.state)

    def get_one_token(self):
        """Get next token and record elapsed time."""
        st = time.time()
        self.state["next_tokens_scores"], self.state["outputs"] = get_one_token(
            self.model, self.state
        )
        time_used = time.time() - st

        # Apply current ensemble weight
        weighted_scores = self.model_ensemble_weight * self.state["next_tokens_scores"]

        return weighted_scores, time_used

    def generate_prepare(self, *args, **kwargs):
        """Prepare generation and initialize span state."""
        self.state = generate_prepare(model=self.model, **self.inputs, **kwargs)
        self.state["model_kwargs"]["use_cache"] = self.use_cache

        # Reset span state in span mode
        if self.span_mode and hasattr(self, "span_state"):
            self.span_state["tokens_in_span"] = 0
            logger.info(f"Model {self.model_name} span state reset for new generation")

    def get_max_position_embeddings(self):
        return self.model.config.max_position_embeddings

    def get_model_name(self):
        return self.model_name

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_inputs_for_model(
        self, chat_list, min_max_position_embeddings=4096, apply_chat_template=False
    ):
        # Calculate the truncation length as 75% of the minimum max_position_embeddings
        truncation_length = int(min_max_position_embeddings * 0.75)
        input_texts = []

        # Apply the chat template and collect the processed text
        for chat in chat_list:
            if apply_chat_template:
                # Assume the tokenizer has an apply_chat_template method
                processed_text = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                processed_text = chat[0]["content"]
            input_texts.append(processed_text)

        self.inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            max_length=truncation_length,
            truncation=True,
        ).to(next(self.model.parameters()).device)

        # In span mode, reset span state
        if self.span_mode and hasattr(self, "span_state"):
            self.span_state["current_span_position"] = 0
            self.span_state["tokens_in_span"] = 0

        return self.inputs.input_ids
