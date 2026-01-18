import yaml
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpanConfigValidator:
    """Span-level configuration file validator."""

    def __init__(self):
        self.required_fields = {
            "NORM_TYPE_API_SERVER": str,
            "THRESHOLD_API_SERVER": (int, float),
            "CONFIG_API_SERVER": list,
        }

        self.span_specific_fields = {
            "SPAN_LEVEL_CONFIG": dict,
            "TRAINING_CONFIG": dict,
            "EVALUATION_CONFIG": dict,
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        try:
            # Check required fields
            for field, field_type in self.required_fields.items():
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")

                if not isinstance(config[field], field_type):
                    raise ValueError(f"Field {field} must be of type {field_type}")

            # Check span-specific fields (if present)
            if "SPAN_LEVEL_CONFIG" in config:
                self._validate_span_config(config["SPAN_LEVEL_CONFIG"])

            if "TRAINING_CONFIG" in config:
                self._validate_training_config(config["TRAINING_CONFIG"])

            if "EVALUATION_CONFIG" in config:
                self._validate_evaluation_config(config["EVALUATION_CONFIG"])

            # Check model configurations
            self._validate_model_configs(config.get("CONFIG_API_SERVER", []))

            logger.info("✅ Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False

    def _validate_span_config(self, span_config: Dict[str, Any]):
        """Validate span-level configuration."""
        required_span_fields = {
            "enabled": bool,
            "default_span_length": int,
            "max_span_position": int,
            "weight_update_frequency": str,
        }

        for field, field_type in required_span_fields.items():
            if field not in span_config:
                raise ValueError(f"Missing span config field: {field}")

            if not isinstance(span_config[field], field_type):
                raise ValueError(
                    f"Span config field {field} must be of type {field_type}"
                )

        # Validate span length range
        span_length = span_config.get("default_span_length", 0)
        if span_length <= 0 or span_length > 128:
            raise ValueError(
                f"Span length must be between 1 and 128, got {span_length}"
            )

        # Validate weight update frequency
        valid_frequencies = ["span", "token"]
        if span_config.get("weight_update_frequency") not in valid_frequencies:
            raise ValueError(
                f"Weight update frequency must be one of {valid_frequencies}"
            )

    def _validate_training_config(self, training_config: Dict[str, Any]):
        """Validate training configuration."""
        if "algorithm" not in training_config:
            raise ValueError("Missing training algorithm specification")

        valid_algorithms = ["ppo", "mappo"]
        if training_config["algorithm"] not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")

        # Validate hyperparameters
        if "hyperparameters" in training_config:
            hyperparams = training_config["hyperparameters"]

            # Validate learning rate
            if "learning_rate" in hyperparams:
                lr = hyperparams["learning_rate"]
                if lr <= 0 or lr > 1:
                    raise ValueError(f"Learning rate must be between 0 and 1, got {lr}")

            # Validate batch size
            if "batch_size" in hyperparams:
                batch_size = hyperparams["batch_size"]
                if batch_size <= 0:
                    raise ValueError(f"Batch size must be positive, got {batch_size}")

            # Validate span length
            if "span_length" in hyperparams:
                span_length = hyperparams["span_length"]
                if span_length <= 0 or span_length > 128:
                    raise ValueError(
                        f"Span length must be between 1 and 128, got {span_length}"
                    )

    def _validate_evaluation_config(self, eval_config: Dict[str, Any]):
        """Validate evaluation configuration."""
        if "metrics" in eval_config:
            if not isinstance(eval_config["metrics"], list):
                raise ValueError("Evaluation metrics must be a list")

        if "span_lengths" in eval_config:
            span_lengths = eval_config["span_lengths"]
            if not isinstance(span_lengths, list):
                raise ValueError("Span lengths must be a list")

            for length in span_lengths:
                if not isinstance(length, int) or length <= 0:
                    raise ValueError(f"Invalid span length: {length}")

    def _validate_model_configs(self, model_configs: List[Dict[str, Any]]):
        """Validate model configurations."""
        if not model_configs:
            raise ValueError("At least one model configuration is required")

        for i, model_config in enumerate(model_configs):
            required_model_fields = [
                "weight",
                "name",
                "score",
                "priority",
                "quantization",
            ]

            for field in required_model_fields:
                if field not in model_config:
                    raise ValueError(
                        f"Model config {i} missing required field: {field}"
                    )

            # Validate priority
            valid_priorities = ["primary", "supportive"]
            if model_config["priority"] not in valid_priorities:
                raise ValueError(
                    f"Model {i} priority must be one of {valid_priorities}"
                )

            # Validate quantization setting
            valid_quantization = ["none", "8bit", "4bit"]
            if model_config["quantization"] not in valid_quantization:
                raise ValueError(
                    f"Model {i} quantization must be one of {valid_quantization}"
                )

            # Validate score
            score = model_config.get("score", 0)
            if score <= 0:
                raise ValueError(f"Model {i} score must be positive, got {score}")


class SpanConfigManager:
    """Span-level configuration manager."""

    def __init__(self, config_dir: str = "example_configs"):
        self.config_dir = Path(config_dir)
        self.validator = SpanConfigValidator()
        self.loaded_configs = {}

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration file."""
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Validate configuration
            if not self.validator.validate_config(config):
                raise ValueError(f"Invalid configuration: {config_name}")

            # Store loaded configuration
            self.loaded_configs[config_name] = config

            logger.info(f"✅ Loaded configuration: {config_name}")
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    def get_span_config(self, config_name: str) -> Dict[str, Any]:
        """Get span-level configuration."""
        if config_name not in self.loaded_configs:
            self.load_config(config_name)

        config = self.loaded_configs[config_name]
        return config.get("SPAN_LEVEL_CONFIG", {})

    def get_training_config(
        self, config_name: str, algorithm: str = "ppo"
    ) -> Dict[str, Any]:
        """Get training configuration."""
        if config_name not in self.loaded_configs:
            self.load_config(config_name)

        config = self.loaded_configs[config_name]
        training_config = config.get("TRAINING_CONFIG", {})

        # Get configuration for a specific algorithm
        algo_config = training_config.get(algorithm, {})

        # Merge common configuration with algorithm-specific configuration
        if "hyperparameters" in algo_config:
            return algo_config["hyperparameters"]

        return algo_config

    def get_model_configs(self, config_name: str) -> List[Dict[str, Any]]:
        """Get model configurations."""
        if config_name not in self.loaded_configs:
            self.load_config(config_name)

        config = self.loaded_configs[config_name]
        return config.get("CONFIG_API_SERVER", [])

    def create_span_preset_config(
        self, preset_name: str, span_length: int
    ) -> Dict[str, Any]:
        """Create span preset configuration."""
        base_config = {
            "SPAN_LEVEL_CONFIG": {
                "enabled": True,
                "default_span_length": span_length,
                "max_span_position": 128,
                "weight_update_frequency": "span",
                "span_presets": {
                    "short_text": 4,
                    "medium_text": 8,
                    "long_text": 16,
                    "dynamic": "adaptive",
                },
            }
        }

        logger.info(
            f"Created span preset config: {preset_name} with length {span_length}"
        )
        return base_config

    def merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge configuration dictionaries."""
        import copy

        merged = copy.deepcopy(base_config)

        def deep_merge(base, override):
            for key, value in override.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(merged, override_config)
        return merged

    def save_config(
        self, config: Dict[str, Any], config_name: str, validate: bool = True
    ):
        """Save configuration to disk."""
        if validate and not self.validator.validate_config(config):
            raise ValueError("Invalid configuration, cannot save")

        config_path = self.config_dir / f"{config_name}.yaml"

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"✅ Saved configuration: {config_path}")

        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")


def load_span_config(config_name: str = "span_ensemble_config") -> Dict[str, Any]:
    """Convenience function: load span configuration."""
    manager = SpanConfigManager()
    return manager.load_config(config_name)


def get_training_params(config_name: str, algorithm: str = "ppo") -> Dict[str, Any]:
    """Convenience function: get training parameters."""
    manager = SpanConfigManager()
    return manager.get_training_config(config_name, algorithm)


def create_default_span_configs():
    """Create default span configurations."""
    manager = SpanConfigManager()

    # Create preset configurations for different scenarios
    presets = {"short_text": 4, "medium_text": 8, "long_text": 16, "very_long_text": 32}

    for preset_name, span_length in presets.items():
        config = manager.create_span_preset_config(preset_name, span_length)

        # Save preset configuration
        config_path = manager.config_dir / f"span_preset_{preset_name}.yaml"
        try:
            manager.save_config(config, f"span_preset_{preset_name}", validate=False)
            logger.info(f"Created default preset: {preset_name}")
        except Exception as e:
            logger.warning(f"Failed to create preset {preset_name}: {e}")


if __name__ == "__main__":
    # Test configuration manager
    logging.basicConfig(level=logging.INFO)

    # Create configuration manager
    manager = SpanConfigManager()

    # Load example configuration
    try:
        config = manager.load_config("span_ensemble_config")
        print("✅ Configuration loaded successfully")

        # Get span configuration
        span_config = manager.get_span_config("span_ensemble_config")
        print(f"Span config: {span_config}")

        # Get training configuration
        training_config = manager.get_training_config("span_ensemble_config", "ppo")
        print(f"Training config: {training_config}")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Create default presets
    print("\nCreating default span presets...")
    create_default_span_configs()
