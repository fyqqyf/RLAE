import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
import os

# Import trained model
from train_span import SpanEnsembleWeightPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_span_policy(
    model_path, span_length=8, num_models=2, test_samples=100, batch_size=32, seed=42
):
    """Evaluate span-level policy model."""

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model
    logger.info(f"Loading model from {model_path}")
    policy = SpanEnsembleWeightPolicy(num_models=num_models, span_length=span_length)

    # Load trained weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        policy.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model file not found: {model_path}, using random weights")

    policy.eval()

    # Load test data
    logger.info("Loading test dataset")
    dataset = load_dataset("gsm8k", "main", split="test")

    # Randomly choose test samples
    if len(dataset) > test_samples:
        indices = np.random.choice(len(dataset), test_samples, replace=False)
        test_dataset = dataset.select(indices)
    else:
        test_dataset = dataset

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Evaluation metrics
    results = {
        "span_weights": [],
        "weight_variance": [],
        "span_positions": [],
        "generated_weights_history": [],
    }

    logger.info(f"Starting evaluation with {len(test_dataset)} samples")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            questions = batch["question"]

            # Simulate weight generation at different span positions
            batch_weights = []
            batch_variance = []

            for span_pos in range(5):  # Test first 5 span positions
                # Generate weights for this span
                weights = policy(questions, span_position=span_pos)

                # Record weight statistics
                weights_np = weights.cpu().numpy()
                batch_weights.append(weights_np)

                # Compute variance (degree of dispersion)
                weight_var = np.var(weights_np, axis=1).mean()
                batch_variance.append(weight_var)

                # Record detailed weight history
                for i, w in enumerate(weights_np):
                    results["generated_weights_history"].append(
                        {
                            "batch": batch_idx,
                            "sample": i,
                            "span_position": span_pos,
                            "weights": w.tolist(),
                            "weight_sum": w.sum(),
                            "max_weight_idx": np.argmax(w),
                            "entropy": -np.sum(w * np.log(w + 1e-8)),
                        }
                    )

            results["span_weights"].append(np.array(batch_weights))
            results["weight_variance"].append(batch_variance)
            results["span_positions"].append(list(range(5)))

    # Compute overall statistics
    all_weights = np.concatenate(
        results["span_weights"], axis=1
    )  # [span_pos, total_samples, num_models]

    stats = {
        "avg_weights_by_span": all_weights.mean(axis=1),
        "std_weights_by_span": all_weights.std(axis=1),
        "avg_variance": np.mean(results["weight_variance"]),
        "total_samples": len(test_dataset),
        "span_length": span_length,
        "num_models": num_models,
    }

    # Analyze weight trends
    logger.info("=== Evaluation results ===")
    logger.info(f"Number of test samples: {stats['total_samples']}")
    logger.info(f"Span length: {stats['span_length']}")
    logger.info(f"Number of models: {stats['num_models']}")

    for span_pos in range(5):
        avg_weight = stats["avg_weights_by_span"][span_pos]
        std_weight = stats["std_weights_by_span"][span_pos]

        logger.info(f"Span position {span_pos}:")
        logger.info(f"  Average weights: {avg_weight}")
        logger.info(f"  Std of weights: {std_weight}")
        logger.info(
            f"  Dominant model: {np.argmax(avg_weight)} (weight: {np.max(avg_weight):.3f})"
        )

    logger.info(f"Average weight variance: {stats['avg_variance']:.4f}")

    # Save detailed results
    output_file = f"span_eval_results_sl{span_length}_nm{num_models}.json"
    import json

    with open(output_file, "w") as f:
        json.dump(
            {
                "stats": stats,
                "weights_history": results["generated_weights_history"][
                    :1000
                ],  # Only save first 1000 entries
            },
            f,
            indent=2,
        )

    logger.info(f"Evaluation results saved to: {output_file}")

    return stats


def analyze_weight_patterns(weights_history):
    """Analyze weight change patterns."""

    # Group by span position
    span_groups = {}
    for record in weights_history:
        span_pos = record["span_position"]
        if span_pos not in span_groups:
            span_groups[span_pos] = []
        span_groups[span_pos].append(record)

    # Analyze weight distribution for each span position
    analysis = {}
    for span_pos, records in span_groups.items():
        weights = np.array([r["weights"] for r in records])

        analysis[span_pos] = {
            "avg_weights": weights.mean(axis=0),
            "weight_entropy": weights.mean(axis=0),
            "dominant_model": np.argmax(weights.mean(axis=0)),
            "weight_concentration": np.max(weights.mean(axis=0)),
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate span-level ensemble policy")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--span_length", type=int, default=8, help="Span length")
    parser.add_argument("--num_models", type=int, default=2, help="Number of models")
    parser.add_argument(
        "--test_samples", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run evaluation
    stats = evaluate_span_policy(
        model_path=args.model_path,
        span_length=args.span_length,
        num_models=args.num_models,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    logger.info("Evaluation finished!")


if __name__ == "__main__":
    main()
