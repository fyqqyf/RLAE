import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
import os

# Import trained model
from train_span import SpanMAEnsemblePolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_span_mappo_policy(
    model_path, span_length=8, num_agents=2, test_samples=100, batch_size=32, seed=42
):
    """Evaluate span-level MAPPO policy."""

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model
    logger.info(f"Loading MAPPO model from {model_path}")
    policy = SpanMAEnsemblePolicy(num_agents=num_agents, span_length=span_length)

    # Load trained weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        policy.load_state_dict(state_dict)
        logger.info("MAPPO model loaded successfully")
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
        "agent_weights_by_span": [],
        "weight_diversity": [],
        "cooperation_metrics": [],
        "generated_weights_history": [],
    }

    logger.info(f"Starting MAPPO evaluation with {len(test_dataset)} samples")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            questions = batch["question"]

            # Simulate weight generation at different span positions
            batch_agent_weights = []
            batch_cooperation = []

            for span_pos in range(5):  # Test first 5 spans
                # Generate weights for this span (multi-agent)
                weights = policy(questions, span_position=span_pos)

                # Record multi-agent weights
                weights_np = weights.cpu().numpy()
                batch_agent_weights.append(weights_np)

                # Compute cooperation between agents (weight similarity)
                if weights_np.shape[1] > 1:  # Multiple agents
                    # Compute std of agent weights (smaller means more consistent)
                    agent_std = np.std(weights_np, axis=1).mean()
                    cooperation_score = 1.0 / (1.0 + agent_std)  # Convert to similarity
                    batch_cooperation.append(cooperation_score)

                # Record detailed history
                for i, w in enumerate(weights_np):
                    results["generated_weights_history"].append(
                        {
                            "batch": batch_idx,
                            "sample": i,
                            "span_position": span_pos,
                            "agent_weights": w.tolist(),
                            "weight_sum": w.sum(),
                            "dominant_agent": np.argmax(w),
                            "agent_entropy": -np.sum(w * np.log(w + 1e-8)),
                            "cooperation_score": (
                                cooperation_score if weights_np.shape[1] > 1 else 0.0
                            ),
                        }
                    )

            results["agent_weights_by_span"].append(np.array(batch_agent_weights))
            results["cooperation_metrics"].append(batch_cooperation)

    # Compute MAPPO-specific statistics
    all_weights = np.concatenate(
        results["agent_weights_by_span"], axis=1
    )  # [span_pos, total_samples, num_agents]

    stats = {
        "avg_agent_weights_by_span": all_weights.mean(axis=1),
        "std_agent_weights_by_span": all_weights.std(axis=1),
        "avg_cooperation_by_span": np.mean(results["cooperation_metrics"], axis=0),
        "agent_diversity_score": np.mean([np.std(w) for w in all_weights]),
        "total_samples": len(test_dataset),
        "num_agents": num_agents,
        "span_length": span_length,
    }

    # Analyze multi-agent behavior
    logger.info("=== MAPPO evaluation results ===")
    logger.info(f"Number of test samples: {stats['total_samples']}")
    logger.info(f"Number of agents: {stats['num_agents']}")
    logger.info(f"Span length: {stats['span_length']}")

    for span_pos in range(5):
        avg_weights = stats["avg_agent_weights_by_span"][span_pos]
        std_weights = stats["std_agent_weights_by_span"][span_pos]
        avg_cooperation = stats["avg_cooperation_by_span"][span_pos]

        logger.info(f"Span position {span_pos}:")
        logger.info(f"  Average agent weights: {avg_weights}")
        logger.info(f"  Std of weights: {std_weights}")
        logger.info(f"  Average cooperation: {avg_cooperation:.3f}")

        # Find dominant agent
        dominant_agent = np.argmax(avg_weights)
        logger.info(
            f"  Dominant agent: {dominant_agent} (weight: {avg_weights[dominant_agent]:.3f})"
        )

    logger.info(f"Overall agent diversity: {stats['agent_diversity_score']:.4f}")

    # Save detailed results
    output_file = f"span_mappo_eval_results_sl{span_length}_na{num_agents}.json"
    import json

    with open(output_file, "w") as f:
        json.dump(
            {
                "stats": stats,
                "weights_history": results["generated_weights_history"][:1000],
            },
            f,
            indent=2,
        )

    logger.info(f"MAPPO evaluation results saved to: {output_file}")

    return stats


def analyze_mappo_cooperation(weights_history, num_agents):
    """Analyze cooperation patterns between MAPPO agents."""

    # Group cooperation patterns by span position
    span_cooperation = {}

    for record in weights_history:
        span_pos = record["span_position"]
        if span_pos not in span_cooperation:
            span_cooperation[span_pos] = []

        if "cooperation_score" in record:
            span_cooperation[span_pos].append(record["cooperation_score"])

    # Compute cooperation statistics for each span position
    cooperation_analysis = {}
    for span_pos, scores in span_cooperation.items():
        if scores:
            cooperation_analysis[span_pos] = {
                "avg_cooperation": np.mean(scores),
                "cooperation_std": np.std(scores),
                "high_cooperation_ratio": np.mean([s > 0.8 for s in scores]),
            }

    return cooperation_analysis


def compare_with_baseline(baseline_stats, mappo_stats):
    """Compare MAPPO with single-agent baseline."""

    comparison = {
        "weight_stability": {
            "mappo": np.mean(mappo_stats["std_agent_weights_by_span"]),
            "baseline": np.mean(baseline_stats.get("std_weights_by_span", [0.0])),
        },
        "cooperation_benefit": {
            "multi_agent_diversity": mappo_stats["agent_diversity_score"],
            "cooperation_improvement": "N/A",  # Requires baseline cooperation data
        },
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate span-level MAPPO policy")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained MAPPO model"
    )
    parser.add_argument("--span_length", type=int, default=8, help="Span length")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    parser.add_argument(
        "--test_samples", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--compare_baseline",
        type=str,
        default=None,
        help="Path to baseline model stats for comparison",
    )

    args = parser.parse_args()

    # Set environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run MAPPO evaluation
    stats = evaluate_span_mappo_policy(
        model_path=args.model_path,
        span_length=args.span_length,
        num_agents=args.num_agents,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # If a baseline is provided, compare
    if args.compare_baseline:
        logger.info("Comparing with baseline...")
        # Here you can load baseline stats and compare
        # comparison = compare_with_baseline(baseline_stats, stats)
        # logger.info(f"Comparison results: {comparison}")

    logger.info("MAPPO evaluation finished!")


if __name__ == "__main__":
    main()
