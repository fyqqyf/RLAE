#!/usr/bin/env python3
"""
Span-level PPO/MAPPO unified training launcher.
Supports multiple training modes and configurations.
"""

import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent


def run_training(algo, mode, **kwargs):
    """Run training."""

    # Determine training script path
    if algo == "ppo":
        script = "ppo/train_span.py" if mode == "span" else "ppo/train.py"
    elif algo == "mappo":
        script = "mappo/train_span.py" if mode == "span" else "mappo/train.py"
    else:
        print(f"Error: Unknown algorithm '{algo}'")
        return False

    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        print(f"Error: Script file does not exist '{script_path}'")
        return False

    print(f"\n=== Start {algo.upper()} {mode}-level training ===")
    print(f"Algorithm: {algo}")
    print(f"Mode: {mode}")
    print(f"Script: {script}")
    print(f"Args: {kwargs}")

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    print(f"\nExecute command: {' '.join(cmd)}")

    # Set environment variables
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    # Run training
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, env=env, capture_output=False, text=True
    )

    if result.returncode == 0:
        print(f"✅ {algo.upper()} {mode}-level training finished")
        return True
    else:
        print(f"❌ {algo.upper()} {mode}-level training failed")
        return False


def run_evaluation(algo, **kwargs):
    """Run evaluation."""
    # Determine evaluation script path
    if algo == "ppo":
        script = "ppo/evaluate_span.py"
    elif algo == "mappo":
        script = "mappo/evaluate_span.py"
    else:
        print(f"Error: Unknown algorithm '{algo}'")
        return False

    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        print(f"Error: Script file does not exist '{script_path}'")
        return False

    print(f"\n=== Run {algo.upper()} evaluation ===")
    print(f"Algorithm: {algo}")
    print(f"Script: {script}")
    print(f"Args: {kwargs}")

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    print(f"\nExecute command: {' '.join(cmd)}")

    try:
        # Run evaluation
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False, text=True)

        if result.returncode == 0:
            print(f"✅ {algo.upper()} evaluation finished")
            return True
        else:
            print(f"❌ {algo.upper()} evaluation failed")
            return False

    except Exception as e:
        print(f"❌ Error while running evaluation: {e}")
        return False


def run_tests(test_type):
    """Run tests."""
    print(f"\n=== Run {test_type} tests ===")

    test_scripts = {
        "ppo": "ppo/test_span.py",
        "mappo": "mappo/test_span.py",
        "all": ["ppo/test_span.py", "mappo/test_span.py"],
    }

    if test_type not in test_scripts:
        print(f"Error: Unknown test type '{test_type}'")
        return False

    scripts = test_scripts[test_type]
    if isinstance(scripts, str):
        scripts = [scripts]

    all_passed = True

    for script in scripts:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            print(f"⚠️  Test script does not exist: {script}")
            continue

        print(f"\nRun test: {script}")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(script_path), "-v"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"✅ {script} tests passed")
            else:
                print(f"❌ {script} tests failed")
                print("STDOUT:", result.stdout[-300:])  # Show last 300 characters
                print("STDERR:", result.stderr[-300:])
                all_passed = False

        except Exception as e:
            print(f"❌ Error while running test script: {e}")
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Span-level PPO/MAPPO unified training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Train PPO span model
  python train_span.py --algo ppo --mode span --span_length 8 --batch_size 32
  
  # Train MAPPO span model
  python train_span.py --algo mappo --mode span --span_length 8 --num_agents 2
  
  # Evaluate PPO model
  python train_span.py --eval --algo ppo --model_path policy_epoch_1.pt
  
  # Run tests
  python train_span.py --test all
        """,
    )

    # Main operation modes
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument(
        "--test", type=str, choices=["ppo", "mappo", "all"], help="Run tests"
    )

    # Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "mappo"],
        help="Choose algorithm (ppo or mappo)",
    )

    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["span", "token"],
        default="span",
        help="Training mode: span-level or token-level",
    )

    # Training parameters
    parser.add_argument(
        "--span_length", type=int, default=8, help="Span length (default 8)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default 16)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of epochs (default 2)"
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=2,
        help="Number of agents (MAPPO only, default 2)",
    )

    # Evaluation parameters
    parser.add_argument("--model_path", type=str, help="Model path (for evaluation)")
    parser.add_argument(
        "--test_samples",
        type=int,
        default=100,
        help="Number of test samples (default 100)",
    )

    args = parser.parse_args()

    # Argument validation
    if not any([args.train, args.eval, args.test]):
        print("Error: You must specify one of --train, --eval or --test")
        parser.print_help()
        sys.exit(1)

    if args.train or args.eval:
        if not args.algo:
            print("Error: --algo is required when training or evaluating")
            parser.print_help()
            sys.exit(1)

    # Execute operation
    success = False

    if args.train:
        success = run_training(
            algo=args.algo,
            mode=args.mode,
            span_length=args.span_length,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_agents=args.num_agents if args.algo == "mappo" else None,
        )

    elif args.eval:
        if not args.model_path:
            print("Error: --model_path is required when evaluating")
            sys.exit(1)

        success = run_evaluation(
            algo=args.algo,
            model_path=args.model_path,
            span_length=args.span_length,
            test_samples=args.test_samples,
        )

    elif args.test:
        success = run_tests(args.test)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
