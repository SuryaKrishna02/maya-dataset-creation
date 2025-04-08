#!/usr/bin/env python3
"""
Script to run the translation evaluation pipeline.
"""

import argparse
import os
from typing import List, Optional
from pipeline.pipeline import run_evaluation_pipeline
from datetime import datetime


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the translation evaluation pipeline."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset JSON file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save the evaluation results",
    )

    parser.add_argument(
        "--workers", type=int, default=5, help="Maximum number of worker threads"
    )

    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        help="List of languages to test (e.g., Spanish French Chinese)",
    )

    parser.add_argument(
        "--prompt-types",
        type=str,
        nargs="+",
        choices=["zero-shot", "few-shot", "Zero-Shot", "Few-Shot"],
        help="List of preamble types to test (zero-shot and/or few-shot). Defaults to both if not specified.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="c4ai-aya-expanse-32b",
        help="Cohere model name to use for translations",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during execution",
    )

    return parser.parse_args()


def main():
    """Run the evaluation pipeline with command-line arguments."""
    args = parse_args()

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a results directory structure with timestamp
    results_dir = os.path.join("results", f"result_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Set output path within the timestamped directory
    output_filename = "evaluation_results.json"
    output_path = os.path.join(results_dir, output_filename)

    # Convert prompt types to proper case format if provided
    prompt_types = None
    if args.prompt_types:
        prompt_types = []
        for pt in args.prompt_types:
            if pt.lower() == "zero-shot":
                prompt_types.append("Zero-Shot")
            elif pt.lower() == "few-shot":
                prompt_types.append("Few-Shot")
            else:
                prompt_types.append(pt)  # Should not occur with argparse choices

    # Run the evaluation pipeline
    run_evaluation_pipeline(
        dataset_path=args.dataset,
        output_path=output_path,
        max_workers=args.workers,
        languages=args.languages,
        prompt_types=prompt_types,
        cohere_model=args.model,
        verbose=args.verbose,
    )

    print(f"\nTo analyze the results, run: python analyze.py {output_path}")
    print(
        f"Or run: python eval_and_analyze.py --dataset {args.dataset} --workers {args.workers} "
        + (f"--languages {' '.join(args.languages)} " if args.languages else "")
        + (
            f"--prompt-types {' '.join(args.prompt_types)} "
            if args.prompt_types
            else ""
        )
        + (f"--model {args.model} " if args.model != "c4ai-aya-expanse-32b" else "")
        + (f"--verbose " if args.verbose else "")
    )


if __name__ == "__main__":
    main()
