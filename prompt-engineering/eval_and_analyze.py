#!/usr/bin/env python3
"""
Script to run the translation evaluation pipeline and analyze results in one command.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional
from pipeline.pipeline import run_evaluation_pipeline
from pipeline.analyze_results import analyze_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the translation evaluation pipeline and analyze results."
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
        "--analysis-dir",
        type=str,
        default="results/analysis",
        help="Directory to save the analysis results",
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
    """Run the evaluation pipeline and analysis with command-line arguments."""
    args = parse_args()

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a results directory structure with timestamp
    results_dir = os.path.join("results", f"result_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Set output path within the timestamped directory
    output_filename = "evaluation_results.json"
    output_path = os.path.join(results_dir, output_filename)

    # Create analysis directory within the timestamped directory
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

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
                prompt_types.append(pt)

    print(f"Results will be saved to: {output_path}")

    # Run the evaluation pipeline
    print("\n=== RUNNING EVALUATION ===\n")
    run_evaluation_pipeline(
        dataset_path=args.dataset,
        output_path=output_path,
        max_workers=args.workers,
        languages=args.languages,
        prompt_types=prompt_types,
        cohere_model=args.model,
        verbose=args.verbose,
    )

    # Run the analysis on the generated results
    print("\n=== RUNNING ANALYSIS ===\n")

    # Run the analysis
    analyze_results(output_path, analysis_dir)

    print(f"\nAnalysis results saved to: {analysis_dir}")
    print("\n=== PROCESS COMPLETE ===\n")


if __name__ == "__main__":
    main()
