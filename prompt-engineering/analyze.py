#!/usr/bin/env python3
"""
Script to analyze translation evaluation results.
"""

import argparse
import os
from pipeline.analyze_results import analyze_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze translation evaluation results."
    )

    parser.add_argument(
        "results_file", type=str, help="Path to the evaluation results JSON file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Directory to save the analysis results",
    )

    return parser.parse_args()


def main():
    """Run the analysis with command-line arguments."""
    args = parse_args()

    # Ensure the results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return

    # Determine the parent directory of the results file
    results_dir = os.path.dirname(args.results_file)

    # Check if the results file is in the new directory structure
    # Expected format: results/result_[timestamp]/evaluation_results.json
    if os.path.basename(results_dir).startswith("result_"):
        # Use the timestamp directory for analysis
        analysis_dir = os.path.join(results_dir, "analysis")
        print(f"Using standard directory structure: {analysis_dir}")
    elif args.output_dir:
        # User specified a custom output directory
        analysis_dir = args.output_dir
        print(f"Using custom output directory: {analysis_dir}")
    else:
        # Default to analysis subdirectory in the same folder as results
        analysis_dir = os.path.join(results_dir, "analysis")
        print(f"Using default analysis directory: {analysis_dir}")

    # Ensure the analysis directory exists
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"Analyzing results from: {args.results_file}")
    print(f"Saving analysis to: {analysis_dir}")

    # Run the analysis
    analyze_results(args.results_file, analysis_dir)

    print(f"\nAnalysis complete. Results saved to: {analysis_dir}")
    print("You can find visualization plots in the 'plots' subdirectory.")

    # Provide a hint about where to find the plots
    plots_dir = os.path.join(analysis_dir, "plots")
    if os.path.exists(plots_dir):
        num_plots = len([f for f in os.listdir(plots_dir) if f.endswith(".png")])
        print(f"Generated {num_plots} visualization plots.")


if __name__ == "__main__":
    main()
