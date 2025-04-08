import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation results from a JSON file.

    Args:
        results_path: Path to the results file

    Returns:
        A list of evaluation results
    """
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return []


def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the evaluation results for easier analysis.

    Args:
        results: The evaluation results

    Returns:
        A pandas DataFrame with the results
    """
    # Extract relevant data from each result
    data = []
    for result in results:
        if "error" in result:
            continue

        # Get the reasoning status
        is_reasoning = result.get("is_reasoning", False)
        prompt_type = result["prompt_type"]

        # Create combined prompt type
        combined_prompt_type = prompt_type
        if is_reasoning:
            combined_prompt_type = f"{prompt_type} Reasoning"

        row = {
            "language": result["language"],
            "prompt_name": result["prompt_name"],
            "prompt_type": prompt_type,
            "is_reasoning": is_reasoning,
            "combined_prompt_type": combined_prompt_type,
            "is_preamble": result.get(
                "is_preamble", False
            ),  # Handle legacy results that don't have is_preamble
            "input_text": result["input_text"],
            "bleu_1gram": result["bleu_scores"]["1-gram"],
            "bleu_2gram": result["bleu_scores"]["2-gram"],
            "bleu_3gram": result["bleu_scores"]["3-gram"],
            "bleu_4gram": result["bleu_scores"]["4-gram"],
            "avg_bleu": sum(result["bleu_scores"].values())
            / len(result["bleu_scores"]),
        }
        data.append(row)

    return pd.DataFrame(data)


def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics from the results DataFrame.

    Args:
        df: The results DataFrame

    Returns:
        A dictionary of summary statistics
    """
    summary = {}

    # Overall statistics
    summary["overall"] = {
        "avg_bleu_1gram": df["bleu_1gram"].mean(),
        "avg_bleu_2gram": df["bleu_2gram"].mean(),
        "avg_bleu_3gram": df["bleu_3gram"].mean(),
        "avg_bleu_4gram": df["bleu_4gram"].mean(),
        "avg_bleu_overall": df["avg_bleu"].mean(),
        "total_evaluations": len(df),
    }

    # Statistics by language
    summary["by_language"] = df.groupby("language")["avg_bleu"].mean().to_dict()

    # Statistics by regular prompt type
    summary["by_prompt_type"] = df.groupby("prompt_type")["avg_bleu"].mean().to_dict()

    # Statistics by combined prompt type (including reasoning)
    summary["by_combined_prompt_type"] = (
        df.groupby("combined_prompt_type")["avg_bleu"].mean().to_dict()
    )

    # Statistics by prompt name
    summary["by_prompt_name"] = df.groupby("prompt_name")["avg_bleu"].mean().to_dict()

    # Statistics by whether it's a preamble or prompt
    summary["by_is_preamble"] = df.groupby("is_preamble")["avg_bleu"].mean().to_dict()

    # Statistics by whether it's a reasoning prompt
    summary["by_is_reasoning"] = df.groupby("is_reasoning")["avg_bleu"].mean().to_dict()

    # Best performing combinations
    best_combinations = (
        df.groupby(["language", "prompt_name", "combined_prompt_type", "is_preamble"])[
            "avg_bleu"
        ]
        .mean()
        .reset_index()
    )
    best_combinations = best_combinations.sort_values("avg_bleu", ascending=False)
    summary["best_combinations"] = best_combinations.head(10).to_dict(orient="records")

    return summary


def calculate_average_bleu_per_preamble_language(df: pd.DataFrame) -> None:
    """
    Calculate and display average BLEU scores for each prompt type/name and language.

    Args:
        df: The results DataFrame
    """
    # Define the numeric columns for BLEU scores
    numeric_cols = ["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]

    # Calculate average BLEU scores by language and prompt_type
    print("\nAverage BLEU Scores per Prompt Type and Language:")
    grouped_by_type = df.groupby(["language", "prompt_type"])[numeric_cols].mean()
    print(grouped_by_type)

    # Calculate average BLEU scores by language and combined_prompt_type
    print("\nAverage BLEU Scores per Combined Prompt Type and Language:")
    grouped_by_combined_type = df.groupby(["language", "combined_prompt_type"])[
        numeric_cols
    ].mean()
    print(grouped_by_combined_type)

    # Calculate average BLEU scores by language and prompt_name
    print("\nAverage BLEU Scores per Prompt Name and Language:")
    grouped_by_name = df.groupby(["language", "prompt_name"])[numeric_cols].mean()
    print(grouped_by_name)

    # Calculate average BLEU scores by language, is_preamble
    print("\nAverage BLEU Scores for Prompts vs Preambles per Language:")
    grouped_by_preamble = df.groupby(["language", "is_preamble"])[numeric_cols].mean()
    print(grouped_by_preamble)

    # Calculate average BLEU scores by language, is_reasoning
    print("\nAverage BLEU Scores for Reasoning vs Non-Reasoning Prompts per Language:")
    grouped_by_reasoning = df.groupby(["language", "is_reasoning"])[numeric_cols].mean()
    print(grouped_by_reasoning)


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate plots from the results DataFrame and save them to the output directory.

    Args:
        df: The results DataFrame
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set the style
    sns.set(style="whitegrid")

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Plot average BLEU scores by language
    language_scores = df.groupby("language")[
        ["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]
    ].mean()
    language_scores.plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Average BLEU Scores by Language")
    axes[0, 0].set_ylabel("BLEU Score")

    # 2. Plot average BLEU scores by prompt type
    type_scores = df.groupby("combined_prompt_type")[
        ["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]
    ].mean()
    type_scores.plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("Average BLEU Scores by Prompt Type")
    axes[0, 1].set_ylabel("BLEU Score")

    # 3. Plot average BLEU scores by prompt name
    name_scores = df.groupby("prompt_name")[
        ["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]
    ].mean()
    name_scores.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Average BLEU Scores by Prompt Name")
    axes[1, 0].set_ylabel("BLEU Score")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Heatmap of 4-gram BLEU scores by language and prompt type
    heatmap_data = df.pivot_table(
        values="bleu_4gram",
        index="language",
        columns="combined_prompt_type",
        aggfunc="mean",
    )
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", ax=axes[1, 1])
    axes[1, 1].set_title("4-gram BLEU Score by Language and Prompt Type")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_scores_summary.png"))

    # Additional plots

    # 5. Plot average BLEU scores by language and prompt name
    plt.figure(figsize=(16, 10))
    heatmap_data_name = df.pivot_table(
        values="avg_bleu", index="language", columns="prompt_name", aggfunc="mean"
    )
    sns.heatmap(heatmap_data_name, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Average BLEU Score by Language and Prompt Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_by_language_and_name.png"))

    # 6. Plot distribution of BLEU scores by prompt type
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="language", y="avg_bleu", hue="combined_prompt_type", data=df)
    plt.title("Distribution of BLEU Scores by Language and Prompt Type")
    plt.xlabel("Language")
    plt.ylabel("Average BLEU Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_distribution.png"))

    # 7. Plot comparing prompts vs preambles
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="language", y="avg_bleu", hue="is_preamble", data=df)
    plt.title("Distribution of BLEU Scores: Prompts vs Preambles")
    plt.xlabel("Language")
    plt.ylabel("Average BLEU Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_prompts_vs_preambles.png"))

    # 8. Plot comparing reasoning vs non-reasoning prompts
    plt.figure(figsize=(14, 8))

    # Create a new figure with a clear color distinction
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a custom color palette dictionary that maps False to blue, True to red
    palette = {False: "#1976D2", True: "#D32F2F"}  # Dark blue for No, dark red for Yes

    # Explicitly pass the palette with the mapping
    sns.boxplot(
        x="language", y="avg_bleu", hue="is_reasoning", data=df, palette=palette, ax=ax
    )

    # Add a custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1976D2", edgecolor="black", label="No"),
        Patch(facecolor="#D32F2F", edgecolor="black", label="Yes"),
    ]
    ax.legend(handles=legend_elements, title="Is Reasoning")

    plt.title("Distribution of BLEU Scores: Reasoning vs Non-Reasoning")
    plt.xlabel("Language")
    plt.ylabel("Average BLEU Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_reasoning_vs_non_reasoning.png"))
    plt.close("all")


def analyze_results(results_path: str, output_dir: str = "../results/analysis") -> None:
    """
    Analyze the evaluation results and generate summary statistics and plots.

    Args:
        results_path: Path to the results file
        output_dir: Directory to save the analysis results
    """
    # Load the results
    results = load_results(results_path)
    if not results:
        print("No results to analyze. Exiting.")
        return

    print(f"Analyzing {len(results)} evaluation results")

    # Create a DataFrame from the results
    df = create_summary_dataframe(results)

    # Generate summary statistics
    summary = generate_summary_statistics(df)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the summary statistics
    summary_path = os.path.join(output_dir, "summary_statistics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary statistics saved to {summary_path}")

    # Calculate and display average BLEU scores by prompt and language
    calculate_average_bleu_per_preamble_language(df)

    # Generate and save plots
    plots_dir = os.path.join(output_dir, "plots")
    plot_results(df, plots_dir)

    print(f"Plots saved to {plots_dir}")

    # Print some key statistics
    print("\nKey Statistics:")
    print(f"- Overall average BLEU score: {summary['overall']['avg_bleu_overall']:.4f}")
    print(
        f"- Best performing language: {max(summary['by_language'].items(), key=lambda x: x[1])[0]}"
    )
    print(
        f"- Best performing prompt type: {max(summary['by_prompt_type'].items(), key=lambda x: x[1])[0]}"
    )
    print(
        f"- Best performing combined prompt type: {max(summary['by_combined_prompt_type'].items(), key=lambda x: x[1])[0]}"
    )
    print(
        f"- Best performing prompt name: {max(summary['by_prompt_name'].items(), key=lambda x: x[1])[0]}"
    )

    # Print if prompts or preambles perform better
    prompts_score = summary["by_is_preamble"].get(False, 0)
    preambles_score = summary["by_is_preamble"].get(True, 0)
    better_approach = "Prompts" if prompts_score > preambles_score else "Preambles"
    print(
        f"- Better approach: {better_approach} (Prompts: {prompts_score:.4f}, Preambles: {preambles_score:.4f})"
    )

    # Print if reasoning or non-reasoning prompts perform better
    non_reasoning_score = summary["by_is_reasoning"].get(False, 0)
    reasoning_score = summary["by_is_reasoning"].get(True, 0)
    better_reasoning = (
        "Reasoning" if reasoning_score > non_reasoning_score else "Non-reasoning"
    )
    print(
        f"- Better reasoning approach: {better_reasoning} (Reasoning: {reasoning_score:.4f}, Non-reasoning: {non_reasoning_score:.4f})"
    )

    # Print top 3 best combinations
    print("\nTop 3 Best Performing Combinations:")
    for i, combo in enumerate(summary["best_combinations"][:3]):
        prompt_type = "Preamble" if combo["is_preamble"] else "Prompt"
        print(
            f"{i+1}. Language: {combo['language']}, {prompt_type}: {combo['prompt_name']} ({combo['combined_prompt_type']}), BLEU: {combo['avg_bleu']:.4f}"
        )

    # Save the DataFrame for further analysis
    df_path = os.path.join(output_dir, "results_dataframe.csv")
    df.to_csv(df_path, index=False)
    print(f"Results DataFrame saved to {df_path}")


if __name__ == "__main__":
    # Get the results file path from command line arguments or use default
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Find the most recent results file
        results_dir = Path("../results")
        if results_dir.exists():
            results_files = list(results_dir.glob("*.json"))
            if results_files:
                results_path = str(max(results_files, key=os.path.getmtime))
            else:
                print("No results files found. Please specify a results file path.")
                sys.exit(1)
        else:
            print("Results directory not found. Please specify a results file path.")
            sys.exit(1)

    print(f"Analyzing results from: {results_path}")
    analyze_results(results_path)
