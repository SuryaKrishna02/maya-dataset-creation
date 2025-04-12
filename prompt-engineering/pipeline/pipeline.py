import cohere
import json
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import textwrap
from textwrap import dedent
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from pipeline.prompt_generator import Prompt, LANGUAGES, ALL_PROMPTS, load_all_prompts
from pipeline.eval_utils import calculate_bleu_nltk
from pipeline.clean_utils import (
    normalize_text,
    extract_thinking_content,
    normalize_whitespace,
)
from dotenv import load_dotenv
from pipeline.analyze_results import calculate_average_bleu_per_preamble_language
from cohere.errors.too_many_requests_error import TooManyRequestsError


# Default model if not specified
DEFAULT_MODEL = "c4ai-aya-expanse-32b"

# Maximum number of retries for API rate limiting
MAX_RETRIES = 10

# Initialize Cohere client with API key from environment variable
load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable is not set")
co_client = cohere.ClientV2(api_key=cohere_api_key)


# Function to generate translation using Cohere API
def generate_translation(
    input_text: str,
    prompt: Prompt,
    language: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Generate a translation using the Cohere API.

    Args:
        input_text: The text to translate
        prompt: Either a formatted prompt string or a Prompt object
        language: The target language for translation
        model: The Cohere model to use for translation
        verbose: Whether to print detailed information

    Returns:
        The generated translation and reasoning content
    """
    try:
        # Validate target language
        target_language = language
        if not target_language:
            print("ERROR: Empty target language provided!")
            return "", None

        if verbose:
            print(f"Using target language: {target_language}")
            print(f"Using model: {model}")

        # Prepare the messages for the chat API
        messages = []
        is_reasoning_prompt = False

        # Check if this is a reasoning prompt
        if prompt.prompt_metadata and prompt.prompt_metadata.get("reasoning", False):
            is_reasoning_prompt = True
            if verbose:
                print("Detected reasoning prompt")

        if prompt.is_preamble:
            # For preambles, use as system message and input_text as user message
            system_msg = prompt.format(target_language)

            # For reasoning prompts, add additional instruction to use thinking tags
            if is_reasoning_prompt:
                system_msg += dedent(
                    """
                    Please enclose any thinking or reasoning processes that are not related to the final translation in <thinking>...</thinking> tags. Other than that, the final output should be the translated content with no indicators.
                    The final output should be in the following format:
                    <thinking>
                    {thinking content}
                    </thinking>
                    {translated content}
                    """
                )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": input_text},
            ]
        else:
            # For prompts with {user_input}, use the format method with user_input
            try:
                prompt_content = prompt.format(target_language, user_input=input_text)

                # For reasoning prompts, add additional instruction
                if is_reasoning_prompt:
                    prompt_content += dedent(
                        """
\n\nPlease enclose any thinking or reasoning processes unrelated to the final translation in <thinking>...</thinking> tags. Other than that, the final output should be the translated content with no indicators.
                            
The final output should be in the following format:
<thinking>
{thinking content}
</thinking>
{translated content}
                    """
                    )

                messages = [{"role": "user", "content": prompt_content}]
            except Exception as e:
                print(f"ERROR formatting prompt template: {e}")  # Always print errors
                return "", None

        # Print the messages for debugging when verbose
        if verbose:
            print("\n============= MESSAGES SENT TO MODEL =============\n")
            for msg in messages:
                print(f"ROLE: {msg['role']}")
                print(f"CONTENT:")
                print(textwrap.fill(msg["content"], width=150))
            print("====================================================\n")

        # Generate the translation using the Cohere chat API with exponential backoff
        retry_count = 0
        base_delay = 1  # Base delay in seconds
        max_delay = 60  # Maximum delay in seconds

        while True:
            try:
                # Call the Cohere API
                response = co_client.chat(model=model, messages=messages)
                break  # Exit the loop if successful
            except TooManyRequestsError as e:
                retry_count += 1

                if retry_count > MAX_RETRIES:
                    print(f"ERROR: Maximum retries exceeded for rate limiting: {e}")
                    return "", None

                # Calculate exponential backoff with jitter
                delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
                jitter = random.uniform(0, 0.5 * delay)  # Add up to 50% jitter
                sleep_time = delay + jitter

                if verbose:
                    print(
                        f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds (attempt {retry_count}/{MAX_RETRIES})"
                    )
                else:
                    print(
                        f"Rate limited. Retry {retry_count}/{MAX_RETRIES} in {sleep_time:.2f}s",
                        end="\r",
                    )

                time.sleep(sleep_time)

        # Extract the generated text
        generated_translation = response.message.content[0].text.strip()

        # Debug log the response when verbose
        if verbose:
            print(f"============= RAW MODEL RESPONSE =============")
            print(textwrap.fill(generated_translation, width=120))
            print("==========================\n")

        # Extract reasoning content and filter out thinking tags for reasoning prompts
        reasoning_content = None
        if is_reasoning_prompt:
            # Use the extract_thinking_content function from clean_utils
            generated_translation, reasoning_content = extract_thinking_content(
                generated_translation
            )

            if verbose:
                print("Filtered thinking tags from response")
                if reasoning_content:
                    print(f"Extracted reasoning content: {reasoning_content[:100]}...")

        return generated_translation, reasoning_content
    except Exception as e:
        print(f"ERROR generating translation: {e}")  # Always print errors
        return "", None


def test_single_translation(
    input_text: str,
    gold_translation: str,
    language: str,
    prompt: Prompt,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Test a single translation with a specific prompt and language.

    Args:
        input_text: The text to translate
        gold_translation: The expected translation
        language: The target language
        prompt: The prompt to use
        model: The Cohere model to use for translation
        verbose: Whether to print detailed information

    Returns:
        A dictionary containing the test results
    """
    try:
        # Ensure we have a valid target language
        if not language:
            raise ValueError("A target language must be provided")

        # Generate translation directly using the original prompt and language
        generated_translation, reasoning_content = generate_translation(
            input_text,
            prompt,
            language,
            model=model,
            verbose=verbose,
        )

        # Calculate BLEU scores (1-gram, 2-gram, 3-gram, 4-gram)
        bleu_scores = {}

        # Don't calculate BLEU scores for empty translations
        if not generated_translation:
            bleu_scores = {
                "1-gram": 0.0,
                "2-gram": 0.0,
                "3-gram": 0.0,
                "4-gram": 0.0,
            }
        else:
            # Calculate BLEU scores in parallel for different n-grams
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        calculate_bleu_nltk,
                        gold_translation,
                        generated_translation,
                        language,
                        n_gram,
                    ): n_gram
                    for n_gram in range(1, 5)
                }

                # Use an OrderedDict to ensure consistent ordering
                from collections import OrderedDict

                bleu_scores = OrderedDict()

                # Initialize with zeros to ensure proper order
                for n_gram in range(1, 5):
                    bleu_scores[f"{n_gram}-gram"] = 0.0

                # Collect results as they complete
                for future in as_completed(futures):
                    n_gram = futures[future]
                    try:
                        score = future.result()
                        bleu_scores[f"{n_gram}-gram"] = score
                    except Exception as e:
                        print(
                            f"ERROR calculating BLEU score for {n_gram}-gram: {e}"
                        )  # Always print errors
                        # Score already initialized to 0.0 above

        # Create result dictionary
        result = {
            "input_text": input_text,
            "gold_translation": gold_translation,
            "generated_translation": generated_translation,
            "bleu_scores": bleu_scores,
            "prompt_name": prompt.prompt_name,
            "prompt_type": prompt.prompt_type,
            "is_preamble": prompt.is_preamble,
            "is_reasoning": (
                prompt.prompt_metadata.get("reasoning", False)
                if prompt.prompt_metadata
                else False
            ),
            "language": language,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }

        # Add reasoning content if available
        if reasoning_content:
            result["reasoning_content"] = reasoning_content

        return result

    except Exception as e:
        print(f"ERROR in test_single_translation: {e}")  # Always print errors
        return {
            "error": str(e),
            "input_text": input_text,
            "prompt_name": prompt.prompt_name,
            "prompt_type": prompt.prompt_type,
            "is_preamble": prompt.is_preamble,
            "language": language,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }


def load_test_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load the test dataset from a JSON file.
    """
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the evaluation results to a JSON file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def evaluate_prompts_multithreaded(
    dataset: List[Dict[str, Any]],
    prompts: Optional[List[Prompt]] = None,
    languages: Optional[List[str]] = None,
    cohere_model: str = DEFAULT_MODEL,
    max_workers: int = 5,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate prompts using multithreading for parallel processing.

    Args:
        dataset: The test dataset
        prompts: List of prompts to test (defaults to ALL_PROMPTS)
        languages: List of languages to test (defaults to LANGUAGES)
        cohere_model: The Cohere model to use for translations
        max_workers: Maximum number of worker threads
        verbose: Whether to print detailed information

    Returns:
        A list of evaluation results
    """
    results = []

    # Use default values if not provided
    if prompts is None:
        prompts = ALL_PROMPTS

    if languages is None:
        languages = list(LANGUAGES)

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Create tasks for all combinations of dataset entries, prompts, and languages
        for entry in dataset:
            input_text = entry.get("input_text", "")

            for language in languages:
                # Get the gold translation for this language
                gold_translation = entry.get(f"translation_{language.lower()}", "")

                # Skip if no gold translation is available
                if not gold_translation:
                    continue

                for prompt in prompts:
                    futures.append(
                        executor.submit(
                            test_single_translation,
                            input_text,
                            gold_translation,
                            language,
                            prompt,
                            cohere_model,
                            verbose,
                        )
                    )

        # Process results as they complete
        total_tasks = len(futures)
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)

                # Print progress
                if verbose:
                    print(f"Processed {i+1}/{total_tasks} tasks", end="\r")
                elif (
                    i + 1
                ) % 10 == 0 or i + 1 == total_tasks:  # Print less frequently when not verbose
                    print(f"Processed {i+1}/{total_tasks} tasks", end="\r")

            except Exception as e:
                print(f"\nERROR processing result: {e}")  # Always print errors

    print(f"\nCompleted {len(results)} evaluations")
    return results


def run_evaluation_pipeline(
    dataset_path: str = "../data/test_dataset.json",
    output_path: Optional[str] = "../results/evaluation_results.json",
    max_workers: int = 5,
    languages: Optional[List[str]] = None,
    prompt_types: Optional[List[str]] = None,
    cohere_model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> None:
    """
    Run the complete evaluation pipeline.

    Args:
        dataset_path: Path to the test dataset
        output_path: Path to save the results
        max_workers: Maximum number of worker threads
        languages: List of languages to test (defaults to all languages)
        prompt_types: List of prompt types to test (defaults to both Zero-Shot and Few-Shot)
        cohere_model: The Cohere model to use for translations
        verbose: Whether to print detailed information
    """
    print(f"Starting evaluation pipeline with {max_workers} workers")
    print(f"Using Cohere model: {cohere_model}")

    if output_path is None:
        # Create a results directory structure with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"result_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # Set output path within the timestamped directory
        output_filename = "evaluation_results.json"
        output_path = os.path.join(results_dir, output_filename)

    # Load the test dataset
    dataset = load_test_dataset(dataset_path)
    if not dataset:
        print("ERROR: No dataset entries found. Exiting.")
        return

    print(f"Loaded {len(dataset)} Few-Shot examples for testing")

    # Get all prompts with verbose mode passed down
    all_prompts = load_all_prompts(verbose=verbose)

    # Filter prompts by type if specified, otherwise use both Zero-Shot and Few-Shot
    prompts = all_prompts

    if verbose:
        print(f"Total prompts available: {len(all_prompts)}")

    if prompt_types:
        prompts = [p for p in all_prompts if p.prompt_type in prompt_types]
        if verbose:
            print(f"Filtered to {len(prompts)} prompts of types: {prompt_types}")
    else:
        # Explicitly default to both Zero-Shot and Few-Shot
        prompt_types = ["Zero-Shot", "Few-Shot"]
        if verbose:
            print(f"Using default prompt types: {prompt_types}")

    if len(prompts) == 0:
        print(
            "ERROR: No prompts found! Check that prompt files exist in the prompt_data directory."
        )
        return

    # Run the evaluation
    results = evaluate_prompts_multithreaded(
        dataset=dataset,
        prompts=prompts,
        languages=languages,
        cohere_model=cohere_model,
        max_workers=max_workers,
        verbose=verbose,
    )

    # Save the results - ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_results(results, output_path)

    # Calculate and display average BLEU scores using the function from analyze_results.py
    try:
        import pandas as pd

        # Create a DataFrame from the results
        df = pd.DataFrame(results)

        # Normalize the bleu_scores column (convert from dict to columns)
        if "bleu_scores" in df.columns:
            bleu_scores_df = pd.json_normalize(df["bleu_scores"])
            df = pd.concat([df.drop(columns=["bleu_scores"]), bleu_scores_df], axis=1)

            # Rename columns to match what calculate_average_bleu_per_preamble_language expects
            column_mapping = {
                "1-gram": "bleu_1gram",
                "2-gram": "bleu_2gram",
                "3-gram": "bleu_3gram",
                "4-gram": "bleu_4gram",
            }
            df = df.rename(columns=column_mapping)

            # Calculate and display average BLEU scores
            calculate_average_bleu_per_preamble_language(df)
    except ImportError:
        print(
            "pandas is required for BLEU score analysis. Please install it with 'pip install pandas'."
        )
    except Exception as e:
        print(f"Error calculating average BLEU scores: {e}. Columns are: {df.columns}")

    # Print summary
    print("\nEvaluation Summary:")
    print(f"- Total evaluations: {len(results)}")
    print(f"- Languages tested: {languages or list(LANGUAGES)}")
    print(f"- Prompt types tested: {prompt_types}")
    print(f"- Cohere model used: {cohere_model}")
    print(f"- Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    run_evaluation_pipeline(
        max_workers=10,
        languages=["Spanish", "French", "Chinese"],
        prompt_types=["zero-shot"],
    )
