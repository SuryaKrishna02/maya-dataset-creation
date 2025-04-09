# Translation Evaluation Pipeline

> Note: This was ported over from a previous repository and `prompt-engineering` was the root directory of the project. So when calling functions in the command line, please `cd` into the `prompt-engineering` directory first.
<!-- 
> :warning: This pipeline is currently in development and has not been tested yet. -->


## Features

- Multithreaded evaluation of translation prompts
- Multithreaded BLEU score calculation for faster processing
- Support for multiple languages and two prompt types (few-shot and zero-shot)
  - Few-shot prompt include examples of translations to guide the model
  - Zero-shot prompt do not include examples 
- Support for prompt metadata like reasoning prompts that automatically filter thinking processes


## Directory Structure

```
.
├── data/
│   ├── few_shot_data.json              # Few-shot examples 
│   └── test_dataset.json               # Test dataset for evaluation
├── prompt_data/
│   ├── few_shot/                       # Few-shot prompt templates
│   └── zero_shot/                      # Zero-shot prompt templates
├── pipeline/
│   ├── prompt_generator.py             # Prompt generation utilities
│   ├── pipeline.py                     # Main evaluation pipeline
│   ├── clean_utils.py                  # Text cleaning utilities
│   ├── eval_utils.py                   # Evaluation metrics
│   └── analyze_results.py              # Results analysis
├── results/                            # Evaluation results
│   └── result_[timestamp]/             # Timestamped results directory
│       ├── evaluation_results.json     # Evaluation results
│       └── analysis/                   # Analysis outputs
│           ├── plots/                  # Generated visualizations
│           └── summary_statistics.json # Summary of results
├── eval.py                             # Script to run the evaluation pipeline
├── analyze.py                          # Script to analyse existing results
├── eval_and_analyze.py                 # Script to run evaluation and analysis together
├── example_usage.ipynb                 # Jupyter notebook with code examples
└── requirements.txt                    # Project dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy `.env.template`, add your Cohere API key, and rename the copied file to `.env`.
## Usage

### Running the Evaluation Pipeline

```bash
python eval.py --workers 10 --languages Spanish French Chinese --prompt-types zero-shot
```

This will:
1. Create a timestamped directory in `results/`
2. Save the evaluation results as `evaluation_results.json` in that directory
3. Print a summary of the evaluation results

If you don't specify the `--prompt-types` parameter, the pipeline will default to using both Zero-Shot and Few-Shot prompt types.

Use the `--verbose` flag to display detailed information during execution:

```bash
python eval.py --workers 10 --languages Spanish --prompt-types zero-shot --verbose
```

### Analyzing Results

You can analyze existing results using the command-line script:

```bash
python analyze.py results/result_[timestamp]/evaluation_results.json
```

Analysis outputs will be saved to `results/result_[timestamp]/analysis/`.

### Running Evaluation and Analysis Together

For convenience, you can run the evaluation and analysis in a single command:

```bash
python eval_and_analyze.py --workers 10 --languages Spanish French Chinese --prompt-types zero-shot
```

This will:
1. Create a timestamped directory in `results/`
2. Run the evaluation and save results to `evaluation_results.json` in that directory
3. Automatically run the analysis on those results
4. Save analysis outputs (visualizations, statistics) to the `analysis/` subdirectory

You can also use the `--verbose` flag with this script:

```bash
python eval_and_analyze.py --workers 10 --languages Spanish --prompt-types zero-shot --verbose
```

### Using Python Code

You can also use the pipeline programmatically in your own Python code:

```python
from pipeline.pipeline import run_evaluation_pipeline
from pipeline.analyze_results import analyze_results

# Run evaluation
run_evaluation_pipeline(
    dataset_path="json datset name inside `data/` folder",
    output_path="results/result_20250408/evaluation_results.json",
    max_workers=10,  # Number of parallel workers
    languages=["Spanish", "French", "Chinese"],  # Specific languages to test
    prompt_types=["Zero-Shot"],  # Specific prompt types to test (defaults to both if not specified)
    verbose=False  # Set to True for detailed output
)

# Analyze results
analyze_results(
    "results/result_20250408/evaluation_results.json", 
    "results/result_20250408/analysis"
)
```

For more detailed examples, see the `example_usage.ipynb` Jupyter notebook included in this repository.

## Command-Line Arguments

### eval.py Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Path to the test dataset JSON file | data/test_dataset.json |
| `--output` | Path to save the evaluation results | results/evaluation_results.json |
| `--workers` | Maximum number of worker threads | 5 |
| `--languages` | List of languages to test (e.g., Spanish French Chinese) | All available languages |
| `--prompt-types` | List of prompt types to test (zero-shot and/or few-shot) | Both types |
| `--model` | Cohere model name to use for translations | c4ai-aya-expanse-32b |
| `--verbose` | Print detailed information during execution | False |

### analyze.py Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `results_file` | Path to the evaluation results JSON file (required) | None |
| `--output-dir` | Directory to save the analysis results | results/analysis |

### eval_and_analyze.py Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Path to the test dataset JSON file | data/test_dataset.json |
| `--output` | Path to save the evaluation results | results/evaluation_results.json |
| `--workers` | Maximum number of worker threads | 5 |
| `--languages` | List of languages to test (e.g., Spanish French Chinese) | All available languages |
| `--prompt-types` | List of prompt types to test (zero-shot and/or few-shot) | Both types |
| `--model` | Cohere model name to use for translations | c4ai-aya-expanse-32b |
| `--analysis-dir` | Directory to save the analysis results | results/analysis |
| `--verbose` | Print detailed information during execution | False |

## Test Dataset Format

The test dataset should be a JSON file with the following structure:

```json
[
  {
    "input_text": "buy blue cotton shirt for men",
    "translation_spanish": "comprar camisa de algodón azul para hombres",
    "translation_french": "acheter une chemise en coton bleu pour hommes",
    ...
  },
  ...
]
```

## Prompt Templates

Prompt templates are stored as `.txt` files in the `prompt_data` directory, organized by type (few_shot or zero_shot). Each template can include placeholders like `{user_input}`,`{target_language}` and `{example1}` that will be replaced with language-specific data.

### Basic Prompt Types

A **preamble** is a prompt that does not have `{user_input}` – this will be detected and the input will be added separately.

A prompt could look like this:

```
Translate {user_input} into {target_language}.
```

A preamble could look like this:

```
Translate the input text into {target_language}.
```

### Required Placeholders

- All prompts must include the `{target_language}` placeholder.
- The `{user_input}` placeholder is optional. If not included, the input will be treated as a preamble and the user input will be inserted later.
- Few-shot prompts in both `zero_shot/` and `few_shot/` directories should include example placeholders: `{example1}`, `{example2}`, `{example3}`.

### Prompt Metadata

Prompts can include metadata by adding special markers at the beginning of the file. Currently supported metadata:

- **Reasoning**: Add `#! reasoning` as the first line of a prompt to mark it as a reasoning prompt. This will:
  1. Add instructions to the model to enclose reasoning in `<thinking>...</thinking>` tags
  2. Automatically filter out content between these tags in the final translation

Example reasoning prompt:

```
#! reasoning
You are an expert translator. Translate the following text into {target_language} step by step.

Input Text: "{user_input}"

First, analyze the input:
1. What are the key terms and concepts?
2. Are there any idiomatic expressions?
3. What is the appropriate tone for the translation?

Then, translate the text into {target_language}:
```

The system will automatically remove the metadata line and any text between `<thinking>` tags in the model's response.


## Results Analysis

The analysis script generates:

1. Summary statistics in JSON format
2. Visualizations of BLEU scores by language, prompt type, and prompt name
3. Heatmaps showing performance across languages and prompt types
4. Distribution plots of BLEU scores
5. A CSV file with all evaluation results for further analysis

### Analysis Features

- **Average BLEU Scores by Language and Prompt Type**: Shows how different prompt types perform across languages
- **Average BLEU Scores by Language and Prompt Name**: Shows how specific prompt perform across languages
- **BLEU Score Distribution**: Shows the distribution of BLEU scores for different combinations
- **Top Performing Combinations**: Identifies the best language-prompt combinations

# Tl;dr
```
python eval_and_analyze.py --dataset <dataset-location> --workers 15 --languages Spanish French Chinese Hindi 
```
Is probably all you need to run. Leave `--dataset` empty to default to an inbuilt test set. However, the few-shot prompts will leak most of the answers to the inbuilt test set.
