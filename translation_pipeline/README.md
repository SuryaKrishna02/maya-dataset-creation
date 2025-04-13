# Translation Pipeline

A pipeline for translating datasets using language models. This pipeline is designed to translate content from a dataset into multiple languages using the Aya model.

## Features

- Translate datasets to multiple languages
- Resume translation from any point
- Batch processing with automatic optimal batch size calculation
- Progress tracking and detailed logging
- Error handling and recovery

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- Hugging Face account with access to the specified model

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/translation-pipeline.git
   cd translation-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   # Create a .env file with your Hugging Face token
   echo "HF_ACCESS_TOKEN=your_token_here" > .env
   ```

## Usage

### Configuration

Create a configuration file in YAML format. See `config/config.yaml` for an example:

```yaml
# Model parameters
model_name: "CohereForAI/aya-expanse-32b"
temperature: 0.1
precision: "float16"

# Processing mode
batch_processing: true

# Languages to translate to
languages:
  - hindi
  - spanish
  - french
  - german

# File paths
dataset_path: "data/blip_laion_cc_sbu_558k.json"
human_values_translation_path: "data/human_values_translation.json"
error_log_path: "logs/error.log"
log_path: "logs/translation.log"
sample_dataset_path: "data/sample_dataset.json"
output_dir: "output"

# Optional parameters
max_new_tokens: 100
do_sample: true
optimal_batch_size: null  # Automatically calculated
```

### Running the Pipeline

Run the pipeline with your configuration file:

```bash
python main.py --config config/config.yaml
```

Additional options:

- `--resume`: Resume translation from the last checkpoint
- `--language LANG`: Process only a specific language
- `--batch-size SIZE`: Override the automatic batch size calculation
- `--skip-optimization`: Skip the batch size optimization step
- `--output-only`: Only process output (merge intermediate files)

Examples:

```bash
# Run with default settings
python main.py --config config/config.yaml

# Resume translation
python main.py --config config/config.yaml --resume

# Translate only to Hindi
python main.py --config config/config.yaml --language hindi

# Use a fixed batch size
python main.py --config config/config.yaml --batch-size 8

# Only process the output files
python main.py --config config/config.yaml --output-only
```

## Input Data Structure

The input dataset should be a JSON file with the following structure:

```json
[
  {
    "id": "004539375",
    "image": "00453/004539375.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Render a clear and concise summary of the photo.\n<image>"
      },
      {
        "from": "gpt",
        "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
      }
    ]
  },
  ...
]
```

## Human Values Translation

The human values translation file should be a JSON file mapping original values to translations for each language:

```json
{
  "hindi": {
    "Render a clear and concise summary of the photo.\n<image>": "फोटो का एक स्पष्ट और संक्षिप्त सारांश दें।\n<image>",
    ...
  },
  "spanish": {
    "Render a clear and concise summary of the photo.\n<image>": "Proporciona un resumen claro y conciso de la foto.\n<image>",
    ...
  },
  ...
}
```

## Output Structure

The output files will be saved in the specified output directory, with one file per language:

```
output/
├── hindi_translated.json
├── spanish_translated.json
├── french_translated.json
└── german_translated.json
```

Each output file follows the same structure as the input dataset, but with translated values.

## Folder Structure

```
translation_pipeline/
├── utils/
│   ├── __init__.py
│   ├── config.py            # Configuration handling
│   ├── prompt_templates.py  # Translation prompt templates
│   ├── batch_optimizer.py   # Batch size optimization
│   ├── data_loader.py       # Data loading and processing
│   └── logger.py            # Logging functionality
├── translation/
│   ├── __init__.py
│   ├── translator.py        # Translation logic
│   └── output_processor.py  # Output processing
├── config/
│   └── config.yaml          # Example configuration
├── .env                     # Environment variables
├── main.py                  # Main script
└── README.md                # Documentation
```

## License

[MIT License](LICENSE)
