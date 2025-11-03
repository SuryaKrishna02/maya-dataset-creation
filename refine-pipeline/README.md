# Refine-Pipeline: High-Throughput Visual Understanding Dataset Generation

This pipeline processes multiple JSONL metadata files to generate comprehensive visual understanding datasets using percentage-based prompt distribution and YAML configuration. Each image gets exactly one prompt based on configurable distribution percentages.

## Overview

The refine-pipeline combines metadata from three sources in the `metadata/` folder:
- `depth_maps_metadata.jsonl` - Depth information and spatial analysis (554,633 items)
- `obj_extr_from_desc.jsonl` - Objects extracted from descriptions (536,730 items)  
- `fg_anno.jsonl` - Foreground annotations with bounding boxes (549,089 items)

**Key Enhancement**: Only processes images present in ALL three metadata files (517,640 images), ensuring complete information for each processed image.

The system generates responses using 9 categories of human-like prompts with configurable percentage distribution for balanced dataset creation.

## Features

- **YAML Configuration**: Hierarchical configuration with all parameters organized by category
- **Percentage-based Distribution**: Each image gets exactly one prompt based on configurable percentages
- **Common Images Only**: Processes only images present in ALL metadata files for complete information
- **Diverse Prompts**: 36 unique human-like prompts across 9 categories
- **Multiprocessing Support**: High-throughput processing with configurable worker processes
- **Resume Capability**: Automatic saving allows resuming interrupted processing
- **Reproducible Results**: Fixed random seed for consistent prompt assignment
- **Error Handling**: Robust retry logic and error reporting

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test with Sample Data

```bash
python3 main.py --sample_mode --sample_size 5
```

### 2. Process with Custom Limits

```bash
python3 main.py --max_images 100 --debug
```

### 3. Full Dataset Processing

```bash
python3 main.py
```

All configuration is managed through `config.yaml` - no need for complex command line arguments!

## File Structure

```
refine-pipeline/
├── config.yaml               # YAML configuration file
├── main.py                   # Main processing script  
├── config_loader.py          # Configuration loading and validation
├── data_processor.py         # Metadata combination and processing
├── multiprocessing_handler.py # High-throughput processing engine
├── prompt_templates.py       # Human-like prompt templates
├── prompt_distributor.py     # Percentage-based prompt distribution
├── test_pipeline.py          # Component testing
├── run_example.py           # Usage examples
├── requirements.txt         # Dependencies
├── metadata/                # Input metadata folder
│   ├── depth_maps_metadata.jsonl
│   ├── obj_extr_from_desc.jsonl
│   └── fg_anno.jsonl
└── generated_responses.jsonl # Generated responses (output)
```

## Prompt Categories

The system uses 9 categories of human-like prompts:

1. **Scene Understanding & Context** - Overall scene description
2. **Observational & Analytical** - Detailed visual analysis
3. **Atmospheric & Sensory** - Mood, lighting, and ambiance
4. **Narrative & Contextual** - Story and context inference
5. **Spatial & Relational** - Spatial relationships and layout
6. **Functional & Purpose-Driven** - Purpose and utility analysis
7. **Comparative & Detailed** - Comparisons and contrasts
8. **Perspective & Experience** - First-person experience description
9. **Comprehensive Analysis** - Thorough multi-aspect analysis

## Configuration

All configuration is handled through `config.yaml` with hierarchical organization:

### Data Configuration
```yaml
data:
  metadata_dir: "metadata"        # Metadata folder location
  max_images: null               # Limit processing (null = all)
  require_all_metadata: true     # Only process images in ALL files
```

### Prompt Distribution (Must sum to 100%)
```yaml
prompts:
  prompts_per_image: 1           # Exactly one prompt per image
  distribution:
    scene_understanding: 12      # 12% of images
    observational_analytical: 12 # 12% of images
    # ... (see config.yaml for full distribution)
  shuffle_dataset: true          # Randomize assignment
  random_seed: 42               # For reproducible results
```

### Processing Configuration
```yaml
processing:
  num_workers: null             # null = CPU count
  batch_size: 50               # Images per batch
  save_frequency: 100          # Save every N items
```

### Command Line Overrides
- `--config`: Specify config file (default: config.yaml)
- `--sample_mode`: Enable sample mode
- `--sample_size`: Override sample size
- `--max_images`: Override max images limit
- `--debug`: Enable debug output

## Performance Optimization

### For High Throughput:
Edit `config.yaml`:
```yaml
processing:
  num_workers: 16
  batch_size: 200
  save_frequency: 50

api:
  max_retries: 5
  timeout_seconds: 60
```

### For GPU Optimization:
- Adjust `num_workers` based on GPU memory and API limits
- Increase `batch_size` for better GPU utilization
- Monitor GPU usage and adjust accordingly
- Use `development.debug: true` for detailed monitoring

## Comprehensive Prompt Format

The system creates rich, structured prompts that integrate all metadata sources:

### System Prompt Structure:
1. **Introduction**: Guidance for comprehensive visual analysis
2. **Spatial and Depth Analysis**: Detailed depth map information including:
   - Depth range, mean, median, standard deviation
   - Percentile distributions (10/25/75/90)
   - Spatial distribution by grid regions (3x3)
   - Field of view and focal length data
3. **Object Detection and Localization**: Structured table with:
   - Object descriptions from image analysis
   - Bounding box coordinates [x1, y1, x2, y2]
   - Object depth values
   - Area measurements
4. **Scene Context and Elements**: High-level scene understanding:
   - Extracted objects from descriptions
   - Comprehensive scene narrative
5. **Response Guidelines**: Detailed instructions for generating rich, comprehensive responses

### Example System Prompt Structure:
```
You are provided with rich visual and spatial metadata...

**Spatial and Depth Analysis:**
The depth map spans from 19.62 m to 37.01 m (mean 24.87 m, median 24.05 m, σ=3.65)...

**Object Detection and Localization:**
For image dimensions of height=336 and width=681 pixels:
Descriptive Objects | Bounding Box Coordinates | Object Depth | Area
"a man in a black and yellow outfit" | [256, 187, 309, 328] | 246 | 13489
"a white car" | [184, 0, 538, 233] | 178 | 125579

**Scene Context and Elements:**
Objects: "white car", "hood", "grassy area", "dents"...
Concise Description: The image shows a white car with its hood open...

**Response Guidelines:**
When generating responses, integrate this metadata to provide rich descriptions...
```

## Output Format

Each line in the output JSONL file contains:

```json
{
  "image_path": "00000/000000010.jpg",
  "prompt": "Tell me what's happening in this image...",
  "prompt_category": "scene_understanding", 
  "response": "Generated response describing the scene...",
  "metadata": {
    "image_path": "00000/000000010.jpg",
    "depth_metadata": "The depth map spans from 19.62 m to 37.01 m...",
    "extracted_objects_description": ["white car", "hood", "grassy area"],
    "scene_description": "The image shows a white car with its hood open...",
    "extracted_objects_image": ["a man in a black and yellow outfit", "a white car"],
    "bounding_boxes": [[256, 187, 309, 328], [184, 0, 538, 233]],
    "object_depths": [246, 178],
    "object_sizes": [13489, 125579],
    "image_width": 681,
    "image_height": 336
  },
  "processing_time": 1.23,
  "success": true,
  "error_message": null
}
```

## Dataset Statistics

- **Total Images**: 517,640 (present in all three metadata files)
- **Prompts per Image**: 1 (based on percentage distribution)
- **Prompt Categories**: 9 categories with configurable percentages
- **Expected Output**: 517,640 responses with complete metadata

## Error Handling

The system includes:
- Automatic retries with exponential backoff
- Graceful error handling and logging
- Failed request tracking
- Resume capability for interrupted processing

## Monitoring Progress

The system provides real-time progress updates:
- Processing rate (items/second)
- Success/failure counts
- Average processing time
- Estimated completion time

## Testing and Examples

### Run Tests
```bash
python3 test_pipeline.py
```

### See Examples  
```bash
python3 run_example.py
```

### Monitor Progress
The system provides real-time progress updates showing:
- Processing rate (items/second)
- Success/failure counts  
- Prompt distribution statistics
- Estimated completion time

## Troubleshooting

### Common Issues:

1. **Memory Issues**: Reduce `batch_size` or `num_workers`
2. **API Rate Limits**: Reduce `num_workers` or increase `timeout_seconds`
3. **File Not Found**: Ensure all three JSONL files are in the data directory
4. **GPU Memory**: Monitor GPU usage and adjust batch sizes

### Logs:
Check console output for detailed progress information and error messages.

## License

See LICENSE file in the parent directory.
