"""
Configuration updates for Cohere API integration.

This module adds support for Cohere API configuration in the translation pipeline.
"""

import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Optional


# Load environment variables from .env file
load_dotenv()

@dataclass
class TranslationConfig:
    """Updated configuration class for translation pipeline parameters with Cohere API support."""
    
    # Model parameters
    model_name: str
    temperature: float
    precision: str  # "float16", "float32", "int8", etc.
    
    # Processing mode
    batch_processing: bool
    
    # Languages
    languages: List[str]
    
    # File paths
    dataset_path: str
    human_values_translation_path: str
    error_log_path: str
    log_path: str
    sample_dataset_path: str
    output_dir: str
    intermediate_dir: str
    
    # HuggingFace access token (from environment variable)
    hf_access_token: Optional[str] = None
    
    # Optional parameters with default values
    max_new_tokens: int = 100
    do_sample: bool = True
    optimal_batch_size: Optional[int] = None
    
    # Cohere API parameters
    use_cohere_api: bool = False
    cohere_model: str = "c4ai-aya-expanse-32b"
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Parent validation logic remains unchanged
        # Get HF token from environment if not provided
        if not self.hf_access_token and not self.use_cohere_api:
            self.hf_access_token = os.environ.get("HF_ACCESS_TOKEN")
            if not self.hf_access_token:
                raise ValueError("HuggingFace access token not found in config or environment variables")
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.intermediate_dir, os.path.dirname(self.error_log_path), 
                         os.path.dirname(self.log_path)]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> TranslationConfig:
    """
    Load configuration from a YAML file with Cohere API support.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        TranslationConfig object with loaded configuration
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create output and intermediate directories based on config
        base_output_dir = config_data.get('output_dir', 'output')
        config_data['intermediate_dir'] = os.path.join(base_output_dir, 'intermediate')
        
        # Create and return config object
        return TranslationConfig(**config_data)
    
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError:
        logging.error(f"Error parsing YAML configuration file: {config_path}")
        raise
    except TypeError as e:
        logging.error(f"Invalid configuration parameters: {e}")
        raise ValueError(f"Invalid configuration parameters: {e}")


def save_config(config: TranslationConfig, output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: TranslationConfig object to save
        output_path: Path where to save the configuration
    """
    # Convert dataclass to dictionary
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    
    # Save to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)