"""
Configuration loader for the refine-pipeline.
Handles loading and validation of YAML configuration files.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Data configuration parameters."""
    metadata_dir: str
    depth_metadata_file: str
    object_descriptions_file: str
    foreground_annotations_file: str
    max_images: Optional[int]
    require_all_metadata: bool

@dataclass 
class APIConfig:
    """API configuration parameters."""
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int

@dataclass
class GenerationConfig:
    """Generation parameters."""
    temperature: float
    max_tokens: int
    top_p: float
    presence_penalty: float
    top_k: int

@dataclass
class ProcessingConfig:
    """Processing configuration parameters."""
    num_workers: Optional[int]
    batch_size: int
    save_frequency: int

@dataclass
class PromptConfig:
    """Prompt distribution configuration."""
    prompts_per_image: int
    distribution: Dict[str, int]
    shuffle_dataset: bool
    random_seed: Optional[int]

@dataclass
class OutputConfig:
    """Output configuration parameters."""
    output_file: str
    include_metadata: bool
    include_prompt_category: bool

@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str
    format: str
    show_progress: bool
    progress_frequency: int

@dataclass
class DevelopmentConfig:
    """Development/testing configuration."""
    sample_mode: bool
    sample_size: int
    debug: bool

@dataclass
class Config:
    """Complete configuration object."""
    data: DataConfig
    api: APIConfig
    generation: GenerationConfig
    processing: ProcessingConfig
    prompts: PromptConfig
    output: OutputConfig
    logging: LoggingConfig
    development: DevelopmentConfig

class ConfigLoader:
    """Loads and validates configuration from YAML files."""
    
    @staticmethod
    def load_config(config_path: str) -> Config:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        
        # Validate and convert to config objects
        try:
            config = ConfigLoader._create_config_objects(yaml_data)
            ConfigLoader._validate_config(config)
            return config
        except Exception as e:
            raise ValueError(f"Error processing configuration: {e}")
    
    @staticmethod
    def _create_config_objects(yaml_data: Dict[str, Any]) -> Config:
        """Create configuration objects from YAML data."""
        
        # Data configuration
        data_config = DataConfig(
            metadata_dir=yaml_data['data']['metadata_dir'],
            depth_metadata_file=yaml_data['data']['files']['depth_metadata'],
            object_descriptions_file=yaml_data['data']['files']['object_descriptions'],
            foreground_annotations_file=yaml_data['data']['files']['foreground_annotations'],
            max_images=yaml_data['data'].get('max_images'),
            require_all_metadata=yaml_data['data'].get('require_all_metadata', True)
        )
        
        # API configuration
        api_config = APIConfig(
            base_url=yaml_data['api']['base_url'],
            api_key=yaml_data['api']['api_key'],
            model=yaml_data['api']['model'],
            timeout_seconds=yaml_data['api']['timeout_seconds'],
            max_retries=yaml_data['api']['max_retries']
        )
        
        # Generation configuration
        generation_config = GenerationConfig(
            temperature=yaml_data['generation']['temperature'],
            max_tokens=yaml_data['generation']['max_tokens'],
            top_p=yaml_data['generation']['top_p'],
            presence_penalty=yaml_data['generation']['presence_penalty'],
            top_k=yaml_data['generation']['top_k']
        )
        
        # Processing configuration
        processing_config = ProcessingConfig(
            num_workers=yaml_data['processing']['num_workers'],
            batch_size=yaml_data['processing']['batch_size'],
            save_frequency=yaml_data['processing']['save_frequency']
        )
        
        # Prompt configuration
        prompt_config = PromptConfig(
            prompts_per_image=yaml_data['prompts']['prompts_per_image'],
            distribution=yaml_data['prompts']['distribution'],
            shuffle_dataset=yaml_data['prompts']['shuffle_dataset'],
            random_seed=yaml_data['prompts']['random_seed']
        )
        
        # Output configuration
        output_config = OutputConfig(
            output_file=yaml_data['output']['output_file'],
            include_metadata=yaml_data['output']['include_metadata'],
            include_prompt_category=yaml_data['output']['include_prompt_category']
        )
        
        # Logging configuration
        logging_config = LoggingConfig(
            level=yaml_data['logging']['level'],
            format=yaml_data['logging']['format'],
            show_progress=yaml_data['logging']['show_progress'],
            progress_frequency=yaml_data['logging']['progress_frequency']
        )
        
        # Development configuration
        development_config = DevelopmentConfig(
            sample_mode=yaml_data['development']['sample_mode'],
            sample_size=yaml_data['development']['sample_size'],
            debug=yaml_data['development']['debug']
        )
        
        return Config(
            data=data_config,
            api=api_config,
            generation=generation_config,
            processing=processing_config,
            prompts=prompt_config,
            output=output_config,
            logging=logging_config,
            development=development_config
        )
    
    @staticmethod
    def _validate_config(config: Config) -> None:
        """Validate configuration parameters."""
        
        # Validate prompt distribution sums to 100
        total_percentage = sum(config.prompts.distribution.values())
        if total_percentage != 100:
            raise ValueError(f"Prompt distribution percentages must sum to 100, got {total_percentage}")
        
        # Validate all percentages are positive
        for category, percentage in config.prompts.distribution.items():
            if percentage < 0:
                raise ValueError(f"Prompt percentage for '{category}' must be non-negative, got {percentage}")
        
        # Validate prompts_per_image is 1
        if config.prompts.prompts_per_image != 1:
            logger.warning(f"prompts_per_image is {config.prompts.prompts_per_image}, but requirement specifies 1 prompt per image")
        
        # Validate file paths exist (relative to config file)
        metadata_dir = Path(config.data.metadata_dir)
        if not metadata_dir.exists():
            logger.warning(f"Metadata directory not found: {metadata_dir}")
        
        # Validate generation parameters
        if not (0 <= config.generation.temperature <= 1):
            raise ValueError(f"Temperature must be between 0 and 1, got {config.generation.temperature}")
        
        if not (0 <= config.generation.top_p <= 1):
            raise ValueError(f"top_p must be between 0 and 1, got {config.generation.top_p}")
        
        if config.generation.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {config.generation.max_tokens}")
        
        # Validate processing parameters
        if config.processing.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {config.processing.batch_size}")
        
        if config.processing.num_workers is not None and config.processing.num_workers <= 0:
            raise ValueError(f"num_workers must be positive or null, got {config.processing.num_workers}")
        
        logger.info("Configuration validation passed")
    
    @staticmethod
    def save_config(config: Config, config_path: str) -> None:
        """Save configuration to YAML file."""
        # Convert config back to dictionary format
        config_dict = {
            'data': {
                'metadata_dir': config.data.metadata_dir,
                'files': {
                    'depth_metadata': config.data.depth_metadata_file,
                    'object_descriptions': config.data.object_descriptions_file,
                    'foreground_annotations': config.data.foreground_annotations_file
                },
                'max_images': config.data.max_images,
                'require_all_metadata': config.data.require_all_metadata
            },
            'api': {
                'base_url': config.api.base_url,
                'api_key': config.api.api_key,
                'model': config.api.model,
                'timeout_seconds': config.api.timeout_seconds,
                'max_retries': config.api.max_retries
            },
            'generation': {
                'temperature': config.generation.temperature,
                'max_tokens': config.generation.max_tokens,
                'top_p': config.generation.top_p,
                'presence_penalty': config.generation.presence_penalty,
                'top_k': config.generation.top_k
            },
            'processing': {
                'num_workers': config.processing.num_workers,
                'batch_size': config.processing.batch_size,
                'save_frequency': config.processing.save_frequency
            },
            'prompts': {
                'prompts_per_image': config.prompts.prompts_per_image,
                'distribution': config.prompts.distribution,
                'shuffle_dataset': config.prompts.shuffle_dataset,
                'random_seed': config.prompts.random_seed
            },
            'output': {
                'output_file': config.output.output_file,
                'include_metadata': config.output.include_metadata,
                'include_prompt_category': config.output.include_prompt_category
            },
            'logging': {
                'level': config.logging.level,
                'format': config.logging.format,
                'show_progress': config.logging.show_progress,
                'progress_frequency': config.logging.progress_frequency
            },
            'development': {
                'sample_mode': config.development.sample_mode,
                'sample_size': config.development.sample_size,
                'debug': config.development.debug
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")


def main():
    """Test the config loader."""
    try:
        config = ConfigLoader.load_config("config.yaml")
        print("Configuration loaded successfully!")
        print(f"Data dir: {config.data.metadata_dir}")
        print(f"Model: {config.api.model}")
        print(f"Prompt distribution: {config.prompts.distribution}")
        print(f"Total percentage: {sum(config.prompts.distribution.values())}%")
    except Exception as e:
        print(f"Error loading config: {e}")


if __name__ == "__main__":
    main()
