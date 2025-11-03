"""
Main script for processing the refine-pipeline dataset with comprehensive visual understanding.
This script uses YAML configuration and implements percentage-based prompt distribution.
"""

import argparse
import logging
import sys
from pathlib import Path

from config_loader import ConfigLoader
from data_processor import DataProcessor
from multiprocessing_handler import MultiprocessingHandler
from prompt_distributor import PromptDistributor


def setup_logging(config):
    """Setup logging based on configuration."""
    level = getattr(logging, config.logging.level.upper())
    logging.basicConfig(
        level=level,
        format=config.logging.format
    )


def main():
    """Main function to process the dataset."""
    parser = argparse.ArgumentParser(
        description="Process refine-pipeline dataset using YAML configuration"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--sample_mode', 
        action='store_true',
        help='Override config to run in sample mode'
    )
    
    parser.add_argument(
        '--sample_size', 
        type=int,
        help='Override sample size from config'
    )
    
    parser.add_argument(
        '--max_images', 
        type=int,
        help='Override max_images from config'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Loading configuration from: {args.config}")
        config = ConfigLoader.load_config(args.config)
        
        # Setup logging based on config
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        # Apply command line overrides
        if args.sample_mode:
            config.development.sample_mode = True
            logger.info("Sample mode enabled via command line")
        
        if args.sample_size:
            config.development.sample_size = args.sample_size
            logger.info(f"Sample size overridden to: {args.sample_size}")
        
        if args.max_images:
            config.data.max_images = args.max_images
            logger.info(f"Max images overridden to: {args.max_images}")
        
        if args.debug:
            config.development.debug = True
            config.logging.level = "DEBUG"
            setup_logging(config)
            logger.info("Debug mode enabled")
        
        # Print configuration summary
        logger.info("=== Processing Configuration ===")
        logger.info(f"Metadata directory: {config.data.metadata_dir}")
        logger.info(f"Model: {config.api.model}")
        logger.info(f"Workers: {config.processing.num_workers or 'CPU count'}")
        logger.info(f"Batch size: {config.processing.batch_size}")
        logger.info(f"Require all metadata: {config.data.require_all_metadata}")
        logger.info(f"Output file: {config.output.output_file}")
        logger.info(f"Sample mode: {config.development.sample_mode}")
        
        if config.development.sample_mode:
            logger.info(f"Sample size: {config.development.sample_size}")
        elif config.data.max_images:
            logger.info(f"Max images: {config.data.max_images}")
        
        # Print prompt distribution
        logger.info("Prompt distribution:")
        for category, percentage in config.prompts.distribution.items():
            logger.info(f"  {category}: {percentage}%")
        
        logger.info("=== End Configuration ===")
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Data processor
        data_processor = DataProcessor(".", metadata_dir=config.data.metadata_dir)
        
        # Multiprocessing handler
        handler = MultiprocessingHandler(config)
        
        # Prompt distributor
        prompt_distributor = PromptDistributor(
            distribution=config.prompts.distribution,
            random_seed=config.prompts.random_seed,
            shuffle_dataset=config.prompts.shuffle_dataset
        )
        
        # Process dataset
        if config.development.sample_mode:
            logger.info(f"Processing sample dataset with {config.development.sample_size} images...")
            results = handler.process_sample(
                data_processor, 
                prompt_distributor,
                num_images=config.development.sample_size
            )
        else:
            logger.info("Processing full dataset...")
            results = handler.process_dataset(data_processor, prompt_distributor)
        
        # Print final summary
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        logger.info("=== Final Summary ===")
        logger.info(f"Total processed: {len(results)}")
        logger.info(f"Successful: {len(successful_results)}")
        logger.info(f"Failed: {len(failed_results)}")
        
        if successful_results:
            avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
            logger.info(f"Average processing time: {avg_time:.2f}s")
            
            # Print distribution summary
            distribution_summary = prompt_distributor.get_distribution_summary(
                [PromptAssignment(r.image_path, r.prompt, r.prompt_category) for r in successful_results]
            )
            logger.info("Actual prompt distribution:")
            for category, count in distribution_summary.items():
                percentage = count / len(successful_results) * 100
                logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        logger.info(f"Results saved to: {config.output.output_file}")
        
        # Print some sample results if in sample mode or debug
        if (config.development.sample_mode or config.development.debug) and successful_results:
            logger.info("\n=== Sample Results ===")
            for i, result in enumerate(successful_results[:3]):
                logger.info(f"\n--- Sample {i + 1} ---")
                logger.info(f"Image: {result.image_path}")
                logger.info(f"Category: {result.prompt_category}")
                logger.info(f"Prompt: {result.prompt[:100]}...")
                if result.response:
                    logger.info(f"Response: {result.response[:200]}...")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Import here to avoid circular import in case of missing dependencies
    from prompt_distributor import PromptAssignment
    main()