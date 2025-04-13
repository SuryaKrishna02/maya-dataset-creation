#!/usr/bin/env python
"""
Main script for the translation pipeline.

This script ties together all components of the translation pipeline
and provides a command-line interface for running the pipeline.
"""

import os
import sys
import time
import torch
import argparse
from datetime import datetime
from utils.config import load_config
from utils.logger import setup_logger
from translation.translator import translate_dataset
from translation.output_processor import process_output
from utils.batch_optimizer import calculate_optimal_batch_size


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Translation Pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume translation from the last checkpoint'
    )
    parser.add_argument(
        '--language', 
        type=str,
        help='Process only this language instead of all languages in config'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override the automatic batch size calculation with a fixed batch size'
    )
    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip the batch size optimization step'
    )
    parser.add_argument(
        '--output-only',
        action='store_true',
        help='Only process output (merge intermediate files)'
    )
    
    return parser.parse_args()


def check_system_requirements():
    """Check if the system meets the requirements for running the pipeline."""
    # Check CUDA availability for GPU support
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU will be very slow.")
        return False
    
    # Check available memory
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        print(f"Available GPU memory: {total_memory_gb:.2f} GB")
        
        # Warning if memory is less than 8GB
        if total_memory_gb < 8:
            print("WARNING: Low GPU memory. This may cause out-of-memory errors.")
            return False
    
    return True


def initialize_pipeline(args):
    """Initialize the translation pipeline components."""
    # Load configuration
    config = load_config(args.config)
    
    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.log_path or os.path.join("logs", f"translation_{timestamp}.log")
    error_log_path = config.error_log_path or os.path.join("logs", f"error_{timestamp}.log")
    
    logger = setup_logger(log_path, error_log_path)
    log = logger.get_logger()
    
    # Log system info
    log.info(f"Starting translation pipeline at {datetime.now().isoformat()}")
    log.info(f"Python version: {sys.version}")
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Filter languages if specified
    if args.language:
        if args.language in config.languages:
            config.languages = [args.language]
            log.info(f"Processing only language: {args.language}")
        else:
            log.error(f"Specified language '{args.language}' not found in config")
            sys.exit(1)
    
    return config, logger


def optimize_batch_size(config, logger, args):
    """Calculate the optimal batch size if not already set."""
    log = logger.get_logger()
    
    # Skip optimization if requested or if batch size is provided
    if args.skip_optimization:
        log.info("Skipping batch size optimization as requested")
        return
    
    if args.batch_size:
        log.info(f"Using provided batch size: {args.batch_size}")
        config.optimal_batch_size = args.batch_size
        return
    
    if config.optimal_batch_size:
        log.info(f"Using batch size from config: {config.optimal_batch_size}")
        return
    
    # Calculate optimal batch size
    log.info("Calculating optimal batch size...")
    
    try:
        sample_language = config.languages[0]
        
        batch_size = calculate_optimal_batch_size(
            model_name=config.model_name,
            language=sample_language,
            sample_dataset_path=config.sample_dataset_path,
            precision=config.precision,
            hf_access_token=config.hf_access_token,
            logger=logger
        )
        
        config.optimal_batch_size = batch_size
        log.info(f"Calculated optimal batch size: {batch_size}")
        
    except Exception as e:
        log.error(f"Error calculating optimal batch size: {e}")
        log.info("Using default batch size of 4")
        config.optimal_batch_size = 4


def run_translation(config, logger, args):
    """Run the translation process."""
    log = logger.get_logger()
    
    # Skip translation if output-only is specified
    if args.output_only:
        log.info("Skipping translation process as output-only is specified")
        return True
    
    # Translate dataset
    log.info("Starting translation process...")
    
    try:
        start_time = time.time()
        results = translate_dataset(config, logger)
        end_time = time.time()
        
        # Log results
        success = all(results.values())
        duration = end_time - start_time
        log.info(f"Translation completed in {duration:.2f} seconds")
        
        for language, success in results.items():
            status = "succeeded" if success else "failed"
            log.info(f"Translation for {language}: {status}")
        
        return success
        
    except Exception as e:
        log.error(f"Error in translation process: {e}")
        return False


def process_final_output(config, logger):
    """Process the final output by merging intermediate files."""
    log = logger.get_logger()
    
    # Process output
    log.info("Processing final output...")
    
    try:
        start_time = time.time()
        results = process_output(config, logger)
        end_time = time.time()
        
        # Log results
        success = all(results.values())
        duration = end_time - start_time
        log.info(f"Output processing completed in {duration:.2f} seconds")
        
        for language, success in results.items():
            status = "succeeded" if success else "failed"
            log.info(f"Output processing for {language}: {status}")
        
        return success
        
    except Exception as e:
        log.error(f"Error in output processing: {e}")
        return False


def main():
    """Main entry point for the translation pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Check system requirements
    system_ok = check_system_requirements()
    if not system_ok:
        print("System does not meet all requirements. Continue anyway? (y/n)")
        response = input().lower()
        if response != 'y':
            sys.exit(1)
    
    # Initialize pipeline
    config, logger = initialize_pipeline(args)
    log = logger.get_logger()
    
    try:
        # Optimize batch size
        optimize_batch_size(config, logger, args)
        
        # Run translation
        translation_success = run_translation(config, logger, args)
        
        # Process output
        output_success = process_final_output(config, logger)
        
        # Final result
        if translation_success and output_success:
            log.info("Translation pipeline completed successfully")
            return 0
        else:
            log.warning("Translation pipeline completed with warnings or errors")
            return 1
            
    except KeyboardInterrupt:
        log.info("Translation pipeline interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Unexpected error in translation pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())