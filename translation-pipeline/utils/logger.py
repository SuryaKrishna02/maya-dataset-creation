"""
Logger module for the translation pipeline.

This module provides logging functionality for the translation pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class TranslationLogger:
    """Custom logger for the translation pipeline."""
    
    def __init__(
        self,
        log_file_path: str,
        error_log_path: str,
        level: int = logging.INFO,
        console_output: bool = True,
        log_format: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            log_file_path: Path to the main log file
            error_log_path: Path to the error log file
            level: Logging level (default: INFO)
            console_output: Whether to output logs to console (default: True)
            log_format: Custom log format (optional)
        """
        self.log_file_path = log_file_path
        self.error_log_path = error_log_path
        
        # Create directories for log files if they don't exist
        Path(os.path.dirname(log_file_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(error_log_path)).mkdir(parents=True, exist_ok=True)
        
        # Set up logging format
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Create main logger
        self.logger = logging.getLogger('translation_pipeline')
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create file handler for regular logs
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Create file handler for error logs
        error_handler = logging.FileHandler(error_log_path)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        # Create console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Logger initialized at {datetime.now().isoformat()}")
        self.logger.info(f"Log file: {log_file_path}")
        self.logger.info(f"Error log file: {error_log_path}")
    
    def get_logger(self):
        """Get the configured logger."""
        return self.logger
    
    def log_progress(self, message: str, count: int, total: int):
        """
        Log progress message with percentage.
        
        Args:
            message: Progress message
            count: Current count
            total: Total count
        """
        percentage = (count / total) * 100 if total > 0 else 0
        self.logger.info(f"{message}: {count}/{total} ({percentage:.2f}%)")
    
    def log_batch_process(self, language: str, batch_num: int, total_batches: int, success: bool):
        """
        Log batch processing result.
        
        Args:
            language: Language being processed
            batch_num: Current batch number
            total_batches: Total number of batches
            success: Whether the batch was processed successfully
        """
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Batch {batch_num}/{total_batches} for {language} {status}")
    
    def log_translation_error(
        self,
        language: str,
        input_text: str,
        error_message: str,
        batch_num: Optional[int] = None,
        item_index: Optional[int] = None
    ):
        """
        Log translation error.
        
        Args:
            language: Language being translated to
            input_text: Input text that caused the error
            error_message: Error message
            batch_num: Batch number (optional)
            item_index: Item index within the batch (optional)
        """
        location = ""
        if batch_num is not None:
            location += f"batch {batch_num}"
            if item_index is not None:
                location += f", item {item_index}"
        
        error_info = f"Translation error for {language} {location}: {error_message}"
        self.logger.error(error_info)
        self.logger.error(f"Input text: {input_text}")
    
    def log_config(self, config: dict):
        """
        Log configuration parameters.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Configuration parameters:")
        for key, value in config.items():
            # Don't log sensitive information like tokens
            if "token" not in key.lower():
                self.logger.info(f"  {key}: {value}")
            else:
                self.logger.info(f"  {key}: [REDACTED]")


def setup_logger(log_path: str, error_log_path: str, console_output: bool = True) -> TranslationLogger:
    """
    Set up and configure the logger.
    
    Args:
        log_path: Path to the main log file
        error_log_path: Path to the error log file
        console_output: Whether to output logs to console (default: True)
        
    Returns:
        Configured TranslationLogger object
    """
    return TranslationLogger(log_path, error_log_path, console_output=console_output)