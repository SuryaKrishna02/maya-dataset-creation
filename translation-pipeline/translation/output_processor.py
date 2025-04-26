"""
Output processor module for the translation pipeline.

This module handles processing and merging the intermediate translation files
into the final output files.
"""

import os
import json
import glob
from typing import List, Dict
from utils.config import TranslationConfig
from utils.logger import TranslationLogger
from utils.data_loader import merge_intermediate_files


class OutputProcessor:
    """Output processor for handling the final output of the translation pipeline."""
    
    def __init__(
        self,
        config: TranslationConfig,
        logger: TranslationLogger
    ):
        """
        Initialize the output processor.
        
        Args:
            config: Configuration object
            logger: Logger object
        """
        self.config = config
        self.logger = logger.get_logger()
        self.logger.info("Initializing output processor...")
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def get_language_batches(self, language: str) -> List[str]:
        """
        Get all batch files for a specific language.
        
        Args:
            language: Language to get batches for
            
        Returns:
            List of batch file paths
        """
        batch_pattern = os.path.join(self.config.intermediate_dir, f"{language}_batch_*.json")
        return sorted(
            glob.glob(batch_pattern),
            key=lambda x: int(os.path.basename(x).split('_batch_')[1].split('.')[0])
        )
    
    def validate_batches(self, language: str) -> bool:
        """
        Validate that all batches for a language are present and valid.
        
        Args:
            language: Language to validate batches for
            
        Returns:
            Whether all batches are valid
        """
        batch_files = self.get_language_batches(language)
        
        if not batch_files:
            self.logger.error(f"No batch files found for language: {language}")
            return False
        
        # Check for missing batch numbers
        batch_numbers = [
            int(os.path.basename(f).split('_batch_')[1].split('.')[0])
            for f in batch_files
        ]
        
        expected_batch_numbers = list(range(1, max(batch_numbers) + 1))
        missing_batches = set(expected_batch_numbers) - set(batch_numbers)
        
        if missing_batches:
            self.logger.error(f"Missing batch files for {language}: {missing_batches}")
            return False
        
        # Validate each batch file
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                if not isinstance(batch_data, list):
                    self.logger.error(f"Invalid batch file {batch_file}: not a list")
                    return False
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.error(f"Error loading batch file {batch_file}: {e}")
                return False
        
        return True
    
    def merge_language_batches(self, language: str) -> bool:
        """
        Merge all batches for a language into a single output file.
        
        Args:
            language: Language to merge batches for
            
        Returns:
            Whether the merge was successful
        """
        self.logger.info(f"Merging batches for language: {language}")
        
        # Validate batches first
        if not self.validate_batches(language):
            return False
        
        try:
            # Define output file path
            output_file = os.path.join(self.config.output_dir, f"{language}_translated.json")
            
            # Merge all intermediate files
            merge_intermediate_files(
                self.config.intermediate_dir,
                output_file,
                language
            )
            
            self.logger.info(f"Successfully merged batches for {language} to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging batches for {language}: {e}")
            return False
    
    def process_all_languages(self) -> Dict[str, bool]:
        """
        Process all languages by merging their batches.
        
        Returns:
            Dictionary mapping languages to success status
        """
        results = {}
        
        for language in self.config.languages:
            success = self.merge_language_batches(language)
            results[language] = success
        
        return results


def process_output(
    config: TranslationConfig,
    logger: TranslationLogger
) -> Dict[str, bool]:
    """
    Process the output of the translation pipeline.
    
    Args:
        config: Configuration object
        logger: Logger object
        
    Returns:
        Dictionary mapping languages to success status
    """
    processor = OutputProcessor(config, logger)
    return processor.process_all_languages()