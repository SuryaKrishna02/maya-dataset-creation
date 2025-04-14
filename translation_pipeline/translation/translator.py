"""
Translator module for the translation pipeline.

This module handles the translation process using the provided model and configurations.
"""

import os
import time
import torch
from typing import List, Dict, Any
from utils.config import TranslationConfig
from utils.logger import TranslationLogger
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompt_templates import format_prompt, get_prompt_for_language
from utils.data_loader import (
    load_dataset, 
    load_human_values_translation, 
    create_batches,
    save_batch_to_json,
    save_progress_state,
    get_resume_point
)


class Translator:
    """Translator class for translating dataset entries."""
    
    def __init__(
        self,
        config: TranslationConfig,
        logger: TranslationLogger
    ):
        """
        Initialize the translator.
        
        Args:
            config: Configuration object
            logger: Logger object
        """
        self.config = config
        self.logger = logger.get_logger()
        self.logger.info("Initializing translator...")
        
        # State file for tracking progress
        self.state_file = os.path.join(self.config.output_dir, "translation_state.json")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load translations for human values
        self._load_human_translations()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.hf_access_token
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set precision
        if self.config.precision == "float16":
            dtype = torch.float16
        elif self.config.precision == "float32":
            dtype = torch.float32
        else:
            self.logger.warning(f"Unsupported precision: {self.config.precision}, using float16")
            dtype = torch.float16
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=self.config.hf_access_token
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation
        torch.set_grad_enabled(False)
        
        self.logger.info("Model and tokenizer loaded successfully")
    
    def _load_human_translations(self):
        """Load translations for human values."""
        self.logger.info(f"Loading human values translations from: {self.config.human_values_translation_path}")
        
        try:
            self.human_translations = load_human_values_translation(
                self.config.human_values_translation_path
            )
            # Log which languages were successfully loaded from CSV
            loaded_langs = list(self.human_translations.keys())
            self.logger.info(f"Successfully loaded translations for languages: {loaded_langs}")
            
            # Check if all languages in config are available in the translations
            missing_langs = [lang for lang in self.config.languages if lang not in loaded_langs]
            if missing_langs:
                self.logger.warning(f"Missing translations for configured languages: {missing_langs}")
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f"Error loading human values translations: {e}")
            self.human_translations = {}
            self.logger.info("Expected CSV format should have columns: 'English Sentence', 'Hindi Translation', 'Spanish Translation', etc.")
    
    def translate_batch(
        self,
        batch: List[Dict[str, Any]],
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of entries.
        
        Args:
            batch: Batch of entries to translate
            language: Language to translate to
            
        Returns:
            Batch of translated entries
        """
        # Get prompt template for the language
        prompt_template = get_prompt_for_language(language)
        if not prompt_template:
            self.logger.error(f"No prompt template found for language: {language}")
            return batch

        # Process each entry in the batch
        translated_batch = []
        for entry in batch:
            translated_entry = self._translate_entry(entry, language, prompt_template)
            translated_batch.append(translated_entry)
        
        return translated_batch
    
    def _translate_entry(
        self, 
        entry: Dict[str, Any], 
        language: str,
        prompt_template
    ) -> Dict[str, Any]:
        """
        Translate a single entry.
        
        Args:
            entry: Entry to translate
            language: Language to translate to
            prompt_template: Prompt template to use
            
        Returns:
            Translated entry
        """
        try:
            # Create a copy of the entry
            translated_entry = entry.copy()
            
            # Translate conversations if present
            if 'conversations' in entry and isinstance(entry['conversations'], list):
                translated_entry['conversations'] = self._translate_conversations(
                    entry['conversations'], 
                    language, 
                    prompt_template
                )
            
            return translated_entry
            
        except Exception as e:
            self.logger.error(f"Error translating entry {entry.get('id', 'unknown')}: {e}")
            # Return the original entry on error to maintain structure
            return entry
    
    def _translate_conversations(
        self, 
        conversations: List[Dict[str, Any]], 
        language: str,
        prompt_template
    ) -> List[Dict[str, Any]]:
        """
        Translate conversations in an entry.
        
        Args:
            conversations: Conversations to translate
            language: Language to translate to
            prompt_template: Prompt template to use
            
        Returns:
            Translated conversations
        """
        translated_conversations = []
        
        for conv in conversations:
            # Create a copy of the conversation
            translated_conv = conv.copy()
            
            if conv.get('from') == 'human':
                # For human values, use pre-translated values if available
                original_value = conv.get('value', '')
                translated_conv['value'] = self._get_human_translation(original_value, language)
            
            elif conv.get('from') == 'gpt':
                # For GPT values, use the model to translate
                original_value = conv.get('value', '')
                if original_value:
                    translated_value = self.translate_text(original_value, prompt_template)
                    translated_conv['value'] = translated_value
            
            translated_conversations.append(translated_conv)
        
        return translated_conversations
    
    def _get_human_translation(self, text: str, language: str) -> str:
        """
        Get translation for a human value, preserving <image> tag and newline placement.
        
        Args:
            text: Source text in English to translate
            language: Target language to translate to
            
        Returns:
            Translated text with <image> tag and newlines preserved, or original text if no translation is available
        """
        # Define constants for the image tag patterns
        IMAGE_TAG_START = "<image>\n"
        IMAGE_TAG_END = "\n<image>"
        IMAGE_TAG_PLAIN = "<image>"
        
        # Check if the original text contains the image tag patterns
        has_image_tag_start = text.startswith(IMAGE_TAG_START)
        has_image_tag_end = text.endswith(IMAGE_TAG_END)
        
        # Remove the image tag and surrounding newlines for lookup
        clean_text = text
        if has_image_tag_start:
            clean_text = clean_text.replace(IMAGE_TAG_START, "", 1)
        if has_image_tag_end:
            clean_text = clean_text.replace(IMAGE_TAG_END, "", 1)
        
        # Further cleanup in case there are other image tags
        clean_text = clean_text.replace(IMAGE_TAG_PLAIN, "").strip()
        
        # Try to find the translation for the clean text
        if language in self.human_translations and clean_text in self.human_translations[language]:
            # Get the translation without tags
            translation = self.human_translations[language][clean_text]
            
            # Add back the tags in their original positions
            if has_image_tag_start:
                translation = IMAGE_TAG_START + translation
            if has_image_tag_end:
                translation = translation + IMAGE_TAG_END
            
            return translation
        
        # If no exact match, log the warning and return the original
        self.logger.warning(f"No translation found for human value: {text}")
        return text
    
    def translate_text(
        self,
        text: str,
        prompt_template
    ) -> str:
        """
        Translate a single text using the model.
        
        Args:
            text: Text to translate
            prompt_template: Prompt template to use
            
        Returns:
            Translated text
        """
        try:
            # Format the prompt
            messages = format_prompt(prompt_template, text)
            
            # Convert to model input format
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate translation
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # Decode the output
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract only the model's response
            # This depends on the model and tokenizer, but generally we need to:
            # 1. Remove the input prompt
            # 2. Extract just the assistant's response
            
            # Find where the assistant's response starts
            # This may vary depending on the model and tokenizer
            response_start = output_text.find("<assistant>")
            if response_start != -1:
                # Skip the <assistant> tag
                response_start += len("<assistant>")
                response_text = output_text[response_start:].strip()
            else:
                # If we can't find the specific marker, try a simpler approach
                # Get the part that's not in the input
                response_text = output_text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error translating text: {e}")
            self.logger.error(f"Original text: {text}")
            return text  # Return original text on error
    
    def translate_batch_for_language(
        self,
        language: str,
        batch_data: List[Dict[str, Any]],
        batch_num: int,
        total_batches: int
    ) -> bool:
        """
        Translate a batch for a specific language and save to file.
        
        Args:
            language: Language to translate to
            batch_data: Batch data to translate
            batch_num: Batch number
            total_batches: Total number of batches
            
        Returns:
            Whether the translation was successful
        """
        try:
            self.logger.info(f"Translating batch {batch_num}/{total_batches} for {language}")
            
            # Translate the batch
            translated_batch = self.translate_batch(batch_data, language)
            
            # Save the translated batch
            save_batch_to_json(
                translated_batch,
                self.config.intermediate_dir,
                language,
                batch_num
            )
            
            # Update progress state
            save_progress_state(
                self.state_file,
                language,
                batch_num,
                total_batches,
                "completed"
            )
            
            self.logger.info(f"Batch {batch_num}/{total_batches} for {language} completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error translating batch {batch_num} for {language}: {e}")
            # Update progress state
            save_progress_state(
                self.state_file,
                language,
                batch_num,
                total_batches,
                "error"
            )
            return False
    
    def translate_dataset_for_language(self, language: str) -> bool:
        """
        Translate the entire dataset for a specific language.
        
        Args:
            language: Language to translate to
            
        Returns:
            Whether the translation was successful
        """
        self.logger.info(f"Starting translation for language: {language}")
        
        try:
            # Load dataset
            dataset = load_dataset(self.config.dataset_path)
            
            # Create batches
            batch_size = self.config.optimal_batch_size or 8  # Default to 8 if not specified
            batches = create_batches(dataset, batch_size)
            
            total_batches = len(batches)
            self.logger.info(f"Created {total_batches} batches with batch size {batch_size}")
            
            # Check if we need to resume from a previous point
            current_batch, _ = get_resume_point(self.state_file, language)
            if current_batch is not None:
                self.logger.info(f"Resuming translation for {language} from batch {current_batch+1}")
                start_batch = current_batch + 1
            else:
                start_batch = 0
            
            # Process each batch
            success = True
            for i in range(start_batch, total_batches):
                batch_success = self.translate_batch_for_language(
                    language,
                    batches[i],
                    i + 1,  # 1-indexed batch number for logging
                    total_batches
                )
                
                if not batch_success:
                    success = False
                
                # Add a small delay between batches to prevent rate limiting
                if i < total_batches - 1:
                    time.sleep(1)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error translating dataset for {language}: {e}")
            return False
    
    def translate_dataset(self) -> Dict[str, bool]:
        """
        Translate the dataset for all languages.
        
        Returns:
            Dictionary mapping languages to success status
        """
        self.logger.info("Starting dataset translation for all languages")
        
        results = {}
        
        for language in self.config.languages:
            self.logger.info(f"Processing language: {language}")
            success = self.translate_dataset_for_language(language)
            results[language] = success
            
            # Log overall result for this language
            if success:
                self.logger.info(f"Successfully translated dataset for {language}")
            else:
                self.logger.warning(f"Translation for {language} completed with errors")
        
        return results


def translate_dataset(config: TranslationConfig, logger: TranslationLogger) -> Dict[str, bool]:
    """
    Translate the dataset using the provided configuration.
    
    Args:
        config: Configuration object
        logger: Logger object
        
    Returns:
        Dictionary mapping languages to success status
    """
    translator = Translator(config, logger)
    return translator.translate_dataset()