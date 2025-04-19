"""
Simplified Cohere API integration module for the translation pipeline.

This module provides functionality to translate text using the Cohere API with the ratelimit
library for efficient rate limiting across multiple API keys.
"""

import os
import cohere
import asyncio
import logging
from ratelimit import limits, sleep_and_retry
from typing import List, Dict, Any, Optional

class SimpleCohereTranslator:
    """Simplified translator using Cohere API with ratelimit library."""
    
    def __init__(self, api_keys: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize the translator with multiple API keys.
        
        Args:
            api_keys: List of Cohere API keys
            logger: Logger object (optional)
        """
        self.api_keys = api_keys
        self.logger = logger or logging.getLogger(__name__)
        self.clients = {key: cohere.AsyncClientV2(api_key=key) for key in api_keys}
        
        # Calculate total rate limit
        self.rate_limit = len(api_keys) * 450
        self.period = (len(api_keys) + 1) * 30
        self.logger.info(f"Initialized with {len(api_keys)} API keys. "
                         f"Total rate limit: {self.rate_limit} requests/minute.")
        
        # Create a rate limited function based on the total available API keys
        self.rate_limited_translate = self._create_rate_limited_function()
    
    def _create_rate_limited_function(self):
        """Create a rate limited function based on the total available API keys."""
        @sleep_and_retry
        @limits(calls=self.rate_limit, period=self.period)
        async def _rate_limited_translate(text: str, language: str, api_key: str, system_prompt: str = None) -> str:
            try:
                client = self.clients[api_key]
                
                # Default system prompt if not provided
                if system_prompt is None:
                    system_prompt = (f"Translate the following text to {language}. "
                                    f"Only return the translated text, nothing else.")
                
                # Create messages for the chat
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
                
                # Make the API call
                response = await client.chat(
                    model="c4ai-aya-expanse-32b",
                    messages=messages,
                    temperature=0.1,
                )
                
                # Extract translated text
                return response.message.content[0].text
                    
            except Exception as e:
                self.logger.error(f"Error translating with key {api_key[:5]}...: {e}")
                return text  # Return original text on error
                
        return _rate_limited_translate
    
    async def translate_text(self, text: str, language: str, api_key: str, system_prompt: str = None) -> str:
        """Translate a single text using the rate-limited function."""
        return await self.rate_limited_translate(text, language, api_key, system_prompt)
    
    async def translate_batch(self, texts: List[str], language: str, system_prompt: str = None) -> List[str]:
        """Translate a batch of texts distributing across available API keys."""
        results = [None] * len(texts)
        
        # Distribute texts among API keys
        key_distribution = {}
        for i, text in enumerate(texts):
            key_index = i % len(self.api_keys)
            key = self.api_keys[key_index]
            if key not in key_distribution:
                key_distribution[key] = []
            key_distribution[key].append((i, text))
        
        # Process texts for each key
        tasks = []
        for key, text_pairs in key_distribution.items():
            for idx, text in text_pairs:
                task = asyncio.create_task(self.translate_text(
                    text, language, key, system_prompt
                ))
                tasks.append((idx, task))
        
        # Wait for all tasks and collect results
        for idx, task in tasks:
            try:
                results[idx] = await task
            except Exception as e:
                self.logger.error(f"Failed to translate text at index {idx}: {e}")
                results[idx] = texts[idx]  # Use original on error
        
        return results
    
    async def process_in_batches(self, all_texts: List[str], language: str, 
                                 system_prompt: str = None, batch_size: int = None) -> List[str]:
        """Process all texts in controlled batch sizes with rate limiting."""
        # If batch_size is not specified, use the rate limit as the batch size
        if batch_size is None:
            batch_size = self.rate_limit
        
        all_results = []
        
        # Process entire batch at once if it's smaller than the batch size
        if len(all_texts) <= batch_size:
            self.logger.info(f"Processing batch with {len(all_texts)} texts")
            batch_results = await self.translate_batch(all_texts, language, system_prompt)
            all_results.extend(batch_results)
            self.logger.info(f"Completed {len(all_texts)}/{len(all_texts)} texts")
            return all_results
        
        # Split into batches for larger datasets
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} texts")
            
            # Translate batch
            batch_results = await self.translate_batch(batch, language, system_prompt)
            all_results.extend(batch_results)
            
            # Log progress
            self.logger.info(f"Completed {min(i+batch_size, len(all_texts))}/{len(all_texts)} texts")
            
            # If not the last batch, add a small delay to ensure rate limits
            if i + batch_size < len(all_texts):
                await asyncio.sleep(2)  # Small buffer between batches
        
        return all_results


class SimpleCohereTranslationManager:
    """Manager for translating dataset entries using the simplified Cohere translator."""
    
    def __init__(self, api_keys: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize the translation manager.
        
        Args:
            api_keys: List of Cohere API keys
            logger: Logger object (optional)
        """
        self.translator = SimpleCohereTranslator(api_keys, logger)
        self.logger = logger or logging.getLogger(__name__)
    
    async def translate_batch_for_language(
        self,
        batch_data: List[Dict[str, Any]],
        language: str,
        prompt_template=None,
        human_translations=None
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of dataset entries using optimized batch collection.
        
        Args:
            batch_data: Batch data to translate
            language: Language to translate to
            prompt_template: Prompt template to use (optional)
            human_translations: Dictionary of pre-translated human values (optional)
            
        Returns:
            Translated batch of entries
        """
        # Extract system prompt if available
        system_prompt = None
        if prompt_template and hasattr(prompt_template, 'system_msg'):
            system_prompt = prompt_template.system_msg
        
        # Collect all GPT values and their positions across the entire batch
        all_gpt_texts = []
        positions = []  # (entry_idx, conv_idx) pairs
        
        # First pass: process human values and collect GPT values
        for entry_idx, entry in enumerate(batch_data):
            if 'conversations' in entry and isinstance(entry['conversations'], list):
                for conv_idx, conv in enumerate(entry['conversations']):
                    if conv.get('from') == 'human' and conv.get('value', ''):
                        # Process human values using pre-translated values
                        if human_translations and language in human_translations:
                            original_value = conv.get('value', '')
                            batch_data[entry_idx]['conversations'][conv_idx]['value'] = self._get_human_translation(
                                original_value, language, human_translations
                            )
                    elif conv.get('from') == 'gpt' and conv.get('value', ''):
                        # Collect GPT values for batch translation
                        all_gpt_texts.append(conv.get('value', ''))
                        positions.append((entry_idx, conv_idx))
        
        # If we have GPT values to translate
        if all_gpt_texts:
            self.logger.info(f"Translating {len(all_gpt_texts)} GPT values in a single batch")
            
            # Translate all GPT values in a single batch
            translated_texts = await self.translator.translate_batch(
                all_gpt_texts, language, system_prompt
            )
            
            # Update batch with translations
            for (entry_idx, conv_idx), translated_text in zip(positions, translated_texts):
                batch_data[entry_idx]['conversations'][conv_idx]['value'] = translated_text
        
        return batch_data
    
    async def _translate_entry(
        self, 
        entry: Dict[str, Any], 
        language: str,
        system_prompt=None,
        human_translations=None
    ) -> Dict[str, Any]:
        """
        Translate a single entry.
        
        Args:
            entry: Entry to translate
            language: Language to translate to
            system_prompt: System prompt to use (optional)
            human_translations: Dictionary of pre-translated human values (optional)
            
        Returns:
            Translated entry
        """
        try:
            # Create a copy of the entry
            translated_entry = entry.copy()
            
            # Translate conversations if present
            if 'conversations' in entry and isinstance(entry['conversations'], list):
                translated_entry['conversations'] = await self._translate_conversations(
                    entry['conversations'], 
                    language,
                    system_prompt,
                    human_translations
                )
            
            return translated_entry
            
        except Exception as e:
            self.logger.error(f"Error translating entry {entry.get('id', 'unknown')}: {e}")
            # Return the original entry on error
            return entry
    
    async def _translate_conversations(
        self, 
        conversations: List[Dict[str, Any]], 
        language: str,
        system_prompt=None,
        human_translations=None
    ) -> List[Dict[str, Any]]:
        """
        Translate conversations in an entry.
        
        Args:
            conversations: Conversations to translate
            language: Language to translate to
            system_prompt: System prompt to use (optional)
            human_translations: Dictionary of pre-translated human values (optional)
            
        Returns:
            Translated conversations
        """
        translated_conversations = []
        gpt_texts = []
        gpt_indices = []
        
        # First, collect all GPT values that need translation
        for idx, conv in enumerate(conversations):
            if conv.get('from') == 'gpt' and conv.get('value', ''):
                gpt_texts.append(conv.get('value', ''))
                gpt_indices.append(idx)
        
        # Translate all GPT values in parallel if there are any
        translated_gpt_texts = []
        if gpt_texts:
            # Process in a single batch rather than one by one
            translated_gpt_texts = await self.translator.translate_batch(
                gpt_texts, language, system_prompt
            )
        
        # Create translated conversations
        for idx, conv in enumerate(conversations):
            # Create a copy of the conversation
            translated_conv = conv.copy()
            
            if conv.get('from') == 'human':
                # For human values, use pre-translated values if available
                original_value = conv.get('value', '')
                if human_translations and language in human_translations:
                    translated_conv['value'] = self._get_human_translation(original_value, language, human_translations)
                # If no translations available, keep the original
            
            elif conv.get('from') == 'gpt':
                # For GPT values, use the translated text
                if conv.get('value', ''):
                    gpt_idx = gpt_indices.index(idx)
                    translated_conv['value'] = translated_gpt_texts[gpt_idx]
            
            translated_conversations.append(translated_conv)
        
        return translated_conversations
    
    def _get_human_translation(self, text: str, language: str, human_translations: Dict) -> str:
        """
        Get translation for a human value, preserving <image> tag and newline placement.
        
        Args:
            text: Source text in English to translate
            language: Target language to translate to
            human_translations: Dictionary of pre-translated values
            
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
        if language in human_translations and clean_text in human_translations[language]:
            # Get the translation without tags
            translation = human_translations[language][clean_text]
            
            # Add back the tags in their original positions
            if has_image_tag_start:
                translation = IMAGE_TAG_START + translation
            if has_image_tag_end:
                translation = translation + IMAGE_TAG_END
            
            return translation
        
        # If no exact match, log a warning and return the original
        self.logger.warning(f"No translation found for human value: {text}")
        return text

    async def process_language_with_batches(
        self,
        language: str,
        batches: List[List[Dict[str, Any]]],
        total_batches: int,
        start_batch: int = 0,
        state_file: str = None,
        output_dir: str = None,
        human_translations: Dict = None
    ) -> bool:
        """
        Process all batches for a language using the simplified translator.
        
        Args:
            language: Language to translate to
            batches: List of batches to process
            total_batches: Total number of batches
            start_batch: Batch number to start from (for resuming)
            state_file: Path to the state file for progress tracking
            output_dir: Directory for intermediate files
            human_translations: Dictionary of pre-translated human values (optional)
            
        Returns:
            Whether the translation was successful
        """
        from utils.data_loader import save_batch_to_json, save_progress_state
        from utils.prompt_templates import get_prompt_for_language
        
        prompt_template = get_prompt_for_language(language)
        success = True
        
        # Ensure the intermediate directory exists
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Process batches sequentially
        for batch_idx in range(start_batch, total_batches):
            batch_num = batch_idx + 1  # 1-indexed batch number for logging
            
            self.logger.info(f"Translating batch {batch_num}/{total_batches} for {language}")
            
            try:
                # Translate batch
                translated_batch = await self.translate_batch_for_language(
                    batches[batch_idx],
                    language,
                    prompt_template,
                    human_translations
                )
                
                # Save batch to file
                if output_dir:
                    save_batch_to_json(
                        translated_batch,
                        output_dir,
                        language,
                        batch_num
                    )
                
                # Update progress state
                if state_file:
                    save_progress_state(
                        state_file,
                        language,
                        batch_num,
                        total_batches,
                        "completed"
                    )
                
                self.logger.info(f"Batch {batch_num}/{total_batches} for {language} completed")
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_num} for {language}: {e}")
                success = False
                
                # Update progress state
                if state_file:
                    save_progress_state(
                        state_file,
                        language,
                        batch_num,
                        total_batches,
                        "error"
                    )
        
        return success


# Function to create a translation manager with multiple API keys
def create_translation_manager(
    api_keys: List[str],
    logger: Optional[logging.Logger] = None
) -> SimpleCohereTranslationManager:
    """
    Create a translation manager with the given API keys.
    
    Args:
        api_keys: List of Cohere API keys
        logger: Logger object (optional)
        
    Returns:
        SimpleCohereTranslationManager instance
    """
    return SimpleCohereTranslationManager(api_keys, logger)