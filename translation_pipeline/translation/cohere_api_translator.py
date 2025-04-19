"""
Cohere API integration module for the translation pipeline.

This module provides functionality to translate text using the Cohere API with asyncio
for parallel processing with multiple API keys and rate limiting.
"""

import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
import cohere
from dataclasses import dataclass

# Rate limiting constants
REQUESTS_PER_MINUTE = 450  # Rate limit per API key
MAX_CONCURRENT_REQUESTS = 50  # Maximum concurrent requests


@dataclass
class APIKeyInfo:
    """Class to hold API key information for rate limiting."""
    key: str
    last_request_time: float = 0
    requests_in_current_minute: int = 0
    minute_start_time: float = 0


class CohereAPITranslator:
    """
    Class for translating text using Cohere API with asyncio for parallel processing
    and rate limiting with multiple API keys.
    """
    
    def __init__(self, api_keys: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize the translator with multiple API keys.
        
        Args:
            api_keys: List of Cohere API keys
            logger: Logger object (optional)
        """
        self.api_keys_info = [APIKeyInfo(key=key, minute_start_time=time.time()) for key in api_keys]
        self.logger = logger or logging.getLogger(__name__)
        self.clients = {}  # Lazy initialization of clients
        self.rate_limit_per_key = REQUESTS_PER_MINUTE
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Initialize API rate limit monitoring
        self.total_keys = len(api_keys)
        self.requests_per_minute = self.total_keys * REQUESTS_PER_MINUTE
        self.logger.info(f"Initialized with {self.total_keys} API keys. "
                        f"Total rate limit: {self.requests_per_minute} requests/minute.")

    def _get_client(self, api_key: str) -> cohere.AsyncClientV2:
        """
        Get or create an AsyncClientV2 for the given API key.
        
        Args:
            api_key: Cohere API key
            
        Returns:
            AsyncClientV2 client for the API key
        """
        if api_key not in self.clients:
            # Create new async client
            client = cohere.AsyncClientV2(api_key=api_key)
            self.clients[api_key] = client
        
        return self.clients[api_key]

    async def _get_available_api_key(self) -> APIKeyInfo:
        """
        Get an available API key that hasn't hit the rate limit.
        Implements a fair rotation policy among keys.
        
        Returns:
            Available API key info
        """
        while True:
            current_time = time.time()
            
            # Sort keys by usage count to ensure fair distribution
            self.api_keys_info.sort(key=lambda x: x.requests_in_current_minute)
            
            for key_info in self.api_keys_info:
                # Reset counter if a minute has passed
                if current_time - key_info.minute_start_time >= 60:
                    key_info.requests_in_current_minute = 0
                    key_info.minute_start_time = current_time
                
                # Check if this key is under the rate limit
                if key_info.requests_in_current_minute < self.rate_limit_per_key:
                    # Ensure we wait at least 60/RATE_LIMIT seconds between requests with the same key
                    seconds_per_request = 60.0 / self.rate_limit_per_key
                    time_since_last_request = current_time - key_info.last_request_time
                    
                    if time_since_last_request < seconds_per_request:
                        # Need to wait a bit more for this key
                        await asyncio.sleep(seconds_per_request - time_since_last_request)
                    
                    # Mark this key as used
                    key_info.requests_in_current_minute += 1
                    key_info.last_request_time = time.time()
                    
                    return key_info
            
            # If all keys are at their limit, wait a bit and retry
            self.logger.warning("All API keys are at their rate limit. Waiting before retry...")
            await asyncio.sleep(1)

    async def translate_text(self, text: str, language: str, system_prompt: str = None) -> str:
        """
        Translate a piece of text using Cohere API.
        
        Args:
            text: Text to translate
            language: Target language
            system_prompt: System prompt for translation (optional)
            
        Returns:
            Translated text
        """
        async with self.semaphore:
            try:
                # Get available API key
                key_info = await self._get_available_api_key()
                client = self._get_client(key_info.key)
                
                # Create default system prompt if not provided
                if system_prompt is None:
                    system_prompt = (
                        f"You are an expert translator. Translate the following text "
                        f"to {language}. Only return the translated text, nothing else."
                    )
                
                # Create messages for the chat
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
                
                # Make the API call
                response = await client.chat(
                    model="c4ai-aya-expanse-32b",
                    messages=messages,
                    temperature=0.1,  # Using low temperature for deterministic translation
                )
                
                # Extract translated text
                translated_text = response.message.content[0].text
                
                # Log successful translation
                self.logger.debug(f"Successfully translated text to {language} using Cohere API")
                
                return translated_text
                
            except Exception as e:
                self.logger.error(f"Error translating text using Cohere API: {e}")
                # Return original text on error
                return text

    async def translate_batch(self, texts: List[str], language: str, system_prompt: str = None) -> List[str]:
        """
        Translate a batch of texts using Cohere API with parallelization.
        
        Args:
            texts: List of texts to translate
            language: Target language
            system_prompt: System prompt for translation (optional)
            
        Returns:
            List of translated texts
        """
        # Create tasks for all texts
        tasks = [self.translate_text(text, language, system_prompt) for text in texts]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        return results


class CohereTranslationManager:
    """
    Manager class for handling translations using Cohere API.
    Provides a high-level interface for the translation pipeline.
    """
    
    def __init__(self, api_keys: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize the translation manager.
        
        Args:
            api_keys: List of Cohere API keys
            logger: Logger object (optional)
        """
        self.translator = CohereAPITranslator(api_keys, logger)
        self.logger = logger or logging.getLogger(__name__)
        
    async def translate_batch_for_language(
        self,
        batch_data: List[Dict[str, Any]],
        language: str,
        prompt_template=None
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of entries for a specific language.
        
        Args:
            batch_data: Batch data to translate
            language: Language to translate to
            prompt_template: Prompt template to use (optional)
            
        Returns:
            Translated batch of entries
        """
        # Extract system prompt from template if provided
        system_prompt = None
        if prompt_template and hasattr(prompt_template, 'system_msg'):
            system_prompt = prompt_template.system_msg
        
        # Process each entry in the batch
        translated_batch = []
        for entry in batch_data:
            translated_entry = await self._translate_entry(entry, language, system_prompt)
            translated_batch.append(translated_entry)
        
        return translated_batch
    
    async def _translate_entry(
        self, 
        entry: Dict[str, Any], 
        language: str,
        system_prompt=None
    ) -> Dict[str, Any]:
        """
        Translate a single entry.
        
        Args:
            entry: Entry to translate
            language: Language to translate to
            system_prompt: System prompt to use (optional)
            
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
                    system_prompt
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
        system_prompt=None
    ) -> List[Dict[str, Any]]:
        """
        Translate conversations in an entry.
        
        Args:
            conversations: Conversations to translate
            language: Language to translate to
            system_prompt: System prompt to use (optional)
            
        Returns:
            Translated conversations
        """
        translated_conversations = []
        gpt_texts = []
        gpt_indices = []
        
        # First, collect all GPT values that need translation
        for idx, conv in enumerate(conversations):
            if conv.get('from') == 'gpt':
                original_value = conv.get('value', '')
                if original_value:
                    gpt_texts.append(original_value)
                    gpt_indices.append(idx)
        
        # Translate all GPT values in parallel
        translated_gpt_texts = []
        if gpt_texts:
            translated_gpt_texts = await self.translator.translate_batch(
                gpt_texts, language, system_prompt
            )
        
        # Create translated conversations
        for idx, conv in enumerate(conversations):
            # Create a copy of the conversation
            translated_conv = conv.copy()
            
            if conv.get('from') == 'gpt':
                # For GPT values, use the translated text
                if conv.get('value', ''):
                    gpt_idx = gpt_indices.index(idx)
                    translated_conv['value'] = translated_gpt_texts[gpt_idx]
            elif conv.get('from') == 'human':
                # For human values, we leave them as is (they're handled elsewhere)
                pass
            
            translated_conversations.append(translated_conv)
        
        return translated_conversations
    
    async def process_language_with_batches(
        self,
        language: str,
        batches: List[List[Dict[str, Any]]],
        batch_size: int,
        total_batches: int,
        start_batch: int = 0,
        state_file: str = None,
        output_dir: str = None
    ) -> bool:
        """
        Process all batches for a language with rate limiting and parallel processing.
        
        Args:
            language: Language to translate to
            batches: List of batches to process
            batch_size: Size of each batch
            total_batches: Total number of batches
            start_batch: Batch number to start from (for resuming)
            state_file: Path to the state file for progress tracking
            output_dir: Directory for intermediate files
            
        Returns:
            Whether the translation was successful
        """
        from utils.data_loader import save_batch_to_json, save_progress_state
        
        prompt_template = None
        from utils.prompt_templates import get_prompt_for_language
        prompt_template = get_prompt_for_language(language)
        
        # Process batches in groups to control memory usage
        group_size = min(5, total_batches - start_batch)  # Process 5 batches at a time
        success = True
        
        # Ensure the intermediate directory exists
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for i in range(start_batch, total_batches, group_size):
            end_idx = min(i + group_size, total_batches)
            batch_indices = list(range(i, end_idx))
            
            # Create tasks for this group of batches
            batch_tasks = []
            for batch_idx in batch_indices:
                batch_num = batch_idx + 1  # 1-indexed batch number for logging
                
                # Log batch progress
                self.logger.info(f"Translating batch {batch_num}/{total_batches} for {language} using Cohere API")
                
                # Process batch
                batch_tasks.append(self.translate_batch_for_language(
                    batches[batch_idx],
                    language,
                    prompt_template
                ))
            
            # Execute tasks for this group
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Save results and update progress for each batch
            for idx, batch_idx in enumerate(batch_indices):
                batch_num = batch_idx + 1  # 1-indexed batch number for logging
                translated_batch = batch_results[idx]
                
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
        
        return success


# Function to create a translation manager with multiple API keys
def create_translation_manager(
    api_keys: List[str],
    logger: Optional[logging.Logger] = None
) -> CohereTranslationManager:
    """
    Create a translation manager with the given API keys.
    
    Args:
        api_keys: List of Cohere API keys
        logger: Logger object (optional)
        
    Returns:
        CohereTranslationManager instance
    """
    return CohereTranslationManager(api_keys, logger)