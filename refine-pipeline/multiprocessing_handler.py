"""
High-throughput multiprocessing handler for GPU-optimized inference.
Handles batch processing with multiple workers and efficient resource utilization.
"""


import time
import json
import queue
import openai
import logging
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_loader import Config
from data_processor import ImageMetadata, DataProcessor
from prompt_distributor import PromptDistributor, PromptAssignment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single image-prompt pair."""
    image_path: str
    prompt: str
    prompt_category: Optional[str]
    response: str
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class MultiprocessingHandler:
    """High-throughput multiprocessing handler for dataset generation."""
    
    def __init__(self, config: Config):
        """Initialize the multiprocessing handler."""
        self.config = config
        self.client = None
        self.results_queue = queue.Queue()
        self.results = []
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        
        # Initialize OpenAI client
        self._initialize_client()
        
        # Setup output directory
        self.output_path = Path(self.config.output.output_file)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        self.client = openai.Client(
            base_url=self.config.api.base_url,
            api_key=self.config.api.api_key
        )
        logger.info(f"Initialized OpenAI client with base URL: {self.config.api.base_url}")
    
    def _create_message(self, metadata: ImageMetadata, prompt: str) -> List[Dict[str, str]]:
        """Create a message for the API call."""
        system_content = metadata.to_metadata_string()
        
        return [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    def _make_api_call(self, messages: List[Dict[str, str]]) -> str:
        """Make a single API call with retry logic."""
        for attempt in range(self.config.api.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.api.model,
                    messages=messages,
                    temperature=self.config.generation.temperature,
                    max_tokens=self.config.generation.max_tokens,
                    top_p=self.config.generation.top_p,
                    presence_penalty=self.config.generation.presence_penalty,
                    timeout=self.config.api.timeout_seconds,
                    extra_body={
                        "top_k": self.config.generation.top_k,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == self.config.api.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("All API call attempts failed")
    
    def _process_single_item(self, metadata: ImageMetadata, prompt: str, prompt_category: Optional[str] = None) -> ProcessingResult:
        """Process a single image-prompt pair."""
        start_time = time.time()
        
        try:
            messages = self._create_message(metadata, prompt)
            response = self._make_api_call(messages)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                image_path=metadata.image_path,
                prompt=prompt,
                prompt_category=prompt_category,
                response=response,
                metadata=asdict(metadata),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Failed to process {metadata.image_path}: {error_msg}")
            
            return ProcessingResult(
                image_path=metadata.image_path,
                prompt=prompt,
                prompt_category=prompt_category,
                response="",
                metadata=asdict(metadata),
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def _worker_function(self, work_items: List[tuple]) -> List[ProcessingResult]:
        """Worker function for processing a batch of items."""
        results = []
        
        for metadata, prompt, prompt_category in work_items:
            result = self._process_single_item(metadata, prompt, prompt_category)
            results.append(result)
        
        return results
    
    def _save_results(self, results: List[ProcessingResult], append: bool = True):
        """Save results to JSONL file."""
        mode = 'a' if append else 'w'
        
        with open(self.output_path, mode, encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(results)} results to {self.output_path}")
    
    def _create_work_items_from_assignments(self, assignments: List[PromptAssignment], metadata_dict: Dict[str, ImageMetadata]) -> List[tuple]:
        """Create work items from prompt assignments and metadata."""
        work_items = []
        
        for assignment in assignments:
            if assignment.image_path in metadata_dict:
                metadata = metadata_dict[assignment.image_path]
                work_items.append((metadata, assignment.prompt, assignment.prompt_category))
            else:
                logger.warning(f"No metadata found for image: {assignment.image_path}")
        
        return work_items
    
    def _print_progress(self):
        """Print processing progress."""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        total_processed = self.processed_count + self.failed_count
        
        if total_processed > 0:
            rate = total_processed / elapsed
            success_rate = (self.processed_count / total_processed) * 100
            
            logger.info(
                f"Progress: {total_processed} processed "
                f"({self.processed_count} success, {self.failed_count} failed) "
                f"| Rate: {rate:.2f} items/sec "
                f"| Success: {success_rate:.1f}% "
                f"| Elapsed: {elapsed:.1f}s"
            )
    
    def process_dataset(self, data_processor: DataProcessor, prompt_distributor: PromptDistributor) -> List[ProcessingResult]:
        """Process the entire dataset with multiprocessing using prompt distribution."""
        logger.info("Starting dataset processing...")
        self.start_time = time.time()
        
        # Get all image paths (only those present in all metadata files)
        all_image_paths = data_processor.get_all_image_paths(
            require_all_metadata=self.config.data.require_all_metadata
        )
        
        # Apply max_images limit if specified
        if self.config.data.max_images:
            all_image_paths = all_image_paths[:self.config.data.max_images]
            logger.info(f"Limited to {len(all_image_paths)} images due to max_images setting")
        
        logger.info(f"Processing {len(all_image_paths)} images with 1 prompt each")
        
        # Distribute prompts across images
        prompt_assignments = prompt_distributor.distribute_prompts(all_image_paths)
        
        # Validate assignments
        if not prompt_distributor.validate_assignments(prompt_assignments, len(all_image_paths)):
            raise ValueError("Prompt assignment validation failed")
        
        # Load metadata for all images
        logger.info("Loading metadata for all images...")
        depth_data, obj_desc_data, fg_anno_data = data_processor.load_all_data()
        metadata_dict = {}
        
        for image_path in all_image_paths:
            metadata = data_processor.combine_metadata(image_path, depth_data, obj_desc_data, fg_anno_data)
            metadata_dict[image_path] = metadata
        
        # Create work items from assignments
        work_items = self._create_work_items_from_assignments(prompt_assignments, metadata_dict)
        logger.info(f"Created {len(work_items)} work items")
        
        all_results = []
        
        # Process in batches
        batch_size = self.config.processing.batch_size
        num_workers = self.config.processing.num_workers or mp.cpu_count()
        
        for i in range(0, len(work_items), batch_size):
            batch_work_items = work_items[i:i + batch_size]
            
            # Split work items across workers
            chunk_size = max(1, len(batch_work_items) // num_workers)
            work_chunks = [
                batch_work_items[j:j + chunk_size]
                for j in range(0, len(batch_work_items), chunk_size)
            ]
            
            # Process with ThreadPoolExecutor for I/O bound tasks
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._worker_function, chunk): chunk
                    for chunk in work_chunks
                }
                
                batch_results = []
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        batch_results.extend(chunk_results)
                        
                        # Update counters
                        for result in chunk_results:
                            if result.success:
                                self.processed_count += 1
                            else:
                                self.failed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                        self.failed_count += len(future_to_chunk[future])
            
            # Save batch results
            all_results.extend(batch_results)
            self._save_results(batch_results, append=len(all_results) > len(batch_results))
            
            # Print progress
            if self.config.logging.show_progress and len(all_results) % self.config.logging.progress_frequency == 0:
                self._print_progress()
        
        # Final summary
        total_time = time.time() - self.start_time
        logger.info(f"Processing complete! Total time: {total_time:.2f}s")
        logger.info(f"Successful: {self.processed_count}, Failed: {self.failed_count}")
        logger.info(f"Results saved to: {self.output_path}")
        
        return all_results
    
    def process_sample(self, data_processor: DataProcessor, prompt_distributor: PromptDistributor, num_images: int = 5) -> List[ProcessingResult]:
        """Process a sample of images for testing."""
        logger.info(f"Processing sample of {num_images} images...")
        
        # Get sample image paths
        all_image_paths = data_processor.get_all_image_paths(
            require_all_metadata=self.config.data.require_all_metadata
        )
        sample_image_paths = all_image_paths[:num_images]
        
        # Distribute prompts for sample
        prompt_assignments = prompt_distributor.distribute_prompts(sample_image_paths)
        
        # Load metadata for sample images
        depth_data, obj_desc_data, fg_anno_data = data_processor.load_all_data()
        metadata_dict = {}
        
        for image_path in sample_image_paths:
            metadata = data_processor.combine_metadata(image_path, depth_data, obj_desc_data, fg_anno_data)
            metadata_dict[image_path] = metadata
        
        # Create work items
        work_items = self._create_work_items_from_assignments(prompt_assignments, metadata_dict)
        
        self.start_time = time.time()
        results = self._worker_function(work_items)
        
        # Update counters
        for result in results:
            if result.success:
                self.processed_count += 1
            else:
                self.failed_count += 1
        
        # Save results
        self._save_results(results, append=False)
        self._print_progress()
        
        return results


def main():
    """Test the multiprocessing handler."""
    from config_loader import ConfigLoader
    
    try:
        # Load configuration
        config = ConfigLoader.load_config("config.yaml")
        
        # Initialize components
        data_processor = DataProcessor(".")
        handler = MultiprocessingHandler(config)
        prompt_distributor = PromptDistributor(
            distribution=config.prompts.distribution,
            random_seed=config.prompts.random_seed,
            shuffle_dataset=config.prompts.shuffle_dataset
        )
        
        # Process a small sample
        results = handler.process_sample(data_processor, prompt_distributor, num_images=3)
        
        # Print some results
        for i, result in enumerate(results[:3]):
            print(f"\n--- Result {i + 1} ---")
            print(f"Image: {result.image_path}")
            print(f"Category: {result.prompt_category}")
            print(f"Prompt: {result.prompt[:100]}...")
            print(f"Success: {result.success}")
            if result.success:
                print(f"Response length: {len(result.response)}")
                print(f"Response preview: {result.response[:200]}...")
    
    except Exception as e:
        print(f"Error in test: {e}")


if __name__ == "__main__":
    main()
