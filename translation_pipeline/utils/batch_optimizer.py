"""
Batch optimizer module for the translation pipeline.

This module helps determine the optimal batch size for GPU processing
to avoid out-of-memory errors.
"""

import gc
import torch
import logging
from typing import List, Optional, Tuple
from utils.logger import TranslationLogger
from utils.data_loader import load_sample_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompt_templates import format_prompt, get_prompt_for_language


def estimate_memory_usage(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_texts: List[str],
    prompt_template,
    dtype=torch.float16
) -> Tuple[float, float]:
    """
    Estimate memory usage for a batch of input texts during inference.
    
    Args:
        model: Model to use for estimation
        tokenizer: Tokenizer to use for estimation
        input_texts: List of input texts to estimate memory usage for
        prompt_template: Prompt template to use
        dtype: Data type to use for tensors
        
    Returns:
        Tuple of (gpu_memory_usage_gb, input_length_stats)
    """
    # Format inputs
    formatted_inputs = []
    for text in input_texts:
        messages = format_prompt(prompt_template, text)
        formatted_inputs.append(messages)
    
    # Process all inputs using chat template
    batch_input_ids = []
    for messages in formatted_inputs:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        batch_input_ids.append(input_ids)
    
    # Pad to max length in batch
    max_length = max(ids.size(1) for ids in batch_input_ids)
    padded_input_ids = []
    attention_masks = []
    
    for input_ids in batch_input_ids:
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(input_ids.size())
        
        # Calculate padding needed
        padding_length = max_length - input_ids.size(1)
        
        if padding_length > 0:
            # Pad input_ids with padding token
            padded = torch.nn.functional.pad(
                input_ids,
                (0, padding_length),
                value=tokenizer.pad_token_id
            )
            # Extend attention mask with zeros for padding
            mask_padding = torch.zeros(input_ids.size(0), padding_length)
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
        else:
            padded = input_ids
        
        padded_input_ids.append(padded)
        attention_masks.append(attention_mask)
    
    # Calculate memory usage for inference only
    # Each token consumes memory based on its embedding size and model parameters
    if dtype == torch.float16:
        bytes_per_token = 2  # bytes for float16
    else:
        bytes_per_token = 4  # bytes for float32
    
    # Estimate memory consumption based on model size
    model_params_gb = sum(p.numel() for p in model.parameters()) * bytes_per_token / (1024**3)
    
    # Estimate KV cache size for attention during generation
    # KV cache size = 2 (K and V) * num_layers * batch_size * seq_length * hidden_size * bytes_per_param
    num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 12
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768
    batch_size = len(input_texts)
    
    # During inference, memory is dominated by:
    # 1. Model weights
    # 2. KV cache for attention
    # 3. Temporary activations
    kv_cache_gb = 2 * num_layers * batch_size * max_length * hidden_size * bytes_per_token / (1024**3)
    
    # Activations memory (rough estimate)
    activations_multiplier = 0.3  # Only a fraction of activations needed for inference vs training
    activations_gb = model_params_gb * activations_multiplier
    
    # Total memory usage (no gradient storage or optimizer states needed for inference)
    gpu_memory_usage_gb = model_params_gb + kv_cache_gb + activations_gb
    
    # Store input length stats
    input_length_stats = max_length
    
    # Clear memory
    del batch_input_ids, padded_input_ids, attention_masks
    torch.cuda.empty_cache()
    gc.collect()
    
    return gpu_memory_usage_gb, input_length_stats


def find_optimal_batch_size(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_texts: List[str],
    prompt_template,
    min_batch_size: int = 1,
    max_batch_size: int = 32,
    target_memory_usage_gb: Optional[float] = None,
    logger: Optional[TranslationLogger] = None
) -> int:
    """
    Find the optimal batch size for GPU processing.
    
    Args:
        model: Model to use for estimation
        tokenizer: Tokenizer to use for estimation
        sample_texts: Sample texts to use for estimation
        prompt_template: Prompt template to use
        min_batch_size: Minimum batch size to consider
        max_batch_size: Maximum batch size to consider
        target_memory_usage_gb: Target GPU memory usage in GB (default: None, uses 80% of available memory)
        logger: Logger to use (optional)
        
    Returns:
        Optimal batch size
    """
    # If target memory not specified, use 80% of available memory
    if target_memory_usage_gb is None:
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            target_memory_usage_gb = total_memory_gb * 0.8
        else:
            # If CUDA not available, use a conservative estimate
            target_memory_usage_gb = 4.0
    
    # Logger for verbose output
    log = logger.get_logger() if logger else logging.getLogger()
    
    log.info(f"Finding optimal batch size (target memory: {target_memory_usage_gb:.2f} GB)")
    
    # Start with a single example to estimate per-example memory usage
    memory_usage_1sample, _ = estimate_memory_usage(
        model, tokenizer, sample_texts[:1], prompt_template
    )
    
    # Estimate memory usage per sample
    memory_per_sample = memory_usage_1sample
    
    # Calculate batch size based on memory usage
    estimated_batch_size = max(
        min_batch_size,
        min(
            max_batch_size,
            int(target_memory_usage_gb / memory_per_sample)
        )
    )
    
    log.info(f"Estimated memory per sample: {memory_per_sample:.2f} GB")
    log.info(f"Estimated optimal batch size: {estimated_batch_size}")
    
    # Test the estimated batch size
    try:
        # Ensure we have enough samples
        actual_samples = min(len(sample_texts), estimated_batch_size)
        test_samples = sample_texts[:actual_samples]
        
        # Try a test run to verify
        log.info(f"Testing batch size {actual_samples}...")
        
        # Process inputs using chat template
        formatted_inputs = []
        for text in test_samples:
            messages = format_prompt(prompt_template, text)
            formatted_inputs.append(messages)
        
        batch_input_ids = []
        for messages in formatted_inputs:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            batch_input_ids.append(input_ids)
        
        # Pad to max length in batch
        max_length = max(ids.size(1) for ids in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        
        for input_ids in batch_input_ids:
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(input_ids.size())
            
            # Calculate padding needed
            padding_length = max_length - input_ids.size(1)
            
            if padding_length > 0:
                # Pad input_ids with padding token
                padded = torch.nn.functional.pad(
                    input_ids,
                    (0, padding_length),
                    value=tokenizer.pad_token_id
                )
                # Extend attention mask with zeros for padding
                mask_padding = torch.zeros(input_ids.size(0), padding_length)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
            else:
                padded = input_ids
            
            padded_input_ids.append(padded)
            attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        batch_input_ids = torch.cat(padded_input_ids, dim=0).to(model.device)
        batch_attention_mask = torch.cat(attention_masks, dim=0).to(model.device)
        
        # Generate outputs without gradient tracking
        with torch.no_grad():
            _ = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=10,  # Just generate a few tokens for testing
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        log.info(f"Batch size {actual_samples} works without OOM errors")
        
        # Clear memory
        del batch_input_ids, batch_attention_mask, padded_input_ids, attention_masks
        torch.cuda.empty_cache()
        gc.collect()
        
    except RuntimeError as e:
        # If we hit an OOM error, reduce the batch size
        if "CUDA out of memory" in str(e):
            log.warning(f"OOM error with batch size {actual_samples}, reducing batch size")
            # Clear memory after OOM error
            torch.cuda.empty_cache()
            gc.collect()
            # Reduce by 50% and try again (recursive)
            new_max = max(min_batch_size, int(estimated_batch_size * 0.5))
            return find_optimal_batch_size(
                model, tokenizer, sample_texts, prompt_template,
                min_batch_size, new_max, target_memory_usage_gb, logger
            )
        else:
            # Re-raise if not an OOM error
            raise
    
    # Return a slightly conservative batch size (90% of estimated) to account for variations
    final_batch_size = max(min_batch_size, int(estimated_batch_size * 0.9))
    log.info(f"Final optimal batch size: {final_batch_size}")
    
    return final_batch_size


def calculate_optimal_batch_size(
    model_name: str,
    language: str,
    sample_dataset_path: str,
    precision: str = "float16",
    hf_access_token: Optional[str] = None,
    min_batch_size: int = 1,
    max_batch_size: int = 32,
    logger: Optional[TranslationLogger] = None
) -> int:
    """
    Calculate the optimal batch size for the given model and dataset.
    
    Args:
        model_name: Name of the model to use
        language: Language to calculate batch size for
        sample_dataset_path: Path to the sample dataset
        precision: Precision to use for the model (default: "float16")
        hf_access_token: HuggingFace access token (default: None)
        min_batch_size: Minimum batch size to consider (default: 1)
        max_batch_size: Maximum batch size to consider (default: 32)
        logger: Logger to use (optional)
        
    Returns:
        Optimal batch size
    """
    log = logger.get_logger() if logger else logging.getLogger()
    
    # Load tokenizer and model
    log.info(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_access_token)
    
    # Set precision
    dtype = torch.float16 if precision == "float16" else torch.float32
    
    # Load model in half precision and set for inference only
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_access_token
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation explicitly
    torch.set_grad_enabled(False)
    
    # Load sample dataset
    log.info(f"Loading sample dataset: {sample_dataset_path}")
    sample_data = load_sample_dataset(sample_dataset_path)
    
    # Extract GPT values for testing
    gpt_values = []
    for entry in sample_data:
        if 'conversations' in entry and isinstance(entry['conversations'], list):
            for conv in entry['conversations']:
                if conv.get('from') == 'gpt':
                    gpt_values.append(conv.get('value', ''))
    
    # Ensure we have some values to test with
    if not gpt_values:
        log.warning("No GPT values found in sample dataset, using default batch size")
        return min_batch_size
    
    # Get prompt template for language
    prompt_template = get_prompt_for_language(language)
    if not prompt_template:
        log.warning(f"No prompt template found for language '{language}', using default batch size")
        return min_batch_size
    
    # Find optimal batch size
    optimal_batch_size = find_optimal_batch_size(
        model=model,
        tokenizer=tokenizer,
        sample_texts=gpt_values,
        prompt_template=prompt_template,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        logger=logger
    )
    
    # Clean up memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return optimal_batch_size