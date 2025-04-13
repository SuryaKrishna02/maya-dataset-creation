"""
Data loader module for the translation pipeline.

This module handles loading and processing data for the translation pipeline.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple, Optional


def load_dataset(file_path: str) -> Dict[str, Any]:
    """
    Load dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dataset as a dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in dataset file: {file_path}")


def load_human_values_translation(file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load human values translation dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing translations
        
    Returns:
        Dictionary mapping original values to translations for each language
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        
        # Validate structure: translations[language][original_text] = translated_text
        for lang, trans_dict in translations.items():
            if not isinstance(trans_dict, dict):
                raise ValueError(f"Invalid format for language '{lang}' in translations file")
        
        return translations
    except FileNotFoundError:
        raise FileNotFoundError(f"Human values translation file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in human values translation file: {file_path}")


def validate_dataset(dataset: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate dataset structure.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if it's a list
    if not isinstance(dataset, list):
        errors.append("Dataset should be a list of entries")
        return False, errors
    
    # Check each entry
    for i, entry in enumerate(dataset):
        errors.extend(validate_entry(entry, i))
    
    return len(errors) == 0, errors


def validate_entry(entry: Dict[str, Any], index: int) -> List[str]:
    """
    Validate a single entry in the dataset.
    
    Args:
        entry: Entry to validate
        index: Index of the entry
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required fields
    for field in ['id', 'image', 'conversations']:
        if field not in entry:
            errors.append(f"Entry {index} missing required field: {field}")
    
    # Check conversations structure
    if 'conversations' in entry:
        errors.extend(validate_conversations(entry['conversations'], index))
    
    return errors


def validate_conversations(conversations: Any, entry_index: int) -> List[str]:
    """
    Validate conversations in an entry.
    
    Args:
        conversations: Conversations to validate
        entry_index: Index of the parent entry
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not isinstance(conversations, list):
        errors.append(f"Entry {entry_index}: 'conversations' should be a list")
        return errors
    
    for j, conv in enumerate(conversations):
        if not isinstance(conv, dict):
            errors.append(f"Entry {entry_index}, conversation {j}: Should be a dictionary")
        elif 'from' not in conv or 'value' not in conv:
            errors.append(f"Entry {entry_index}, conversation {j}: Missing 'from' or 'value' field")
    
    return errors


def create_batches(dataset: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Split dataset into batches.
    
    Args:
        dataset: Dataset to split
        batch_size: Size of each batch
        
    Returns:
        List of batches, where each batch is a list of dataset entries
    """
    return [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]


def extract_human_values(dataset: List[Dict[str, Any]]) -> List[str]:
    """
    Extract unique human values from the dataset.
    
    Args:
        dataset: Dataset
        
    Returns:
        List of unique human values
    """
    human_values = set()
    
    for entry in dataset:
        if 'conversations' in entry and isinstance(entry['conversations'], list):
            for conv in entry['conversations']:
                if conv.get('from') == 'human':
                    human_values.add(conv.get('value', ''))
    
    return list(human_values)


def extract_gpt_values(dataset: List[Dict[str, Any]]) -> List[str]:
    """
    Extract unique GPT values from the dataset.
    
    Args:
        dataset: Dataset
        
    Returns:
        List of unique GPT values
    """
    gpt_values = set()
    
    for entry in dataset:
        if 'conversations' in entry and isinstance(entry['conversations'], list):
            for conv in entry['conversations']:
                if conv.get('from') == 'gpt':
                    gpt_values.add(conv.get('value', ''))
    
    return list(gpt_values)


def save_batch_to_json(
    batch: List[Dict[str, Any]],
    output_path: str,
    language: str,
    batch_num: int
) -> None:
    """
    Save a batch of translated data to a JSON file.
    
    Args:
        batch: Batch to save
        output_path: Directory to save the file in
        language: Language code
        batch_num: Batch number
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{language}_batch_{batch_num}.json")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)


def load_sample_dataset(file_path: str, sample_size: int = 10) -> List[Dict[str, Any]]:
    """
    Load a sample of the dataset for batch size optimization.
    
    Args:
        file_path: Path to the dataset file
        sample_size: Number of entries to sample
        
    Returns:
        List of sample entries
    """
    full_dataset = load_dataset(file_path)
    
    if isinstance(full_dataset, list):
        # If it's a list, take the first sample_size entries
        return full_dataset[:min(sample_size, len(full_dataset))]
    elif isinstance(full_dataset, dict) and 'entries' in full_dataset:
        # If it has an 'entries' key with a list
        entries = full_dataset['entries']
        if isinstance(entries, list):
            return entries[:min(sample_size, len(entries))]
    
    # If we can't determine the structure, return an empty list
    return []


def merge_intermediate_files(
    intermediate_dir: str,
    output_path: str,
    language: str
) -> None:
    """
    Merge intermediate batch files into a single output file.
    
    Args:
        intermediate_dir: Directory containing intermediate files
        output_path: Path to save the merged file
        language: Language code
    """
    merged_data = []
    batch_files = [f for f in os.listdir(intermediate_dir) if f.startswith(f"{language}_batch_") and f.endswith(".json")]
    
    # Sort batch files by batch number
    batch_files.sort(key=lambda x: int(x.split('_batch_')[1].split('.')[0]))
    
    # Merge data from all batch files
    for batch_file in batch_files:
        file_path = os.path.join(intermediate_dir, batch_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            merged_data.extend(batch_data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error loading batch file {batch_file}: {e}")
    
    # Save merged data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)


def save_progress_state(
    state_file: str,
    language: str,
    batch_num: int,
    total_batches: int,
    status: str
) -> None:
    """
    Save progress state to a file for resuming later.
    
    Args:
        state_file: Path to the state file
        language: Current language
        batch_num: Current batch number
        total_batches: Total number of batches
        status: Status of the current process
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    # Load existing state if it exists
    state = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If there's an error loading the state file, start fresh
            state = {}
    
    # Update state
    if 'languages' not in state:
        state['languages'] = {}
    
    state['languages'][language] = {
        'current_batch': batch_num,
        'total_batches': total_batches,
        'status': status,
        'timestamp': os.path.getmtime(state_file) if os.path.exists(state_file) else 0
    }
    
    # Save state
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def load_progress_state(state_file: str) -> Dict[str, Any]:
    """
    Load progress state from a file.
    
    Args:
        state_file: Path to the state file
        
    Returns:
        State dictionary, or empty dict if file doesn't exist or is invalid
    """
    if not os.path.exists(state_file):
        return {}
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return state
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def get_resume_point(state_file: str, language: str) -> Tuple[Optional[int], int]:
    """
    Get the point to resume translation from.
    
    Args:
        state_file: Path to the state file
        language: Language to check
        
    Returns:
        Tuple of (current_batch, total_batches), where current_batch is None if 
        no previous progress exists
    """
    state = load_progress_state(state_file)
    
    if 'languages' in state and language in state['languages']:
        lang_state = state['languages'][language]
        return lang_state.get('current_batch'), lang_state.get('total_batches', 0)
    
    return None, 0