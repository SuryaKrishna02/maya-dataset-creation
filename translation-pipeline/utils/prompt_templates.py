"""
Prompt templates module for the translation pipeline.

This module defines prompt templates for different languages used in the translation pipeline.
"""

import json
from textwrap import dedent
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Prompt:
    """Class to define a prompt template with system and user messages."""
    system_msg: str
    user_msg: str


# Constant for message placeholder
MESSAGE_PLACEHOLDER = "Input Text: {message}"
SYSTEM_PLACEHOLDER = dedent("""
                ## You are an expert translator. Translate the following text into {target_language} by breaking down the task step by step. Think briefly, then provide the final translation. (Note: In the final output, only the final translation should be output.)

                Note: Do not include any explanations or reasoning.

                ###Instructions to follow before translation:
                1. Identify the key objects, attributes, and context of the text.
                2. Decide on the best translation for each component.
                3. Assemble the components, ensuring the structure and tone match the original.

                Output Format:
                ```json
                {{
                    "translated_text": "<Translation in {target_language}>"
                }}
                ```
""")


# Hindi translation prompt
HINDI_PROMPT = Prompt(
    system_msg=SYSTEM_PLACEHOLDER,
    user_msg=MESSAGE_PLACEHOLDER
)

# Spanish translation prompt
SPANISH_PROMPT = Prompt(
    system_msg=SYSTEM_PLACEHOLDER,
    user_msg=MESSAGE_PLACEHOLDER
)

# French translation prompt
FRENCH_PROMPT = Prompt(
    system_msg=SYSTEM_PLACEHOLDER,
    user_msg=MESSAGE_PLACEHOLDER
)

# Chinese translation prompt
CHINESE_PROMPT = Prompt(
    system_msg=SYSTEM_PLACEHOLDER,
    user_msg=MESSAGE_PLACEHOLDER
)

# Collection of all prompts
LANGUAGE_PROMPTS = {
    "hindi": HINDI_PROMPT,
    "spanish": SPANISH_PROMPT,
    "french": FRENCH_PROMPT,
    "chinese": CHINESE_PROMPT,
}

def get_prompt_for_language(language: str) -> Optional[Prompt]:
    """
    Get the prompt template for a specific language.
    
    Args:
        language: Language code (lowercase)
        
    Returns:
        Prompt template for the specified language, or None if not found
    """
    return LANGUAGE_PROMPTS.get(language.lower())


def load_prompts_from_file(file_path: str) -> Dict[str, Prompt]:
    """
    Load prompt templates from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing prompt templates
        
    Returns:
        Dictionary mapping language codes to Prompt objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        # Convert the loaded JSON data to Prompt objects
        prompts = {}
        for lang, prompt_data in prompts_data.items():
            if 'system_msg' in prompt_data and 'user_msg' in prompt_data:
                # Clean up the system message using dedent
                system_msg = dedent(prompt_data['system_msg'])
                prompts[lang.lower()] = Prompt(
                    system_msg=system_msg,
                    user_msg=prompt_data['user_msg']
                )
        
        if not prompts:
            raise ValueError(f"No valid prompt templates found in {file_path}")
        
        return prompts
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading prompts from file: {e}")
        # Fall back to hardcoded prompts
        return LANGUAGE_PROMPTS


def format_prompt(prompt: Prompt, message: str) -> List[Dict[str, str]]:
    """
    Format a prompt for use with the model.
    
    Args:
        prompt: Prompt template
        message: Message to be inserted into the user message template
        
    Returns:
        Formatted conversation messages for the model
    """
    return [
        {"role": "system", "content": prompt.system_msg},
        {"role": "user", "content": prompt.user_msg.format(message=message)}
    ]