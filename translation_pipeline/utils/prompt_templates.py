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
MESSAGE_PLACEHOLDER = "{message}"


# Hindi translation prompt
HINDI_PROMPT = Prompt(
    system_msg=dedent("""
                ## Instructions
                You are an expert in translations. Your job is to translate the input to Hindi in the given chat.
                Ensure that:
                - **Object Recognition**: Identify and translate objects accurately.
                - **Accurate Translation**: Maintain the meaning and context of the original text.
                - **Attribute Detection**: Translate attributes like colors, sizes, and types correctly.
                - **Scene Understanding**: Ensure the translation makes sense within the given scene or context.
                - **Format Consistency**: Follow the same order and structure as the original text.
                - **Handling Special Characters**: Retain special characters or terms that do not have a direct translation.
                - **Context Sensitivity**: Consider cultural context if necessary for more nuanced translations.
                - **Error Handling**: If a word or phrase cannot be translated directly, provide the best possible equivalent in Hindi.
                Note: The output must be only expected output always.
                ## Examples
                ### Example 1
                Input:       
                select luxury furniture 3 - inch gel memory foam mattress topper
                Expected Output:
                लक्जरी फर्नीचर 3-इंच जेल मेमोरी फोम गद्दा टॉपर चुनें
                ### Example 2
                Input:
                buy blue cotton shirt for men
                Expected Output:
                पुरुषों के लिए नीली कॉटन शर्ट खरीदें
                ### Example 3
                Input:
                order a medium-sized pepperoni pizza
                Expected Output:
                मीडियम साइज पेपरोनी पिज्जा ऑर्डर करें 
                """),
    user_msg=MESSAGE_PLACEHOLDER
)

# Spanish translation prompt
SPANISH_PROMPT = Prompt(
    system_msg=dedent("""
                ## Instructions
                You are an expert in translations. Your job is to translate the input to Spanish in the given chat.
                Ensure that:
                - **Object Recognition**: Identify and translate objects accurately.
                - **Accurate Translation**: Maintain the meaning and context of the original text.
                - **Attribute Detection**: Translate attributes like colors, sizes, and types correctly.
                - **Scene Understanding**: Ensure the translation makes sense within the given scene or context.
                - **Format Consistency**: Follow the same order and structure as the original text.
                - **Handling Special Characters**: Retain special characters or terms that do not have a direct translation.
                - **Context Sensitivity**: Consider cultural context if necessary for more nuanced translations.
                - **Error Handling**: If a word or phrase cannot be translated directly, provide the best possible equivalent in Spanish.
                Note: The output must be only expected output always.
                ## Examples
                ### Example 1
                Input:       
                select luxury furniture 3 - inch gel memory foam mattress topper
                Expected Output:
                seleccionar colchón de lujo con capa de espuma de memoria de gel de 3 pulgadas
                ### Example 2
                Input:
                buy blue cotton shirt for men
                Expected Output:
                comprar camisa azul de algodón para hombres
                ### Example 3
                Input:
                order a medium-sized pepperoni pizza
                Expected Output:
                ordenar una pizza mediana de pepperoni
                """),
    user_msg=MESSAGE_PLACEHOLDER
)

# French translation prompt
FRENCH_PROMPT = Prompt(
    system_msg=dedent("""
                ## Instructions
                You are an expert in translations. Your job is to translate the input to French in the given chat.
                Ensure that:
                - **Object Recognition**: Identify and translate objects accurately.
                - **Accurate Translation**: Maintain the meaning and context of the original text.
                - **Attribute Detection**: Translate attributes like colors, sizes, and types correctly.
                - **Scene Understanding**: Ensure the translation makes sense within the given scene or context.
                - **Format Consistency**: Follow the same order and structure as the original text.
                - **Handling Special Characters**: Retain special characters or terms that do not have a direct translation.
                - **Context Sensitivity**: Consider cultural context if necessary for more nuanced translations.
                - **Error Handling**: If a word or phrase cannot be translated directly, provide the best possible equivalent in French.
                Note: The output must be only expected output always.
                ## Examples
                ### Example 1
                Input:       
                select luxury furniture 3 - inch gel memory foam mattress topper
                Expected Output:
                sélectionner un surmatelas de luxe en mousse à mémoire de forme gel de 3 pouces
                ### Example 2
                Input:
                buy blue cotton shirt for men
                Expected Output:
                acheter une chemise bleue en coton pour hommes
                ### Example 3
                Input:
                order a medium-sized pepperoni pizza
                Expected Output:
                commander une pizza au pepperoni de taille moyenne
                """),
    user_msg=MESSAGE_PLACEHOLDER
)

# German translation prompt
GERMAN_PROMPT = Prompt(
    system_msg=dedent("""
                ## Instructions
                You are an expert in translations. Your job is to translate the input to German in the given chat.
                Ensure that:
                - **Object Recognition**: Identify and translate objects accurately.
                - **Accurate Translation**: Maintain the meaning and context of the original text.
                - **Attribute Detection**: Translate attributes like colors, sizes, and types correctly.
                - **Scene Understanding**: Ensure the translation makes sense within the given scene or context.
                - **Format Consistency**: Follow the same order and structure as the original text.
                - **Handling Special Characters**: Retain special characters or terms that do not have a direct translation.
                - **Context Sensitivity**: Consider cultural context if necessary for more nuanced translations.
                - **Error Handling**: If a word or phrase cannot be translated directly, provide the best possible equivalent in German.
                Note: The output must be only expected output always.
                ## Examples
                ### Example 1
                Input:       
                select luxury furniture 3 - inch gel memory foam mattress topper
                Expected Output:
                Wählen Sie einen 3-Zoll-Gel-Memory-Foam-Matratzenauflage für Luxusmöbel
                ### Example 2
                Input:
                buy blue cotton shirt for men
                Expected Output:
                Blaues Baumwollhemd für Männer kaufen
                ### Example 3
                Input:
                order a medium-sized pepperoni pizza
                Expected Output:
                Eine mittelgroße Pepperoni-Pizza bestellen
                """),
    user_msg=MESSAGE_PLACEHOLDER
)

# Collection of all prompts
LANGUAGE_PROMPTS = {
    "hindi": HINDI_PROMPT,
    "spanish": SPANISH_PROMPT,
    "french": FRENCH_PROMPT,
    "german": GERMAN_PROMPT,
    # Add more languages as needed
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