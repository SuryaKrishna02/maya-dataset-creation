import re
from typing import Optional, List, Dict, Any, Union, Tuple


# Normalize and clean up special characters or punctuation
def normalize_text(text: str) -> str:
    """
    Normalize and clean up text by removing special characters and punctuation.

    Args:
        text: The text to normalize

    Returns:
        The normalized text
    """
    # Remove special characters and punctuation
    text = re.sub(
        r"[^\w\s]", "", text
    )  # Removes all characters except word characters and whitespace
    return text.strip().lower()


def extract_thinking_content(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract content between <thinking> tags and return both the cleaned text and the thinking content.

    Args:
        text: The input text that may contain <thinking> tags

    Returns:
        A tuple containing (cleaned_text, thinking_content)
        thinking_content will be None if no thinking tags are found
    """
    # Extract content between <thinking> tags
    thinking_matches = re.findall(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
    reasoning_content = None

    if thinking_matches:
        reasoning_content = "\n".join(thinking_matches)

    # Remove content between <thinking> tags
    cleaned_text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)

    # Clean up any remaining tags
    cleaned_text = re.sub(r"<thinking>|</thinking>", "", cleaned_text)

    # Normalize whitespace
    cleaned_text = normalize_whitespace(cleaned_text)

    return cleaned_text, reasoning_content


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text, replacing multiple spaces with a single space.

    Args:
        text: The text to normalize

    Returns:
        Text with normalized whitespace
    """
    return re.sub(r"\s+", " ", text).strip()
