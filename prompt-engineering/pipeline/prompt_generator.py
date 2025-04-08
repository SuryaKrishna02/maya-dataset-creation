import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Set, List, Union
from pydantic import BaseModel


# Load languages from few_shot_data.json
data_dir = Path(__file__).parent.parent / "data"
language_data_path = data_dir / "few_shot_data.json"

try:
    with open(language_data_path, "r", encoding="utf-8") as f:
        language_data = json.load(f)
    LANGUAGES = set(language_data.keys())
except (FileNotFoundError, json.JSONDecodeError):
    print(
        f"Error loading language data from {language_data_path}. Language set will be empty."
    )
    LANGUAGES = set()  # Empty set if file not found or invalid


class Prompt(BaseModel):
    """Pydantic class for a prompt template. Main function is `format` which returns the prompt with the variables substituted."""

    prompt_name: str
    prompt_type: Literal["Zero-Shot", "Few-Shot"]
    prompt_template: str = ""
    is_preamble: bool = False
    prompt_metadata: Optional[Dict[str, bool]] = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.load_template()
        self.check_if_preamble()
        self.extract_metadata()

    def load_template(self) -> str:
        """Load the prompt template from the corresponding file."""
        # Determine the directory based on prompt type
        type_dir = self.prompt_type.lower().replace(
            "-", "_"
        )  # Convert to lowercase and replace hyphen with underscore

        template_path = (
            Path(__file__).parent.parent
            / "prompt_data"
            / type_dir
            / f"{self.prompt_name}.txt"
        )

        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        return self.prompt_template

    def check_if_preamble(self) -> None:
        """Check if this is a preamble (doesn't contain {user_input})."""
        self.is_preamble = "{user_input}" not in self.prompt_template

    def extract_metadata(self) -> None:
        """Extract metadata from prompt template (lines starting with #!)"""
        self.prompt_metadata = {}
        lines = self.prompt_template.split("\n")
        cleaned_lines = []

        for line in lines:
            if line.startswith("#!"):
                # Extract metadata
                metadata_line = line[2:].strip().lower()
                if metadata_line == "reasoning":
                    self.prompt_metadata["reasoning"] = True
            else:
                cleaned_lines.append(line)

        # Update the template with metadata lines removed
        self.prompt_template = "\n".join(cleaned_lines)

    def format(
        self,
        language: str,
        user_input: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format the prompt template with language data, few-shot examples (if prompt_type is "Few-Shot") and additional variables.

        If user_input is provided and this is not a preamble, it will replace {user_input} in the template.
        Otherwise, {user_input} will be preserved for later substitution.

        Args:
            language: The target language for translation
            user_input: Optional text to substitute for {user_input} placeholder
            variables: Optional dictionary of additional variables to substitute

        Returns:
            A new Prompt object with the formatted template
        """
        # Check if language is supported
        if language not in LANGUAGES:
            available_languages = ", ".join(sorted(LANGUAGES))
            raise ValueError(
                f"Language '{language}' not found in language data. Available languages: {available_languages}"
            )

        # Load language data from JSON
        data_dir = Path(__file__).parent.parent / "data"
        language_data_path = data_dir / "few_shot_data.json"

        with open(language_data_path, "r", encoding="utf-8") as f:
            all_language_data = json.load(f)

        # Get language-specific examples
        language_examples = all_language_data.get(language, {})

        # Create the format variables dictionary
        format_vars = {"target_language": language}
        format_vars.update(language_examples)
        if variables:
            format_vars.update(variables)

        # Format the template
        formatted_template = self.prompt_template
        for var_placeholder, content in format_vars.items():
            try:
                # Create a format string with the specific placeholder
                placeholder_str = "{" + var_placeholder + "}"
                if placeholder_str in formatted_template:
                    formatted_template = formatted_template.replace(
                        placeholder_str, str(content)
                    )
            except Exception as e:
                print(f"Error replacing {var_placeholder}: {e}")

        if not self.is_preamble and user_input is not None:
            formatted_template = formatted_template.replace("{user_input}", user_input)
        return formatted_template


def load_all_prompts(verbose: bool = False) -> List[Prompt]:
    """
    Load all prompts from the few-shot and zero-shot directories.

    Args:
        verbose: Whether to print detailed loading information

    Returns:
        A list of Prompt objects
    """
    prompts: List[Prompt] = []
    base_dir = Path(__file__).parent.parent / "prompt_data"

    # Load zero-shot prompts
    zero_shot_dir = base_dir / "zero_shot"
    if zero_shot_dir.exists():
        if verbose:
            print(f"Loading Zero-Shot prompts from {zero_shot_dir}")
        for file_path in zero_shot_dir.glob("*.txt"):
            prompt_name = file_path.stem
            if verbose:
                print(f"  - Loading {prompt_name}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    has_user_input = "{user_input}" in content
                    if verbose:
                        print(f"    - Has user_input placeholder: {has_user_input}")

                prompt = Prompt(prompt_name=prompt_name, prompt_type="Zero-Shot")
                if verbose:
                    print(f"    - Is preamble: {prompt.is_preamble}")
                prompts.append(prompt)
            except Exception as e:
                print(f"ERROR loading {prompt_name}: {e}")  # Always print errors

    # Load few-shot prompts
    few_shot_dir = base_dir / "few_shot"
    if few_shot_dir.exists():
        if verbose:
            print(f"Loading Few-Shot prompts from {few_shot_dir}")
        for file_path in few_shot_dir.glob("*.txt"):
            prompt_name = file_path.stem
            if verbose:
                print(f"  - Loading {prompt_name}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    has_user_input = "{user_input}" in content
                    if verbose:
                        print(f"    - Has user_input placeholder: {has_user_input}")

                prompt = Prompt(prompt_name=prompt_name, prompt_type="Few-Shot")
                if verbose:
                    print(f"    - Is preamble: {prompt.is_preamble}")
                prompts.append(prompt)
            except Exception as e:
                print(f"ERROR loading {prompt_name}: {e}")  # Always print errors

    return prompts


# Create a list of all available prompts
ALL_PROMPTS: List[Prompt] = load_all_prompts(verbose=False)
