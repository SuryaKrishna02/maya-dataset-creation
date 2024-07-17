import csv
import json

def load_json(file_path: str) -> dict:
    """
    Load a JSON file from the given file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(data: dict, file_path: str) -> None:
    """
    Save a Python dictionary as a JSON file.

    Args:
        data (dict): The data to be saved as JSON.
        file_path (str): The path where the JSON file should be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def update_csv(csv_file_path: str, id: str, gpt_original: str, gpt_modified: str) -> None:
    """
    Append a row of data to a CSV file.

    Args:
        csv_file (str): The path to the CSV file.
        id (str): The ID of the processed item.
        gpt_original (str): The original GPT message.
        gpt_modified (str): The translated GPT message.
    """
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([id, gpt_original, gpt_modified])