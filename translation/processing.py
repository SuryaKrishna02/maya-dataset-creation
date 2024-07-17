import sys
sys.path.append("../")
import json
import cohere
import logging
import cupy as cp
from utils.helpers import update_csv
from utils.constants import MODEL_NAME
from translation.prompts import HINDI_PROMPT
from ratelimit import sleep_and_retry, limits

@sleep_and_retry
@limits(calls=10000, period=60)  # 10,000 calls per 1 minute according cohere production key documentation
async def cohere_chat(co: cohere.AsyncClient, message: str) -> str | None:
    """
    Send a message to the Cohere API and get the response.

    Args:
        co (cohere.AsyncClient): The Cohere API client.
        message (str): The message to send to the API.

    Returns:
        str: The response from the Cohere API, or None if an error occurred.
    """
    try:
        if not isinstance(message, str):
            message = str(message)

        print(f"Sending message to Cohere: {message}")

        response = await co.chat(
            model=MODEL_NAME,
            # This is bit mouthful for simple prompt but when constructing complex messages this will be easier to manage.
            message=HINDI_PROMPT.message.format(message=message),
            temperature=0.3,
            preamble=HINDI_PROMPT.preamble,
            prompt_truncation='AUTO'
        )
        return response.text
    except Exception as e:
        print(f"Error in Cohere API call: {str(e)}")
        return None

def process_batch_gpu(batch):
    """
    Process a batch of items using GPU acceleration.

    Args:
        batch (list): A list of items to be processed.

    Returns:
        list: The processed batch, sorted by conversation length.
    """
    conversations = [item['conversations'] for item in batch]
    conv_lengths = cp.array([len(json.dumps(conv)) for conv in conversations])
    sorted_indices = cp.argsort(conv_lengths)
    sorted_batch = [batch[i] for i in sorted_indices.get()]
    return sorted_batch

def extract_translation(response):
    """
    Extract translated value from the Cohere API response.

    Args:
        response (str): The response string from the Cohere API.

    Returns:
        tuple: A tuple containing the translated message and any error message.
    """
    translation = response.strip()
    error = None
    if not translation:
        error = f"Failed to extract translation. Response: {response}"
    return translation, error

async def process_item(item, co, csv_file):
    """
    Process a single item by sending it to the Cohere API for translation,
    updating the original data, and saving the results to a CSV file.

    Args:
        item (dict): The item to be processed.
        co (cohere.AsyncClient): The Cohere API client.
        csv_file (str): The path to the CSV file for saving results.

    Returns:
        bool: True if the item was processed successfully, False otherwise.
    """
    conversations = item['conversations']
    try:
        gpt_message = next(conv['value'] for conv in conversations if conv['from'] == 'gpt')
        response = await cohere_chat(co, gpt_message)

        if response:
            translation, error = extract_translation(response)

            if error:
                print(f"Error processing item {item['id']}: {error}")
                logging.error(f"Item {item['id']} - {error}")
                return False

            # Update the original conversations with translation
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    conv['value'] = translation

            update_csv(csv_file, item['id'], gpt_message, translation)
            return True
        else:
            print(f"Error: No response from Cohere API for item {item['id']}")
            logging.error(f"Item {item['id']} - No response from Cohere API")
    except Exception as e:
        print(f"Error processing item {item['id']}: {str(e)}")
        logging.exception(f"Item {item['id']} - Unexpected error")
    return False
