import sys
from os import environ
from dotenv import load_dotenv

load_dotenv()

sys.path.append("../")

MODEL_NAME = "c4ai-aya-23"
TRANSLATION_BATCH_SIZE = 10 # Maximum is 165
COHERE_API_KEY = environ.get("COHERE_API_KEY")
ISO_639_1_CODES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese"
}
API_CALLS_PER_MINUTE = 10_000
API_CALLS_TIME_PERIOD = 60