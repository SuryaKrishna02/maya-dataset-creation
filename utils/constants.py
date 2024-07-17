import sys
from os import environ
from dotenv import load_dotenv

load_dotenv()

sys.path.append("../")

MODEL_NAME = "c4ai-aya-23"
TRANSLATION_BATCH_SIZE = 10 # Maximum is 165
COHERE_API_KEY = environ.get("COHERE_API_KEY")