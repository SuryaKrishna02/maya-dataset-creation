import sys
from os import environ
from dotenv import load_dotenv

load_dotenv()

sys.path.append("../")

COHERE_API_KEY = environ.get("COHERE_API_KEY")
MODEL_NAME = "c4ai-aya-23"