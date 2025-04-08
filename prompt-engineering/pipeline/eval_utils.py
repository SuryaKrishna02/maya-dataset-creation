import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from janome.tokenizer import Tokenizer as JTokenizer
import jieba
from typing import List, Union, Optional, Tuple, Dict, Any


# Initialize NLTK tokenizers
nltk.download('punkt_tab')
janome_tokenizer = JTokenizer()

# Function to calculate BLEU score using NLTK for different n-grams with smoothing
def calculate_bleu_nltk(gold_translation: str, generated_translation: str, language: str, n_gram: int = 4) -> float:
    """
    Calculate BLEU score using NLTK for different n-grams with smoothing. Returns the BLEU score.
    """
    # Tokenize the translations based on the language
    if language == "Chinese":
        reference = [list(jieba.cut(gold_translation))]
        hypothesis = list(jieba.cut(generated_translation))
    elif language == "Japanese":
        reference = [[token.surface for token in janome_tokenizer.tokenize(gold_translation)]]
        hypothesis = [token.surface for token in janome_tokenizer.tokenize(generated_translation)]
    else:
        # Default tokenization for other languages
        reference = [nltk.word_tokenize(gold_translation)]
        hypothesis = nltk.word_tokenize(generated_translation)

    # Apply smoothing function
    smoothing = SmoothingFunction().method1

    # Calculate BLEU score based on n-gram
    if n_gram == 1:
        weight = (1.0, 0.0, 0.0, 0.0)
    elif n_gram == 2:
        weight = (0.5, 0.5, 0.0, 0.0)
    elif n_gram == 3:
        weight = (0.33, 0.33, 0.33, 0.0)
    else:
        weight = (0.25, 0.25, 0.25, 0.25)

    return sentence_bleu(reference, hypothesis, weights=weight, smoothing_function=smoothing)

