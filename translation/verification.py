import sys
sys.path.append("../")
import cohere
import os
from utils.helpers import update_csv
from utils.constants import MODEL_NAME
from translation.prompts import HINDI_PROMPT
from ratelimit import sleep_and_retry, limits
from utils.constants import COHERE_API_KEY, TRANSLATION_BATCH_SIZE
from utils.schemas import Prompt
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
import instructor
from pydantic import BaseModel, Field
import evaluate
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
import re


class LLMJudgeTranslationScore(BaseModel):
    """A Pydantic model for the response of the LLM judge translation scoring task."""
    equivalence_of_meaning: int = Field(None, ge=1, le=5)
    appropriate_wording: int = Field(None, ge=1, le=5)
    fully_consistent: int = Field(None, ge=1, le=5)


@sleep_and_retry
@limits(calls=10000, period=60)  # 10,000 calls per 1 minute according cohere production key documentation
def llm_as_judge_translation_score(
        reference: str,
        prediction: str, 
        reference_lang: str,
        prediction_lang: str, 
        model_id: str = "c4ai-aya-23"
    ) -> list[dict[str, int]]:
    """Sends a prompt to the Cohere API for scoring translations. Requires ISO language code for reference and prediction languages."""
    TRANSLATION_SCORING_PREAMBLE = Prompt(
        preamble="""## Instructions
    You are an expert in translations. Your job is to score translations based on three criteria: equivalence of meaning, appropriate wording, and fully consistent.

    For each of these criteria, assign a score from 1 to 5, where 1 is the lowest and 5 is the highest.

    - **Equivalence of Meaning**: How well does the translation convey the same meaning as the original text?
    - **Appropriate Wording**: How well does the translation use appropriate language and terminology?
    - **Fully Consistent**: How well does the translation maintain consistency with the original text?""",
        message="""{message}"""
    )
    message = f"""Message from {reference_lang}: {reference}
    Message to {prediction_lang}: {prediction}"""
    
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    co = cohere.Client(api_key=COHERE_API_KEY)
    client = instructor.from_cohere(co)
    response = client.chat.completions.create(
        response_model=LLMJudgeTranslationScore,
        messages=[
            {"role": "system", "content": TRANSLATION_SCORING_PREAMBLE.preamble},
            {"role": "system", "content": TRANSLATION_SCORING_PREAMBLE.message.format(message=message)}
        ],
        model=model_id,
        max_retries=3
    )
    return response


def compute_corpus_level_chrf(predictions, references, lowercase=False):
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=predictions, references=references, word_order=2, lowercase=lowercase)
    return results


def compute_sentence_level_chrf(predictions, references, lowercase=False):
    scores = []
    chrf = evaluate.load("chrf")
    for pred, ref in zip(predictions, references):
        score = chrf.compute(predictions=[pred], references=[[ref]], word_order=2, lowercase=lowercase)
        scores.append(score)
    return scores


def check_repeated_tokens(
    prediction: str, 
    reference: str,
    tokenizer: CohereTokenizerFast 
    ) -> bool:
    """Check if the prediction has repeated tokens (two or more identical tokens that occur consecutively) that don't appear in the reference text."""
    prediction_tokens = tokenizer.tokenize(prediction)
    reference_tokens = tokenizer.tokenize(reference)
    for i in range(len(prediction_tokens) - 1):
        if prediction_tokens[i] == prediction_tokens[i + 1] and prediction_tokens[i] not in reference_tokens:
            return True
    return False


def run_validation(
    dataset : Dataset,
    model_id: str = "CohereForAI/aya-23-35B",
    api_model_id: str = "c4ai-aya-23",
    do_repeated_tokens: bool = True,
    do_back_translate_chrf: bool = False,
    do_llm_judge_translation_score: bool = False, 
    reference_col_name: str = "",
    prediction_col_name: str  = ""
) -> dict:
    """Run validation on a dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    references = dataset[reference_col_name]
    predictions = dataset[prediction_col_name]
 
    repeated_tokens = []
    back_translate_chrf = []
    llm_judge_equivalence_of_meaning = []
    llm_judge_appropriate_wording = []
    llm_judge_fully_consistent = []

    for reference, prediction in zip(references, predictions):
        if do_repeated_tokens:
            repeated = check_repeated_tokens(prediction, reference, tokenizer)
            repeated_tokens.append(repeated)
        if do_back_translate_chrf:
            score = compute_sentence_level_chrf([prediction], [reference])
            score = score[0]['score']
            back_translate_chrf.append(score)
        if do_llm_judge_translation_score:
            response = llm_as_judge_translation_score(reference, prediction, "en", "es", api_model_id)
            llm_judge_equivalence_of_meaning.append(response.equivalence_of_meaning)
            llm_judge_appropriate_wording.append(response.appropriate_wording)
            llm_judge_fully_consistent.append(response.fully_consistent)

    results = {
        "repeated_tokens": repeated_tokens,
        "back_translate_chrf": back_translate_chrf, 
        "llm_judge_equivalence_of_meaning": llm_judge_equivalence_of_meaning,
        "llm_judge_appropriate_wording": llm_judge_appropriate_wording,
        "llm_judge_fully_consistent": llm_judge_fully_consistent
    }
    return results


def strip_image_tag(text: str) -> str:
    """Strip image tags from text."""
    return text.replace("<image>", "")


def flatten_conversations(example):
    # Extract and flatten conversations from the columns
    english_turns = [conv['value'] for conv in example['conversations_english']]
    hindi_turns = [conv['value'] for conv in example['conversations_hindi']]
    
    # Strip <image> tags
    english_turns = [re.sub(r'<image>', '', turn) for turn in english_turns]
    hindi_turns = [re.sub(r'<image>', '', turn) for turn in hindi_turns]
    
    # Join turns into a single string
    english_turns = '\n'.join(english_turns).strip('\n')
    hindi_turns = '\n'.join(hindi_turns).strip('\n')

    return {
        'conversations_english': english_turns,
        'conversations_hindi': hindi_turns 
    }


def filter_chrf_scores(example, threshold=0.5):
    """Filter CHRF scores based on a threshold."""
    example["back_translate_chrf_int"] = [0 if score >= threshold else 1 for score in example["back_translate_chrf"]]
    return example


### Sample run
if __name__ == "__main__":
    dataset = load_dataset("DrishtiSharma/combined_5k_samples_hindi_english_blip_laion")

    # prepare ds
    transformed_dataset = DatasetDict()
    for split in dataset.keys():
        transformed_dataset[split] = dataset[split].map(flatten_conversations, remove_columns=['conversations_english', 'conversations_hindi'])

    # tmp for testing
    transformed_dataset['train'] = transformed_dataset['train'].select(range(10))

    # tests to run
    repeated_tokens = True
    back_translate_chrf = True
    llm_judge_translation_score = False

    results_train = run_validation(
        dataset=transformed_dataset['train'],
        model_id="CohereForAI/aya-23-35B",
        do_repeated_tokens=repeated_tokens,
        do_back_translate_chrf=back_translate_chrf,
        do_llm_judge_translation_score=llm_judge_translation_score,
        reference_col_name="conversations_english",
        prediction_col_name="conversations_hindi"
    )

    for split in transformed_dataset.keys():
        if repeated_tokens:
            transformed_dataset[split] = transformed_dataset[split].add_column(
                "repeated_tokens", 
                results_train['repeated_tokens']
            )
        if back_translate_chrf:
            transformed_dataset[split] = transformed_dataset[split].add_column(
                "back_translate_chrf", 
                results_train['back_translate_chrf']
            )
        if llm_judge_translation_score:
            transformed_dataset[split] = transformed_dataset[split].add_column(
                "llm_judge_equivalence_of_meaning",
                  results_train['llm_judge_equivalence_of_meaning']
            )
            transformed_dataset[split] = transformed_dataset[split].add_column(
                "llm_judge_appropriate_wording",
                results_train['llm_judge_appropriate_wording']
            )
            transformed_dataset[split] = transformed_dataset[split].add_column(
                "llm_judge_fully_consistent", 
                results_train['llm_judge_fully_consistent']
            )

    # apply filter for back translate chrf
    for split in transformed_dataset.keys():
        transformed_dataset[split] = transformed_dataset[split].map(filter_chrf_scores, batched=True)
            
    print(transformed_dataset['train'][0])

    