import sys
sys.path.append("../")
import cohere
from utils.constants import MODEL_NAME, COHERE_API_KEY, ISO_639_1_CODES, API_CALLS_PER_MINUTE, API_CALLS_TIME_PERIOD
from ratelimit import sleep_and_retry, limits
from utils.schemas import Prompt
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
import instructor
from pydantic import BaseModel, Field
import evaluate
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

from collections import defaultdict
from datasets import Dataset
from tqdm import tqdm
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio


def compute_corpus_level_chrf(predictions, references, lowercase=False):
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=predictions, references=references, word_order=2, lowercase=lowercase)
    return results


@sleep_and_retry
@limits(calls=API_CALLS_PER_MINUTE, period=API_CALLS_TIME_PERIOD)  # 10,000 calls per 1 minute according cohere production key documentation
async def translate(
    client: cohere.AsyncClient,
    reference: str,
    to_lang: str,
):
    """Translate text to a target language."""
    TRANSLATION_PROMPT = Prompt(
        preamble="""## Instructions
        You are an expert in translations. Your job is to translate text into a given language.""",
        message="""{reference}
        Translate the above message into {to_lang}."""
    )
    try:
        response = await client.chat(
            chat_history=[
                {"role": "SYSTEM", "message": TRANSLATION_PROMPT.preamble}
            ],
            message=TRANSLATION_PROMPT.message.format(reference=reference, to_lang=to_lang),
            model=MODEL_NAME
        )
        return response
    except Exception as e:
        return e 


@sleep_and_retry
@limits(calls=API_CALLS_PER_MINUTE, period=API_CALLS_TIME_PERIOD)  # 10,000 calls per 1 minute according cohere production key documentation
async def translate(
    client: cohere.AsyncClient,
    reference: str,
    to_lang: str,
):
    """Translate text to a target language."""
    TRANSLATION_PROMPT = Prompt(
        preamble="""## Instructions
        You are an expert in translations. Your job is to translate text into a given language.""",
        message="""{reference}
        Translate the above message into {to_lang}."""
    )
    try:
        response = await client.chat(
            chat_history=[
                {"role": "SYSTEM", "message": TRANSLATION_PROMPT.preamble}
            ],
            message=TRANSLATION_PROMPT.message.format(reference=reference, to_lang=to_lang),
            model=MODEL_NAME
        )
        return response
    except Exception as e:
        return e 
async def fetch_all_translations(
    client: cohere.AsyncClient,
    references: list[str],
    to_lang: str,
):
    """Fetch translations for a list of references."""
    
    # async def safe_translate(client, reference, to_lang):
    #     try:
    #         return await translate(client, reference, to_lang)
    #     except cohere.CohereError as e: 
    #         return "" 

    tasks = [translate(client, reference, to_lang) for reference in references]
    translations = await tqdm_asyncio.gather(*tasks)
    translation_str = []
    for translation in translations:
        try:
            txt = translation.text 
        except:
            txt = ""
        translation_str.append(txt)
    return translation_str 

# async def fetch_all_translations(
#     client: cohere.client.Client,
#     references: list[str],
#     to_lang: str,
# ):
#     """Fetch translations for a list of references."""
#     tasks = []
#     for reference in references:
#         tasks.append(translate(client, reference, to_lang))
#         translations = await tqdm_asyncio.gather(*tasks)
#     return translations


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
    for i in range(len(prediction_tokens) - 2):
        # if two consecutive tokens are identical and are in not in the original text
        # ie: are not two spaces or newlines, then return true
        if prediction_tokens[i] == prediction_tokens[i + 1] and\
            prediction_tokens[i] not in reference_tokens:
            return True
    return False


def strip_image_tag(text: str) -> str:
    """Strip image tags from text."""
    return text.replace("<image>", "")


def filter_chrf_scores(example, threshold=0.5):
    """Filter CHRF scores based on a threshold."""
    example["back_translate_chrf_int"] = [0 if score >= threshold else 1 for score in example["back_translate_chrf"]]
    return example


def flatten_conversations(example):
    # Extract and flatten conversations from the columns
    turns = [conv['value'] for conv in example['conversations']]

    # Strip <image> tags
    turns = [turn.replace('<image>', '') for turn in turns]

    # Join turns into a single string
    turns = '\n'.join(turns).strip('\n')

    return {
        'id': example['id'],
        'conversations': turns 
    }


def find_english_texts(examples):
    english_texts = defaultdict(str)
    for id, is_english, conversation in zip(examples['id'], examples['is_english'], examples['conversations']):
        if is_english:
            english_texts[id] = conversation
    return {'english_texts': [english_texts[id] for id in examples['id']]}


def add_reference_text(examples, english_texts):
    reference_texts = []
    for id, is_english in zip(examples['id'], examples['is_english']):
        if is_english:
            reference_texts.append("")
        else:
            reference_texts.append(english_texts.get(id, ""))
    return {'reference_text': reference_texts}


def make_splits(dataset, n_splits):
    n_examples = len(dataset)
    split_size = n_examples // n_splits
    splits = []
    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size
        splits.append(dataset.select(range(start, end)))
    return splits


def add_reference_text_column(dataset):
    # First pass: find English texts for each id
    english_texts_dataset = dataset.map(
        find_english_texts,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Finding English texts"
    )
    english_texts = {id: text for id, text in zip(dataset['id'], english_texts_dataset['english_texts']) if text}

    # Second pass: add reference_text column
    return dataset.map(
        add_reference_text,
        fn_kwargs={'english_texts': english_texts},
        batched=True,
        desc="Adding reference text column"
    )


def run_validation(
    dataset : Dataset,
    model_id: str = "CohereForAI/aya-23-35B",
    do_repeated_tokens: bool = True,
    do_back_translate_chrf: bool = False,
    reference_col_name: str = "",
    prediction_col_name: str  = "",
    back_translation_col: str = ""
) -> dict:
    """Run validation on a dataset."""

    if do_repeated_tokens:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    references = dataset[reference_col_name]
    predictions = dataset[prediction_col_name]
    back_translations = dataset[back_translation_col]

    repeated_tokens = []
    back_translate_chrf = []
    for reference, prediction, back_translation in tqdm(zip(references, predictions, back_translations)):
        if do_repeated_tokens:
            repeated = check_repeated_tokens(prediction, reference, tokenizer)
            repeated_tokens.append(repeated)
        if do_back_translate_chrf:
            score = compute_sentence_level_chrf([back_translation], [reference])
            score = score[0]['score']
            back_translate_chrf.append(score)
    
    results = {
        "repeated_tokens": repeated_tokens,
        "back_translate_chrf": back_translate_chrf
    }
    return results
    

async def main():
    dataset = load_dataset("kkr5155/Maya-llava-pretrain")

    # add language
    num_examples = len(dataset['train'])
    num_languages = 8
    # languages in order of file tree on hf hub
    languages = [
        "ar",
        "zh",
        "en",
        "fr",
        "hi",
        "jp",
        "ru",
        "ja"
    ]
    dataset = dataset.map(lambda example, index: {'language': languages[index // (num_examples // num_languages)]}, with_indices=True)

    dataset['train'] = dataset['train'].map(lambda example: {"is_english": example['language'] == 'en'})

    # prepare ds
    transformed_dataset = DatasetDict()
    for split in dataset.keys():
        transformed_dataset[split] = dataset[split].map(flatten_conversations)
        transformed_dataset[split] = add_reference_text_column(transformed_dataset[split])

        # No need to validate english examples -- filter out english
        transformed_dataset[split] = transformed_dataset[split].filter(lambda example: not example['language'] == 'en')  

    ## TMP --  backtranslate
    LANG_TO_BACKTRANSLATE = "ar" 

    transformed_dataset['train'] = transformed_dataset['train'].filter(lambda example: example['language']==LANG_TO_BACKTRANSLATE)

    # tests to run
    repeated_tokens = False
    back_translate_chrf = True

    # backtranslate
    if back_translate_chrf:
        if "back_translations" not in transformed_dataset['train'].column_names:
            co = cohere.AsyncClient(api_key=COHERE_API_KEY)
            # for now all backtranslations are into english, add lang id later
            LANG_VERBOSE = ISO_639_1_CODES["en"]
            # do back translation
            references = transformed_dataset['train']["reference_text"]

            size_of_chunk = 160
            reference_chunks = [references[i:i+size_of_chunk] for i in range(0, len(references), size_of_chunk)]
            all_results = []
            for chunk in reference_chunks:
                results = await fetch_all_translations(co, chunk, LANG_VERBOSE)
                all_results.extend(results) 

            transformed_dataset['train'] = transformed_dataset['train'].add_column(
                "back_translations",
                results
            )
            # Save to upload later
            transformed_dataset['train'].save_to_disk(f'validation_results_back_translate_lang_{LANG_TO_BACKTRANSLATE}')
    
    # # tmp for testing
    # # transformed_dataset['train'] = transformed_dataset['train'].select(range(50))
    
    results_train = run_validation(
        dataset=transformed_dataset['train'],
        model_id="CohereForAI/aya-23-35B",
        do_repeated_tokens=repeated_tokens,
        do_back_translate_chrf=back_translate_chrf,
        reference_col_name="reference_text",
        prediction_col_name="conversations",
        back_translation_col="back_translations"
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


### Sample run
if __name__ == "__main__":
    asyncio.run(main())