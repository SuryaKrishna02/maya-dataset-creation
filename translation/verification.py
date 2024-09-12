import sys
sys.path.append("../")
import cohere
from utils.constants import MODEL_NAME, COHERE_API_KEY, ISO_639_1_CODES, API_CALLS_PER_MINUTE, API_CALLS_TIME_PERIOD
from prompts import ARABIC_TRANSLATION_PROMPT, \
    HINDI_PROMPT, \
    FRENCH_TRANSLATION_PROMPT, \
    CHINESE_TRANSLATION_PROMPT, \
    JAPANESE_TRANSLATION_PROMPT, \
    RUSSIAN_TRANSLATION_PROMPT, \
    SPANISH_TRANSLATION_PROMPT
from ratelimit import sleep_and_retry, limits
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
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
import time


def compute_corpus_level_chrf(predictions, references, lowercase=False):
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=predictions, references=references, word_order=2, lowercase=lowercase)
    return results


@sleep_and_retry
@limits(calls=API_CALLS_PER_MINUTE, period=API_CALLS_TIME_PERIOD)  # 10,000 calls per 1 minute according cohere production key documentation
async def translate(
    client: cohere.AsyncClient,
    reference: str,
    preamble: str
):
    """Translate text to a target language."""
    try:
        response = await client.chat(
            preamble=preamble,
            message=reference,
            model=MODEL_NAME,
            temperature=0.3
        )
        return response
    except Exception as e:
        return e 


@sleep_and_retry
@limits(calls=API_CALLS_PER_MINUTE, period=API_CALLS_TIME_PERIOD)
async def fetch_all_translations(
    client: cohere.AsyncClient,
    references: list[str],
    preamble: str
):
    """Fetch translations for a list of references."""
    tasks = [translate(client, reference, preamble) for reference in references]
    translations = await tqdm_asyncio.gather(*tasks)
    translation_str = []
    for translation in translations:
        try:
            txt = translation.text 
        except:
            print(translation)
            txt = ""
        translation_str.append(txt)
    return translation_str 


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
    return {
        'id': example['id'],
        'conversations': example['conversations'][-1]['value']
    }


def make_splits(dataset, n_splits):
    n_examples = len(dataset)
    split_size = n_examples // n_splits
    splits = []
    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size
        splits.append(dataset.select(range(start, end)))
    return splits


def find_english_texts(examples):
    english_texts = defaultdict(str)
    for id, lang, conversation in zip(examples['id'], examples['language'], examples['conversations']):
        if lang == 'en':
            english_texts[id] = conversation
    return {'english_texts': [english_texts[id] for id in examples['id']]}


def add_reference_text(examples, english_texts):
    reference_texts = []
    for id, language in zip(examples['id'], examples['language']):
        if language == 'en':
            reference_texts.append("")
        else:
            reference_texts.append(english_texts.get(id, "id_not_found"))
    return {'reference_text': reference_texts}


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


def upload_ds_from_local(path: str, username: str, private: bool =False):
    ds = load_dataset(path)
    ds.push_to_hub(f"{username}/{path}", private=private)


def join_datasets(paths: list[str]):
    datasets = [load_dataset(path) for path in paths]
    return datasets[0].concatenate(datasets[1:])


async def do_backtranslations(
    ds: Dataset,
    co: cohere.AsyncClient,
    languages_to_backtranslate: list[str] = ["ar", "zh", "fr", "hi", "jp", "ru", "es"]
):
    all_results = [] 
    for lang in languages_to_backtranslate:
        tmp_ds = ds.filter(lambda example: example['language'] == lang)
        references = tmp_ds["conversations"]
        if lang == "ar":
            prompt = ARABIC_TRANSLATION_PROMPT
        elif lang == "zh":
            prompt = CHINESE_TRANSLATION_PROMPT
        elif lang == "fr":
            prompt = FRENCH_TRANSLATION_PROMPT
        elif lang == "hi":
            prompt = HINDI_PROMPT
        elif lang == "jp":
            prompt = JAPANESE_TRANSLATION_PROMPT
        elif lang == "ru":
            prompt = RUSSIAN_TRANSLATION_PROMPT
        elif lang == "es":
            prompt = SPANISH_TRANSLATION_PROMPT
        else:
            raise ValueError(f"Language {lang} not supported.")

        size_of_chunk = 100 
        reference_chunks = [references[i:i+size_of_chunk] for i in range(0, len(references), size_of_chunk)]
        for chunk in reference_chunks:
            results = await fetch_all_translations(co, chunk, prompt.preamble) 
            all_results.extend(results)
            time.sleep(1.0) # weird thing is that raising batch size from 16 to 100 and adding manual sleep improved throughput 

    ds = ds.add_column(
        "back_translations",
        all_results
    )
    return ds


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
    

def main():
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
        "es"
    ]
    dataset = dataset.map(lambda example, index: 
                          {'language': languages[index // (num_examples // num_languages)]},
                            with_indices=True,
                            desc="Adding language column")


    transformed_dataset = DatasetDict()
    for split in dataset.keys():
        transformed_dataset[split] = dataset[split].map(flatten_conversations, desc="Flattening conversations")

    ## Do backtranslations if needed
    if "back_translations" not in transformed_dataset['train'].column_names:
        co = cohere.AsyncClient(api_key=COHERE_API_KEY)
        transformed_dataset['train'] = asyncio.run(do_backtranslations(transformed_dataset['train'], co))
        transformed_dataset['train'].save_to_disk(f'validation_results_back_translate')

    # prepare ds
    for split in transformed_dataset.keys():
        transformed_dataset[split] = add_reference_text_column(transformed_dataset[split])

        # No need to validate english examples -- filter out english
        transformed_dataset[split] = transformed_dataset[split].filter(lambda example: not example['language'] == 'en')  

    # tests to run
    repeated_tokens = False
    back_translate_chrf = True

    # tmp for testing
    # transformed_dataset['train'] = transformed_dataset['train'].select(range(50))
    
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
    main()