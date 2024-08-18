from dataclasses import dataclass
from utils.constants import MODEL_NAME, COHERE_API_KEY
from nltk.translate.bleu_score import sentence_bleu
import ast
import argparse
import cohere
import csv

@dataclass
class Prompt:
    translate_to: str
    preamble: str
    message: str 


def check_prompt(prompt):
    if not isinstance(prompt, Prompt):
        return "The input is not a valid Prompt object. Please provide a valid Prompt object."
    
    if not isinstance(prompt.translate_to, str):
        return "'translate_to' should be of type 'str'. Please modify the prompt."

    if not isinstance(prompt.preamble, str) or not isinstance(prompt.message, str):
        return "Both 'preamble' and 'message' should be of type 'str'. Please modify the prompt."

    if not prompt.translate_to.strip():
        return "'translate_to' should not be empty. Please provide valid content."

    if not prompt.preamble.strip() or not prompt.message.strip():
        return "Both 'preamble' and 'message' should not be empty. Please provide valid content."

    return None



def build_subset(csv_file: str, lang: str)-> list:
  pair_list =[]
  with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
      if row['English Sentence'] != 'English Sentence':
        pair = [row['English Sentence'], row[f"{lang} Translation"]]
        pair_list.append(pair)
  return pair_list

def fetch_translation_by_aya(input_text: str, prompt: Prompt) -> str:
  co = cohere.Client(api_key=COHERE_API_KEY)
  message = f"Input: {input_text}"
  preamble = prompt.preamble
  response = co.chat(
    model = MODEL_NAME,
    message=message,
    preamble = preamble
  )
  return response.text

def calculate_bleu(ground_truth: str, hypothesis: str):
    ground_truth_tokens = ground_truth.split()
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu([ground_truth_tokens], hypothesis_tokens)

def evaluate(lang: str, pair_list: list, prompt: Prompt) -> list:
  eval_list = [['English', 'Ground_truth', 'Aya_output', 'Bleu_score']]
  for pair in pair_list:
    english_sentence = pair[0]
    ground_truth = pair[1]
    hypothesis = fetch_translation_by_aya(english_sentence, prompt)
    bleu_score = calculate_bleu(ground_truth, hypothesis)
    eval_list.append([english_sentence, ground_truth, hypothesis, bleu_score])
  print(f"evaluation_done for {lang}")
  return eval_list



def create_prompt(strings: list[str]) -> Prompt:
    """
    Processes raw input prompts and creates instances of the `Prompt` class.
    """
    if len(strings) != 3:
        raise ValueError("List must contain exactly three elements: [translate_to, preamble, message]")
    
    translate_to, preamble, message = strings
    return Prompt(translate_to=translate_to, preamble=preamble, message=message)


def process_prompts(base_path: str, eval_data: str, prompt_list: list[list[str]]):
  """
  Processes the specified prompts, checks their structure, and runs the report generation.

  Parameters:
  base_path: Path of directory to store reports
  eval_data: path to evaluation dataset
  prompt_list: List of raw prompts to be processed. Raw prompts are lists of size 3
  containing the translate_to, premable and message values. 
  Returns:
  None
  """
  for (i, raw_prompt) in enumerate(prompt_list):
      prompt = create_prompt(raw_prompt)
      lang = (prompt.translate_to).lower().capitalize()
      validation_error = check_prompt(prompt)

      if validation_error:
          print(f"Validation failed for '{lang} prompt': {validation_error}")
      
      pair_list = build_subset(eval_data, lang)
      print(f"Generating report for 'prompt_{i} {lang} '...")
      eval_list = evaluate(lang, pair_list, prompt)
      for row in eval_list:
        if len(row) != 4:
            raise ValueError("Each row must contain exactly 4 elements.")

      file_name = f"{base_path}prompt_{i}_{lang}_eval_report.csv"
      with open(file_name, 'w', newline='') as file:
          writer = csv.writer(file)
          writer.writerows(eval_list)

def main():
  def extract_list_of_lists(file_path):
      """
      Parameters:
      file_path - Path to .txt file that has list of raw prompts to be processed.
      Eg: [['Hindi', 'Translate', 'Input'], ['Spanish', 'Translate to Spanish', 'Input Text']]
      """
    with open(file_path, 'r') as file:
        # Read the content of the file
        content = file.read()
        
        # Safely evaluate the content to a Python object (list of lists)
        list_of_lists = ast.literal_eval(content)
        return list_of_lists
  parser = argparse.ArgumentParser(description="Generate evaluation reports for different set of language prompts")
  parser.add_argument('base_path', type=str, help="Directory path to save the generated reports.")
  parser.add_argument('eval_csv', type=str, help="Path to the evaluation dataset CSV file containing english sentences and their translations in other languages.")
  parser.add_argument('prompt_file', type = str, help="List of prompts to be processed")

  args = parser.parse_args()
  base_path = args.base_path
  input_csv = args.eval_csv
  selected_prompt_list = extract_list_of_lists(args.prompt_file)
  
  process_prompts(base_path, input_csv, selected_prompt_list)

if __name__ == "__main__":
    main()
