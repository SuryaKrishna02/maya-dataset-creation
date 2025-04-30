from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import pandas as pd


dataset = load_dataset(
  "kkr5155/Maya-llava-pretrain",
  data_files="english_blip_laion_cc_sbu_558k.json",  
)

model_id = "meta-llama/Meta-Llama-Guard-2-8B"
device = "cuda:0"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
llm = LLM(model=model_id, download_dir=os.environ.get('TRANSFORMERS_CACHE', ''))
sampling_params = SamplingParams(max_tokens=100)

def moderate(chats):
  pmompts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
  output = llm.generate(pmompts, sampling_params=sampling_params)
  return output


demo_msg = [
  {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
  {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
] # Output: safe

batch_size = 8
train_data = dataset['train']

unsafe = []

for i in tqdm(range(0, len(train_data), batch_size), total=len(train_data) // batch_size):
  batch = train_data[i:i+batch_size]
  # print(batch)
  msgs = []
  gpts = []
  for conv in batch['conversations']:
    gpt = conv[1]['value'].replace("<image>", "")
    msg = [
      {"role": "user", "content": conv[0]['value'].replace("<image>", "")},
      {"role": "assistant", "content": gpt},
    ]
    gpts.append(gpt)
    msgs.append(msg)

  outputs = moderate(msgs)
  # print(outputs)
  
  for output, id, gpt in zip(outputs, batch['id'], gpts):
    output = output.outputs[0].text.strip()
    # print(output, id, gpt)
    if output != "safe":
      unsafe.append({"id": id, "output": output, "gpt": gpt})
      
      if len(unsafe) % 10 == 0:
        # pd.DataFrame(unsafe).to_json('data/unsafe__english_blip_laion_cc_sbu_558k.json')
        pd.DataFrame(unsafe).to_csv('data/unsafe__english_blip_laion_cc_sbu_558k.csv')
  
  # break


pd.DataFrame(unsafe).to_csv('data/unsafe__english_blip_laion_cc_sbu_558k.csv')


