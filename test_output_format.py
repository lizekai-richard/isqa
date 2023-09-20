import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings
from huggingface_hub import snapshot_download
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel


feedback_data = load_dataset("json", data_files="/mnt/data/zekai/feedback_data.json")['train']
example = feedback_data[21]

prompt = """Below is a question paired with its context, please return the answer and \
    the most relevant evidence in the format of: (Answer: ### Evidence:). If the question is unanswerable, \
    directly return 'unanswerable' \
    ###Question: {question} \
    ###Context: {context} \
    ###Response: """.format(question=example['question'], context=example['evidence'])


tokenizer = LlamaTokenizer.from_pretrained("/mnt/data/zekai/vicuna_7b")
model = LlamaForCausalLM.from_pretrained("/mnt/data/zekai/vicuna_7b", device_map="auto", load_in_8bit=True)

lora_bin_path = os.path.join("/mnt/data/zekai/feedback_model/checkpoint-600", "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path):
    pytorch_bin_path = os.path.join("/mnt/data/zekai/feedback_model/checkpoint-600", "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
    else:
        assert ('Checkpoint is not Found!')
model = PeftModel.from_pretrained(model, "/mnt/data/zekai/feedback_model/checkpoint-600")

input_ids = tokenizer(prompt, max_length=512, padding='max_length', return_tensors='pt').input_ids.cuda()
output_ids = model.generate(input_ids=input_ids, max_new_tokens=100)
output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

print(output)