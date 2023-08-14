import sys
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import argparse
import warnings
from datasets import load_dataset
import os
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--lora_path", type=str)
parser.add_argument("--use_typewriter", type=int, default=0)
parser.add_argument("--use_local", type=int, default=1)
parser.add_argument("--data_path", type=str, default="sample/instruct/scimrc_for_inst_tune.json")
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--num_beams", type=int, default=1)
args = parser.parse_args()
print(args)


tokenizer = LlamaTokenizer.from_pretrained(args.model_path, padding_side='left')
tokenizer.pad_token_id = 0  # we want pad token to be different from eos token

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path
BATCH_SIZE = 4

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and args.use_local:
    pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
    else:
        assert ('Checkpoint is not Found!')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto", #device_map={"": 0},
    )
    model = StreamPeftGenerationMixin.from_pretrained(
        model, LORA_WEIGHTS, torch_dtype=torch.float16,
        device_map="auto", #device_map={"": 0}
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = StreamPeftGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = StreamPeftGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:{instruction}
                    ### Input:{input}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
                    ### Instruction:{instruction}"""


if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(generation_config, input_ids):
    # prompt = generate_prompt(instruction, input=input)
    # inputs = tokenizer(prompt, max_length=2048, truncation=True, return_tensors="pt")
    # input_ids = inputs["input_ids"].to(device)
    # generation_config = GenerationConfig(
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     num_beams=num_beams,
    #     bos_token_id=1,
    #     eos_token_id=2,
    #     pad_token_id=0,
    #     max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
    #     min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
    #     **kwargs,
    # )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            output_scores=False,
            repetition_penalty=1.3,
        )
        # print(generation_output.size())
        output = generation_output[:, len(input_ids[0]):]
        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output


class SciMRCDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __getitem__(self, index):
        example = self.data[index]
        instruction = example['instruction']
        input = example['input']
        _id = example['id']

        prompt = generate_prompt(instruction, input=input)
        inputs = tokenizer(prompt, max_length=self.max_length, padding='max_length', 
                           truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]

        return _id, input_ids
    
    def __len__(self):
        return len(self.data)


data = load_dataset("json", data_files=args.data_path)['train']
dataset = SciMRCDataset(tokenizer, data.select(range(2000)), 2048)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def predict(temperature=0.1, top_p=0.75, top_k=40, num_beams=2, max_new_tokens=128, min_new_tokens=1,
            repetition_penalty=2.0, **kwargs):

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
    )

    outputs = []
    for _ids, input_ids in tqdm(data_loader):
        input_ids = input_ids.to(device)
        preds = evaluate(generation_config, input_ids)

        for _id, pred in zip(_ids, preds):
            outputs.append({
                'id': _id,
                'summary': pred
            })

    return outputs


outputs = predict(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)

with open('summaries_after_tune.json', 'w') as f:
    json.dump(outputs, f)
