import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
from torchmetrics.text.rouge import ROUGEScore
import argparse
import warnings
from datasets import load_dataset
from inference_helper import StreamPeftGenerationMixin, StreamLlamaForCausalLM
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


# fix the path for local checkpoint

def load_model(args):
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
            args.model_path,
            load_in_8bit=args.use_8bit,
            torch_dtype=torch.float16,
            device_map="auto",  # device_map={"": 0},
        )
        model = StreamPeftGenerationMixin.from_pretrained(
            model,
            args.lora_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = StreamPeftGenerationMixin.from_pretrained(
            model,
            args.lora_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        model = StreamPeftGenerationMixin.from_pretrained(
            model,
            args.lora_path,
            device_map={"": device},
        )

    if not args.use_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:{instruction}
                    ### Input:{input}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
                    ### Instruction:{instruction}"""


def evaluate(generation_config, model, input_ids):
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
    def __init__(self, tokenizer, data):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data

    def __getitem__(self, index):
        example = self.data[index]
        input = example['text'][:8000]
        _id = example['id']
        summary = example['summary']
        question = example['question']
        answer = example['answer']

        # prompt = generate_prompt(instruction, input=input)
        prompt = "###Summarize the following academic paper: {paper} \n ###Summary:".format(paper=input)
        inputs = self.tokenizer(prompt, max_length=2048, padding='max_length',
                                truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]

        return _id, input_ids, summary, question, answer
    
    def __len__(self):
        return len(self.data)


def compute_metrics(preds, labels):
    rouge = ROUGEScore()
    return rouge(preds, labels)


def predict(args, data_loader):

    model = load_model(args)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=args.max_new_tokens,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=args.min_new_tokens,  # min_length=min_new_tokens+input_sequence
    )

    outputs = []
    predictions, labels = [], []
    for batch in tqdm(data_loader):
        _ids, input_ids, summaries, questions, answers = batch
        input_ids = input_ids.cuda()
        preds = evaluate(generation_config, model, input_ids)
        for _id, pred, question, answer in zip(_ids, preds, questions, answers):
            outputs.append({
                'id': _id,
                'summary': pred,
                'question': question,
                'answer': answer
            })

        labels.extend(summaries)
        predictions.extend(preds)

    rouge_score = compute_metrics(predictions, labels)
    print(rouge_score)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--save_path", type=str, default="/path/to/save")
    parser.add_argument("--output_path", type=str, default="lora-Vicuna")
    parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_size", type=int, default=800)
    parser.add_argument("--use_8bit", type=bool, default=True)
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--use_typewriter", type=int, default=0)
    parser.add_argument("--use_local", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--min_new_tokens", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "disable"

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp = args.world_size != 1

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, padding_side='left')
    tokenizer.pad_token_id = 0  # we want pad token to be different from eos token

    data = load_dataset("json", data_files=args.data_path)['train']
    dataset = SciMRCDataset(tokenizer, data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    outputs = predict(args, data_loader)

    with open(args.save_path, 'w') as f:
        json.dump(outputs, f)
