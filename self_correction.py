import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
import argparse
from inference_helper import StreamPeftGenerationMixin


class SciMRCDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __getitem__(self, index):
        example = self.data[index]
        input = example['text'][:8000]
        _id = example['id']
        summary = example['summary']
        question = example['question']
        answer = example['answer']

        # prompt = generate_prompt(instruction, input=input)
        prompt = "###Summarize the following academic paper: {paper} \n ###Summary:".format(paper=input)
        inputs = self.tokenizer(prompt, max_length=self.max_length, padding='max_length',
                                truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]

        return _id, input_ids, summary, question, answer

    def __len__(self):
        return len(self.data)


def load_base_model(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=args.use_8bit,
        device_map="auto",
    )
    return tokenizer, model


def load_feedback_model(args):
    base_model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=args.use_8bit,
        device_map='auto'
    )
    model = StreamPeftGenerationMixin.from_pretrained(
        base_model,
        args.lora_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model


def generate_prompt_for_feedback_model(summary, question):
    prompt =  """Below is a question paired with its context, please return the answer and \
    the most relevant evidence in the format of: (Answer: ### Evidence:). If the question is unanswerable, \
    directly return 'unanswerable' \
    ###Question: {question} \
    ###Context: {context} \
    ###Response: """.format(question=question, context=summary)
    return prompt


def generate_feedback(args, model, tokenizer, summary, question, answer):
    prompt = generate_prompt_for_feedback_model(summary, question)
    input_ids = tokenizer(
        prompt,
        max_length=args.feedback_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids

    output_ids = model.generate(
        input_ids,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=args.min_new_tokens,  # min_length=min_new_tokens+input_sequence
    )

    output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    if "unanswerable" in output:
        return None

    answer, evidence = "", ""
    if "###" in output:
        answer, evidence = output.split("###")

    else:
        pass


def refine_step(step_number, base_model, feedback_model, summary, questions, answers):
    feedback_signals = generate_feedback(feedback_model, summary, questions, answers)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--output_path", type=str, default="/path/to/output")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--lora_path", type=str, default="/path/to/adapter")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--feedback_max_length", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--use_8bit", type=bool, default=True)
    args = parser.parse_args()

