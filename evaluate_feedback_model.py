import json
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel


class EvalDataset(Dataset):
    def __init__(self, tokenizer, data):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data

    def __getitem__(self, index):
        example = self.data[index]
        question = example['question']
        context = example['evidence']
        answer = example['answer']
        supporting_fact = example['supporting_fact']

        prompt = """Below is a question paired with its context, please return your response in two parts:\n1. the 
                    answer to the question\n2. the most relevant evidence in the context to answer the question.\nIf the 
                    question is unanswerable, directly return 'unanswerable'.\n
                    ###Question: {question}\n
                    ###Context: {context}\n
                    ###Response: """
        inputs = self.tokenizer(prompt.format(question=question, context=context),
                                max_length=512, padding='max_length',
                                truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]

        return input_ids, answer, supporting_fact

    def __len__(self):
        return len(self.data)


def load_model(args):
    device_map = "auto"
    print(args.model_path)

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
    )
    print(args.lora_path)
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    return model


def evaluate(args, model, tokenizer, data_loader):
    saved_results = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, answers, supporting_facts = batch
            input_ids = input_ids.cuda()

            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty
            )

            output_ids = output_ids[:, len(input_ids[0]):]
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for i in range(len(outputs)):
                output = outputs[i]
                answer = answers[i]
                supporting_fact = supporting_facts[i]
                saved_results.append({
                    'prediction': output,
                    'answer': answer,
                    'supporting_fact': supporting_fact
                })

    return saved_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--output_path", type=str, default="/path/to/output")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--lora_path", type=str, default="/path/to/lora")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use_8bit", type=bool, default=True)
    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "disable"

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp = args.world_size != 1

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = load_model(args)

    data = load_dataset("json", data_files=args.data_path)['train']
    test_dataset = EvalDataset(tokenizer, data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    predictions = evaluate(args, model, tokenizer, test_loader)

    with open(args.output_path, "w") as f:
        json.dump(predictions, f)
