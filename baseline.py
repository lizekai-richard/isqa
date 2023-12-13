import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, T5ForConditionalGeneration, T5Tokenizer


class SciMRCDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __getitem__(self, index):
        example = self.data[index]
        input_text = example['text'][:6000]
        _id = example['id']
        summary = example['summary']
        qa_pairs = example['qa_pairs']

        # prompt = generate_prompt(instruction, input=input)
        prompt = "Please summarize the following scientific document.\n###Paper: {text}\n###Summary:".\
            format(text=input_text)
        inputs = self.tokenizer(prompt, max_length=self.max_length, padding='max_length',
                                truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]

        return {
            'id': _id,
            'input_ids': input_ids,
            'summary': summary,
            'qa_pairs': qa_pairs
        }

    def __len__(self):
        return len(self.data)


def load_model(args):
    if "t5" not in args.model_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token_id = 0

        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            load_in_8bit=True,
            device_map='auto'
        )
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)

        model = T5ForConditionalGeneration.from_pretrained(
            args.model_path,
            load_in_8bit=True,
            device_map='auto'
        )
    return tokenizer, model


@torch.inference_mode()
def inference(args, model, tokenizer, dataset):
    results_to_save = []
    for data in tqdm(dataset):
        _ids, input_ids, summaries, qa_pairs = data['id'], data['input_ids'], data['summary'], \
        data['qa_pairs']

        output_ids = model.generate(
            input_ids.cuda(),
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature
        )

        if model.config.is_encoder_decoder:
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            output = tokenizer.decode(output_ids[0, len(input_ids[0]):], skip_special_tokens=True)
        results_to_save.append({
            'id': _ids,
            'pred': output,
            'label': summaries,
            'qa_pairs': qa_pairs
        })
    return results_to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--output_path", type=str, default="/path/to/output")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data = load_dataset("json", data_files=args.data_path)['train']
    tokenizer, model = load_model(args)

    dataset = SciMRCDataset(tokenizer, data, args.max_length)
    # print(dataset[0])
    # data_loader = DataLoader(dataset, batch_size=args.batch_size)

    results = inference(args, model, tokenizer, dataset)

    with open(args.output_path, "w") as f:
        json.dump(results, f)
