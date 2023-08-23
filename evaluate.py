import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import transformers
import argparse
import warnings
from peft import PeftModel
from torchmetrics.text.rouge import ROUGEScore

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install " \
   "git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training


class SciMRCDataset(Dataset):
    def __init__(self, tokenizer, data):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data

    def __getitem__(self, index):
        example = self.data[index]
        instruction = example['question'] + " Reply N.A. if the question is unanswerable."
        input = example['input']
        _id = example['id']
        answer = example['answer']

        # prompt = generate_prompt(instruction, input=input)
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. 
            Write a response that appropriately completes the request.
            ### Instruction:{instruction}
            ### Input:{input}
            ### Response:
        """
        inputs = self.tokenizer(prompt, max_length=2048, padding='max_length',
                                truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]

        return _id, input_ids, answer

    def __len__(self):
        return len(self.data)


def prepare_model(args):

    device_map = "auto"
    if args.ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // args.world_size
    print(args.model_path)

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(model, args.lora_path)

    if args.use_8bit is True:
        warnings.warn(
            "If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: "
            "pip install bitsandbytes==0.37.2"
        )
        model = prepare_model_for_int8_training(model)

    model.eval()
    return model


def compute_metrics(preds, labels):
    rouge = ROUGEScore()
    return rouge(preds, labels)


def evaluate(args, model, data_loader):

    labels, predictions = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            _ids, input_ids, answers = batch
            input_ids = input_ids.cuda()

            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repitition_penalty
            )

            output_ids = output_ids[:, len(input_ids[0]):]
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            labels.extend(answers)
            predictions.extend(output)

    rouge_score = compute_metrics(predictions, labels)
    print(rouge_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--lora_path", type=str, default="/path/to/lora")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--min_new_tokens", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "disable"

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp = args.world_size != 1

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = prepare_model(args)

    data = load_dataset("json", data_files=args.data_path)
    data = data.select(range(len(data))[-4000:])
    test_dataset = SciMRCDataset(tokenizer, data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    evaluate(args, model, test_loader)

