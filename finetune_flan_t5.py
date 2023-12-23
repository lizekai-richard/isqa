import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import prepare_model_for_int8_training

tokenizer = T5Tokenizer.from_pretrained("/mnt/data/zekai/flan-t5-large")


def generate_prompt_for_data(data_point):
    user_prompt = """"Below is a question about an article. Please return both the answer and the evidence sentence, 
separated by <sep>:\n{context}\n{question}""".format(question=data_point['question'], context=data_point['context'])

    response = f"""{data_point['answer']} <sep> {data_point['evidence']}"""

    tokenized_prompt = tokenizer(
        user_prompt,
        truncation=True,
        max_length=1024,
        padding="max_length",
    )

    tokenized_label = tokenizer(
        response,
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    return {
        "input_ids": tokenized_prompt['input_ids'],
        "attention_mask": tokenized_prompt['attention_mask'],
        "labels": tokenized_label['input_ids']
    }


def prepare_data(args):
    data = load_dataset("json", data_files=args.data_path)
    train_data = data["train"].shuffle().map(generate_prompt_for_data)
    val_data = data["test"].map(generate_prompt_for_data)

    return train_data, val_data


def train(args):

    model = T5ForConditionalGeneration.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        torch_type=torch.float16,
        device_map="auto"
    )
    model = prepare_model_for_int8_training(model)

    train_data, val_data = prepare_data(args)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=100,
            evaluation_strategy="steps" if args.test_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_steps if args.test_size > 0 else None,
            save_steps=args.save_steps,
            output_dir=args.output_path,
            save_total_limit=10,
            load_best_model_at_end=True if args.test_size > 0 else False,
            ddp_find_unused_parameters=False if args.ddp else None,
            report_to="wandb" if args.wandb else [],
            ignore_data_skip=args.ignore_data_skip
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("\n If there's a warning about missing keys above, please disregard :)")

    trainer.train()
    model.save_pretrained(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--output_path", type=str, default="/path/to/output")
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "disable"

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp = args.world_size != 1

    train(args)
