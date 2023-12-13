import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings
from huggingface_hub import snapshot_download
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install " \
   "git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

tokenizer = LlamaTokenizer.from_pretrained("/mnt/data/zekai/vicuna_7b", add_eos_token=True)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
# tokenizer.padding_side = "left"  # Allow batched inference


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def generate_prompt_for_mrc(data_point):
    user_prompt = """Below is a question paired with its context, please return your response in two parts: \
    1. the answer to the question\n2. the most relevant evidence in the context to answer the question.\n \
    If the question is unanswerable, directly return 'unanswerable'\
    ###Question: {question} \
    ###Context: {context} \
    ###Response: """.format(question=data_point['question'], context=data_point['evidence'])

    len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=args.max_length,
                )["input_ids"]
            )
            - 1
    )  # no eos token
    if data_point['unanswerable']:
        full_tokens = tokenizer(
            user_prompt + "unanswerable",
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )["input_ids"][:-1]
    else:
        full_tokens = tokenizer(
            user_prompt + "1.Answer:{answer}\n2.Evidence:{evidence}".format(answer=data_point["answer"],
                                                                            evidence=data_point['supporting_fact']),
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""Below is an instruction that describes a task, paired with an input that provides further context. 
            Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes 
            the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )
    len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=args.max_length,
                )["input_ids"]
            )
            - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


def prepare_data(args):
    data = load_dataset("json", data_files=args.data_path)
    if args.test_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.test_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_prompt_for_mrc)
        val_data = train_val["test"].shuffle().map(generate_prompt_for_mrc)
    else:
        train_data = data["train"].shuffle().map(generate_prompt_for_mrc)
        val_data = None

    return train_data, val_data


def prepare_model(args):
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]

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

    if args.use_8bit is True:
        warnings.warn(
            "If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: "
            "pip install bitsandbytes==0.37.2"
        )
        model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()
    return model


def train(args):

    model = prepare_model(args)
    train_data, val_data = prepare_data(args)

    now_max_steps = max((len(train_data)) // args.batch_size * args.epochs, args.epochs)
    if args.resume_from_checkpoint:
        if args.lora_remote_checkpoint is not None:
            snapshot_download(repo_id=args.lora_remote_checkpoint, allow_patterns=["*.pt", "*.bin", "*.json"],
                              local_dir=args.resume_from_checkpoint)
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            pytorch_bin_path = checkpoint_name
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            if os.path.exists(checkpoint_name):
                os.rename(checkpoint_name, pytorch_bin_path)
                warnings.warn(
                    "The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
            else:
                args.resume_from_checkpoint = (
                    None  # So the trainer won't try loading its state
                )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

        train_args_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")

        if os.path.exists(train_args_path):
            import json
            base_train_args = json.load(open(train_args_path, 'r'))
            base_max_steps = base_train_args["max_steps"]
            resume_scale = base_max_steps / now_max_steps
            if base_max_steps > now_max_steps:
                warnings.warn("epoch {} replace to the base_max_steps {}".format(args.epochs, base_max_steps))
                args.max_step = base_max_steps
            else:
                args.max_step = now_max_steps
    else:
        args.max_step = now_max_steps

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=60,
            num_train_epochs=args.epochs,
            # max_steps=args.max_step,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if args.test_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_steps if args.test_size > 0 else None,
            save_steps=args.save_steps,
            output_dir=args.output_path,
            save_total_limit=30,
            load_best_model_at_end=True if args.test_size > 0 else False,
            ddp_find_unused_parameters=False if args.ddp else None,
            report_to="wandb" if args.wandb else [],
            ignore_data_skip=args.ignore_data_skip
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[PeftSavingCallback]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

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
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=int, default=0.5)
    parser.add_argument("--use_8bit", type=bool, default=True)
    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "disable"

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp = args.world_size != 1

    train(args)
