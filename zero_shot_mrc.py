"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import string
import re
from collections import Counter
from tqdm import tqdm
import torch
from torchmetrics.text.rouge import ROUGEScore
from datasets import load_dataset

from fastchat.model import load_model, get_conversation_template, add_model_args


def save_preds(pred_answers):
    with open("eval_ans.txt", "w", encoding='utf-8') as f:
        for ans in pred_answers:
            f.write(ans + "\n")


def normalize_answer(s):
    def remove_redundant_whitespace(text):
        return text.strip()

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ''.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_redundant_whitespace(remove_articles(remove_punc(lower(s)))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_metrics(ground_truth, predicted_answer):
    # f1 = exact_match = total = 0
    # for t, p in zip(ground_truth, predicted_answer):
    #     total += 1
    #     cur_EM = exact_match_score(p, t)
    #     cur_f1, _, _ = f1_score(p, t)
    #     exact_match += cur_EM
    #     f1 += cur_f1

    # exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total

    # return {'exact_match': exact_match, 'f1': f1}
    rouge = ROUGEScore()
    scores = rouge(predicted_answer, ground_truth)
    return {
        'rouge1': scores['rouge1_fmeasure'],
        'rouge2': scores['rouge2_fmeasure'],
        'rougeL': scores['rougeL_fmeasure']
    }


@torch.inference_mode()
def inference(args, model, tokenizer, prompt):

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--prompt", type=str, default="Answer the question: {qn}\n Based on the following document: {doc}\n")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    dataset = load_dataset("json", data_files=args.dataset)['train']

    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    ref_answers, pred_answers = [], []
    for i in tqdm(range(args.test_size)):
        example = dataset[i]
        question = example['question']
        paper = example['text'][:10000]
        answer = example['answer']
        msg = args.prompt.format(doc=paper, qn=question)

        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        pred = inference(args, model, tokenizer, prompt)

        ref_answers.append(answer)
        pred_answers.append(pred)

    metrics = compute_metrics(ref_answers, pred_answers)
    print(metrics)
    save_preds(pred_answers)
    