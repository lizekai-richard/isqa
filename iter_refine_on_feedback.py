import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import re
import string
import random
import torch
import spacy
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel, prepare_model_for_int8_training
import argparse

# nlp = spacy.load("/home/users/nus/e0817846/.conda/envs/vicuna/lib/python3.10/site-packages/en_core_web_sm/
# en_core_web_sm-3.7.0")
nlp = spacy.load("en_core_web_sm")


def load_base_model(args):
    if "vicuna" in args.base_model_path or "llama" in args.base_model_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model_path)

        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"

        model = LlamaForCausalLM.from_pretrained(
            args.base_model_path,
            load_in_8bit=args.use_8bit,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        model = prepare_model_for_int8_training(model)
    elif "t5" in args.base_model_path:
        tokenizer = T5Tokenizer.from_pretrained(args.base_model_path)

        model = T5ForConditionalGeneration.from_pretrained(
            args.base_model_path,
            load_in_8bit=args.use_8bit,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = prepare_model_for_int8_training(model)
    return tokenizer, model


def load_feedback_model(args):
    feedback_tokenizer = LlamaTokenizer.from_pretrained(args.feedback_model_path)
    feedback_model = LlamaForCausalLM.from_pretrained(
        args.feedback_model_path,
        load_in_8bit=args.use_8bit,
        device_map='auto'
    )
    feedback_model = PeftModel.from_pretrained(
        feedback_model,
        args.lora_path,
        device_map="auto"
    )
    feedback_model = prepare_model_for_int8_training(feedback_model)
    return feedback_tokenizer, feedback_model


def normalize_answer(s):
    def remove_redundant_whitespace(text):
        return text.strip()

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_special_tokens(text):
        return re.sub(r'\\u25b6\\ufe0f', '', text)

    return white_space_fix(remove_redundant_whitespace(remove_articles(remove_punc(remove_special_tokens(lower(s))))))


def token_level_f1_score(pred, label):
    normalized_pred, normalized_label = normalize_answer(pred), normalize_answer(label)

    prediction_tokens = normalized_pred.split()
    ground_truth_tokens = normalized_label.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def generate_prompt_for_feedback_model(summary, question):
    prompt = """Below is a question paired with its context, please return your response in two parts:\n1. the answer 
                to the question\n2. the most relevant evidence in the context to answer the question.\nIf the question 
                is unanswerable, directly return 'unanswerable'.\n
                ###Question: {question}\n
                ###Context: {context}\n
                ###Response: """.format(question=question, context=summary)
    return prompt


@torch.inference_mode()
def generate_feedback(args, model, tokenizer, summary, question, answer):
    prompt = generate_prompt_for_feedback_model(summary, question)
    input_ids = tokenizer(
        prompt,
        max_length=args.feedback_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids.cuda()

    output_ids = model.generate(
        input_ids=input_ids,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.fed_max_new_tokens,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=args.fed_min_new_tokens,  # min_length=min_new_tokens+input_sequence
    )

    output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)
    # print("Answer+Evidence:", output)
    if ("unanswerable" or "Unanswerable") in output:
        return None, 0

    ans_index, sp_index = -1, -1
    ans_prefix, sp_prefix = None, None
    if "Answer:" in output or "answer:" in output or "1." in output:
        ans_index = output.find("Answer:")
        ans_prefix = "Answer:"
        if ans_index == -1:
            ans_index = output.find("answer:")
            ans_prefix = "answer:"
        if ans_index == -1:
            ans_index = output.find("1.")
            ans_prefix = "1."
    if "Evidence:" in output or "evidence:" in output or "2." in output:
        sp_index = output.find("Evidence:")
        sp_prefix = "Evidence:"
        if sp_index == -1:
            sp_index = output.find("evidence:")
            sp_prefix = "evidence:"
        if sp_index == -1:
            sp_index = output.find("2.")
            sp_prefix = "2."

    if ans_index == -1 or sp_index == -1:
        return None, 0
    feedback_ans = output[ans_index + len(ans_prefix): sp_index]
    feedback_sp = output[sp_index + len(sp_prefix):]
    return feedback_sp, token_level_f1_score(feedback_ans, answer)


def feedback_step(args, tokenizer, feedback_model, summary, question, answer):
    feedback_dict = {}
    feedback_signal, score = generate_feedback(args, feedback_model, tokenizer, summary, question, answer)
    # print("Feedback: ", feedback_signal)
    if feedback_signal is None:
        return None

    feedback_dict['score'] = score
    if score >= args.threshold:
        feedback_dict['fact'] = feedback_signal
    else:
        feedback_dict['non_fact'] = feedback_signal

    return feedback_dict


@torch.inference_mode()
def refine_step(args, tokenizer, base_model, text, feedback):
    prompt = """
        Below is a scientific paper. Please write a summary based on facts and non-facts we provide.\n###Paper: {text}\n###Facts: {facts}\n###Non-Facts: {non_facts}\n###Summary:
    """
    facts = ""
    for i, fact in enumerate(feedback['facts']):
        facts += "{num}. {fact}\n".format(num=i, fact=fact)

    non_facts = ""
    for i, non_fact in enumerate(feedback['non_facts']):
        non_facts += "{num}. {non_fact}\n".format(num=i, non_fact=non_fact)

    prompt = prompt.format(text=text, facts=facts, non_facts=non_facts)
    input_ids = tokenizer(
        prompt,
        max_length=args.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids.cuda()

    output_ids = base_model.generate(
        input_ids,
        num_beams=args.num_beams,
        max_new_tokens=args.gen_max_new_tokens,
        min_new_tokens=args.gen_min_new_tokens,
        temperature=args.temperature
    )

    if base_model.config.is_encoder_decoder:
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
    else:
        output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
    return output


def create_mini_batch(indices, mini_batch_size):
    if mini_batch_size <= len(indices):
        return random.sample(indices, mini_batch_size)
    else:
        return indices


def similarity_filter(new_feedback, old_feedbacks):
    p = nlp(new_feedback)
    for of in old_feedbacks:
        q = nlp(of)
        similarity = p.similarity(q)
        if similarity > 0.8:
            return False
    return True


@torch.inference_mode()
def batched_correction_stage(args, base_model, base_tokenizer, feedback_model, feedback_tokenizer, dataset):
    results_to_save = {}
    avg_f1_score = 0.0
    tot_cnt = 0
    with torch.no_grad():
        for example in tqdm(dataset):
            _id = example['id']
            text = example['text'][:6000]
            qa_pairs = example['qa_pairs']
            gold_summary = example['summary']

            results_to_save[_id] = []

            initial_prompt = """
                Please summarize the following scientific document.\n###Paper: {text}\n###Summary:
            """.format(text=text)

            input_ids = base_tokenizer(
                initial_prompt,
                max_length=args.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).input_ids.cuda()

            output_ids = base_model.generate(
                input_ids,
                num_beams=args.num_beams,
                max_new_tokens=args.gen_max_new_tokens,
                min_new_tokens=args.gen_min_new_tokens,
                temperature=args.temperature
            )

            if not base_model.config.is_encoder_decoder:
                pred_summary = base_tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
            else:
                pred_summary = base_tokenizer.decode(output_ids[0], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)

            feedback = {'facts': [], 'non_facts': []}
            indices = list(range(len(qa_pairs)))
            best_score = 0
            for i in range(args.num_correction_steps):
                curr_batch = create_mini_batch(indices, args.correction_batch_size)
                batch_scores = 0
                batch_cnt = 0
                for j in curr_batch:
                    question, answer = qa_pairs[j][0], qa_pairs[j][1]
                    feedback_dict = feedback_step(args, feedback_tokenizer, feedback_model, pred_summary, question,
                                                  answer)
                    if feedback_dict is None or feedback_dict['score'] == 0:
                        continue
                    # max_score_for_cur_example = max(max_score_for_cur_example, feedback_dict['score'])

                    if "fact" in feedback_dict:
                        if similarity_filter(feedback_dict['fact'], feedback['facts']):
                            feedback['facts'].append(feedback_dict['fact'])
                    elif "non_fact" in feedback_dict:
                        if similarity_filter(feedback_dict['non_fact'], feedback['non_facts']):
                            feedback['non_facts'].append(feedback_dict['non_fact'])
                    batch_scores += feedback_dict['score']
                    batch_cnt += 1

                if batch_cnt > 0:
                    batch_avg_score = batch_scores / batch_cnt
                    best_score = max(best_score, batch_avg_score)
                    results_to_save[_id].append({
                        'output': pred_summary,
                        'feedback': feedback,
                        'f1-score': batch_avg_score
                    })
                    pred_summary = refine_step(args, base_tokenizer, base_model, text, feedback)

            if best_score > 0:
                avg_f1_score += best_score
                tot_cnt += 1
    print("Average F1 score after self-correction is: ", avg_f1_score / tot_cnt)
    return results_to_save


@torch.inference_mode()
def correction_stage(args, base_model, tokenizer, feedback_model, dataset):
    results_to_save = {}
    avg_f1_score = 0.0
    tot_cnt = 0
    with torch.no_grad():
        for example in tqdm(dataset):
            _id = example['id']
            text = example['text'][:6000]
            qa_pairs = example['qa_pairs']
            gold_summary = example['summary']

            results_to_save[_id] = []

            initial_prompt = """
                Please summarize the following scientific document.\n###Paper: {text}\n###Summary:
            """.format(text=text)

            input_ids = tokenizer(
                initial_prompt,
                max_length=args.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).input_ids.cuda()

            output_ids = base_model.generate(
                input_ids,
                num_beams=args.num_beams,
                max_new_tokens=args.gen_max_new_tokens,
                min_new_tokens=args.gen_min_new_tokens,
                temperature=args.temperature
            )

            if base_model.config.is_encoder_decoder:
                pred_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                pred_summary = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            feedback = {'facts': [], 'non_facts': []}
            prev_score = 0
            max_score_for_cur_example = 0.0
            for question, answer in qa_pairs:
                # print("Summary: ", pred_summary)
                # print("Score: ", avg_score_for_cur_example)
                feedback_dict = feedback_step(args, tokenizer, feedback_model, pred_summary, question, answer)
                if feedback_dict is None:
                    continue
                max_score_for_cur_example = max(max_score_for_cur_example, feedback_dict['score'])

                if "fact" in feedback_dict:
                    feedback['facts'].append(feedback_dict['fact'])
                    results_to_save[_id].append({
                        'output': pred_summary,
                        'fact': feedback_dict['fact'],
                        'f1-score': feedback_dict['score']
                    })
                elif "non_fact" in feedback_dict:
                    feedback['non_facts'].append(feedback_dict['non_fact'])
                    results_to_save[_id].append({
                        'output': pred_summary,
                        'non_fact': feedback_dict['non_fact'],
                        'f1-score': feedback_dict['score']
                    })

                pred_summary = refine_step(args, tokenizer, base_model, text, feedback)

            if max_score_for_cur_example > 0:
                avg_f1_score += max_score_for_cur_example
                tot_cnt += 1

    print("Average F1 score after self-correction is: ", avg_f1_score / tot_cnt)
    return results_to_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--save_path", type=str, default="/path/to/save")
    parser.add_argument("--base_model_path", type=str, default="/path/to/base/model")
    parser.add_argument("--feedback_model_path", type=str, default="/path/to/feedback/model")
    parser.add_argument("--lora_path", type=str, default="/path/to/adapter")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
    parser.add_argument("--ignore_data_skip", type=str, default="False")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--feedback_max_length", type=int, default=512)
    parser.add_argument("--fed_min_new_tokens", type=int, default=1)
    parser.add_argument("--fed_max_new_tokens", type=int, default=100)
    parser.add_argument("--gen_min_new_tokens", type=int, default=100)
    parser.add_argument("--gen_max_new_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--temperature", type=int, default=1.3)
    parser.add_argument("--use_8bit", type=bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--patience", type=float, default=0.01)
    parser.add_argument("--num_correction_steps", type=int, default=10)
    parser.add_argument("--correction_batch_size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base_tokenizer, base_model = load_base_model(args)
    feedback_tokenizer, feedback_model = load_feedback_model(args)

    dataset = load_dataset("json", data_files=args.data_path)['train']
    # results_to_save = correction_stage(args, base_model, tokenizer, feedback_model, dataset.select(range(200, 300)))
    results_to_save = batched_correction_stage(args, base_model, base_tokenizer, feedback_model, feedback_tokenizer,
                                               dataset.select(range(2)))

    with open(args.save_path, "w") as f:
        json.dump(results_to_save, f)
