import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import re
import string
from utils.preprocessing import process_example
from collections import Counter
from tqdm import tqdm
import torch
from torchmetrics.text.rouge import ROUGEScore
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class FactualityMetric:

    def __init__(self, model_name, metrics_name, prompt, save_path, device="cuda:0", max_length=512, max_new_tokens=100, min_new_tokens=1,
                 num_beams=2, repetition_penalty=1.3):
        
        self.metrics_name = metrics_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        self.save_path = save_path
        self.device = torch.device(device)
        self.prompt = prompt
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.predictions = []

    def compute_metrics(self, question, summary, answer):

        input_text = self.prompt.format(context=summary, question=question)
        input_ids = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            num_beams=self.num_beams
        )

        if self.model.config.is_encoder_decoder:
            pred = self.tokenizer.decode(output_ids[0], skip_sepcial_tokens=True)
        else:
            pred = self.tokenizer.decode(output_ids[0][len(input_ids):], skip_sepcial_tokens=True)

        self.predictions.append(pred)
        metrics = {}
        if self.metrics_name == 'f1':
            f1, precision, recall = self.token_level_f1_score(pred, answer)
            metrics['f1'] = f1
            metrics['precision'] = precision
            metrics['recall'] = recall
        elif self.metrics_name == 'rouge':
            rouge = ROUGEScore()
            _pred = self.normalize_answer(pred)
            _answer = self.normalize_answer(answer)

            if "unanswerable" in _pred:
                metrics['rouge1'] = -1
                metrics['rouge2'] = -1
                metrics['rougeL'] = -1
            else:
                score = rouge(_pred, _answer)
                metrics['rouge1'] = score['rouge1_fmeasure']
                metrics['rouge2'] = score['rouge2_fmeasure']
                metrics['rougeL'] = score['rougeL_fmeasure']
        return metrics

    def normalize_answer(self, s):
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
            return re.sub(r'<pad>|<\\s>', '', text)

        return white_space_fix(remove_redundant_whitespace(remove_articles(remove_punc(remove_special_tokens(lower(s))))))

    def token_level_f1_score(self, pred, label):
        # normalized_pred, normalized_label = self.normalize_answer(pred), self.normalize_answer(label)
        # if normalized_pred == "unanswerable":
        #     return -1, -1, -1
        # prediction_tokens = normalized_pred.split()
        # ground_truth_tokens = normalized_label.split()

        prediction_tokens = process_example(pred)
        ground_truth_tokens = process_example(label)

        if "unanswerable" in prediction_tokens:
            return -1, -1, -1
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def save_predictions(self):
        with open(self.save_path, "w") as f:
            json.dump(self.predictions, f)


if __name__ == '__main__':

    prompt = """
        Read the context: {context} \n Answer the question: {question}. 
        If the question is unanswerable, return "unanswerable" \n Answer:
    """

    data_path = "/mnt/data/zekai/summaries_after_tune_1200.json"

    data = load_dataset("json", data_files=data_path)['train']
    metric = FactualityMetric("google/flan-t5-large", "rouge", prompt, "qa_metric_result_1200.json",
                              device="cuda:0")

    tot_f1, tot_prec, tot_rec = 0.0, 0.0, 0.0
    tot_rouge1, tot_rouge2, tot_rougeL = 0.0, 0.0, 0.0
    cnt = 0
    for example in tqdm(data):
        summary = example['summary']
        summary = re.sub(r"\n|\*", "", summary)
        question = example['question']
        answer = example['answer']

        metrics = metric.compute_metrics(question, summary, answer)

        if "f1" in metrics.keys():
            f1, precision, recall = metrics['f1'], metrics['precision'], metrics['recall']
            if f1 <= 0 or precision <= 0 or recall <= 0: continue
            tot_f1 += f1
            tot_prec += precision
            tot_rec += recall

        elif "rouge1" in metrics.keys():
            rouge1, rouge2, rougeL = metrics['rouge1'], metrics['rouge2'], metrics['rougeL']
            if rouge1 <= 0 or rouge2 <= 0 or rougeL <= 0:
                continue
            tot_rouge1 += rouge1
            tot_rouge2 += rouge2
            tot_rougeL += rougeL
            
        cnt += 1

    if metric.metrics_name == 'f1':
        avg_f1 = tot_f1 / cnt
        avg_prec = tot_prec / cnt
        avg_rec = tot_rec / cnt
        print({"f1": avg_f1, "precision": avg_prec, "recall": avg_rec})
    elif metric.metrics_name == 'rouge':
        avg_rouge1 = tot_rouge1 / cnt
        avg_rouge2 = tot_rouge2 / cnt
        avg_rougeL = tot_rougeL / cnt
        print({'rouge1': avg_rouge1, 'rouge2': avg_rouge2, 'rougeL': avg_rougeL})

    # metric.save_predictions()

