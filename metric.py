import json
import re
import string
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class FactualityMetric:

    def __init__(self, model_name, prompt, save_path, device="cuda", max_length=512, max_new_tokens=100, min_new_tokens=10,
                 num_beams=2, repetition_penalty=1.3):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device)
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
        f1, precision, recall = self.token_level_f1_score(pred, answer)
        return f1, precision, recall

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

        return white_space_fix(remove_redundant_whitespace(remove_articles(remove_punc(lower(s)))))

    def token_level_f1_score(self, pred, label):
        normalized_pred, normalized_label = self.normalize_answer(pred), self.normalize_answer(label)

        if normalized_pred == "unanswerable":
            return -1, -1, -1

        prediction_tokens = normalized_pred.split()
        ground_truth_tokens = normalized_label.split()
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

    data_path = "../ExperimentResults/summaries_after_tune.json"

    data = load_dataset("json", data_files=data_path)['train']
    metric = FactualityMetric("google/flan-t5-large", prompt, "mps")

    tot_f1, tot_prec, tot_rec = 0.0, 0.0, 0.0
    for example in data:
        summary = example['summary']
        question = example['question']
        answer = example['answer']

        f1, precision, reccall = metric.compute_metrics(question, summary, answer)

        if f1 <= 0 or precision <= 0 or reccall <= 0:
            continue

        tot_f1 += f1
        tot_prec += precision
        tot_rec += reccall

    avg_f1 = tot_f1 / len(data)
    avg_prec = tot_prec / len(data)
    avg_rec = tot_rec / len(data)
    print({"f1": avg_f1, "precision": avg_prec, "recall": avg_rec})

