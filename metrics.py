import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import random
import re
import string
import torch
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from transformers import pipeline
from utils.generate_qa_pairs import generate_qa


class FactualityMetric:

    def __init__(self, args):
        
        # self.metrics_name = args.metrics_name
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, device_map="auto")
        self.nlp_qa = pipeline(
            task="question-answering",
            model=args.model_path,
            tokenizer=args.model_path,
            trust_remote_code=True,
            device_map="auto"
            # torch_dtype=torch.bfloat16
        )
        self.save_path = args.save_path
        # self.device = torch.device(args.device)
        self.prompt = args.prompt
        self.max_length = args.max_length
        self.max_new_tokens = args.max_new_tokens
        self.min_new_tokens = args.min_new_tokens
        self.num_beams = args.num_beams
        self.repetition_penalty = args.repetition_penalty
        self.n_questions = args.n_questions
        self.qg_model_path = args.qg_model_path
        self.saved_results = {}

    def compute_metrics(self, data, predictions):

        avg_scores = 0
        for pred in tqdm(predictions):
            _id = pred['id']
            pred_summary = pred['pred']

            # qa_pairs_from_ds = pred["qa_pairs"]
            qa_pairs = generate_qa(pred_summary, n_qa_pairs=self.n_questions, qg_model_path=self.qg_model_path)
            if len(qa_pairs) == 0:
                continue
            # qa_pairs.extend(qa_pairs_from_ds)
            random.shuffle(qa_pairs)

            context = ""
            for d in data:
                if d['id'] == _id:
                    context = d['text'][:5000]
            assert context != ""

            avg_score = 0
            for question, answer in qa_pairs:
                metric = self.compute_metrics_step(question, context, answer)
                avg_score += metric['f1']
            avg_score /= len(qa_pairs)

            avg_scores += avg_score
        avg_scores /= len(predictions)
        return avg_scores
    
    def compute_metrics_from_refine(self, data, predictions):

        tot_scores = 0.0
        cnt = 0
        for key in tqdm(predictions):
            self.saved_results[key] = []
            prediction = predictions[key]
            context = ""
            for d in data:
                if d['id'] == key:
                    context = d['text'][:5000]
            assert context != ""

            cur_fact_score = 0
            for pred in prediction:
                summary = pred['output']

                qa_pairs = generate_qa(summary, n_qa_pairs=self.n_questions, qg_model_path=self.qg_model_path)
                if len(qa_pairs) == 0:
                    continue
                random.shuffle(qa_pairs)

                avg_score = 0.0
                for question, answer in qa_pairs:
                    metric = self.compute_metrics_step(question, context, answer)
                    avg_score += metric['f1']
                avg_score /= len(qa_pairs)

                self.saved_results[key].append(avg_score)
                cur_fact_score = max(cur_fact_score, avg_score)
            
            if cur_fact_score > 0:
                tot_scores += cur_fact_score
                cnt += 1
        tot_avg_score  = tot_scores / cnt
        self.save_metrics()
        return tot_avg_score

    @torch.inference_mode()
    def compute_metrics_step(self, question, context, answer):

        # input_text = self.prompt.format(context=context, question=question)
        # input_ids = self.tokenizer(
        #     input_text,
        #     max_length=self.max_length,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt'
        # ).input_ids.cuda()

        # output_ids = self.model.generate(
        #     input_ids,
        #     max_new_tokens=self.max_new_tokens,
        #     min_new_tokens=self.min_new_tokens,
        #     num_beams=self.num_beams
        # )

        # if self.model.config.is_encoder_decoder:
        #     pred = self.tokenizer.decode(output_ids[0], skip_sepcial_tokens=True)
        # else:
        #     pred = self.tokenizer.decode(output_ids[0][len(input_ids):], skip_sepcial_tokens=True)
        pipeline_input = {
            'question': question,
            'context': context
        }

        res = self.nlp_qa(pipeline_input)
        pred = res['answer']

        metrics = {}
        f1, precision, recall = self.token_level_f1_score(pred, answer)
        metrics['f1'] = f1
        metrics['precision'] = precision
        metrics['recall'] = recall
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
        normalized_pred, normalized_label = self.normalize_answer(pred), self.normalize_answer(label)
        if normalized_pred == "unanswerable":
            return -1, -1, -1
        prediction_tokens = normalized_pred.split()
        ground_truth_tokens = normalized_label.split()

        # prediction_tokens = process_example(pred)
        # ground_truth_tokens = process_example(label)

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

    def save_metrics(self):
        with open(self.save_path, "w") as f:
            json.dump(self.saved_results, f)


class RougeMetric:

    def __init__(self):
        self.rouge = ROUGEScore()

    def compute_metrics(self, predictions):
        pred_summaries = []
        gold_summaries = []
        for pred in predictions:
            pred_summary = pred['pred']
            gold_summary = pred['label']

            pred_summaries.append(pred_summary)
            gold_summaries.append(gold_summary)

        rouge_score = self.rouge(pred_summaries, gold_summaries)
        return {
            'rouge1': rouge_score['rouge1_fmeasure'],
            'rouge2': rouge_score['rouge2_fmeasure'],
            'rougeL': rouge_score['rougeL_fmeasure']
        }

    def compute_metrics_from_refine(self, data, predictions):
        avg_rouge_1 = 0.0
        avg_rouge_2 = 0.0
        avg_rouge_l = 0.0
        cnt = 0
        for key in tqdm(predictions):
            prediction = predictions[key]

            label = ""
            for d in data:
                if d['id'] == key:
                    label = d['summary']
            # assert label != ""
            if label == "":
                continue

            max_rouge_1 = 0
            max_rouge_2 = 0
            max_rouge_l = 0
            max_avg_rouge = 0
            for pred in prediction:
                summary = pred['output']

                rouge_score = self.rouge(summary, label)

                avg_rouge = (rouge_score['rouge1_fmeasure'] + rouge_score['rouge2_fmeasure'] + 
                rouge_score['rougeL_fmeasure']) / 3

                if max_avg_rouge < avg_rouge:
                    max_avg_rouge = avg_rouge
                    max_rouge_1 = rouge_score['rouge1_fmeasure']
                    max_rouge_2 = rouge_score['rouge2_fmeasure']
                    max_rouge_l = rouge_score['rougeL_fmeasure']
            
            if max_avg_rouge > 0:
                cnt += 1
                avg_rouge_1 += max_rouge_1
                avg_rouge_2 += max_rouge_2
                avg_rouge_l += max_rouge_l
        
        avg_rouge_1 /= cnt
        avg_rouge_2 /= cnt
        avg_rouge_l /= cnt

        return {
            'rouge1': avg_rouge_1,
            'rouge2': avg_rouge_2,
            'rougeL': avg_rouge_l
        }

            
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--data_path", type=str, default="/path/to/data")
    parser.add_argument("--prediction_path", type=str, default="/path/to/predicition")
    parser.add_argument("--save_path", type=str, default="/path/to/save")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--qg_model_path", type=str, default="/path/to/qg/model")
    parser.add_argument("--from_refine", type=str, default="True")

    args = parser.parse_args()

    args.prompt = """
        "Read this and answer the question\n\n{context}\n\n{question}"
    """
    with open(args.data_path, "r") as f:
        data = json.load(f)

    with open(args.prediction_path, "r") as f:
        predictions = json.load(f)

    qags = FactualityMetric(args)
    rouge = RougeMetric()

    if args.from_refine == "True":
        qags_scores = qags.compute_metrics_from_refine(data=data, predictions=predictions)
        print("Factuality Score: ", qags_scores)

        rouge_scores = rouge.compute_metrics_from_refine(data=data, predictions=predictions)
        print("Rouge Score: ", rouge_scores)

    else:
        qags_scores = qags.compute_metrics(data=data, predictions=predictions)
        print("Factuality Score: ", qags_scores)

        rouge_scores = rouge.compute_metrics(data=data, predictions=predictions)
        print("Rouge Score: ", rouge_scores)

    
    

    

