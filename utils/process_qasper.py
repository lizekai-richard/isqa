import os
import json
import uuid
from enum import Enum
from tqdm import tqdm
from datasets import load_dataset
import spacy

nlp = spacy.load("en_core_web_sm")

with open("../Data/qasper-train-v0.3.json", "r") as f:
    train_data = json.load(f)

with open("../Data/qasper-dev-v0.3.json", "r") as f:
    dev_data = json.load(f)


class AnswerType(Enum):
    EXTRACTIVE = 1
    ABSTRACTIVE = 2
    BOOLEAN = 3
    NONE = 4


def extract_answer_and_evidence(answer):

    if answer.get("unanswerable", False):
        answer_string = "Unanswerable"
        answer_type = AnswerType.NONE
    elif answer.get("yes_no") is not None:
        answer_string = "Yes" if answer["yes_no"] else "No"
        answer_type = AnswerType.BOOLEAN
    elif answer.get("extractive_spans", []):
        answer_string = ", ".join(answer["extractive_spans"])
        answer_type = AnswerType.EXTRACTIVE
    else:
        answer_string = answer.get("free_form_answer", "")
        answer_type = AnswerType.ABSTRACTIVE

    return answer_string, answer_type


processed_data = []

for article in tqdm(train_data.values()):
    _id = str(uuid.uuid4())

    summary = article['abstract']

    text = ""
    full_text = article['full_text']
    for sec in full_text:
        for para in sec['paragraphs']:
            text += para

    qa_pairs = []
    for question_answer in article["qas"]:
        question = question_answer['question']
        all_answers = []
        for answer_annotation in question_answer["answers"]:
            answer, answer_type = extract_answer_and_evidence(
                answer_annotation["answer"]
            )
            all_answers.append(answer)

        qa_pairs.append([question, all_answers])

    processed_data.append({
        'id': _id,
        'text': text,
        'summary': summary,
        'qa_pairs': qa_pairs
    })

for article in tqdm(dev_data.values()):
    _id = str(uuid.uuid4())

    summary = article['abstract']

    text = ""
    full_text = article['full_text']
    for sec in full_text:
        for para in sec['paragraphs']:
            text += para

    qa_pairs = []
    for question_answer in article["qas"]:
        question = question_answer['question']
        all_answers = []
        for answer_annotation in question_answer["answers"]:
            answer, answer_type = extract_answer_and_evidence(
                answer_annotation["answer"]
            )
            all_answers.append(answer)

        qa_pairs.append([question, all_answers])

    processed_data.append({
        'id': _id,
        'text': text,
        'summary': summary,
        'qa_pairs': qa_pairs
    })

with open("generator_data_qasper.json", "w") as f:
    json.dump(processed_data, f)
