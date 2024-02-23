import json
import uuid
from tqdm import tqdm

scimrc_data = []
with open("../../Data/SciMRC/new_smrc_train.jsonl", "r") as f:
    for d in f:
        scimrc_data.append(json.loads(d))

print("Here")
processed_data_for_generator = []
processed_data_for_corrector = []
processed_easy_for_corrector = []
processed_medium_for_corrector = []
processed_hard_for_corrector = []
total_len = 0.0
total_examples = 0.0
cnt = 0

for example in tqdm(scimrc_data):
    _id = str(uuid.uuid4())
    total_examples += 1

    text = ""
    full_text = example['full_text']
    for sec in full_text:
        for para in sec['paragraphs']:
            text += para

    total_len += len(text)

    summary = example['abstract']
    qa_pairs = []
    qas = example['qas']
    for qa in qas:
        question = qa['question']
        question_level = int(qa['annotateType'])
        answer = qa['answers'][0]

        unanswerable = answer['answer']['unanswerable']
        answer_text = answer['answer']['free_form_answer']
        evidence = answer['answer']['evidence'][0] if len(answer['answer']['evidence']) > 0 else ''
        supporting_fact = answer['answer']['highlighted_evidence'][0]

        if unanswerable:
            qa_pairs.append((question, 'unanswerable'))
        else:
            qa_pairs.append((question, answer_text))

        processed_data_for_corrector.append({
            'id': _id,
            # 'text': text,
            'evidence': evidence,
            # 'summary': summary,
            'question': question,
            'unanswerable': unanswerable,
            'answer': answer_text,
            'supporting_fact': supporting_fact
        })
        if question_level == 1:
            processed_easy_for_corrector.append({
                'id': _id,
                # 'text': text,
                'evidence': evidence,
                # 'summary': summary,
                'question': question,
                'unanswerable': unanswerable,
                'answer': answer_text,
                'supporting_fact': supporting_fact
            })
        elif question_level == 2:
            processed_medium_for_corrector.append({
                'id': _id,
                # 'text': text,
                'evidence': evidence,
                # 'summary': summary,
                'question': question,
                'unanswerable': unanswerable,
                'answer': answer_text,
                'supporting_fact': supporting_fact
            })
        elif question_level == 3:
            processed_hard_for_corrector.append({
                'id': _id,
                # 'text': text,
                'evidence': evidence,
                # 'summary': summary,
                'question': question,
                'unanswerable': unanswerable,
                'answer': answer_text,
                'supporting_fact': supporting_fact
            })

    processed_data_for_generator.append({
        'id': _id,
        'text': text,
        'summary': summary,
        'qa_pairs': qa_pairs
    })

print(len(processed_data_for_generator))
print(len(processed_data_for_corrector))
print(len(processed_easy_for_corrector))  # 826
print(len(processed_medium_for_corrector))  # 1427
print(len(processed_hard_for_corrector))  # 2620

with open("../../Data/SciMRC/feedback_data.json", "w") as f:
    json.dump(processed_data_for_corrector, f)

with open("../../Data/SciMRC/feedback_data_easy.json", "w") as f:
    json.dump(processed_easy_for_corrector, f)

with open("../../Data/SciMRC/feedback_data_medium.json", "w") as f:
    json.dump(processed_medium_for_corrector, f)

with open("../../Data/SciMRC/feedback_data_hard.json", "w") as f:
    json.dump(processed_hard_for_corrector, f)

with open("../../Data/SciMRC/generator_data.json", "w") as f:
    json.dump(processed_data_for_generator, f)