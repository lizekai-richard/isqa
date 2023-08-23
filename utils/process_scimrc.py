import json
import uuid

scimrc_data = []
with open("../Data/SciMRC/new_smrc_train.jsonl", "r") as f:
    for d in f:
        scimrc_data.append(json.loads(d))

print("Here")
processed_data = []
processed_easy = []
processed_medium = []
processed_hard = []
total_len = 0.0
total_examples = 0.0
cnt = 0
avg_evi_len = 0.0

for example in scimrc_data:
    _id = str(uuid.uuid4())
    total_examples += 1

    text = ""
    full_text = example['full_text']
    for sec in full_text:
        for para in sec['paragraphs']:
            text += para

    total_len += len(text)

    summary = example['abstract']
    
    qas = example['qas']
    for qa in qas:
        question = qa['question']
        question_level = int(qa['annotateType'])

        for answer in qa['answers']:
            unanswerable = answer['answer']['unanswerable']
            answer_text = answer['answer']['free_form_answer']
            evidence = answer['answer']['evidence'][0]
            print("evidence:" + evidence + "  answer: " + answer_text)
            processed_data.append({
                'id': _id,
                'text': text,
                'summary': summary,
                'question': question,
                'answer': answer_text
            })
            if question_level == 1:
                processed_easy.append({
                    'id': _id,
                    'text': text,
                    'summary': summary,
                    'question': question,
                    'answer': answer_text
                })
            elif question_level == 2:
                processed_medium.append({
                    'id': _id,
                    'text': text,
                    'summary': summary,
                    'question': question,
                    'answer': answer_text
                })
            elif question_level == 3:
                processed_hard.append({
                    'id': _id,
                    'text': text,
                    'summary': summary,
                    'question': question,
                    'answer': answer_text
                })
        break

print(avg_evi_len / len(scimrc_data))
print(len(processed_data))
print(len(processed_easy)) #826
print(len(processed_medium)) #1427
print(len(processed_hard)) #2620

with open("../results/data/processed_scimrc.json", "w") as f:
    json.dump(processed_data, f)

with open("../results/data/scimrc_easy.json", "w") as f:
    json.dump(processed_easy, f)

with open("../results/data/scimrc_medium.json", "w") as f:
    json.dump(processed_medium, f)

with open("../results/data/scimrc_hard.json", "w") as f:
    json.dump(processed_hard, f)






