import json
from tqdm import tqdm
from datasets import load_dataset

train_data, val_data = load_dataset("hotpot_qa", 'distractor', split=['train', 'validation'])


def process_example(example):
    processed_example = {}

    _id = example['id']
    question = example['question']
    answer = example['answer']

    title2text = {}
    title2sents = {}
    for i, title in enumerate(example['context']['title']):
        text = " ".join(example['context']['sentences'][i])
        title2text[title] = text
        title2sents[title] = example['context']['sentences'][i]

    context = ""
    evidence = ""
    for i in range(len(example['supporting_facts']['title'])):
        title = example['supporting_facts']['title'][i]
        sent_id = example['supporting_facts']['sent_id'][i]

        context += (title2text[title] + "\n")

        if sent_id < len(title2sents[title]):
            evidence += (title2sents[title][sent_id] + " ")

    return {
        'id': _id,
        'context': context,
        'question': question,
        'answer': answer,
        'evidence': evidence
    }


processed_train, processed_val = [], []

for example in tqdm(train_data):
    processed_train.append(process_example(example))

for example in tqdm(val_data):
    processed_val.append(process_example(example))

# train_data = train_data.map(process_example)
# val_data = val_data.map(process_example)

# train_data.remove_columns(['supporting_facts', 'level', 'type'])
# val_data.remove_columns(['supporting_facts', 'level', 'type'])

with open("feedback_data_hotpot_qa_train.json", "w") as f:
    json.dump(processed_train, f)

with open("feedback_data_hotpot_qa_val.json", "w") as f:
    json.dump(processed_val, f)
