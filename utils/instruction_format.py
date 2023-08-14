from datasets import load_dataset
import json


scimrc_data = load_dataset("json", data_files="../results/data/processed_scimrc.json")['train']

mrc_data = []
summ_data = []
for example in scimrc_data:
    _id = example['id']
    question = example['question']
    text = example['text']
    answer = example['answer']
    summary = example['summary']

    mrc_data.append({
        'id': _id,
        'instruction': question,
        'input': text,
        'output': answer
    })

    summ_data.append({
        'id': _id,
        'instruction': "Summarize the given article",
        'input': text,
        'output': summary
    })

with open("scimrc_for_inst_tune.json", "w") as f:
    json.dump(mrc_data, f)

with open("scimrc_for_summ.json", "w") as f:
    json.dump(summ_data, f)

