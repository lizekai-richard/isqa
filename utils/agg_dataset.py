import json
from tqdm import tqdm
from datasets import load_dataset

mrc_data_path = "../results/data/mrc_medium.json"
summ_data_path = "allenai/mup"
agg_data_path = "./agg.json"

with open(mrc_data_path, "r") as f:
    mrc_data = json.load(f)

val_ds = load_dataset(summ_data_path)['validation']

agg_data = []

for example in tqdm(mrc_data):
    paper_id = example['paper_id']
    qa_pairs = example['qa_pairs']
    for _example in val_ds:
        _paper_id = _example['paper_id']
        if _paper_id == paper_id:
            text = _example['text']
            break
    
    for qa_pair in qa_pairs:
        question = qa_pair[0]
        answer = qa_pair[1]
        agg_data.append({
            'text': text,
            'question': question,
            'answer': answer
        })


with open(agg_data_path, "w") as f:
    json.dump(agg_data, f)

    