import json
import uuid

with open("../results/data/processed_scimrc.json", "r") as f:
    ori_data = json.load(f)


tran_data = []

for example in ori_data:
    question = example['question']
    text = example['text'][:10000]
    answer = example['answer']

    id = str(uuid.uuid4())

    from_human = {
        'from': 'human',
        'value': "Answer the question: {qn} given the following article: {text}".format(qn=question, text=text)
    }

    from_gpt = {
        'from': 'gpt',
        'value': "Here is the answer: {ans}".format(ans=answer)
    }

    tran_data.append({
        'id': id,
        'conversations': [from_human, from_gpt]
    })


with open("scimrc_for_vicuna.json", "w") as f:
    json.dump(tran_data, f)