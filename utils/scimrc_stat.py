import json

scimrc_data = []
with open("../../Data/SciMRC/new_smrc_train.jsonl", "r") as f:
    for d in f:
        scimrc_data.append(json.loads(d))

tot_len = 0.0
tot_cnt = 0.0
# for example in scimrc_data:
#
#     qas = example['qas']
#     for qa in qas:
#         question = qa['question']
#         question_level = int(qa['annotateType'])
#         print(qa['answers'][0]['answer'])
#         break
#     break

example = scimrc_data[567]
print(type(example['qas'][0]['answers'][0]['answer']['unanswerable']))