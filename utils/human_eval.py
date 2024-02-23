import json
import random
import nltk
from nltk import sent_tokenize
import pandas as pd

nltk.download('punkt')
random.seed(42)

with open("../../Data/generator_data.json", "r") as f:
    data = json.load(f)

with open("../../ExperimentResults/baseline_results_vicuna.json", "r") as f:
    baseline = json.load(f)

with open("../../ExperimentResults/generic_feedback_results_vicuna.json", "r") as f:
    generic = json.load(f)

with open("../../ExperimentResults/batched_refine_results_vicuna_all.json", "r") as f:
    iter_refine = json.load(f)

paper_keys = list(iter_refine.keys())
for_human_evaluation = {
    'key': [],
    'ref summary': [],
    'summary1': [],
    'summary2': [],
    'summary3': []
}
src_texts = []

eval_sample_cnt = 0
key_set = set()

while eval_sample_cnt < 30:

    cur_key = random.choice(paper_keys)
    # print(len(generic[cur_key]))
    if cur_key in key_set or len(generic[cur_key]) < 4 or len(iter_refine[cur_key]) < 4:
        continue

    key_set.add(cur_key)

    src_text = ""
    ref_summary = ""

    for d in data:
        if d['id'] == cur_key:
            src_text = d['text'][:6000]
            ref_summary = d['summary']
            break

    summary_baseline = ""
    for d in baseline:
        if d['id'] == cur_key:
            summary_baseline = d['pred']
            break

    summary_generic = generic[cur_key][3]['output']
    summary_iter_refine = iter_refine[cur_key][3]['output']

    if src_text == "" or ref_summary == "" or summary_baseline == "" or summary_generic == "" or \
            summary_iter_refine == "":
        continue

    summary_baseline = summary_baseline.replace("\n", " ").strip()
    summary_generic = summary_generic.replace("\n", " ").strip()
    summary_iter_refine = summary_iter_refine.replace("\n", " ").strip()

    summary_baseline_sents = sent_tokenize(summary_baseline)
    summary_generic_sents = sent_tokenize(summary_generic)
    summary_iter_refine_sents = sent_tokenize(summary_iter_refine)

    summary_baseline = "\n".join(summary_baseline_sents)
    summary_generic = "\n".join(summary_generic_sents)
    summary_iter_refine = "\n".join(summary_iter_refine_sents)

    for_human_evaluation['key'].append(cur_key)
    for_human_evaluation['ref summary'].append(ref_summary)
    for_human_evaluation['summary1'].append(summary_baseline)
    for_human_evaluation['summary2'].append(summary_generic)
    for_human_evaluation['summary3'].append(summary_iter_refine)
    src_texts.append(src_text)
    
    eval_sample_cnt += 1

df = pd.DataFrame(for_human_evaluation)
df.to_csv("for_human_eval.csv")

if __name__ == '__main__':
    pass
