import json
import spacy

nlp = spacy.load("en_core_web_lg")

with open("../../ExperimentResults/batched_correction_results_vicuna.json", "r") as f:
    data = json.load(f)

cases = {}
for key in data:
    cur_example = data[key]
    cases[key] = []
    for t in range(len(cur_example) - 1):
        feedback = cur_example[t]['feedback']
        summary = cur_example[t + 1]['output']
        cases[key].append([summary, feedback])
    if not cases[key]:
        del cases[key]


def plot_helper(paper_key):
    paper = cases[paper_key]
    step2sim = {}
    print(paper[0])
    for step, pair in enumerate(paper):
        summary = pair[0]
        feedback = pair[1]

        s = nlp(summary)

        avg_fact_sim_score = 0.0
        avg_non_fact_sim_score = 0.0

        for fact in feedback['facts']:
            f = nlp(fact)
            sim = s.similarity(f)
            avg_fact_sim_score += sim

        for non_fact in feedback['non_facts']:
            nf = nlp(non_fact)
            sim = s.similarity(nf)
            avg_non_fact_sim_score += sim

        if len(feedback['facts']) > 0:
            avg_fact_sim_score /= len(feedback['facts'])
        if len(feedback['non_facts']) > 0:
            avg_non_fact_sim_score /= len(feedback['non_facts'])

        step2sim[step] = [avg_fact_sim_score, avg_non_fact_sim_score]

    return step2sim


paper_keys = list(cases.keys())
print(plot_helper(paper_keys[100]))




