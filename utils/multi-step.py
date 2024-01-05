import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman']  # 中文字体为楷体，英文字体为新罗马字体
plt.rcParams['font.size'] = 36  # 坐标轴字号为16
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

with open("../../ExperimentResults/refine_vicuna_qags_scores.json", "r") as f:
    qags_vicuna = json.load(f)

with open("../../ExperimentResults/refine_llama2_qags_scores.json", "r") as f:
    qags_llama2 = json.load(f)

with open("../../ExperimentResults/refine_flan_t5_qags_scores.json", "r") as f:
    qags_flan_t5 = json.load(f)

with open("../../ExperimentResults/quest_eval_results_vicuna.json", "r") as f:
    quest_eval_vicuna = json.load(f)

with open("../../ExperimentResults/quest_eval_results_llama2.json", "r") as f:
    quest_eval_llama2 = json.load(f)

with open("../../ExperimentResults/quest_eval_results_flan_t5.json", "r") as f:
    quest_eval_flan_t5 = json.load(f)


def total_score(data):
    score = 0
    cnt = 0
    for i, key in enumerate(data):
        # if i >= 20: break
        max_score = 0
        for gen in data[key]:
            max_score = max(max_score, gen['f1-score'])
        if max_score > 0:
            score += max_score
            cnt += 1
    return score, cnt


def scores_with_steps(data, step):
    tot_score = 0
    cnt = 0
    for i, key in enumerate(data):
        max_score = 0
        for j, score in enumerate(data[key]):
            if j > step:
                break
            max_score = max(max_score, score)
        if max_score > 0:
            tot_score += max_score
            cnt += 1
    return tot_score / cnt


steps = range(8)
vicuna_scores = [scores_with_steps(quest_eval_vicuna, step) for step in steps]
llama2_scores = [scores_with_steps(quest_eval_llama2, step) for step in steps]
flan_t5_scores = [scores_with_steps(quest_eval_flan_t5, step) for step in steps]

plt.figure(figsize=(18, 12))
plt.xlabel("steps")
plt.ylabel("QuestEval scores")
plt.plot([step + 1 for step in steps], vicuna_scores, marker='^', linewidth=4, markersize=20, label='vicuna')
plt.plot([step + 1 for step in steps], llama2_scores, marker='^', linewidth=4, markersize=20, label='llama2')
plt.plot([step + 1 for step in steps], flan_t5_scores, marker='^', linewidth=4, markersize=20, label='flan-t5')
plt.grid(linestyle='--')
plt.legend()
plt.savefig("quest_eval_multi_step.pdf")
