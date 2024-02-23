import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Arial']  # 中文字体为楷体，英文字体为新罗马字体
plt.rcParams['font.size'] = 36  # 坐标轴字号为16
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

with open("../../ExperimentResults/refine_vicuna_qags_scores.json", "r") as f:
    qags_vicuna = json.load(f)

with open("../../ExperimentResults/refine_llama2_qags_scores_new.json", "r") as f:
    qags_llama2 = json.load(f)

with open("../../ExperimentResults/refine_flan_t5_qags_scores.json", "r") as f:
    qags_flan_t5 = json.load(f)

with open("../../ExperimentResults/quest_eval_results_vicuna.json", "r") as f:
    quest_eval_vicuna = json.load(f)

with open("../../ExperimentResults/quest_eval_results_llama2_new.json", "r") as f:
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


steps = range(9)
vicuna_scores = [scores_with_steps(qags_vicuna, step) for step in steps]
llama2_scores = [scores_with_steps(qags_llama2, step) for step in steps]
flan_t5_scores = [scores_with_steps(qags_flan_t5, step) for step in steps]

# print(vicuna_scores)
# print(llama2_scores)
# print(flan_t5_scores)

# llama2_scores = [0.549391816658651, 0.58913188880105965, 0.60245654937962624, 0.6094792627114727, 0.6123632382474606,
#                  0.6133980060939932, 0.6140655669129472, 0.6140655669129472, 0.6140655669129472]

vicuna_scores = [0.28395475802, 0.31566531151557803, 0.32506117523562206, 0.3288220905484095, 0.32966488022794656,
                 0.3296655130513326, 0.3296660271444216, 0.3296667494847326, 0.32966662298813495]
llama2_scores = [0.2890402329358911, 0.31780361149341234, 0.3286480018591378, 0.3334858238973965,
                 0.33576547688308003, 0.336275427612103, 0.33627814562925925, 0.3362780398221568, 0.3362780675651568]
flan_t5_scores = [0.29658302187490393, 0.3075233356039662, 0.3185233356039662, 0.32179506592156026,
                  0.3230983852968746, 0.3234574605984222, 0.32365101794411477, 0.3236514582333127,
                  0.3236549417670976]

plt.figure(figsize=(18, 12))
plt.xlabel("steps")
plt.ylabel("QuestEval scores")
plt.plot(steps, vicuna_scores, marker='^', linewidth=4, markersize=20, label='vicuna')
plt.plot(steps, llama2_scores, marker='o', linewidth=4, markersize=20, label='llama2')
plt.plot(steps, flan_t5_scores, marker='s', linewidth=4, markersize=20, label='flan-t5')
plt.grid(linestyle='--')
plt.legend()
plt.savefig("quest_eval_multi_step.pdf")
