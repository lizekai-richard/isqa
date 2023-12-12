import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman']  # 中文字体为楷体，英文字体为新罗马字体
plt.rcParams['font.size'] = 36  # 坐标轴字号为16
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# with open("/mnt/data/zekai/batched_correction_results_300.json", "r") as f:
#     data_ft = json.load(f)

# with open("finetune_vs_non_finetune.json", "r") as f:
#     data_non_ft = json.load(f)

with open("../../Report/batched_correction_results_300.json", "r") as f:
    data = json.load(f)

with open("../../Report/batched_correction_results_300_500.json", "r") as f:
    _data = json.load(f)

data.update(_data)
# with open("correction_results_300.json", "r") as f:
#     data_300 = json.load(f)

# with open("correction_results_400.json", "r") as f:
#     data_400 = json.load(f)

# with open("correction_results_500.json", "r") as f:
#     data_500 = json.load(f)

step_to_scores = {}


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
    score = 0
    cnt = 0
    for i, key in enumerate(data):
        max_score = 0
        for j, gen in enumerate(data[key]):
            if j > step:
                break
            max_score = max(max_score, gen['f1-score'])
        if max_score > 0:
            score += max_score
            cnt += 1
    return score / cnt

# print(total_score(data_ft)[0] / 300, total_score(data_ft)[1] / 300)
# print(total_score(data_non_ft)[0] / 50, total_score(data_non_ft)[1] / 50)
# scores = total_score(data_100) + total_score(data_200) + total_score(data_300) + total_score(data_400) + total_score(data_500)
# print(scores / 500)
# score, cnt = total_score(data)
# print(score / cnt)


steps = range(8)
scores = [scores_with_steps(data, step) for step in steps]

plt.figure(figsize=(18, 12))
plt.xlabel("steps")
plt.ylabel("score")
plt.plot([step + 1 for step in steps], scores, marker='^', linewidth=4, markersize=20)
plt.grid(linestyle='--')
plt.savefig("1.pdf")
