import json

with open("correction_results_100.json", "r") as f:
    data_100 = json.load(f)

with open("correction_results_200.json", "r") as f:
    data_200 = json.load(f)

with open("correction_results_300.json", "r") as f:
    data_300 = json.load(f)

with open("correction_results_400.json", "r") as f:
    data_400 = json.load(f)

with open("correction_results_500.json", "r") as f:
    data_500 = json.load(f)


def total_score(data):
    score = 0
    for key in data:
        max_score = 0
        for gen in data[key]:
            max_score = max(max_score, gen['f1-score'])
        score += max_score
    return score


print(total_score(data_100) / 100)
scores = total_score(data_100) + total_score(data_200) + total_score(data_300) + total_score(data_400) + total_score(data_500)
print(scores / 500)
