import json

with open("../../ExperimentResults/iter_refine_results_llama2_all.json", "r") as f:
    data = json.load(f)

cnt = 0
for key in data:
    if len(data[key]) == 0:
        continue
    cnt += 1

print(cnt)
