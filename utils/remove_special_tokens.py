import json


with open("baseline_results_llama2.json", "r") as f:
    data = json.load(f)


def clean_up(text):
    if "###Paper:" in text:
        text = text.replace("###Paper:", "")
    if "\u040b" in text:
        text = text.replace("\u040b", "")
    if "\u2194" in text:
        text = text.replace("\u2194", " ")
    if "\u0409" in text:
        text = text.replace("\u0409", "")
    if "\u201c" in text:
        text = text.replace("\u201c", "")
    if "\u201d" in text:
        text = text.replace("\u201d", "")
    if "\u201c" in text:
        text = text.replace("\u201c", "")
    return text


new_data = []
for d in data:
    pred = d['pred']
    pred = clean_up(pred)
    if not pred:
        continue

    new_data.append({
        'pred': pred,
        'label': d['label'],
        'qa_pairs': d['qa_pairs']
    })


with open("baseline_results_llama2.json", "w") as f:
    json.dump(new_data, f)

