import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import string
import re
import json
import torch
from collections import Counter
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, prepare_model_for_int8_training

with open("/mnt/data/zekai/generator_data.json", "r") as f:
    data = json.load(f)

with open("correction_results_250.json", "r") as f:
    gen = json.load(f)

paper = ""
for example in data:
    if example['id'] == "4fb32a07-e82c-4592-ad7c-ca8f3a35a634":
        paper = example['text']
        qa_pairs = example['qa_pairs']

def normalize_answer(s):
    def remove_redundant_whitespace(text):
        return text.strip()

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def remove_special_tokens(text):
        return re.sub(r'\\u25b6\\ufe0f', '', text)

    return white_space_fix(remove_redundant_whitespace(remove_articles(remove_punc(remove_special_tokens(lower(s))))))

def token_level_f1_score(pred, label):
    normalized_pred, normalized_label = normalize_answer(pred), normalize_answer(label)
    
    prediction_tokens = normalized_pred.split()
    ground_truth_tokens = normalized_label.split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def load_base_model():
    tokenizer = LlamaTokenizer.from_pretrained("/mnt/data/zekai/vicuna_7b")
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        "/mnt/data/zekai/vicuna_7b",
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model = prepare_model_for_int8_training(model)
    return tokenizer, model


def load_feedback_model():
    base_model = LlamaForCausalLM.from_pretrained(
        "/mnt/data/zekai/vicuna_7b",
        load_in_8bit=True,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(
        base_model,
        "/mnt/data/zekai/feedback_model_new/checkpoint-600",
        device_map="auto"
    )
    model = prepare_model_for_int8_training(model)
    return model


initial_prompt = """
Please summarize the following scientific document.\n###Paper: {text}\n###Summary:
""".format(text=paper[:6000])

tokenizer, base_model = load_base_model()
feedback_model = load_feedback_model()
# input_ids = tokenizer(
#     initial_prompt,
#     max_length=2048,
#     padding='max_length',
#     truncation=True,
#     return_tensors='pt'
# ).input_ids.cuda()

# output_ids = base_model.generate(
#     input_ids,
#     num_beams=2,
#     max_new_tokens=200,
#     min_new_tokens=1
# )

pred_summary = """
The vulnerability of machine learning models to adversarial attacks is a well-known issue. In natural language processing (NLP), generating adversarial examples is more challenging than in computer vision due to the discrete nature of the input space and ensuring semantic coherence with the original sentence. Recent works for attacking text models rely on introducing errors at the character level or adding/deleting words, but these techniques often result in unnatural-looking adversarial examples that can be easily identified by humans.

The authors propose a novel technique called BAE (BERT-based Adversarial Examples) that uses a language model to generate adversarial examples. BAE perturbs an input sentence by either replacing a token or inserting a new token in the sentence, by masking a part of the input and using a language model to fill in the mask. BAE relies on the powerful BERT masked language model for ensuring grammatical
"""

fed_prompt = """Below is a question paired with its context, please return your response in two parts:\n1. the answer to the question\n2. the most relevant evidence in the context to answer the question.\nIf the question is unanswerable, directly return 'unanswerable'.
###Question: {question}
###Context: {context}
###Response: """.format(question=qa_pairs[0][0], context=pred_summary)

# input_ids = tokenizer(
#     fed_prompt,
#     max_length=512,
#     padding='max_length',
#     truncation=True,
#     return_tensors='pt'
# ).input_ids.cuda()

# output_ids = feedback_model.generate(
#     input_ids=input_ids,
#     num_beams=2,
#     max_new_tokens=100,  # max_length=max_new_tokens+input_sequence
#     min_new_tokens=1,  # min_length=min_new_tokens+input_sequence
# )

output = """
▶1. Answer:The vulnerability of machine learning models to adversarial attacks is a well-known issue.
2.Evidence:The vulnerability of machine learning models to adversarial attacks is a well-known issue.)
"""

ans_index, sp_index = -1, -1
ans_prefix, sp_prefix = None, None
if ("Answer:" or "answer:" or "1.") in output:
    ans_index = output.find("Answer:")
    ans_prefix = "Answer:"
    if ans_index == -1: 
        ans_index = output.find("answer:")
        ans_prefix = "answer:"
    if ans_index == -1: 
        ans_index = output.find("1.")
        ans_prefix = "1."
if ("Evidence:" or "evidence:" or "2.") in output:
    sp_index = output.find("Evidence:")
    sp_prefix = "Evidence:"
    if sp_index == -1: 
        sp_index = output.find("evidence:")
        sp_prefix = "evidence:"
    if sp_index == -1:
        sp_index = output.find("2.")
        sp_prefix = "2."

if ans_index == -1 or sp_index == -1:
    print("Not found")
feedback_ans = output[ans_index + len(ans_prefix): sp_index]
feedback_sp = output[sp_index + len(sp_prefix):]
score = token_level_f1_score(feedback_ans, qa_pairs[0][1])
print(feedback_ans)
print(feedback_sp)
print("F1-score: ", score)


prompt = """
    Below is a scientific paper. Please provide a summary that include facts and exclude non-facts.\n###Paper: {text}\n###Facts: {facts}\n###Non-Facts: {non_facts}\n###Summary:
"""
if score < 0.5:
    prompt = prompt.format(text=paper[:6000], facts="", non_facts="1." + feedback_sp)
else:
    prompt = prompt.format(text=paper[:6000], facts="1. " + feedback_sp, non_facts="")

input_ids = tokenizer(
    prompt,
    max_length=2048,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
).input_ids.cuda()


output_ids = base_model.generate(
    input_ids,
    num_beams=2,
    do_sample=True,
    max_new_tokens=200,
    min_new_tokens=50
)

output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True,
                         clean_up_tokenization_spaces=True)
print(output)

# decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
# print(decoded_prompt)