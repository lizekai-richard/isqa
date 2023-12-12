import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
from datasets import load_dataset
import warnings
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

generator_data = load_dataset("json", data_files="/mnt/data/zekai/generator_data.json")['train']
example = generator_data[12]
# initial_prompt = """
#     Please summarize the following scientific document.\n###Paper: {text}\n###Summary:
# """.format(text=example['text'][:6000])

tokenizer = LlamaTokenizer.from_pretrained("/mnt/data/zekai/vicuna_7b")
# base_model = LlamaForCausalLM.from_pretrained("/mnt/data/zekai/vicuna_7b", device_map="auto", load_in_8bit=True)

# input_ids = tokenizer(initial_prompt, max_length=2048, truncation=True,
#                       padding='max_length', return_tensors='pt').input_ids.cuda()
# output_ids = base_model.generate(input_ids=input_ids, max_new_tokens=200, num_beams=2)
# output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

summary = """
Archivlink://arxiv.org/abs/1810.04805v1/summary">Bidirectional Encoder Representations from Transformers (BERT) is a novel 
Transformer-based model that has achieved state-of-the-art performance in various natural language processing tasks. However, 
Transformers are not well-suited for handling long sequences, as they can only consume a limited context of symbols as input. 
To address this issue, the authors propose a method that employs BERT's architecture to classify long texts by performing 
segmentation and using another layer on top of the segment representations. They introduce two extensions, Recurrence over 
BERT (RoBERT) and Transformer over BERT (ToBERT), which enable the application of BERT in classification of long texts. 
The paper achieves state-of-the-art results on the Fisher topic classification task and significant improvement on the CSAT 
prediction.
"""

feedback_model = LlamaForCausalLM.from_pretrained("/mnt/data/zekai/vicuna_7b", device_map="auto", load_in_8bit=True)
feedback_model = PeftModel.from_pretrained(feedback_model, "/mnt/data/zekai/feedback_model/checkpoint-600")

question = example['qa_pairs'][1][0]
answer = example['qa_pairs'][1][1]
feedback_prompt = """Below is a question paired with its context, please return your response in two parts:\n1. the answer to the question\n2. the most relevant evidence in the context to answer the question. 
If the question is unanswerable, directly return 'unanswerable'.
###Question: {question}
###Context: {context}
###Response: """.format(question=question, context=summary)

input_ids = tokenizer(feedback_prompt, max_length=512, truncation=True,
                      padding='max_length', return_tensors='pt').input_ids.cuda()
output_ids = feedback_model.generate(input_ids=input_ids, max_new_tokens=100, num_beams=2)
output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
print(output)

"""
    Feedback Output Format Test
"""

feedback_data = load_dataset("json", data_files="/mnt/data/zekai/feedback_data.json")['train']
example = feedback_data[1287]
prompt = """Below is a question paired with its context, please return your response in two parts:
1. the answer to the question 2. the most relevant evidence in the context to answer the question. 
If the question is unanswerable, directly return 'unanswerable'.
###Question: {question}
###Context: {context}
###Response: """.format(question=example['question'], context=example['evidence'])

tokenizer = LlamaTokenizer.from_pretrained("/mnt/data/zekai/vicuna_7b")
model = LlamaForCausalLM.from_pretrained("/mnt/data/zekai/vicuna_7b", device_map="auto", load_in_8bit=True)

lora_bin_path = os.path.join("/mnt/data/zekai/feedback_model/checkpoint-600", "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path):
    pytorch_bin_path = os.path.join("/mnt/data/zekai/feedback_model/checkpoint-600", "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
    else:
        assert ('Checkpoint is not Found!')
model = PeftModel.from_pretrained(model, "/mnt/data/zekai/feedback_model/checkpoint-600")

input_ids = tokenizer(prompt, max_length=512, padding='max_length', return_tensors='pt').input_ids.cuda()
output_ids = model.generate(input_ids=input_ids, max_new_tokens=100)
output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

print(output)
