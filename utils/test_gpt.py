import json
import time
import random
import numpy as np
import torch
from openai import OpenAI

# random.seed(42)
# np.random.seed(42)
# torch.random.manual_seed(42)


with open("../../Data/generator_data.json", "r") as f:
    data = json.load(f)

text = data[0]['text'][:6000]
summary = """The paper discusses the limitations of Neural Machine Translation (NMT) when dealing with low-resource 
or zero-resource language pairs. NMT heavily relies on large-scale parallel data, resulting in poor performance on 
low-resource or zero-resource language pairs. One common alternative to avoid pivoting in NMT is transfer learning, 
which leverages a high-resource pivot-target model to initialize a low-resource source-target model. However, 
this approach still performs poorly in extreme low-resource or zero-resource translation scenarios. The authors 
propose a new transfer learning approach for NMT that uses cross-lingual language model pre-training to enable a high 
performance on zero-shot translation. They propose a novel pre-training method called BRidge Language Modeling (BRLM) 
that can effectively alleviate the distance between different source language spaces. Their proposed approach 
significantly improves zero-shot translation performance, consistently surpassing pivoting and"""

messages = [
    {'role': 'system', 'content': 'You will be given a paragraph and a corresponding summary. Rigorously based on the '
                                  'paragraph, please extract all facts and non-facts existed in the summary. In your '
                                  'response, please use "Facts:" and "Non-facts" to denote extracted facts and '
                                  'non-facts, respectively. Also, no need to correct the non-facts. Extract the '
                                  'original sentences directly. If no facts or non-facts are found, please return '
                                  '"None".'},
    {'role': 'user', 'content': 'Read the paragraph:\n\n{para}\n\nPlease extract all facts and non-facts in the '
                                'following summary:\n\n{summary}'.format(para=text, summary=summary)}
]

client = OpenAI(api_key="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")

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages
)
output = response.choices[0].message.content
print(output)

messages.append({
    'role': 'assistant', 'content': output
})

new_summary1 = """This paper explores transfer learning in a common zero-shot scenario where there are a lot of 
source-pivot and pivot-target parallel data but no source-target parallel data. The authors propose a new transfer 
learning approach for NMT that uses cross-lingual pre-training to enable a high performance on zero-shot translation. 
They also propose a novel pre-training method called BRidge Language Modeling (BRLM) that can effectively alleviate 
the distance between different source language spaces. Their proposed approach significantly improves zero-shot 
translation performance, consistently surpassing pivoting and multilingual approaches. In addition, the performance 
on supervised translation direction remains the same level or even better when using their method."""

messages.append({
    'role': 'user', 'content': "Here is another summary. Please extract all facts and non-facts:\n\n{summary}".format(
        summary=new_summary1)
})

time.sleep(3)
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages
)
output = response.choices[0].message.content
print(output)

messages.append({
    'role': 'assistant', 'content': output
})

new_summary2 = """1. The paper proposes a novel transfer learning approach for NMT that uses cross-lingual language 
model pre-training to enable a high performance on zero-shot translation.\n    2. The proposed approach, 
called BRidge Language Modeling (BRLM), investigates the performance of two existing cross-lingual pre-training 
methods in the zero-shot translation scenario and designs a novel pre-training method to make full use of the 
source-pivot bilingual data to obtain a universal encoder for different languages.\n    3. The proposed approach 
significantly improves zero-shot translation performance, consistently surpassing pivoting and multilingual 
approaches.\n    4. The paper conducts experiments on two public datasets, WMT 2014 and WMT 2017, and their proposed 
approach outperforms several strong baseline systems."""

messages.append({
    'role': 'user', 'content': "Here is another summary. Please extract all facts and non-facts:\n\n{summary}".format(
        summary=new_summary2)
})

time.sleep(3)
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages
)
output = response.choices[0].message.content
print(output)

