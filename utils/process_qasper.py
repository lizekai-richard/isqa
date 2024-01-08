import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import json
import uuid
from tqdm import tqdm
from datasets import load_dataset
from generate_qa_pairs import generate_qa
from nltk import sent_tokenize, word_tokenize
import spacy 

nlp = spacy.load("en_core_web_sm")

def similarity_score(sent, text):
    p = nlp(sent)
    q = nlp(text)
    score = p.similarity(q)
    return score

train_data, val_data, _ = load_dataset("allenai/qasper", split=['train', 'validation', 'test'])

# with open("/mnt/data/zekai/generator_data.json", "r") as f:
#     train_data = json.load(f)

def generate_qa_pairs_for_dataset(dataset):
    processed_data_for_generator = []
    chunk_size = 256
    for example in tqdm(dataset):

        text = ""
        for section in example['full_text']['paragraphs']:
            text += '\n'.join(section)
        
        if len(text) < 10000:
            continue

        summary = example['abstract']
        _id = str(uuid.uuid4())

        sents = sent_tokenize(text[:10000])
        word_count = 0
        seg_count = 0
        all_text = ""
        text_chunk = ""
        text_chunks = []
        qa_pairs = []
        for i in range(len(sents)):
            words = word_tokenize(sents[i])
            word_count += len(words)
            text_chunk += sents[i]

            if word_count > chunk_size:
                # answers = generate_answer_candidates(text_chunk, n_candidates=2)

                all_text += text_chunk
                text_chunks.append(text_chunk)
                word_count = 0
                text_chunk = ""
                seg_count += 1

                if seg_count >= 8:
                    break

        qa_pairs = generate_qa(text_chunks, n_qa_pairs=2, qg_model_path="default")
        if len(qa_pairs) == 0:
            # qa_pairs = example['qa_pairs']
            continue
        # qa_pairs.sort(lambda x: similarity_score(x[0], all_text), reverse=True)
        # selected_qa_pairs = qa_pairs[:16] if len(qa_pairs) > 16 else qa_pairs
        processed_data_for_generator.append({
            'id': _id,
            'text': all_text,
            'summary': summary,
            'qa_pairs': qa_pairs
        })

        # total_qa_pair_cnt += len(qa_pairs)
        # qa_pairs = []
        # for i in range(5):
        #     qa_pairs_chunk = generate_qa(text[i * 1000: (i + 1) * 1000], n_qa_pairs=2, qg_model_path="lmqg/t5-large-squad-qg")
        #     qa_pairs.extend(qa_pairs_chunk)
        # processed_data_for_generator.append({
        #     'id': _id,
        #     'text': text,
        #     'summary': summary,
        #     'qa_pairs': qa_pairs
        # })
    return processed_data_for_generator


train_generator_data = generate_qa_pairs_for_dataset(train_data)
val_generator_data = generate_qa_pairs_for_dataset(val_data)
all_data_for_generator = train_generator_data + val_generator_data
# print(total_qa_pair_cnt / len(processed_data_for_generator))
# print(len(processed_data_for_feedback))
# print(processed_data_for_feedback[200])

# with open("feedback_data_qasper.json", "w") as f:
#     json.dump(processed_data_for_feedback, f)

print(len(all_data_for_generator))
with open("/mnt/data/zekai/generator_data_qasper.json", "w") as f:
    json.dump(all_data_for_generator, f)
