import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import spacy
import random
import ipdb
from lmqg import TransformersQG


ANS_TOK = "[ANS]"
NO_ANS_TOK = "[NO_ANS]"


def extract_ans(txts):
    """ extract entities from a sentence using spacy

    rules:
        - entities (non-pronoun)
            - each portion of a person's name
        - noun chunks (non-pronoun)
            - adjectives within noun chunks
            - nouns w/ dependencies that are proper nouns, roughly nouns modifying proper nouns
            - if the head of a noun chunk if a verb, the entire noun chunk ?
    """
    nlp = spacy.load("en_core_web_lg")
    all_ans = list()
    for doc in nlp.pipe(txts, disable=[]):
        ans = list()
        for ent in doc.ents:
            ans.append(ent.text)
        for chunk in doc.noun_chunks:
            ans.append(chunk.text)
        ans = list(set(ans))
        all_ans.append(ans)
    return all_ans


def prepare_ans_conditional_data(text, n_ans_per_txt=10, use_no_ans=False):
    """ Given a text file, extract possible answer candidates for each line.

    Will generate n_ans_per_text instances for each line in txt
    """

    if use_no_ans:
        print("\twith NO_ANS option!")
    else:
        print("\twithout NO_ANS option!")

    print("Extracting entities...")
    all_anss = extract_ans(text)
    print("\tDone!")
    print(f"\tMin ans count: {min(len(a) for a in all_anss)}")
    print(f"\tMax ans count: {max(len(a) for a in all_anss)}")

    print("Writing...")
    txts_w_ans = list()
    all_txt = list()
    all_ans = list()
    for txt, anss in zip(text, all_anss):
        if use_no_ans:
            if len(anss) > n_ans_per_txt - 1:
                anss = random.sample(anss, k=n_ans_per_txt - 1)
            anss += [NO_ANS_TOK] * (n_ans_per_txt - len(anss))
            assert NO_ANS_TOK in anss, ipdb.set_trace()
        else:
            if len(anss) < n_ans_per_txt:
                # extra_anss = random.choices(anss, k=n_ans_per_txt - len(anss))
                # anss += extra_anss
                continue
            if len(anss) > n_ans_per_txt:
                anss = random.sample(anss, n_ans_per_txt)
            assert len(anss) == n_ans_per_txt, ipdb.set_trace()

        for ans in anss:
            txts_w_ans.append(f"{txt} {ANS_TOK} {ans}")
            all_txt.append(txt)
            all_ans.append(ans)

    return all_ans, all_txt


def generate_answer_candidates(text, n_candidates):
    all_ans, all_txt = prepare_ans_conditional_data(text=text, n_ans_per_txt=n_candidates)
    return all_ans

def generate_qa(text, n_qa_pairs, qg_model_path):
    model = TransformersQG(language="en", model="lmqg/t5-large-squad-qg")

    try:
        all_ans, all_txt = prepare_ans_conditional_data(text, n_ans_per_txt=n_qa_pairs)
        questions = model.generate_q(list_context=all_txt, list_answer=all_ans)
        qa_pairs = [[question, answer] for question, answer in zip(questions, all_ans)]
    except Exception:
        qa_pairs = []
    # try:
    #     qa_pairs = model.generate_qa(list_context=text, num_questions=n_qa_pairs)
    # except Exception:
    #     qa_pairs = []
    # assert len(questions) == len(all_ans)
    # qa_pairs = [[question, answer] for question, answer in zip(questions, all_ans)]

    return qa_pairs