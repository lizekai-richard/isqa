"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchmetrics.text.rouge import ROUGEScore
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, prepare_model_for_int8_training
from fastchat.model import load_model, get_conversation_template, add_model_args


def save_preds(saved_result):
    with open("summary_before_tune_13b.json", "w") as f:
        json.dump(saved_result, f)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    rouge = ROUGEScore()
    return rouge(preds, labels)


@torch.inference_mode()
def inference(args, model, tokenizer, prompt):

    input_ids = tokenizer([prompt], truncation=True).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        # do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=2
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--prompt", type=str, default="Please summarize the following scientific paper:\n###Paper: {text}\n###Summary: ")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    dataset = load_dataset("json", 
                           data_files="/mnt/data/zekai/generator_data.json")['train']


    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        "/mnt/data/zekai/vicuna_13b",
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model = prepare_model_for_int8_training(model)

    ref_summaries, pred_summaries = [], []
    saved_result = []
    for i in tqdm(range(args.test_size)):
        example = dataset[i]
        summary = example['summary']
        paper = example['text'][:6000]
        qa_pairs = example['qa_pairs']
        # question = example['question']
        # answer = example['answer']
        
        # msg = args.prompt.format(text=paper)

        # conv = get_conversation_template(args.model_path)
        # conv.append_message(conv.roles[0], msg)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        # print(prompt)
        prompt = args.prompt.format(text=paper)
        pred = inference(args, model, tokenizer, prompt)

        ref_summaries.append(summary)
        pred_summaries.append(pred)
        for qa_pair in qa_pairs:
            question = qa_pair[0]
            answer = qa_pair[1]
            saved_result.append({
                'summary': pred,
                'question': question,
                'answer': answer
            })

    metrics = compute_metrics((pred_summaries, ref_summaries))
    save_preds(saved_result)
    print(metrics)



    # prompt = """
    #     Please summarize the following article: 1 INTRODUCTION . Statistical learning theory studies the 
    #     learning properties of machine learning algorithms , and more fundamentally , the conditions under 
    #     which learning from finite data is possible . In this context , classical learning theory focuses 
    #     on the size of the hypothesis space in terms of different complexity measures , such as combinatorial 
    #     dimensions , covering numbers and Rademacher/Gaussian complexities ( Shalev-Shwartz & Ben-David , 2014 
    #     ; Boucheron et al. , 2005 ) . Another more recent approach is based on defining suitable notions of 
    #     stability with respect to perturbation of the data ( Bousquet & Elisseeff , 2001 ; Kutin & Niyogi , 
    #     2002 ) . In this view , the continuity of the process that maps data to estimators is crucial , rather 
    #     than the complexity of the hypothesis space . Different notions of stability can be considered , depending 
    #     on the data perturbation and metric considered ( Kutin & Niyogi , 2002 ) . Interestingly , the stability 
    #     and complexity approaches to characterizing the learnability of problems are not at odds with each other , 
    #     and can be shown to be equivalent as shown in Poggio et al . ( 2004 ) and Shalev-Shwartz et al . ( 2010 ) . 
    #     In modern machine learning overparameterized models , with a larger number of parameters than the size of 
    #     the training data , have become common . The ability of these models to generalize is well explained by 
    #     classical statistical learning theory as long as some form of regularization is used in the training process 
    #     ( Bühlmann & Van De Geer , 2011 ; Steinwart & Christmann , 2008 ) . However , it was recently shown - first 
    #     for deep networks ( Zhang et al. , 2017 ) , and more recently for kernel methods ( Belkin et al. , 2019 ) - 
    #     that learning is possible in the absence of regularization , i.e. , when perfectly fitting/interpolating the 
    #     data . Much recent work in statistical learning theory has tried to find theoretical ground for this empirical 
    #     finding . Since learning using models that interpolate is not exclusive to deep neural networks , we study 
    #     generalization in the presence of interpolation in the case of kernel methods . We study both linear and kernel 
    #     least squares problems in this paper . Our Contributions : • We characterize the generalization properties of 
    #     interpolating solutions for linear and kernel least squares problems using a stability approach . While the 
    #     ( uniform ) stability properties of regularized kernel methods are well known ( Bousquet & Elisseeff , 2001 ) , 
    #     we study interpolating solutions of the unregularized ( `` ridgeless '' ) regression problems . • We obtain an 
    #     upper bound on the stability of interpolating solutions , and show that this upper bound is minimized by the
    #     minimum norm interpolating solution .
    # """