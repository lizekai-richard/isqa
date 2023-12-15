import json
import uuid
from datasets import load_dataset


train_data, val_data, _ = load_dataset("allenai/qasper", split=['train', 'validation', 'test'])

processed_data_for_generator = []
processed_data_for_feedback = []
total_qa_pair_cnt = 0

for example in train_data:

    text = ""
    for section in example['full_text']['paragraphs']:
        text += '\n'.join(section)

    summary = example['abstract']

    qa_pairs = []
    questions = []
    answers = []
    for i, _answers in enumerate(example['qas']['answers']):
        for _answer in _answers['answer']:
            answer_string = ""
            unanswerable = False
            if _answer.get("unanswerable", False):
                answer_string = "unanswerable"
                unanswerable = True
            elif _answer.get("yes_no") is not None:
                answer_string = "Yes" if _answer["yes_no"] else "No"
            elif _answer.get("extractive_spans", []):
                if len(_answer["extractive_spans"]) > 1:
                    answer_string = ", ".join(_answer["extractive_spans"])
                else:
                    answer_string = _answer["extractive_spans"][0]
            else:
                answer_string = _answer.get("free_form_answer", "")

            if answer_string:
                questions.append(example['qas']['question'][i])
                answers.append(answer_string)

            context_spans = [x.replace("\n", " ").strip() for x in _answer["evidence"]]
            context_spans = [x for x in context_spans if x != ""]

            if not context_spans:
                context = text[:5000]
            else:
                context = " ".join(context_spans)

            evidence_spans = [x.replace("\n", " ").strip() for x in _answer["highlighted_evidence"]]
            evidence_spans = [x for x in evidence_spans if x != ""]

            if not evidence_spans:
                continue
            evidence = " ".join(evidence_spans)

            processed_data_for_feedback.append({
                'question': example['qas']['question'][i],
                'evidence': context,
                'answer': answer_string,
                'supporting_fact': evidence,
                'unanswerable': unanswerable
            })

    assert len(questions) == len(answers)
    for q, a in zip(questions, answers):
        qa_pairs.append([q, a])

    _id = str(uuid.uuid4())

    total_qa_pair_cnt += len(qa_pairs)
    processed_data_for_generator.append({
        'id': _id,
        'text': text,
        'summary': summary,
        'qa_pairs': qa_pairs
    })


# print(total_qa_pair_cnt / len(processed_data_for_generator))
print(len(processed_data_for_feedback))
print(processed_data_for_feedback[200])

with open("feedback_data_qasper.json", "w") as f:
    json.dump(processed_data_for_feedback, f)


