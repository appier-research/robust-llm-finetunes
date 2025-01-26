import re
import os
import json
import torch

def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt").to('cuda')
    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids = target_ids

        with torch.no_grad():
            input_ids =input_ids.to('cuda')
            target_ids= target_ids.to('cuda')
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )

def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )

def ensure_logging_dir():
    if not os.path.exists('logging'):
        os.makedirs('logging')

def write_jsonl(filename, data, path='logging'):
    ensure_logging_dir()
    full_path = os.path.join(path, filename)
    with open(full_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

def write_final_log(logging_file, stats, path='logging'):
    final_log = {"final_score": f"{stats['hit']}/{stats['total']}"}
    if 'negative_markings' in stats:
        final_log.update({
            "correct_answers": f"{stats['correct']}/{stats['total']}",
            "negative_markings": f"{stats['negative_markings']}/{stats['total']}"
        })
    write_jsonl(logging_file, final_log, path)


def evaluate_neg_mark(log_entry, extracted_answer, stats):
    if extracted_answer == 'E':
        stats['negative_markings'] += 1
        log_entry["result"] = "No answer"
    elif extracted_answer == log_entry["ground_truth"]:
        stats['hit'] += 1
        stats['correct'] += 1
        log_entry["result"] = "Correct"
    else:
        stats['hit'] -= 0.33
        log_entry["result"] = "Incorrect"

def evaluate_regular(log_entry, extracted_answer, stats):
    if extracted_answer == log_entry["ground_truth"]:
        stats['hit'] += 1
        stats['correct'] += 1
        log_entry["is_correct"] = True
    else:
        log_entry["is_correct"] = False

def evaluate_response(log_entry, mode, stats):
    match = re.search(r"Answer\s*:\s*(\w+)", log_entry["normalized_response"])
    if match:
        extracted_answer = normalize_extracted_answer(match.group(1))
        log_entry["extracted_answer"] = extracted_answer
        if mode == "neg_mark":
            evaluate_neg_mark(log_entry, extracted_answer, stats)
        else:
            evaluate_regular(log_entry, extracted_answer, stats)
    else:
        log_entry["extracted_answer"] = None
        log_entry["result" if mode == "neg_mark" else "is_correct"] = "No answer extracted" if mode == "neg_mark" else False

    stats['total'] += 1
    log_entry["current_score"] = f"{stats['hit']}/{stats['total']}"
    if mode == "neg_mark":
        log_entry.update({
            "correct_answers": stats['correct'],
            "negative_markings": stats['negative_markings']
        })