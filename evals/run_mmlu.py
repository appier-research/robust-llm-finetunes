import os
import json
import argparse
import random
from typing import List
from tqdm import tqdm
from datasets import load_dataset
from llms.sglang_inference import SGLangInference
from .mmlu_utils import write_jsonl, normalize_extracted_answer, normalize_response, evaluate_response, write_final_log

BASELINE_PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

"""
NEGATIVE_MARKINGS_INSTRUCTION = """Answer the following multiple choice question with negative marking. For each incorrect answer you will be deducted for one score, you will not be deduct any scores for chosing E. No answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

"""

choices = ['A', 'B', 'C', 'D']


def format_question(row, mode, leetspeak=False, option_remapping=None):
    question = row['question']
    answer_index = row['answer']
    original_choices = row['choices']

    if 'None of the above' in original_choices and ('nota' in mode or 'no_answer' == mode):
        for idx, choice in enumerate(choices):
            question += f'\n{choice}. {original_choices[idx]}'
        return question

    if mode == "nota_e":
        # Remove the correct answer and shift remaining choices
        modified_choices = [choice for i, choice in enumerate(original_choices) if i != answer_index]
        modified_choices.append("None of the above")  # Add "None of the above" as the last choice

        for idx, choice in enumerate(choices):
            question += f'\n{choice}. {modified_choices[idx]}'
    elif mode == "nota_neg":
        # Remove the correct answer and shift remaining choices
        modified_choices = [choice for i, choice in enumerate(original_choices) if i != answer_index]
        # Add "None of the above" as the last choice
        modified_choices.append(", ".join([ f"Not {choice}" for choice in  modified_choices]))
        for idx, choice in enumerate(choices):
            question += f'\n{choice}. {modified_choices[idx]}'
    elif mode == "nota_other":
        # Find one none answer candidate
        modified_choices = [choices[i] for i, choice in enumerate(original_choices) if i != answer_index]
        selected = random.choice(modified_choices)
        assert selected != choices[answer_index]
        for idx, choice in enumerate(choices):
            if choice == selected:
                question += f'\n{choice}. None of the above'
            else:
                question += f'\n{choice}. {original_choices[idx]}'
    else:
        for idx, choice in enumerate(choices):
            if (mode == "no_answer" or "nota" in mode ) and \
                idx == answer_index and \
                choice == choices[answer_index]:
                choice_name = "None of the above"
                if "v1" in mode:
                    choice_name = "Answer not found"
                elif "v2" in mode:
                    choice_name = "No valid options"
                elif "v3" in mode:
                    choice_name = "None of the options given is correct"
                question += f'\n{choice}. {choice_name}'
            elif 'keyed_replacement' == mode and idx == answer_index:
                choice_name = option_remapping[original_choices[idx]]
                question += f'\n{choice}. {choice_name}'
            else:
                question += f'\n{choice}. {original_choices[idx]}'

        if mode == "neg_mark":
            question += '\nE. No answer'
        elif "aota" in mode:
            question += '\nE. All of the above'

    return question


def process_question(row, llm, system_prompt, mode, stats, leetspeak=False, temp=0.0, remapper=None):
    log_entry = {"question_id": stats['total']}
    question = format_question(row, mode, leetspeak=leetspeak, option_remapping=remapper)

    if mode == "nota_e" or mode == "nota_neg":
        ground_truth = "D"  # "None of the above" is always D in nota_e mode
    else:
        ground_truth = choices[row['answer']]

    log_entry.update({
        "original_question": row['question'],
        "formatted_question": question,
        "ground_truth": ground_truth,
        "full_prompt": system_prompt + "\n\n" + question
    })

    res_text, res_info = llm(log_entry["full_prompt"], max_tokens=2048, temperature=temp)
    log_entry.update({
        "llm_response": res_text,
        "llm_info": res_info,
        "normalized_response": normalize_response(res_text)
    })

    evaluate_response(log_entry, mode, stats)
    return log_entry

def load_existing_entries(logging_file, log_path='logging'):
    full_path = os.path.join(log_path, logging_file)

    stats = {'total': 0, 'hit': 0, 'correct': 0, 'negative_markings': 0, 'existing_entries': {}}
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if 'question_id' in entry:
                    stats['existing_entries'][entry['question_id']] = entry
                    stats['total'] += 1
                    stats['hit'] += int(entry['current_score'].split('/')[0])
                    if 'correct_answers' in entry:
                        stats['correct'] += entry['correct_answers']
                        stats['negative_markings'] += entry['negative_markings']
                    elif entry.get('is_correct', False):
                        stats['correct'] += 1
    return stats

def eval_dataset(llm, subject_name, mode="baseline", leetspeak=False, split='test', remapper = None):
    if mode == "nota_mine":
        dataset = load_dataset('appier-ai-research/mmlu-w-nota', subject_name, split=split)
    else:
        dataset = load_dataset('edinburgh-dawg/mmlu-redux', subject_name, split=split)
    logging_file = f"{subject_name}_{mode}_{str(llm).split('/')[-1]}.jsonl"
    if leetspeak:
        logging_file = f"{subject_name}_leetspeak-{mode}_{str(llm).split('/')[-1]}.jsonl"

    system_prompts = {
        "nota": BASELINE_PROMPT,
        "nota_aota": BASELINE_PROMPT,
        "aota": BASELINE_PROMPT,
        "baseline": BASELINE_PROMPT,
        "no_answer": BASELINE_PROMPT,
        "nota_e": BASELINE_PROMPT,
        "nota_mine": BASELINE_PROMPT,
        "nota_neg": BASELINE_PROMPT,
        "nota_other": BASELINE_PROMPT,
        "keyed_replacement": BASELINE_PROMPT,
        "neg_mark": NEGATIVE_MARKINGS_INSTRUCTION,
    }
    log_path = 'logging'
    if split == 'validation':
        log_path = 'val_logging'
    stats = load_existing_entries(logging_file, log_path=log_path)
    full_path = os.path.join(log_path, logging_file)

    with open(full_path, 'a') as log_file:
        for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True, initial=stats['total'])):
            if idx in stats['existing_entries']:
                continue
            system_prompt = BASELINE_PROMPT
            if mode in system_prompts:
                system_prompt = system_prompts[mode]
            # qid_remapper = remapper[str(idx)] if mode == 'keyed_replacement' else None
            qid_remapper = None
            log_entry = process_question(row, llm, system_prompt, mode, stats, leetspeak=leetspeak, remapper=qid_remapper)
            json.dump(log_entry, log_file)
            log_file.write('\n')
            log_file.flush()

    write_final_log(logging_file, stats)
    return stats['hit'], stats['total'], stats['correct'], stats['negative_markings']

VALID_DATASETS = [
        'anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics',
        'conceptual_physics', 'econometrics', 'electrical_engineering', 'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_geography', 'high_school_macroeconomics',
        'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'high_school_us_history', 'human_aging', 'logical_fallacies',
        'machine_learning', 'miscellaneous', 'philosophy', 'professional_accounting', 'professional_law', 'public_relations', 'virology', 'astronomy'
    ]

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM on MMLU datasets")
    parser.add_argument("model_name", type=str, help="Name of the LLM model to use")
    parser.add_argument("--lora", type=str, help="Lora model", default=None)
    parser.add_argument("--series", type=str, help="Name of the LLM model to use", default=None)
    parser.add_argument("--use_sglang", action="store_true", help="Use SglAng inference engine")
    parser.add_argument("--datasets", nargs='+', default=['all'], choices=VALID_DATASETS + ['all'],
                        help="List of datasets to evaluate (default: all)")
    parser.add_argument("--mode",
                        choices=[
                            "baseline", "neg_mark", "no_answer", "nota", "nota_v1", "nota_mine", "keyed_replacement",
                            "nota_v2", "nota_v3", "nota_aota", "aota", "nota_e", "nota_neg", "nota_other"
                        ],
                        default="baseline",
                        help="Evaluation mode (default: baseline)")
    parser.add_argument("--leetspeak", action="store_true", default=False,
                        help="Enable leetspeak mode (default: False)")
    parser.add_argument("--split", type=str, choices=["test", "validation"], default="test",
                        help="Dataset split to evaluate (default: test)")
    return parser.parse_args()

def get_datasets_to_evaluate(selected_datasets: List[str]) -> List[str]:
    if 'all' in selected_datasets:
        return VALID_DATASETS
    return selected_datasets

def main():
    args = parse_arguments()
    if args.use_sglang:
        llm = SGLangInference(args.model_name, args.lora)
    else:
        llm = get_llm(args.model_name, args.series)
    datasets_to_evaluate = get_datasets_to_evaluate(args.datasets)

    results = {}
    for dataset in datasets_to_evaluate:
        print(f"Evaluating {dataset}...")
        hit, total, correct, negative_markings = eval_dataset(llm, dataset,
                    mode=args.mode, leetspeak=args.leetspeak, split=args.split, remapper=None
                )
        results[dataset] = {
            "score": hit / total,
            "correct": correct,
            "total": total
        }
        if args.mode == "neg_mark":
            results[dataset]["negative_markings"] = negative_markings

        print(f"Dataset: {dataset}")
        print(f"Score: {hit}/{total} ({hit/total:.2%})")
        if args.mode == "neg_mark":
            print(f"Correct answers: {correct}/{total} ({correct/total:.2%})")
            print(f"Negative markings: {negative_markings}/{total} ({negative_markings/total:.2%})")
        print()

    # Print overall results
    total_score = sum(result["score"] for result in results.values()) / len(results)
    print(f"Overall score across all evaluated datasets: {total_score:.2%}")

if __name__ == "__main__":
    main()
