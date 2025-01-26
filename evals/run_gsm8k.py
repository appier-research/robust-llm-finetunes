import os
# os.environ['HF_HOME'] = '../hf_cache'
# os.environ['TRANSFORMERS_CACHE'] = '../hf_cache'
import re
import torch
from tqdm import tqdm
import json
from pathlib import Path
import datetime
import argparse
from datasets import load_dataset
from transformers import pipeline
from .normalization import gsm8k_normalizer


def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()
    if '\n' in answer:
        answer = answer.split('\n')[-1]

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer

instruction_prompt = """
You are given a grade school math question. Please answer the question in the following format:

Q: <Question>
A: <Think step by step here> #### <number only answer>

Format requirements : you must first output your reasoning before finalized with the " #### " format followed by the final numeric answer
"""
# instruction_prompt = """
# You are given a grade school math question.

# Format requirements : you must first output your reasoning before finalized with the " #### " format followed by the final numeric answer
# """


def create_unique_filename(base_name, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def main(load_adapter, few_shot=0, resume=None, 
    base_model="unsloth/llama-3-8b-Instruct-bnb-4bit", 
    torch_dtype=None, run_val=False):
    if run_val and '*' in load_adapter:
        validate_checkpoint(load_adapter, base_model=base_model, torch_dtype=torch_dtype, few_shot=few_shot)
        return 2
    if run_val:
        log_dir = Path("logging")
        log_dir.mkdir(exist_ok=True)
        if few_shot == 0:
            log_dir = Path("logging/gsm8k-val")
            log_dir.mkdir(exist_ok=True)
        else:
            log_dir = Path("logging/gsm8k-val-{}".format(few_shot))
            log_dir.mkdir(exist_ok=True)
    else:
        log_dir = Path("logging")
        log_dir.mkdir(exist_ok=True)
        if few_shot == 0:
            log_dir = Path("logging/gsm8k-test")
            log_dir.mkdir(exist_ok=True)
        else:
            log_dir = Path("logging/gsm8k-test-{}".format(few_shot))
            log_dir.mkdir(exist_ok=True)

    dataset = load_dataset("openai/gsm8k", "main")
    if torch_dtype=='bf16':
        pipe = pipeline("text-generation", base_model, device_map='auto', torch_dtype=torch.bfloat16)
    else:
        pipe = pipeline("text-generation", base_model)
    if load_adapter:
        pipe.model.load_adapter(load_adapter)
    else:
        load_adapter = base_model
    # if 'mistral' in base_model:
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(base_model)
    #     pipe.model.generation_config.pad_token_id = tokenizer.unk_token_id
    total = 0
    correct_cnt = 0
    fmt_fail_cnt = 0

    added = set()
    if resume is not None and os.path.exists(resume):
        jsonl_path = resume
        with open(resume, 'r') as f:
            for idx, line in enumerate(f):
                payload = json.loads(line)
                if payload['passed']:
                    correct_cnt += 1
                total += 1
                added.add(payload['id'])
        print('resume from {}/{}'.format(correct_cnt, total))
        jsonl_filename = resume.replace(str(log_dir)+'/','')
    else:
        adapter_uniq_name = Path(load_adapter.replace('outputs/', '').replace('/','-').replace('./','').replace('0.', '0-')).stem
        jsonl_filename = create_unique_filename(adapter_uniq_name if load_adapter else base_model.split('/')[-1], "jsonl")
        if torch_dtype == 'bf16':
            jsonl_filename = 'bf16_'+jsonl_filename
        jsonl_path = log_dir / jsonl_filename
        print(jsonl_path)

    few_shot_text = ''
    if few_shot > 0:
        for idx, row in enumerate(dataset['train']):
            if idx > (few_shot-1):
                break
            few_shot_text += '\nQ: {}\nA: {}'.format(row['question'], row['answer'])
    few_shot_text = few_shot_text.strip()

    with tqdm(total=len(dataset['test']), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset['test']):
            if idx in added:
                pbar.update(1)
                continue

            input_text = instruction_prompt.strip()
            if few_shot > 0:
                input_text += '\nExamples:\n'+few_shot_text+'\n'
                input_text += 'Now answer this question:\n'
            input_text += 'Q: '+row['question']+'\nA:'
            messages = [
                {"role": "user", "content": input_text},
            ]
            # if 'mistral' in base_model:
            #     # system_prompt = messages[0]['content']
            #     # messages = [ {"role": "user", "content": system_prompt+'\n'+messages[1]['content'] }]
            #     from transformers import AutoTokenizer
            #     tokenizer = AutoTokenizer.from_pretrained(base_model)
            #     messages = tokenizer.apply_chat_template(messages, tokenize=False)
            generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
            failed_format = 0
            if '#### ' not in generation:
                failed_format = 1
            final_ans = extract_ans_from_response(generation)
            gt = extract_ans_from_response(row['answer'])
            correct = 0
            if final_ans == gt:
                correct = 1

            # Log iteration data to JSONL file
            log_entry = {
                "id": idx,
                "prompt": input_text,
                "generation": generation,
                "passed": correct,
                'fmt_err': failed_format
            }
            correct_cnt += correct
            total += 1
            with open(jsonl_path, "a") as fout:
                fout.write(json.dumps(log_entry)+'\n')

            pbar.update(1)
            pbar.set_description("[pass={:.2f}, {}]".format(100*correct_cnt/total, len(input_text)))

    print(correct_cnt/total)
    # Write final pass score to a text file
    final_score = correct_cnt / total
    score_filename = log_dir / "final_scores.txt"
    with open(score_filename, "a") as f:
        f.write(f"{jsonl_filename}: {final_score:.4f}\n")

def validate_checkpoint(load_adapter, 
        base_model="unsloth/llama-3-8b-Instruct-bnb-4bit", 
        torch_dtype=None,
        few_shot=8
    ):
    import glob
    dataset = load_dataset("arrow", data_files={"validation":"dataset/ground_truth/gsm8k/data-00000-of-00001.arrow"})["validation"]

    checkpoint2scores = {}
    for checkpoint in glob.glob(load_adapter):
        # prevent acc of prev checkpoint
        if torch_dtype == 'bf16':
            pipe = pipeline("text-generation", base_model,
                            device_map='cuda', torch_dtype=torch.bfloat16)
        else:
            pipe = pipeline("text-generation", base_model)
        print(checkpoint)
        pipe.model.load_adapter(checkpoint)
        total = 0
        correct_cnt = 0
        format_failure_rate = 0
        for idx, row in enumerate(dataset):
            input_text = instruction_prompt.strip()
            if few_shot > 0:
                input_text += '\nExamples:\n'+few_shot_text+'\n'
                input_text += 'Now answer this question:\n'
            input_text += 'Q: '+row['question']+'\nA:'
            messages = [
                {"role": "user", "content": input_text},
            ]
            generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
            failed_format = 0
            if '#### ' not in generation:
                failed_format = 1
                format_failure_rate += 1
            final_ans = extract_ans_from_response(generation)
            gt = extract_ans_from_response(row['answer'])
            correct = 0
            if final_ans == gt:
                correct_cnt += 1
                correct = 1
            total += 1
        checkpoint2scores[checkpoint] = correct_cnt / total
        del pipe
    print(base_model)
    for ckpt, pass_score in checkpoint2scores.items():
        print(ckpt, pass_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with an optional adapter.")
    parser.add_argument("--load_adapter", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--base_model", type=str, help="Path to the adapter file to load", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--shots", type=int, help="Path to the adapter file to load", default=0)
    parser.add_argument("--resume", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--torch_dtype", type=str, help="torch dtype for model", default=None)
    parser.add_argument("--run_val",  dest='run_val', action='store_true', help="run validation split", default=False)

    args = parser.parse_args()

    main(args.load_adapter, 
        args.shots,
        args.resume, 
        args.base_model,
        torch_dtype=args.torch_dtype,
        run_val=args.run_val
    )