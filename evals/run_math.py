import os
import re
import json
import datetime
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import pipeline

from .common import check_equality, ANSWER_PATTERN
from .openai_sampler import ChatCompletionSampler
from .normalization import math_normalizer
from llms.sglang_inference import SGLangInference


QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

train_prompt = "Given a mathematics problem, determine the answer.\n"

def create_unique_filename(base_name, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def validate_checkpoint(load_adapter, 
        base_model="unsloth/llama-3-8b-Instruct-bnb-4bit", 
        torch_dtype=None
    ):
    import glob
    dataset = load_dataset("appier-ai-research/robust-finetuning", "math-resample")['validation']
    checkpoint2scores = {}
    mini_sampler = ChatCompletionSampler(model="gpt-4o-mini-2024-07-18")
    for checkpoint in glob.glob(load_adapter):
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
            problem = row['problem']
            test_question = QUERY_TEMPLATE.format(Question=problem)
            messages = [
                {"role": "user", "content": test_question},
            ]
            generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
            match = re.search(ANSWER_PATTERN, generation)
            extracted_answer = match.group(1) if match else math_normalizer(generation)
            gold = math_normalizer(row['solution'])
            correct = float(check_equality(mini_sampler, gold, extracted_answer))
            correct_cnt += correct
            total += 1
            if total == 200:
                break
        checkpoint2scores[checkpoint] = correct_cnt / total
        print(checkpoint, correct_cnt / total)
        del pipe
    print(base_model)
    for ckpt, pass_score in checkpoint2scores.items():
        print(ckpt, pass_score)

def main(load_adapter, few_shot=0, resume=None, base_model="unsloth/llama-3-8b-Instruct-bnb-4bit", 
            torch_dtype='bf16', run_val=False, use_sglang=False):
    if run_val and '*' in load_adapter:
        validate_checkpoint(load_adapter, base_model=base_model, torch_dtype=torch_dtype)
        return 2

    log_dir = Path("logging")
    log_dir.mkdir(exist_ok=True)
    if few_shot == 0:
        log_dir = Path("logging/math-test")
        log_dir.mkdir(exist_ok=True)
    else:
        log_dir = Path("logging/math-test-{}".format(few_shot))
        log_dir.mkdir(exist_ok=True)

    dataset = load_dataset("appier-ai-research/robust-finetuning", "math-resample")

    if use_sglang:
        pipe = SGLangInference(base_model, load_adapter)
        load_adapter = load_adapter or base_model
    elif 'gpt' in base_model:
        load_adapter = "openai_"+base_model
    else:
        if torch_dtype=='bf16':
            pipe = pipeline("text-generation", base_model, device_map='cuda', torch_dtype=torch.bfloat16)
        else:
            pipe = pipeline("text-generation", base_model)
        if load_adapter:
            pipe.model.load_adapter(load_adapter)
        else:
            load_adapter = base_model
    total = 0
    correct_cnt = 0
    fmt_fail_cnt = 0
    mini_sampler = ChatCompletionSampler(model="gpt-4o-mini-2024-07-18")

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
        jsonl_path = log_dir / jsonl_filename
        print(jsonl_path)

    few_shot_text = ''
    if few_shot > 0:
        for idx, row in enumerate(dataset['train']):
            if idx > (few_shot-1):
                break
            few_shot_text += '\nProblem: {}\nAnswer: {}'.format(row['problem'], row['solution'])
    few_shot_text = few_shot_text.strip()

    with tqdm(total=len(dataset['test']), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset['test']):
            if idx in added:
                pbar.update(1)
                continue

            problem = row['problem']
            test_question = QUERY_TEMPLATE.format(Question=problem)
            # test_question = "\nProblem: " + problem + "\n" + "Answer: $"
            # input_text = train_prompt
            # if few_shot > 0:
            #     input_text += '\nExamples:\n'+few_shot_text+'\n'
            #     input_text += 'Now answer this question:\n'
            # input_text += test_question
            if use_sglang:
                generation, _ = pipe(test_question, max_tokens=512, temperature=0.0)
            else:
                messages = [
                    {"role": "user", "content": test_question},
                ]
                generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
            match = re.search(ANSWER_PATTERN, generation)
            extracted_answer = match.group(1) if match else math_normalizer(generation)
            gold = math_normalizer(row['solution'])
            correct = float(check_equality(mini_sampler, gold, extracted_answer))
            log_entry = {
                "id": idx,
                "prompt": test_question,
                "generation": generation,
                "passed": correct,
                "pred": extracted_answer,
                "gold": gold
            }
            correct_cnt += correct
            total += 1
            with open(jsonl_path, "a") as fout:
                fout.write(json.dumps(log_entry)+'\n')

            pbar.update(1)
            pbar.set_description("[pass={:.2f}, {}]".format(100*correct_cnt/total, len(test_question)))

    print(correct_cnt/total)
    # Write final pass score to a text file
    final_score = correct_cnt / total
    score_filename = log_dir / "final_scores.txt"
    with open(score_filename, "a") as f:
        f.write(f"{jsonl_filename}: {final_score:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with an optional adapter.")
    parser.add_argument("--load_adapter", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--base_model", type=str, help="Path to the adapter file to load", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--shots", type=int, help="Path to the adapter file to load", default=0)
    parser.add_argument("--resume", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--torch_dtype", type=str, help="torch dtype for model", default=None)
    parser.add_argument("--run_val",  dest='run_val', action='store_true', help="run validation split", default=False)
    parser.add_argument("--use_sglang", action="store_true", help="Use SGLang inference engine")
    args = parser.parse_args()
    main(args.load_adapter, args.shots, args.resume, args.base_model,
        torch_dtype=args.torch_dtype,
        run_val=args.run_val,
        use_sglang=args.use_sglang
    )
