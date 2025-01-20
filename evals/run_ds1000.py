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
from .ds1000_utils import execution
from datasets import load_dataset
from transformers import pipeline

def extract_after_exec_content(code_context: str) -> str:
    return re.findall(pattern=r'exec_context.*"""', string=code_context, flags=re.DOTALL)


prompt = """
Here is the user's requirements for solving a programming problem (enclosed in '''):
'''
{question}
'''

You need to provide your solution in python code to satisfy the user's requirements. Your code will be tested as follows (enclosed in '''):
'''
{exec_context}

code = exec_context.replace("[insert]", <your_code>)
a_test_case = generate_test_case()
test_input, expected_result = a_test_case
test_env = {{"test_input": test_input}}
exec(code, test_env)
assertEqual(test_env["result"], expected_result)
'''

Now, generate your code directly in the following format:
```python
<your_code>
```
"""

def create_unique_filename(base_name, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def main(load_adapter, resume=None, base_model=None, torch_dtype=None, sanity_check=False):
    log_dir = Path("logging")
    log_dir.mkdir(exist_ok=True)
    dataset = load_dataset("appier-ai-research/test_ds1000", split="test", token=os.environ["HF_TOKEN"])
    # dataset = load_dataset("xlangai/DS-1000", split="test", token=os.environ["HF_TOKEN"])
    # dataset = load_dataset("appier-ai-research/convo_ds1000", split="validation", token=os.environ["HF_TOKEN"])
    if sanity_check:
        total = 0
        oai_correct_cnt = 0
        correct_cnt = 0
        with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
            for idx, row in enumerate(dataset):
                match = row['reference_code']
                completion_id = 0
                code_context = row['code_context']
                test_program = (
                    "import matplotlib\nmatplotlib.use('agg')\n" + code_context + '\n'
                    + f'code = {repr(match)}\n'
                    + 'test_execution(code)\n'
                    + ('test_string(code)\n'  if 'test_string(' in code_context  else '\n')
                )
                results = execution.check_correctness_single_proc(
                    test_program,
                    timeout=128,
                    completion_id=completion_id
                )
                correct = results['passed']
                if correct:
                    correct_cnt += 1
                total += 1
                pbar.update(1)
                pbar.set_description("[pass={:.2f}]".format(100*correct_cnt/total))
        return None


    if torch_dtype=='bf16':
        pipe = pipeline("text-generation", base_model, device_map='auto', torch_dtype=torch.bfloat16)
    else:
        pipe = pipeline("text-generation", base_model)
    if load_adapter:
        pipe.model.load_adapter(load_adapter)
    else:
        load_adapter = base_model
    total = 0
    correct_cnt = 0
    added = set()
    if resume is not None and os.path.exists(resume):
        jsonl_path = resume
        with open(resume, 'r') as f:
            for idx, line in enumerate(f):
                payload = json.loads(line)
                if payload['passed']:
                    correct_cnt += 1
                total += 1
                added.add(idx)
        print('resume from {}/{}'.format(correct_cnt, total))
        jsonl_filename = resume.replace(str(log_dir)+'/','')
    else:
        adapter_name = load_adapter.replace('outputs/', '').replace('/','-').replace('./','').replace('0.', '0-')
        jsonl_filename = create_unique_filename(Path(adapter_name).stem if load_adapter else base_model.split('/')[-1], "jsonl")
        if torch_dtype == 'bf16':
            jsonl_filename = 'bf16_'+jsonl_filename
        jsonl_path = log_dir / jsonl_filename
        print(jsonl_path)

    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            if idx in added:
                pbar.update(1)
                continue

            input_text = prompt.format(
                        question=row['prompt'].strip(),
                        exec_context=extract_after_exec_content(row['code_context'])
                    ).lstrip()
            # input_text = row['conversations'][0]['value']
            if len(input_text) > 4096:
                continue
            if total > 0:
                pbar.set_description("[pass={:.2f}, {}]".format(100*correct_cnt/total, len(input_text)))
            messages = [
                {"role": "user", "content": input_text},
            ]
            generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
            try:
                match = re.search(
                    pattern=r"```python(.*?)```",
                    string=generation,
                    flags=re.DOTALL
                ).group(1).strip()
            except Exception as e:
                match = generation
            completion_id = 0
            code_context = row['code_context']
            test_program = (
                "import matplotlib\nmatplotlib.use('agg')\n" + code_context + '\n'
                + f'code = {repr(match)}\n'
                + 'test_execution(code)\n'
                + ('test_string(code)\n'  if 'test_string(' in code_context  else '\n')
            )
            results = execution.check_correctness_single_proc(
                test_program,
                timeout=128,
                completion_id=completion_id
            )
            correct = results['passed']
            if correct:
                correct_cnt += 1
            total += 1
            pbar.update(1)
            pbar.set_description("[pass={:.2f}, {}]".format(100*correct_cnt/total, len(input_text)))
            # Log iteration data to JSONL file
            log_entry = {
                "prompt": input_text,
                "generation": generation,
                "passed": correct
            }
            with open(jsonl_path, "a") as fout:
                fout.write(json.dumps(log_entry)+'\n')

    print(correct_cnt/total)
    # Write final pass score to a text file
    final_score = correct_cnt / total
    score_filename = log_dir / "final_scores.txt"
    with open(score_filename, "a") as f:
        f.write(f"{jsonl_filename}: {final_score:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with an optional adapter.")
    parser.add_argument("--load_adapter", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--resume", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--base_model", type=str, help="Path to the adapter file to load", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--torch_dtype", type=str, help="torch dtype for model", default=None)
    parser.add_argument("--env_sanity_check",  dest='env_sanity_check', action='store_true', help="torch dtype for model", default=False)
    
    args = parser.parse_args()
    main(args.load_adapter, args.resume, args.base_model, torch_dtype=args.torch_dtype, sanity_check=args.env_sanity_check)