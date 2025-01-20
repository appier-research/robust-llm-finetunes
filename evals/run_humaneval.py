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
from openai import OpenAI
    

from .humaneval_utils.executor import check_correctness

def create_unique_filename(base_name, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def extract_code(text):
    pattern = r'```(?:python)?\s*(.*?)```?'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    return code_blocks


def find_code(completion):
    # https://github.com/openai/simple-evals/blob/main/humaneval_eval.py
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion
    extracted_answer = extracted_answer[
        extracted_answer.find(":\n    ") + 2 :
    ]  # remove signature
    if '```' in extracted_answer:
        extracted_answer = extracted_answer.split('```', maxsplit=1)[0]
    return extracted_answer

def instruct_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation, make sure its enclose in this:\n```python\n<code completion here>\n```\nCode:\n\n{prompt}\n\n### Response:"""

def oai_template(prompt: str) -> str:
    instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"
    return instruction + prompt

def extract_after_exec_content(code_context: str) -> str:
    return re.findall(pattern=r'exec_context.*"""', string=code_context, flags=re.DOTALL)

def use_custom_gpt4(messages, model_name):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'] , organization=os.environ['OPENAI_ORGANIZATION'])

    response = client.chat.completions.create(
      model=model_name,
      messages=messages,
      temperature=1,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    ).choices[0].message.content
    return response


def main(load_adapter, base_model="unsloth/llama-3-8b-Instruct-bnb-4bit", torch_dtype=None, sanity_check=False, template='none'):
    assert template in ('none', 'simple', 'openai')

    log_dir = Path("logging")
    log_dir.mkdir(exist_ok=True)
    log_dir = Path("logging/humaneval-test")
    log_dir.mkdir(exist_ok=True)

    dataset = load_dataset("evalplus/humanevalplus", split="test")
    oai_dataset = load_dataset("openai/openai_humaneval", split="test")

    if sanity_check:
        total = 0
        oai_correct_cnt = 0
        correct_cnt = 0
        with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
            for idx, row in enumerate(dataset):
                oai_row = oai_dataset[idx]
                prompt = row['prompt']
                match = row['canonical_solution']
                result = check_correctness(row, match, timeout=10, completion_id=idx)
                if result['result'] == 'passed':
                    correct_cnt += 1
                    correct = 1
                result = check_correctness(oai_row, match, timeout=10, completion_id=idx) 
                if result['result'] == 'passed':
                    oai_correct_cnt += 1
                total += 1
                pbar.update(1)
                pbar.set_description("[pass={:.2f}, oai-pass={:.2f}]".format(100*correct_cnt/total, 100*oai_correct_cnt/total))
        return None
    if 'gpt' in base_model:
        load_adapter = "openai_"+base_model
    else:
        if torch_dtype == 'bf16':
            pipe = pipeline("text-generation", base_model, device_map='auto', torch_dtype=torch.bfloat16)
        else:
            pipe = pipeline("text-generation", base_model)
        if load_adapter:
            pipe.model.load_adapter(load_adapter)
    total = 0
    oai_correct_cnt = 0
    correct_cnt = 0
    adapter_name = load_adapter.replace('outputs/', '').replace('/','-').replace('./','').replace('0.', '0-') if load_adapter else None
    jsonl_filename = create_unique_filename(Path(adapter_name).stem if load_adapter else base_model.split('/')[-1], "jsonl")
    jsonl_filename = 'temp-'+template+'_'+jsonl_filename
    if torch_dtype == 'bf16':
        jsonl_filename = 'bf16_'+jsonl_filename
    jsonl_path = log_dir / jsonl_filename
    print(jsonl_path)
    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            oai_row = oai_dataset[idx]
            # input_text = row['prompt']
            prompt = row['prompt']
            if template == 'none':
                input_text = prompt
            elif template == 'simple':
                input_text = instruct_prompt(prompt)
            elif template == 'openai':
                input_text = oai_template(prompt)
            messages = [
                # {"role": "system", "content": "You are an expert program assistant"},
                {"role": "user", "content": input_text},
            ]
            # if 'mistral' not in base_model:
            #     messages = [{"role": "system", "content": "You are an expert program assistant"}]+messages
            if 'gpt' in base_model:
                generation = use_custom_gpt4(messages, base_model)
            else:
                generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']

            match = find_code(generation)

            final_output = {}
            # print(match)
            # print('--------')
            # print(row['canonical_solution'])
            result = check_correctness(row, match, timeout=10, completion_id=idx)
            final_output['humanevalplus'] = result
            correct = 0
            if result['result'] == 'passed':
                correct_cnt += 1
                correct = 1
            

            result = check_correctness(oai_row, match, timeout=10, completion_id=idx) 
            final_output['humaneval'] = result
            if result['result'] == 'passed':
                oai_correct_cnt += 1
            total += 1

            pbar.update(1)
            pbar.set_description("[pass={:.2f}, oai-pass={:.2f}]".format(100*correct_cnt/total, 100*oai_correct_cnt/total))
            log_entry = {
                "prompt": input_text,
                "generation": generation,
                "passed": correct,
                "oai_passed": result['result'] == 'passed',
                "final_output": final_output
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
    parser.add_argument("--base_model", type=str, help="Path to the adapter file to load", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--torch_dtype", type=str, help="torch dtype for model", default=None)
    parser.add_argument("--env_sanity_check",  dest='env_sanity_check', action='store_true', help="torch dtype for model", default=False)
    parser.add_argument("--template", type=str, help='Prompt template to validate result', choices=['none', 'simple', 'openai'], default='openai')
    
    args = parser.parse_args()
    main(args.load_adapter, args.base_model, args.torch_dtype,
        sanity_check=args.env_sanity_check,
        template=args.template
    )
