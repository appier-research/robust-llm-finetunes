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
from .humaneval_utils.executor import check_correctness
prefix_instruct = "Please refer the given test cases and generate a python function for my problem. Make sure the written code is wrapped in code block : ```python\n<your code>\n```\n"

def create_unique_filename(base_name, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def extract_code(text):
    pattern = r'```(?:python)?\s*(.*?)```?'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks[:len('python')] == 'python':
        return code_blocks[len('python')+1:]
    return code_blocks


def extract_after_exec_content(code_context: str) -> str:
    return re.findall(pattern=r'exec_context.*"""', string=code_context, flags=re.DOTALL)


def validate_checkpoint(load_adapter, 
        base_model="unsloth/llama-3-8b-Instruct-bnb-4bit", 
        torch_dtype=None
    ):
    import glob
    dataset = load_dataset("arrow", data_files={"validation":"dataset/ground_truth/mbpp/validation/data-00000-of-00001.arrow"}, split="validation" )

    checkpoint2scores = {}
    for checkpoint in glob.glob(load_adapter):
        # prevent acc of prev checkpoint
        if torch_dtype == 'bf16':
            pipe = pipeline("text-generation", base_model,
                            device_map='cuda', torch_dtype=torch.bfloat16)
        else:
            pipe = pipeline("text-generation", base_model, device_map='cuda')
        print(checkpoint)
        pipe.model.load_adapter(checkpoint)
        total = 0
        correct_cnt = 0
        for idx, row in enumerate(dataset):
            prompt = row['text']
            q = row['text']
            tests = row['test_list']
            input_text = prefix_instruct+">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
            messages = [
                {"role": "user", "content": input_text},
            ]
            generation = pipe(messages, max_new_tokens=1024)[0]['generated_text'][-1]['content']
            try:
                match = extract_code(generation)[0].rstrip()
            except Exception as e:
                match = generation.rstrip()
            if match[:len('python')] == 'python':
                match = match[len('python')+1:]
            result = check_correctness(row, match, timeout=10, completion_id=idx)
            final_output = result
            if result['result'] == 'passed':
                correct_cnt += 1
            total += 1

        checkpoint2scores[checkpoint] = correct_cnt / total
        del pipe
    print(base_model)
    for ckpt, pass_score in checkpoint2scores.items():
        print(ckpt, pass_score)


def main(load_adapter, base_model="unsloth/llama-3-8b-Instruct-bnb-4bit", 
    resume=None, torch_dtype=None, sanity_check=False, run_val=False):

    if run_val and (load_adapter and '*' in load_adapter):
        validate_checkpoint(load_adapter, base_model=base_model, torch_dtype=torch_dtype)
        return 2

    log_dir = Path("logging")
    log_dir.mkdir(exist_ok=True)
    if run_val:
        log_dir = Path("logging/mbpp-validation")
        log_dir.mkdir(exist_ok=True)
    else:
        log_dir = Path("logging/mbpp-test")
        log_dir.mkdir(exist_ok=True)
    if run_val:
        dataset = load_dataset("arrow", data_files={"validation":"dataset/ground_truth/mbpp/validation/data-00000-of-00001.arrow"}, split="validation" )
    else:
        dataset = load_dataset("arrow", data_files={"test":"dataset/ground_truth/mbpp/test/data-00000-of-00001.arrow"}, split="test" )
    
    if sanity_check:
        total = 0
        oai_correct_cnt = 0
        correct_cnt = 0
        for split in ['train','validation','test']:
            dataset = load_dataset("arrow", data_files={split:f"dataset/ground_truth/mbpp/{split}/data-00000-of-00001.arrow"}, split=split )
            print(split)
            with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
                for idx, row in enumerate(dataset):
                    prompt = row['text']
                    q = row['text']
                    tests = row['test_list']
                    result = check_correctness(row, row['code'], timeout=10, completion_id=idx)
                    final_output = result
                    if result['result'] == 'passed':
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
        
        # pipe.resize_token_embeddings(len(tokenizer))
        pipe.model.load_adapter(load_adapter)
    else:
        load_adapter = base_model
    # if 'mistral' in base_model:
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(base_model)
    #     pipe.model.generation_config.pad_token_id = tokenizer.unk_token_id
    total = 0
    oai_correct_cnt = 0
    correct_cnt = 0
    prefix = Path(load_adapter.replace('outputs/', '').replace('/','-')).stem if load_adapter else base_model.split('/')[-1]
    jsonl_filename = create_unique_filename(prefix, "jsonl")
    jsonl_path = log_dir / jsonl_filename
    print(jsonl_path)

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
        jsonl_filename = create_unique_filename(Path(adapter_name).stem if load_adapter else "default", "jsonl")
        if torch_dtype == 'bf16':
            jsonl_filename = 'bf16_'+jsonl_filename
        jsonl_path = log_dir / jsonl_filename
        print(jsonl_path)

    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            if idx in added:
                pbar.update(1)
                continue

            prompt = row['text']
            q = row['text']
            tests = row['test_list']
            input_text = prefix_instruct+">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
            messages = [
                {"role": "user", "content": input_text},
            ]
            # if 'mistral' in base_model:
            #     # system_prompt = messages[0]['content']
            #     # messages = [ {"role": "user", "content": system_prompt+'\n'+messages[1]['content'] }]
            #     from transformers import AutoTokenizer
            #     tokenizer = AutoTokenizer.from_pretrained(base_model)
            #     messages = tokenizer.apply_chat_template(messages, tokenize=False)
            generation = pipe(messages, max_new_tokens=1024)[0]['generated_text'][-1]['content']
            try:
                match = extract_code(generation)[0].rstrip()
            except Exception as e:
                match = generation.rstrip()
            if match[:len('python')] == 'python':
                match = match[len('python')+1:]
            # print('-------')
            # print(row['code'])
            # print(match)
            result = check_correctness(row, match, timeout=10, completion_id=idx)
            final_output = result
            if result['result'] == 'passed':
                correct_cnt += 1
            total += 1

            pbar.update(1)
            pbar.set_description("[pass={:.2f}, oai-pass={:.2f}]".format(100*correct_cnt/total, 100*oai_correct_cnt/total))
            log_entry = {
                "id": idx,
                "prompt": input_text,
                "generation": generation,
                "passed": result['result'] == 'passed',
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
    parser.add_argument("--resume", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--torch_dtype", type=str, help="torch dtype for model", default=None)
    parser.add_argument("--env_sanity_check",  dest='env_sanity_check', action='store_true', help="torch dtype for model", default=False)
    parser.add_argument("--run_val",  dest='run_val', action='store_true', help="run validation split", default=False)
    args = parser.parse_args()
    main(args.load_adapter,
        base_model=args.base_model,
        resume=args.resume,
        torch_dtype=args.torch_dtype,
        sanity_check=args.env_sanity_check,
        run_val=args.run_val
    )
