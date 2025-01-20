import os
# os.environ['HF_HOME'] = '/tmp/hf_cache'
# os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
import re
import torch
from tqdm import tqdm
import json
from pathlib import Path
import datetime
import argparse
from datasets import load_dataset
from transformers import pipeline
from .ifeval_utils.evaluation_main import check_correctness

def create_unique_filename(base_name, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


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


def main(load_adapter, resume=None, base_model=None, torch_dtype=None):
    log_dir = Path("logging/ifeval")
    log_dir.mkdir(exist_ok=True)
    dataset = load_dataset("appier-ai-research/robust-finetuning","instruction_following_eval", split="train")
    # dataset = load_dataset("xlangai/DS-1000", split="test", token=os.environ["HF_TOKEN"])
    # dataset = load_dataset("appier-ai-research/convo_ds1000", split="validation", token=os.environ["HF_TOKEN"])
    if 'gpt' in base_model:
        load_adapter = "openai_"+base_model
    else:
        if torch_dtype=='bf16':
            pipe = pipeline("text-generation", base_model, device_map='auto', torch_dtype=torch.bfloat16)
        else:
            pipe = pipeline("text-generation", base_model)
        if load_adapter:
            pipe.model.load_adapter(load_adapter)
    total = 0
    oai_correct_cnt = 0
    correct_cnt = 0
    prefix = Path(load_adapter.replace('outputs/', '').replace('/','-')).stem if load_adapter else "default"
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
        jsonl_filename = create_unique_filename(Path(adapter_name).stem if load_adapter else base_model.split('/')[-1], "jsonl")
        jsonl_path = log_dir / jsonl_filename
        print(jsonl_path)
    results = []
    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            if idx in added:
                pbar.update(1)
                continue

            prompt = row['prompt']
            input_text = prompt#instruct_prompt(prompt)

            messages = [
                {"role": "user", "content": prompt}
            ]
            if 'mistral' not in base_model:
                messages = [{"role": "system", "content": "You are a helpful assistant."}]+messages
            if 'gpt' in base_model:
                generation = use_custom_gpt4(messages, base_model)
            else:
                generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
            results= {"prompt": input_text, "response": generation}

            result = check_correctness(row, results)
            print(result)
            final_output = result
            if result[-1]==1:
                correct_cnt += 1
            total += 1

            pbar.update(1)
            pbar.set_description("[pass={:.2f}, oai-pass={:.2f}]".format(100*correct_cnt/total, 100*oai_correct_cnt/total))
            log_entry = {
                "id": idx,
                "prompt": input_text,
                "generation": generation,
                "passed": result[-1]==1,
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
    parser.add_argument("--resume", type=str, help="Path to the adapter file to load", default=None)
    parser.add_argument("--base_model", type=str, help="Path to the adapter file to load", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--torch_dtype", type=str, help="torch dtype for model", default=None)
    
    args = parser.parse_args()
    main(args.load_adapter, args.resume, args.base_model, torch_dtype=args.torch_dtype)