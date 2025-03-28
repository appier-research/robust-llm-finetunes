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
from .utils import normalize_extracted_answer, normalize_response
from openai import OpenAI

def check_correctness(a, res):
    answerKey = a[1]
    generation = res
    failed = False
    correct = False
    # print(generation)
    if 'Answer: ' not in generation.strip():
        generation = 'Answer: ' + generation
    # print(generation)
    normalized_response = normalize_response(generation)
    # print(normalized_response)
    match = re.search(r"Answer\s*:\s*(\w+)", normalized_response)
    # print(match)
    if match:
        extracted_answer = normalize_extracted_answer(match.group(1))
        correct = extracted_answer == answerKey or extracted_answer == a[0]
    elif len(generation)==1:
        correct = generation==answerKey
        
     
    return correct
    #if ans[0].strip()!='' and ans[0].lower().strip() == res.lower().strip():
    #    return [1]
    #elif ans[0].strip()!='' and ans[0].strip().lower() != res.lower().strip():
    #    return [ans[1] in res]
    #else:
    #    return [0]

def templated(p):
    # print(p)
    question, choices, answerKey = p["question"], p["choices"], p["answerKey"]
    
    choice = [l+'. '+t for t, l in zip(choices['text'],choices['label'])]
    # print(choice, answerKey)
    q = 'Question:\n'+question
    c = "Choices:\n"+'\n'.join(choice)
    a = choice[ord(answerKey)-ord(choices['label'][0])]
    question_text=q+c
    messages = [
            { 'role': 'user', 'content': "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n"+question_text+"\n" }
        ]
    #remove "Think step by step before answering." before question text
    pair = {
        "conversations": 
            #[
            #    {"role": "system", "content": "Please read the question and choices, and then choose one correct answer option from the choices."},
            #    {"role": "user", "content": q+c }
            #]
        messages
    }
    return pair, (a, answerKey)

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


def main(load_adapter, resume=None, base_model=None, torch_dtype=None, sanity_check=False, run_val=False):
    dataset = load_dataset("arrow", data_files={"train":"dataset/ground_truth/arc_challenge/train/data-00000-of-00001.arrow","validation":"dataset/ground_truth/arc_challenge/validation/data-00000-of-00001.arrow","test":"dataset/ground_truth/arc_challenge/test/data-00000-of-00001.arrow"})
    if run_val:
        log_dir = Path("logging/arc-val")
        log_dir.mkdir(exist_ok=True)
        dataset = dataset['validation']
    else:
        log_dir = Path("logging/arc-test")
        log_dir.mkdir(exist_ok=True)
        dataset = dataset['test']
    if sanity_check:
        total = 0
        oai_correct_cnt = 0
        correct_cnt = 0
        for split in ['validation','test']: #train
            dataset_ = dataset[split]
            print(split)
            with tqdm(total=len(dataset_), dynamic_ncols=True) as pbar:
                for idx, row in enumerate(dataset_):
                    _, answers = templated(row)
                    result = check_correctness(answers, "Answer: "+row["answerKey"])
                    if result:
                        correct_cnt += 1
                    total += 1
    
                    pbar.update(1)
                    pbar.set_description("[pass={:.2f}]".format(100*correct_cnt/total))
        return None
    # dataset = dataset['test']
    if 'gpt' in base_model:
        load_adapter = "openai_"+base_model
    else:
        if torch_dtype=='bf16':
            pipe = pipeline("text-generation", base_model, device_map='auto', torch_dtype=torch.bfloat16)
        else:
            pipe = pipeline("text-generation", base_model,device_map='auto', torch_dtype=torch.bfloat16 )
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
        if load_adapter:
            adapter_name = load_adapter.replace('outputs/', '').replace('/','-').replace('./','').replace('0.', '0-')
        jsonl_filename = create_unique_filename(Path(adapter_name).stem if load_adapter else base_model.split('/')[-1], "jsonl")
        if pipe and torch_dtype == 'bf16':
            jsonl_filename = 'bf16_'+jsonl_filename
        jsonl_path = log_dir / jsonl_filename
        print(jsonl_path)
    results = []
    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            if idx in added:
                pbar.update(1)
                continue

            prompt, answers = templated(row)

            messages = prompt['conversations']
            # if 'mistral' in base_model:
            #     system_prompt = messages[0]['content']
            #     messages = [ {"role": "user", "content": system_prompt+'\n'+messages[1]['content'] }]
                # from transformers import AutoTokenizer
                # tokenizer = AutoTokenizer.from_pretrained(base_model)
                # messages = tokenizer.apply_chat_template(messages, tokenize=False)
            input_text = messages[-1]['content']
            if 'gpt' in base_model:
                generation = use_custom_gpt4(messages, base_model)
            else:
                generation = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
            results= {"prompt": input_text, "response": generation}
            print('[GEN]',generation, answers)
            print(generation)
            result = check_correctness(answers, generation)
            print(result)
            final_output = result
            if result:
                correct_cnt += 1
            total += 1

            pbar.update(1)
            pbar.set_description("[pass={:.2f}, oai-pass={:.2f}]".format(100*correct_cnt/total, 100*oai_correct_cnt/total))
            log_entry = {
                "id": idx,
                "prompt": input_text,
                "generation": generation,
                "passed": result,
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
    parser.add_argument("--env_sanity_check",  dest='env_sanity_check', action='store_true', help="check validation or testing set", default=False)
    parser.add_argument("--run_val",  dest='run_val', action='store_true', help="do validation accuracy", default=False)
    

    args = parser.parse_args()
    main(args.load_adapter, args.resume, args.base_model, torch_dtype=args.torch_dtype, sanity_check=args.env_sanity_check, run_val=args.run_val)
