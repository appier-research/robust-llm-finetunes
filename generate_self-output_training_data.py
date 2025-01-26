import os
import re
import torch
from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path
import datetime
import argparse
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from utils import calculate_perplexity
from utils import normalize_extracted_answer, normalize_response
from typing import Dict, List, Union
from openai import OpenAI
import time

PROMPT_TEMPLATE = """Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your own response.
### Instruction:
{instruction}
### Reference Answer:
{original_response}
### Response:"""

def gen_mbpp_sample(args, pipe, base_name, temp=0.7, samples=32, split="train"):
    from evals.humaneval_utils.executor import check_correctness
    def extract_code(text):
        pattern = r'```(?:python)?\s*(.*?)```?'
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return code_blocks

    def extract_after_exec_content(code_context: str) -> str:
        return re.findall(pattern=r'exec_context.*"""', string=code_context, flags=re.DOTALL)

    print(base_name)
    base_name = base_name.replace('/','-')
    output_filename = f'dataset/{args.mode}/mbpp_{base_name}_sample_{split}_t_{temp}_{samples}.jsonl'
    added = set()
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                payload = json.loads(line)
                added.add(payload['row_id'])
    dataset = "dataset/ground_truth/mbpp"
    dataset = load_dataset("arrow", data_files={f"{split}":os.path.join(dataset,f"{split}/data-00000-of-00001.arrow")})[split]
    print("MBPP", split)
    for row_id, row in tqdm(enumerate(dataset)):
        if row_id in added:
            continue
        # print(row)
        prompt = PROMPT_TEMPLATE.format(instruction=row['conversations'][0]['value'], original_response=row['conversations'][1]['value'])
        negative_samples = [ row['conversations'][1]['value'] ]
        positive_samples = []
        if args.mode=='self-output':
            messages = [{ 'role': 'user', 'content': row['conversations'][0]['value'] } ]
        else:
            messages = [{ 'role': 'user', 'content': prompt } ]
        response = row['conversations'][1]['value']
        gt_answer = {'content': row['conversations'][1]['value'], 'src': 'ground_truth', 'correct': True, 'order': -1 }
        
        if pipe:
            generations = pipe([messages]*samples, max_new_tokens=512,
                top_p=0.95,
                temperature=temp,
                do_sample=True
            )
        # print(generations)
        for idx, gen in enumerate(generations):
            if pipe:
                generation = gen[0]['generated_text'][-1]['content']
            # generation = gen
            print(gen)
            failed = False
            try:
                match = extract_code(generation)[0].rstrip()
            except Exception as e:
                match = generation.rstrip()
                failed = True

            if failed:
                sample = {
                    'content': generation,
                    'src': base_name,
                    'correct': False,
                    'order': idx,
                    'result': 'extraction_failed'
                }
                negative_samples.append(sample)
            else:
                row['test'] = '\n'.join(row['test_list'])
                result = check_correctness(row, match, timeout=10, completion_id=row_id)
                sample = {
                    'content': generation,
                    'src': base_name,
                    'correct': result['result'] == 'passed',
                    'order': idx,
                    'result': json.dumps(result)
                }
                if result['result'] == 'passed':
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)
        # we do not found correct answer, use original prompt as ground truth
        if len(positive_samples) == 0:
            positive_samples = [ gt_answer ]
        else:
            positive_samples.append(gt_answer)        
        row = {
            'instruction': row['conversations'][0]['value'],
            'accept': positive_samples,
            'rejection': negative_samples,
            'row_id': row_id
        }
        with open(output_filename, 'a') as f:
            f.write(json.dumps(row)+'\n')
    return output_filename

def groq_math_sample(args, pipe, base_name, temp=0.7, split='train', samples=16):
    from evals.normalization import math_normalizer
    from evals.openai_sampler import ChatCompletionSampler
    from evals.common import check_equality
    base_postfix = base_name.split("/")[-1]

    QUERY_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

    print('MATH')
    dataset = "dataset/ground_truth/math"
    dataset = load_dataset("arrow", data_files={f"{split}":os.path.join(dataset,f"{split}/data-00000-of-00001.arrow")})[split]
    output_filename = f'dataset/{args.mode}/math-t2_{base_postfix}_dpo_sample_{split}_t_{temp}.jsonl'
    added = set()
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                payload = json.loads(line)
                added.add(payload['row_id'])
    print(len(added))
    mini_sampler = ChatCompletionSampler(model="gpt-4o-mini-2024-07-18")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True,
                device_map="cuda"
            )
    for row_id, row in tqdm(enumerate(dataset)):
        if row_id in added:
            continue
        problem = row['problem']
        if args.mode=='self-output':
            test_question = QUERY_TEMPLATE.format(Question=problem)
        else:
            test_question = PROMPT_TEMPLATE.format(instruction=row['conversations'][0]['value'], original_response=row['conversations'][1]['value'])

        gt_answer = {
            'content': row['conversations'][1]['value'],
            'src': 'ground_truth',
            'correct': True,
            'order': -1
        }
        negative_samples = []
        positive_samples = []
        messages = [{ 'role': 'user', 'content': test_question } ]
        response = row['conversations'][1]['value']
        idx = 0
        text_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sub_size = 8
        for _ in range(sub_size):
            model_inputs = tokenizer([text_chat]*(samples//sub_size), return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=temp
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for idx, gen in enumerate(generations):
                # use this if its pipe
                # pred_response = gen[0]['generated_text'][-1]['content']
                
                pred_response = gen
                pred = math_normalizer(pred_response)
                gold = math_normalizer(row['solution'])
                correct = float(check_equality(mini_sampler, gold, pred))
                sample = {
                    'content': pred_response,
                    'src': base_model,
                    'correct': correct,
                    'order': idx
                }
                if correct > 0:
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)

        # we do not found correct answer, use original prompt as ground truth
        if len(positive_samples) == 0:
            positive_samples = [ gt_answer ]
            negative_samples = negative_samples
        else:
            positive_samples.append(gt_answer)
        row = {
            'instruction': row['conversations'][0]['value'],
            'accept': positive_samples,
            'rejection': negative_samples,
            'row_id': row_id
        }
        with open(output_filename, 'a') as f:
            f.write(json.dumps(row)+'\n')
    return output_filename

def cal_ppl_file(args, file= "test.jsonl", base_model=""):
    output_filename= file.replace('.jsonl', '_wrejppl.jsonl')
    print(base_model, file, output_filename)
    train = pd.read_json(file, lines=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map='cuda')
    for ind,row in tqdm(train.iterrows()):
        row = row.to_dict()
        accept =[]
        REJECT =[]
        if args.rejection!="None":
            REJECT = row['rejection']
            if len(REJECT)==0 or (len(REJECT)==1 and  ('src' in REJECT[0] and REJECT[0]['src']=='ground_truth')):
                continue
            for text in REJECT:
                if type(text)==dict and 'content' in text:
                    text['ppl'] = calculate_perplexity(text['content'], model, tokenizer)
                else:
                    continue
        accept = row['accept']
       
        if len(accept)==0 or (len(accept)==1 and  ('src' in accept[0] and accept[0]['src']=='ground_truth')):
            continue
        for text in accept:
            if type(text)==dict and 'content' in text:
                text['ppl'] = calculate_perplexity(text['content'], model, tokenizer)
            else:
                continue
        if len(accept)!=0:
            row['accept'] = accept
        if len(REJECT)!=0:
            row['rejection']=REJECT
        with open(output_filename, 'a') as f:
            f.write(json.dumps(row)+'\n')
def main(args):
    start = time.time()
    base_model = args.base_model
    task =args.task
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if args.mode=='self-output':
        nsamples=32
    else:
        nsamples=1
    if task=='mbpp':
        pipe = pipeline("text-generation",
                    base_model,
                    device_map='auto',
                    torch_dtype=torch.bfloat16
                    )
        train_file = gen_mbpp_sample(args, pipe, base_model, temp=0.7, samples=nsamples, split="train")
        val_file = gen_mbpp_sample(args, pipe, base_model, temp=0.7, samples=nsamples, split="validation")
    elif task=='math':
        pipe = None
        train_file = groq_math_sample(args, pipe, base_model, temp=0.7, samples=nsamples, split="train")
        val_file = groq_math_sample(args, pipe, base_model, temp=0.7, samples=nsamples, split="validation")
    print('finish self-generating, elasped time:', time.time()-start)
    start =time.time()
    for file in [train_file, val_file]:
        cal_ppl_file(args, file, base_model)

    def lowest(l):
        return 0
    def rand(l):
        import random
        return random.choice(range(l))
    pick = lowest
    new_train = []
    new_val = []
    pick_first = 0
    pick_last = 0
    train_gt = 0
    val_gt = 0
    avg_diff = []
    avg_len = []
    rej_rate = []
    base_model_name = base_model.replace('/','-').replace('.','-').lower()
    path = f'dataset/{args.mode}/{task}/{base_model_name}-lowest-ppl'
    if args.rejection == 'incorrect':
        path +='_incorrect'
    elif args.rejection == 'lowest_all_correct': 
        path +='_correct'
    elif args.rejection == 'random':
        path += '_correct_random'
    neg_case = 0
    avg_ppl =[]
    train = pd.read_json(train_file.replace('.jsonl','_wrejppl.jsonl'), lines=True)
    val = pd.read_json(val_file.replace('.jsonl','_wrejppl.jsonl'), lines=True)
    for ind, rephrase in train.iterrows():
        
        rephrase['rejection'] = [i for i in rephrase['rejection'] if type(i)is dict and i['src']!='ground_truth']
        rephrase['accept'] = [i for i in rephrase['accept'] if type(i)is dict and i['src']!='ground_truth']

        if args.rejection == 'incorrect':
            sorted_pool = sorted(rephrase['rejection'], key=lambda x: x["ppl"], reverse=False)
        elif args.rejection == 'lowest_all_correct': 
            sorted_pool = sorted(rephrase['accept'], key=lambda x: x["ppl"], reverse=False)
        elif args.rejection == 'random':
            sorted_pool = sorted(rephrase['accept']+rephrase['rejection'], key=lambda x: x["ppl"], reverse=False)
        else:
            sorted_pool = sorted(rephrase['accept'], key=lambda x: x["ppl"], reverse=False)
        leng = len(sorted_pool)
        if args.rejection!="None" and (len(rephrase['rejection'])==0 or len(rephrase['accept'])==0):
            neg_case+=1
            continue
        accept = sorted_pool[pick(leng)]
        
        if accept['src']=='ground_truth':
            train_gt+=1
            continue
        if pick(leng)==0:
            pick_first+=1
        elif pick(leng)==leng-1:
            pick_last+=1
        if not accept['correct'] or accept['correct'] =='False':
            rej_rate.append(1)
        else:
            rej_rate.append(0)
        avg_diff.append(accept['ppl'] - sorted_pool[0]['ppl'])
        avg_ppl.append(accept['ppl'])
        avg_len.append(len(tokenizer.encode(accept['content'], add_special_tokens=False)))
        conversations=[
            {"role": "user", "content": rephrase['instruction']},
            {"role": "assistant", "content": accept['content']}
        ]
        new_train.append({"conversations": conversations})
    print("number of training  instances: ",len(new_train), '\nnumber of average sentence ppl: ', sum(avg_ppl)/len(avg_ppl),'\naverage sentence tokens length: ', sum(avg_len)/len(avg_len))
    print('Correctness: ',sum(rej_rate)/len(rej_rate))
    pick_first = 0
    pick_last = 0
    avg_ppl = []
    for ind, rephrase in val.iterrows():

        rephrase['accept'] = [i for i in rephrase['accept']if type(i)is dict and  i['src']!='ground_truth']
        sorted_pool = sorted(rephrase['accept'], key=lambda x: x["ppl"], reverse=False)
        leng = len(rephrase['accept'])
        if leng==0:
            neg_case+=1
            continue
        accept = sorted_pool[pick(leng)]
        if accept['src']=='ground_truth':
            val_gt+=1
            continue
        if pick(leng)==0:
            pick_first+=1
        elif pick(leng)==leng-1:
            pick_last+=1
        avg_diff.append(accept['ppl'] - sorted_pool[0]['ppl'])
        avg_ppl.append(accept['ppl'])
        conversations=[
            {"role": "user", "content": rephrase['instruction']},
            {"role": "assistant", "content": accept['content']}
        ]
        new_val.append({"conversations": conversations})
    # print(len(new_val), pick_first, pick_last, val_gt, sum(avg_diff)/len(avg_diff), sum(avg_ppl)/len(avg_ppl))
    from datasets import Dataset
    dataset = load_dataset("arrow", data_files={"train": f"dataset/ground_truth/{task}/train/data-00000-of-00001.arrow", "validation": f"dataset/ground_truth/{task}/validation/data-00000-of-00001.arrow", "test": f"dataset/ground_truth/{task}/test/data-00000-of-00001.arrow"})
    dataset['train'] = Dataset.from_list(new_train)
    dataset['validation'] = Dataset.from_list(new_val)
    dataset.save_to_disk(path)

    print("Self-output data generation done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with an optional adapter.")
    parser.add_argument("--mode", type=str, help="mode for self-output or rephrase", default="self-output", choices=["self-output", "rephrase"])
    parser.add_argument("--base_model", type=str, help="Path to the model to load", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--task", type=str, help="Path to the model to load", default="mbpp", choices=['mbpp', 'math'])
    parser.add_argument("--rejection", type=str, help="Path to the model to load", default="None", choices = ["None", "lowest_all_correct", "incorrect", "random"])
    args = parser.parse_args()
    
    main(args)

