import os
import json
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import argparse

def calculate_token_metrics_with_surprisal(logits, target_ids):
    """
    Calculate per-token perplexity and entropy
    
    Args:
        logits: Model output logits of shape [prefix_len, vocab_size]
        target_ids: Target token ids
        prefix_len: Length of the sequence to analyze
    
    Returns:
        token_ppls: Per-token perplexity
        token_entropy: Per-token entropy
    """
    # Get the relevant logits and targets
    logits_len = len(logits)
    # logits = logits[prefix_len:, :]
    # targets = target_ids[0, prefix_len:]
    
    # Get logits and shift them to align with targets
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    # Calculate loss per token (using CrossEntropyLoss)
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    ).view(shift_labels.size())
    perplexity_per_token = torch.exp(loss_per_token.float())

    return perplexity_per_token.float().cpu().numpy(), loss_per_token.float().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with an optional adapter.")
    parser.add_argument("--base_model", type=str, help="Path to the model to load", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--task", type=str, help="Task data to generate stm training data", default="mbpp", choices=['mbpp', 'math'])
    parser.add_argument("--stm", type=str, help="Whether or not to apply stm data alternatives", default="None", choices=['None','stm_dpf', 'dpf'])
    parser.add_argument("--stm_adapter", type=str, help="Path to the adapter to load for stm filters", choices=['stm_dpf', 'dpf'])
    args = parser.parse_args()
    model_path = args.base_model
    task = args.task
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda')
    model = model.eval()

    max_length = 2048
    messages_key = 'conversations'
    dataset = load_dataset("arrow", data_files={"train": f"dataset/ground_truth/{task}/train/data-00000-of-00001.arrow"})['train']
    output_file = model_path.replace('/','-').replace('.','-') + '_'+ task
    if args.stm:
        output_file +'_'+args.stm
    data = []
    for example in tqdm(dataset, dynamic_ncols=True):
        prompt = example[messages_key][0]
        prompt = [ {'role': 'user', 'content': prompt['value']} ]
        # prompt = [ {'role': 'user', 'content': prompt['content']} ]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        message = example[messages_key]
        # message = [ 
        #     {'role': 'user', 'content': message[0]['content']},
        #     {'role': 'assistant', 'content': message[1]['content']}
        # ]
        message = [ 
            {'role': 'user', 'content': message[0]['value']},
            {'role': 'assistant', 'content': message[1]['value']}
        ]
        formatted_message = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        tokenized_message = tokenizer(
            formatted_message,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors='pt',
            add_special_tokens=False,
        )
        tokenized_message = tokenized_message.to(model.device)
        with torch.no_grad():
            outputs = model(tokenized_message.input_ids, labels=tokenized_message['input_ids'])
        token_ppls, token_surprisal = calculate_token_metrics_with_surprisal(
            outputs.logits[0], 
            tokenized_message['input_ids']
        )
        token_ppls = token_ppls[0, :].tolist()
        token_ppls = [10] + token_ppls
        example['ppl'] = token_ppls
        example['text'] = formatted_message
        data.append(example)
    if args.stm and args.stm!="None":
        model.load_adapter(args.stm_adapter)
        diff_data = []
    
        for idx, example in enumerate(tqdm(dataset, dynamic_ncols=True)):
            prompt = example[messages_key][0]
            prompt = [ {'role': 'user', 'content': prompt['value']} ]
            formatted_prompt = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            message = example[messages_key]
            message = [ 
                {'role': 'user', 'content': message[0]['value']},
                {'role': 'assistant', 'content': message[1]['value']}
            ]
            formatted_message = tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            )
            tokenized_message = tokenizer(
                formatted_message,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors='pt',
                add_special_tokens=False,
            )
            tokenized_message = tokenized_message.to(model.device)
            with torch.no_grad():
                outputs = model(tokenized_message.input_ids, labels=tokenized_message['input_ids'])
            token_ppls, token_surprisal = calculate_token_metrics_with_surprisal(
                outputs.logits[0], 
                tokenized_message['input_ids']
            )
            token_ppls = token_ppls[0, :].tolist()
            token_ppls = [10] + token_ppls
            token_ppls = np.array(token_ppls)
            ref_ppl = np.array(data[idx]['ppl'])
            diff_ppl = token_ppls - ref_ppl
            diff_ppl[0] = 10
            if args.stm=='stm_dpf':
                ref_ppl[diff_ppl > 0] = 88888
            example = data[idx]
            if args.stm=='stm_dpf':
                example['ppl'] = ref_ppl
            else:
                example['ppl'] = diff_ppl
            example['text'] = formatted_message
            diff_data.append(example)
        data =diff_data

    Dataset.from_list(data).save_to_disk('dataset/stm/'+output_file)
