"""
export WANDB_ENTITY="XXXX"
export WANDB_PROJECT="XXXX"
export HF_TOKEN="XXXX" <= read access
python train_with_soft.py --learning_rate 2e-4 --threshold -1
"""
import gc
import os
import argparse
import torch
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from trl.trainer import SFTConfig
from trl.trainer.utils import (
    peft_module_casting_to_bf16,
)
import torch.nn.functional as F
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from collate_fn import CustomizedDataCollatorForChatML
from typing import Any, Literal, Optional, Union
from accelerate import PartialState
import deepspeed

def _low_ppl_weights(ppl: torch.Tensor, tau: float = 0.75) -> torch.Tensor:
    """
    Monotone decreasing soft weights with maximum at ppl=1:
        w = exp(-((ppl - 1)/tau)^2)
    Args:
      ppl: per-token perplexity (can contain NaNs for masked tokens)
      tau: width of the bump (smaller -> learn only very low-ppl tokens)
    """
    return torch.exp(-((ppl - 1.0) / tau) ** 2)

class PeftWithLowPPLFocus(PeftModelForCausalLM):
    ppl_tau: float = 0.75  # tune this

    def forward(self, *args: Any, **kwargs: Any):
        labels = kwargs['labels']
        completion_ppl = kwargs.pop('completion_ppl', None)

        # Forward for logits
        preds = self.get_base_model()(*args, **kwargs)  # CausalLMOutput...
        logits = preds.logits  # (B, T, V)

        # Valid-token mask
        mask = (labels != -100)  # (B, T)
        logits = logits[:, :-1, :]        # (B, T-1, V)
        labels = labels[:, 1:]            # (B, T-1)
        mask   = mask[:, 1:]              # align with labels
        completion_ppl = completion_ppl[:, 1:]

        # Per-token NLL on valid positions
        flat_logits = logits[mask]          # (N_valid, V)
        flat_labels = labels[mask]          # (N_valid,)
        # print(completion_ppl[mask].mean())
        # Memory-efficient gathered log-softmax
        token_nll = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='none')

        if completion_ppl is not None:
            flat_ppl = completion_ppl[mask].to(token_nll.dtype)  # (N_valid,)
            # standard_loss = token_nll.mean()
            # print(f"DEBUG: Standard (unweighted) loss: {standard_loss.item()}")
            # Soft weights that are highest near ppl=1 and smoothly -> 0 as ppl increases
            raw_w = _low_ppl_weights(flat_ppl, tau=getattr(self, "ppl_tau", 0.75))

            # Mask NaNs/Infs (e.g., prompt/pad) to zero so they don't contribute
            w = torch.nan_to_num(raw_w, nan=0.0, posinf=0.0, neginf=0.0)
            w = torch.clip(w, min=0, max=1).detach()
            # Normalize: average over weighted tokens (keeps scale stable)
            num_valid_tokens = mask.sum().clamp_min(1e-8)
            loss = (w * token_nll).sum() / num_valid_tokens
            weight_mean = w.mean() if w.numel() > 0 else torch.tensor(0.0, device=logits.device)
        else:
            # Fallback: standard mean over valid tokens
            loss = token_nll.mean()
            weight_mean = torch.tensor(1.0, device=logits.device)

        preds.loss = loss
        preds.token_nll_mean = token_nll.mean()
        preds.low_ppl_weight_mean = weight_mean
        return preds

# Add argument parser
parser = argparse.ArgumentParser(description='Train model with custom learning rate and threshold')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate for training (default: 2e-4)')
parser.add_argument('--threshold', type=float, default=0.75,
                    help='Threshold for data collator (default: -1)')
parser.add_argument('--dataset', type=str, default="mbpp",
                    help='Threshold for data collator (default: mbpp)')
parser.add_argument('--model_name', type=str, default="llama3",
                    help='Threshold for data collator (default: mbpp)')
parser.add_argument('--target_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help='Threshold for data collator (default: mbpp)')
parser.add_argument('--debug_loss', action='store_true')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
args_cli = parser.parse_args()

# model_name = 'gemma'
# target_model = "google/gemma-2-2b-it"

# model_name = 'mistral'
# target_model = 'unsloth/mistral-7b-instruct-v0.3'

#multi-gpu training setup
# device_string = PartialState().process_index

model_name = args_cli.model_name
target_model = args_cli.target_model
dataset_name = args_cli.dataset


data_path = os.path.join('dataset/stm/',target_model.replace('/','-').replace('.','-')+'_'+dataset_name)
dataset = load_from_disk(data_path)
tokenizer = AutoTokenizer.from_pretrained(target_model)
tokenizer.pad_token = tokenizer.eos_token
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
threshold = str(args_cli.threshold).replace(".","-")
args = SFTConfig(output_dir=f"{model_name}_{dataset_name}_config_lr{args_cli.learning_rate}_soft-exp-decay-tau={threshold}",
    gradient_checkpointing=True,
    num_train_epochs = 3,
    learning_rate=args_cli.learning_rate,  # Use command line argument
    save_strategy="epoch",
    save_steps=1,
    use_liger=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True,
    warmup_steps=20,
    logging_steps=1,
    packing=False,
    lr_scheduler_type="cosine",
    # local_rank=args_cli.local_rank,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    # deepspeed="deepspeed_configs/zero3_bf16.json",
    optim='adamw_bnb_8bit',
)
args.skip_prepare_dataset = True
args.remove_unused_columns = False
if args_cli.debug_loss:
    trainer = SFTTrainer(
        target_model,
        peft_config=peft_config,        
        train_dataset=dataset,
        args=args,
        data_collator=CustomizedDataCollatorForChatML(tokenizer, threshold=-1)  # Use command line argument
    )
    trainer.train()
else:
    base_model = AutoModelForCausalLM.from_pretrained(target_model,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="eager",
                        device_map="auto",
                        # device_map={'':device_string}
                        # device_map="auto"
    )
    base_model.enable_input_require_grads() # important
    target_model = PeftWithLowPPLFocus(base_model, peft_config=peft_config, ppl_tau=args_cli.threshold)
    trainer = SFTTrainer(
        target_model,
        train_dataset=dataset,
        max_seq_length = 2048,
        args=args,
        peft_config=peft_config,
        data_collator=CustomizedDataCollatorForChatML(tokenizer, output_ppl=True, threshold=-1)  # Use command line argument
    )
    trainer.train()
