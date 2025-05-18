"""
export WANDB_ENTITY="XXXX"
export WANDB_PROJECT="XXXX"
export HF_TOKEN="XXXX" <= read access
python train_with_mask.py --learning_rate 2e-4 --threshold -1
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

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

class PeftWithKL(PeftModelForCausalLM):

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        self.base_model.disable_adapter_layers()
        labels = kwargs['labels']
        outputs = self.get_base_model()(*args, **kwargs)
        ref_loss = outputs.loss
        ref_logits = outputs.logits
        ref_logits /= self.kl_temperature + 1e-7

        mask          = labels != -100             # boolean mask of valid positions
        flat_logits   = ref_logits[mask]               # shape (N_valid, num_classes)
        flat_index    = labels[mask]               # shape (N_valid,)
        ref_logprobs   = selective_log_softmax(flat_logits, flat_index)
        self.base_model.enable_adapter_layers()
        del outputs, flat_logits
        torch.cuda.empty_cache()
        gc.collect()

        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            preds = self.get_base_model()(*args, **kwargs)
        pred_logits = preds.logits

        flat_logits  = pred_logits[mask]               # shape (N_valid, num_classes)
        logprobs   = selective_log_softmax(flat_logits, flat_index)
        xentropy_loss = preds.loss

        # I wasn't sure if this stays equiv after grad acc
        # https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L508
        logr = ref_logprobs - logprobs
        kl = -logr if self.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
        mean_kl = kl.mean()
        non_score_reward = -self.kl_coef * kl
        mean_non_score_reward = non_score_reward.mean()
        preds.kl_div = mean_non_score_reward
        preds.loss += mean_non_score_reward
        return preds

# Add argument parser
parser = argparse.ArgumentParser(description='Train model with custom learning rate and threshold')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate for training (default: 2e-4)')
parser.add_argument('--threshold', type=float, default=-1,
                    help='Threshold for data collator (default: -1)')
parser.add_argument('--kl_esti_weight', type=float, default=0.1,
                    help='KL-div normalized weights')
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
# args = SFTConfig(output_dir=f"{model_name}_{dataset_name}_config_lr{args_cli.learning_rate}_t{args_cli.threshold}_with_kl",
#     gradient_checkpointing=True,
#     use_liger=False,
#     learning_rate=args_cli.learning_rate,  # Use command line argument
#     save_strategy="epoch",
#     save_steps=1,
#     bf16=True,
#     # dataset_text_field='problem', # uncomment this if you got a weird `text` field missing error
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     warmup_steps=10,
#     logging_steps=1,
#     packing=False
# )
threshold = str(args_cli.threshold).replace(".","-")
kl_esti_weight = str(args_cli.kl_esti_weight).replace(".","-")
args = SFTConfig(output_dir=f"{model_name}_{dataset_name}_config_lr{args_cli.learning_rate}_t{threshold}_klr{kl_esti_weight}_fix",
    gradient_checkpointing=False,
    num_train_epochs = 3,
    learning_rate=args_cli.learning_rate,  # Use command line argument
    save_strategy="epoch",
    save_steps=1,
    use_liger=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
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
        data_collator=CustomizedDataCollatorForChatML(tokenizer, threshold=args_cli.threshold)  # Use command line argument
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
    target_model = PeftWithKL(base_model, peft_config=peft_config)
    target_model.kl_temperature = 1.0
    target_model.kl_coef = args_cli.kl_esti_weight
    # http://joschu.net/blog/kl-approx.html
    target_model.kl_estimator = 'k3'
    trainer = SFTTrainer(
        target_model,
        train_dataset=dataset,
        args=args,
        peft_config=peft_config,
        data_collator=CustomizedDataCollatorForChatML(tokenizer, threshold=args_cli.threshold, apply="highest")  # Use command line argument
    )
    trainer.train()