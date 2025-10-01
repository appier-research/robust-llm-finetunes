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

# we found a bug in the previous code : label not shifted
class PeftWithKL(PeftModelForCausalLM):

    def forward(self, *args: Any, **kwargs: Any):
        labels: torch.Tensor = kwargs["labels"]  # (B, T)
        # We'll compute the loss ourselves; do NOT let inner model compute a loss.
        kwargs_labels = {k: v for k, v in kwargs.items()}
        device = labels.device

        # ---------------------------
        # 1) Reference model pass (no adapters, no grad, no labels)
        # ---------------------------
        # Ensure adapters are disabled only for this block and always re-enabled.
        self.base_model.disable_adapter_layers()
        try:
            with torch.no_grad():
                ref_outputs = self.get_base_model()(*args, **kwargs_labels, return_dict=True)
                ref_logits = ref_outputs.logits  # (B, T, V)
        finally:
            self.base_model.enable_adapter_layers()

        # Temperature on the reference distribution
        ref_logits = ref_logits / (getattr(self, "kl_temperature", 1.0) + 1e-7)

        # ---------------------------
        # 2) PEFT model pass (adapters ON), still no labels to avoid base loss
        # ---------------------------
        with self._enable_peft_forward_hooks(*args, **kwargs):
            peft_outputs = self.get_base_model()(*args, **kwargs_labels, return_dict=True)
            pred_logits = peft_outputs.logits  # (B, T, V)

        # ---------------------------
        # 3) Causal LM shift + mask
        # ---------------------------
        # labels:    (B, T)   -> shift to (B, T-1)
        # logits:    (B, T,V) -> (B, T-1, V)
        mask = (labels != -100)
        labels_shift = labels[:, 1:]
        mask_shift   = mask[:, 1:]
        pred_logits  = pred_logits[:, :-1, :]
        ref_logits   = ref_logits[:,  :-1, :]

        # Flatten valid positions
        flat_mask      = mask_shift          # (B, T-1) -> boolean
        flat_labels    = labels_shift[flat_mask]          # (N_valid,)
        flat_pred_log  = pred_logits[flat_mask]           # (N_valid, V)
        flat_ref_log   = ref_logits[flat_mask]            # (N_valid, V)

        # ---------------------------
        # 4) Cross-entropy on PEFT (our main supervised loss)
        # ---------------------------
        # selective_log_softmax(gather) equals F.cross_entropy with ignore_index if done on flat rows
        pred_logps_taken = selective_log_softmax(flat_pred_log, flat_labels)  # log p_pred(y*)
        ce_loss = (-pred_logps_taken).mean()

        # ---------------------------
        # 5) KL proxy on taken-action log-probs (PPO-style reward shaping)
        #    r = exp(logp_ref - logp_pred)
        # ---------------------------
        ref_logps_taken = selective_log_softmax(flat_ref_log, flat_labels)   # log p_ref(y*)
        logr = ref_logps_taken - pred_logps_taken  # log r

        kl_estimator = getattr(self, "kl_estimator", "k1")
        if kl_estimator == "k1":
            # KL proxy: -log r
            kl_per_token = -logr
        else:
            # k3: (r - 1) - log r  where r = exp(logr)
            r = torch.exp(logr)
            kl_per_token = (r - 1.0) - logr

        mean_kl = kl_per_token.mean()
        kl_coef = torch.as_tensor(getattr(self, "kl_coef", 0.0), device=device, dtype=pred_logps_taken.dtype)

        # Final loss = CE + kl_coef * KL_proxy
        loss = ce_loss + kl_coef * mean_kl

        # Package outputs similar to HF/TRL expectations
        out = peft_outputs  # keep logits etc for downstream features
        out.loss = loss
        out.kl_div = (kl_coef * mean_kl)  # log it if you like
        out.ce_loss = ce_loss
        out.mean_kl = mean_kl
        return out

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
