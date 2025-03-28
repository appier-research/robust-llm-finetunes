"""
export WANDB_ENTITY="XXXX"
export WANDB_PROJECT="XXXX"
export HF_TOKEN="XXXX" <= read access
python train_with_mask.py --learning_rate 2e-4 --threshold -1
"""
import os
import argparse
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from trl.trainer import SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer
from collate_fn import CustomizedDataCollatorForChatML
import torch
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
# Add argument parser
parser = argparse.ArgumentParser(description='Train model with custom learning rate and threshold')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate for training (default: 2e-5)')
parser.add_argument('--threshold', type=float, default=2.5,
                    help='Threshold for data collator (default: 2.5)')
parser.add_argument('--dataset', type=str, default="mbpp",
                    help='Data task to train (default: mbpp)')
parser.add_argument('--model_name', type=str, default="llama3",
                    help='base model name to train (default: llama3)')
parser.add_argument('--target_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help='base model path to train')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

args_cli = parser.parse_args()



# Set device if using deepspeed
# device = torch.device("cuda", args_cli.local_rank) if torch.cuda.is_available() else torch.device("cpu")

# Initialize DeepSpeed
# deepspeed.init_distributed()


# model_name = 'gemma'
# target_model = "google/gemma-2-2b-it"

# model_name = 'mistral'
# target_model = 'unsloth/mistral-7b-instruct-v0.3'


model_name = args_cli.model_name
target_model = args_cli.target_model
dataset_name = args_cli.dataset
data_path = os.path.join('dataset/stm/',target_model.replace('/','-').replace('.','-')+'_'+dataset_name)
dataset = load_from_disk(data_path)

#multi-gpu training setup
device_string = PartialState().process_index
tokenizer = AutoTokenizer.from_pretrained(target_model)
tokenizer.pad_token = tokenizer.eos_token
target_model = AutoModelForCausalLM.from_pretrained(
    args_cli.target_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={'':device_string}
)
for param in target_model.get_input_embeddings().parameters():
    param.requires_grad = False
for param in target_model.get_output_embeddings().parameters():
    param.requires_grad = False

    
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     target_modules="all-linear",
#     task_type="CAUSAL_LM",
# )
args = SFTConfig(output_dir="/workspace/sharing/johnsonwu/trained_models/"+f"{model_name}_{dataset_name}_config_lr{args_cli.learning_rate}_t{args_cli.threshold}_sft",
    gradient_checkpointing=True,
    num_train_epochs = 1,
    use_liger=False,
    learning_rate=args_cli.learning_rate,  # Use command line argument
    save_strategy="epoch",
    save_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,
    warmup_steps=20,
    logging_steps=1,
    packing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    deepspeed="deepspeed_configs/zero3.json",
    optim='adamw_8bit',
)
args.skip_prepare_dataset = True
args.remove_unused_columns = False
trainer = SFTTrainer(
    target_model,
    train_dataset=dataset,
    args=args,
    # peft_config=peft_config,
    data_collator=CustomizedDataCollatorForChatML(tokenizer, threshold=args_cli.threshold, apply="None")  # Use command line argument
)
trainer.train()
