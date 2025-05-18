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
# os.environ["CUDA_VISIBLE_DEVICES"] ="0"
# from unsloth.chat_templates import get_chat_template
# from unsloth import FastLanguageModel, is_bfloat16_supported

# Add argument parser
parser = argparse.ArgumentParser(description='Train model with custom learning rate and threshold')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate for training (default: 2e-5)')
parser.add_argument('--threshold', type=float, default=2.5,
                    help='Threshold for data collator (default: 2.5)')
parser.add_argument('--dataset', type=str, default="mbpp",
                    help='Data task to train (default: mbpp)')
parser.add_argument('--model_name', type=str, default="gemma2-2b-it",
                    help='base model name to train (default: llama3)')
parser.add_argument('--target_model', type=str, default="google/gemma-2-2b-it",
                    help='base model path to train')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

args_cli = parser.parse_args()



# Set device if using deepspeed
# device = torch.device("cuda", args_cli.local_rank) if torch.cuda.is_available() else torch.device("cpu")

# Initialize DeepSpeed
# deepspeed.init_distributed()


# model_name = 'gemma'
# target_model = "google/gemma-2-9b-it"

# model_name = 'mistral'
# target_model = 'unsloth/mistral-7b-instruct-v0.3'
max_seq_length = 512
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
    attn_implementation="eager",
    device_map={'':device_string}
)
# for name, param in target_model.named_parameters():
#     if name in ["lm_head.weight","model.embed_tokens.weight"]:
#         param.requires_grad =False
    
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    use_dora=True,
)
threshold = str(args_cli.threshold).replace(".","-")
args = SFTConfig(output_dir="/workspace/sharing/johnsonwu/trained_models/"+f"{model_name}_{dataset_name}_config_lr{args_cli.learning_rate}_t{threshold}",
    gradient_checkpointing=True,
    num_train_epochs = 3,
    learning_rate=args_cli.learning_rate,  # Use command line argument
    save_strategy="epoch",
    save_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    bf16=True,
    warmup_steps=20,
    logging_steps=1,
    packing=False,
    lr_scheduler_type="cosine",
    local_rank=args_cli.local_rank,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # deepspeed="deepspeed_configs/zero3_bf16.json",
    optim='adamw_bnb_8bit',
)
args.skip_prepare_dataset = True
args.remove_unused_columns = False
trainer = SFTTrainer(
    target_model,
    train_dataset=dataset,
    max_seq_length = max_seq_length,
    args=args,
    peft_config=peft_config,
    data_collator=CustomizedDataCollatorForChatML(tokenizer, threshold=args_cli.threshold, apply="highest")  # Use command line argument
)
trainer.train()
