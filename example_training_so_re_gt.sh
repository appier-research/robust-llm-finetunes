#training a self-output data model with axolotl
accelerate launch -m axolotl.cli.train training/axolotl_configs/llama3/mbpp-so.yml --deepspeed deepspeed_configs/zero3_bf16.json
#training a ground truth data model with axolotl
accelerate launch -m axolotl.cli.train training/axolotl_configs/llama3/mbpp-gt.yml --deepspeed deepspeed_configs/zero3_bf16.json
#training a rephrase model with axolotl
accelerate launch -m axolotl.cli.train training/axolotl_configs/llama3/mbpp-rephrase.yml --deepspeed deepspeed_configs/zero3_bf16.json
#training astm model 
accelerate launch train_with_mask.py --learning_rate 2e-5 --threshold 2.5
