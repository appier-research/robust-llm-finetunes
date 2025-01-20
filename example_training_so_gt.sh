#training a ground truth data model with axolotl
accelerate launch -m axolotl.cli.train training/mbpp-gt.yml --deepspeed deepspeed_configs/zero3_bf16.json