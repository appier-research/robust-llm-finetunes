# robust-llm-finetunes-
## To started with...
### Environment setup
- install axolotl >=0.5.1 from their [repository](https://github.com/axolotl-ai-cloud/axolotl)
- some basic commands to establish a training/inference envionment:
  ```
  conda create -n robust-sft python=3.10
  conda activate robust-sft
  pip install -r requirements.txt
  ```
To reproduce our results. you need to generate the training data first
## Generation of training task data.
### SO (Self-Output generated data) for Llama3-8B Instruct on MBPP
```
python3 generate_self-output_training_data.py --base_model meta-llama/Meta-Llama-3-8B-Instruct  --task mbpp --mode "self-output"
```
### Rephrase generated data for Llama3-8B Instruct on MBPP
```
python3 generate_self-output_training_data.py --base_model meta-llama/Meta-Llama-3-8B-Instruct  --task mbpp --mode "rephrase"
```
### STM (Selective Token Masking data) for Llama3-8B Instruct on MBPP with naive STM
```
python3 generate_stm_training_data.py --base_model meta-llama/Meta-Llama-3-8B-Instruct --task mbpp
```
### STM (Selective Token Masking data) for Llama3-8B Instruct on MBPP with DPF
- note that when apply alternative stm, you need to fine tune a base model with the data first, and provide the adapter path.
```
python3 generate_stm_training_data.py --base_model meta-llama/Meta-Llama-3-8B-Instruct --task mbpp --stm dfp --stm_adapter <adapter_path_in_distk>
```
The data should be stored ar `dataset/self-output`, `dataset/stm`, `dataset/rephrase`.
## Training of tasks
### train self-output data on Llama3-8B Instruct for on MBPP
- first create an axolotl training config as training/mbpp-gt.yml
- you need to specify the training data path, base model, lora settings, training settings like learning rate, epochs...etc (we have provide our setting in training/mbpp-gt.yml)
- and modify `example_training_so_gt.sh` to add your training config then run
```
sh example_training_so_gt.sh
```
- the trained model should be placed `trained_models/`
## Inference of tasks
### Inference trained adapter models with base model Llama3 on MBPP testing data
- first provide your WANDB_ENTITY, HF_TOKEN, OPENAI_API_KEY and OPENAI_ORGANIZATION in  `example_inference.sh`
- in `example_inference.sh`, you can provide your own `load_adapter` as your trained adapter model, and the base model name to be added on then run:
```
sh example_inference.sh
```
- the results should be placed at `logging/<task>-test`