export WANDB_ENTITY="{your wandb entity}"
export TRANSFORMERS_CACHE="/tmp/hf_cache"
export HF_HOME="/tmp/hf_cache"    
export HF_TOKEN="{your huggingface tokens}"
export OPENAI_API_KEY="{your openai api key}"
export OPENAI_ORGANIZATION="{your openai organization}"

#example of running all evaluation with a trained model adapter and its base model.

python -m evals.run_mbpp --base_model meta-llama/Meta-Llama-3-8B-Instruct --load_adapter trained_models/checkpoint-50 --torch_dtype=bf16
python -m evals.run_math --base_model meta-llama/Meta-Llama-3-8B-Instruct --load_adapter trained_models/checkpoint-50 --torch_dtype=bf16
python -m evals.run_arc --base_model meta-llama/Meta-Llama-3-8B-Instruct --load_adapter trained_models/checkpoint-50 --torch_dtype=bf16
python -m evals.run_gsm8k --base_model meta-llama/Meta-Llama-3-8B-Instruct --load_adapter trained_models/checkpoint-50 --torch_dtype=bf16

