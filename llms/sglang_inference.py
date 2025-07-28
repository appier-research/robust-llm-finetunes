import sglang as sgl

def load_llm(model_path, lora_path=None):
    if lora_path:
        print(f"Loading lora: {lora_path}")
        # lora_name = "gsm8k_lora"
        # print(f"Loading LoRA adapter '{lora_name}' from: {lora_path}")
        # llm.load_lora_adapter(lora_name=lora_name, lora_path=lora_path)
        # lora_to_use = lora_name
        llm = sgl.Engine(model_path=model_path, lora_paths=[lora_path], disable_radix_cache=True, cuda_graph_max_bs=4, mem_fraction_static=0.6)
    else:
        llm = sgl.Engine(model_path=model_path, cuda_graph_max_bs=4)
    return llm


def generate(llm):
    prompts = [prompt] * num_samples
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": 1024,
        "stop": ["Q:", "</s>", "<|im_end|>", 'User:']
    }

    for attempt in range(retries):
        try:
            # Pass lora_to_use to the generate call
            outputs = llm.generate(prompts, sampling_params)
            break  # Success
        except Exception as e:
            print(f"\nError on Q{idx}, attempt {attempt + 1}/{retries}: {e}")
            if attempt + 1 == retries:
                print(f"All retries failed for Q{idx}. Skipping.")
                outputs = []
                break
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    # Extract answers and check correctness
    strict_correct, flexible_correct = 0, 0
    response_details = []
    
    for output in outputs:
        generated_text = output['text'] # this is text

