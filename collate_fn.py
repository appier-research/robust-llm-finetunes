import torch
import json
import numpy as np
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizerBase,
    TrainerState,
    TrainingArguments,
    is_comet_available,
)
def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

@dataclass
class CustomizedDataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    apply: str = "highest"
    output_ppl: bool = False
    messages_key: str = "conversations"
    threshold : float = 2.5

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 2048)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []
        completion_ppl_list = []  # NEW

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                if 'from' in prompt[0]:
                    prompt = [{'role': 'user' if r['from'] == 'human' else 'assistant', 'content': r['value']} for r in prompt]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            if 'input_ids' in example:
                del example['input_ids']

            if "input_ids" not in example:
                message = example[self.messages_key]
                if 'from' in message[0]:
                    message = [{'role': 'user' if r['from'] == 'human' else 'assistant', 'content': r['value']} for r in message]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False
                )
                tokenized_message = self.tokenizer(
                    formatted_message,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                input_ids.append(tokenized_message["input_ids"])
                attention_mask.append(tokenized_message["attention_mask"])
            else:
                input_ids.append(example["input_ids"])
                attention_mask.append(example["attention_mask"])

            tokenized_prompt = self.tokenizer(
                formatted_prompt,
                truncation=True,
                max_length=len(input_ids[-1]),
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )

            prompts_input_ids.append(tokenized_prompt["input_ids"])
            prompt_attention_mask.append(tokenized_prompt["attention_mask"])

            # labels: ignore prompt, keep completion
            label = [self.ignore_index] * len(input_ids[-1])
            completion_start_idx = len(tokenized_prompt["input_ids"])
            label[completion_start_idx:] = input_ids[-1][completion_start_idx:]
            label = torch.tensor(label, dtype=torch.long)

            # NEW: build per-token completion_ppl aligned to full seq; NaN elsewhere
            full_len = len(input_ids[-1])
            ppl_vec = torch.full((full_len,), float('nan'), dtype=torch.float)

            if 'ppl' in example and example['ppl'] is not None:
                completion_ppl = torch.from_numpy(np.array(example['ppl'], dtype=np.float32))
                end_idx = min(full_len, completion_start_idx + len(completion_ppl))
                put_len = max(0, end_idx - completion_start_idx)
                if put_len > 0:
                    ppl_vec[completion_start_idx:end_idx] = completion_ppl[:put_len]

                if self.threshold > 0:
                    if self.apply == "random":
                        import random
                        random.seed(42)
                        cnt = int((completion_ppl > self.threshold).sum().item())
                        if cnt > 0:
                            samplen = random.choices(range(len(label)), k=cnt)
                            label[samplen] = self.ignore_index
                    elif self.apply == "lowest":
                        label[completion_start_idx:end_idx][completion_ppl[:put_len] < self.threshold] = self.ignore_index
                    elif self.apply == "highest":
                        label[completion_start_idx:end_idx][completion_ppl[:put_len] > self.threshold] = self.ignore_index

            labels.append(label)
            completion_ppl_list.append(ppl_vec)

        # to tensors + left pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]
        input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        prompts_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids]
        prompt_attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask]
        prompts_input_ids = pad(prompts_input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", padding_value=0)

        # NEW: left-pad completion_ppl with NaN so it’s easy to mask later
        completion_ppl = pad(completion_ppl_list, padding_side="left", padding_value=float('nan'))
        output_result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompts": prompts_input_ids,
                "prompt_attention_mask": prompt_attention_mask
        }

        if self.output_ppl:
            output_result["completion_ppl"] = completion_ppl
        return output_result

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    collate_fn = CustomizedDataCollatorForChatML(tokenizer)
    examples = []
    with open('llama3_mbpp_gt_inference.jsonl', 'r') as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) > 20:
                break
    collate_fn(examples)
        
