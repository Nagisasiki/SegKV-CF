import torch
import numpy as np
import random
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 device_map=device,
                                                 dtype=torch.float16,
                                                 attn_implementation="flash_attention_2")
    model_config = AutoConfig.from_pretrained(path)

    model = model.eval()
    return model, tokenizer, model_config

def build_chat(tokenizer, prompt, model_name):
    if model_name == "llama-3.1-8b-instruct":
        # print("======== llama3 build chat ========")
        prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
    elif model_name == "mistral-7B-instruct-v0.2":
        # print("======== mistral build chat ========")
        prompt = f'<s>[INST] {prompt} [/INST]'
    elif "qwen" in model_name:
        # print("======== qwen build chat ========")
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        raise ValueError("not supported model_name")

    return prompt


def save_generation(out_path, model_name, generation, data, dataset_name):
    dir_path = f"{out_path}/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = f"{dir_path}/{dataset_name}.jsonl"
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump({"gen": generation, "answers": data["answers"], "all_classes": data["all_classes"],
                   "length": data["length"]}, f, ensure_ascii=False)
        f.write('\n')