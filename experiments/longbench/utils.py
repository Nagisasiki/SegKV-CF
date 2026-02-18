import torch
import numpy as np
import random
import os
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name, path, device, compress=False, kv_comp=None, window_size=None, max_capacity=None, lookahead_steps=None):

    if compress:
        if kv_comp == "segment":
            from kv_compress.monkeypatch import replace_llama_by_segmentkv, replace_mistral_by_segmentkv
            replace_llama_by_segmentkv()
            replace_mistral_by_segmentkv()
        elif kv_comp == "segment_two_stage":
            from kv_compress.monkeypatch import replace_llama_by_segmentkv_two_stage
            replace_llama_by_segmentkv_two_stage()
        elif kv_comp == "snap":
            from kv_compress.monkeypatch import replace_llama_by_snapkv, replace_mistral_by_snapkv
            replace_llama_by_snapkv()
            replace_mistral_by_snapkv()
        elif kv_comp == "chunk":
            from kv_compress.monkeypatch import replace_llama_by_chunkkv, replace_mistral_by_chunkkv
            replace_llama_by_chunkkv()
            replace_mistral_by_chunkkv()
        elif kv_comp == "laq":
            from kv_compress.monkeypatch import replace_llama_by_laqkv
            replace_llama_by_laqkv()
        elif kv_comp == "vanilla":
            pass
        else:
            raise ValueError("not supported kv_comp")

    if model_name not in ["llama-3.1-8b-instruct", "mistral-7B-instruct-v0.2"]:
        raise ValueError(f"Model {model_name} not supported!")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 device_map=device,
                                                 dtype=torch.float16,
                                                 attn_implementation="flash_attention_2")
    model_config = AutoConfig.from_pretrained(path)
    layers = model_config.num_hidden_layers
    if compress:
        for l in range(layers):
            model.model.layers[l].self_attn.config.window_size = window_size
            model.model.layers[l].self_attn.config.max_capacity = max_capacity
            model.model.layers[l].self_attn.config.lookahead_steps = lookahead_steps

            # params for laq
            model.model.layers[l].self_attn.config.lookahead_window_size = 32
            model.model.layers[l].self_attn.config.lookahead_max_capacity = 2048
            model.model.layers[l].self_attn.config.lookahead_size = lookahead_steps
            model.model.layers[l].self_attn.config.stage2_window_sizes = 8

    model = model.eval()
    return model, tokenizer, model_config


def build_chat(tokenizer, prompt, model_name):
    if model_name == "llama-3.1-8b-instruct":
        prompt =  [{ "role": "user", "content": prompt}]
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
