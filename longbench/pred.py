import torch
import argparse
import json
import os
import time
from datasets import load_dataset
from utils import seed_everything, load_model_and_tokenizer, build_chat, save_generation
from kv_compress.mokeypatch import replace_llama, replace_mistral


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mistral-7B-instruct-v0.2",
                        choices=["llama-3.1-8b-instruct",
                                "mistral-7B-instruct-v0.2"])
    parser.add_argument('--datapath', type=str, default="THUDM/LongBench")
    parser.add_argument('--compress', action='store_true', default=True, help="Compress args")
    parser.add_argument('--kv_comp', type=str, default="SegKV_CF", help="Compress args")
    parser.add_argument('--coarse_budget', type=int, default=2048, help="Compress args")
    parser.add_argument('--budget', type=int, default=1024, help="Compress args")
    parser.add_argument('--window_size', type=int, default=32, help="Compress args")
    parser.add_argument('--lookahead_steps', type=int, default=4, help="Compress args")
    parser.add_argument('--kernel_size', type=int, default=7, help="Compress args")
    return parser.parse_args(args)

@torch.inference_mode()
def inference_with_single_gpu(dataset, dataset_name, prompt_format,
                              max_length, max_gen, model_name, path, out_path,
                              compress=False, kv_comp=None, window_size=None, coarse_budget=None, budget=None, lookahead_steps=None, kernel_size=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compress:
        replace_llama(kv_comp)
        replace_mistral(kv_comp)
    model, tokenizer, model_config = load_model_and_tokenizer(path, device)

    idx = 0
    for data in dataset:
        idx += 1
        print(f"Processing {idx}")

        prompt = prompt_format.format(**data)

        if dataset_name not in ["trec", "samsum", "lcc", "repobench-p", "validation"]: # From ada-kv, chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        tokenized_prompt = tokenizer(prompt, truncation=False, return_offsets_mapping=True, return_tensors='pt').to(device)
        context_length = tokenized_prompt.input_ids.shape[-1]
        print(f"context_length: {context_length}")

        layers = model_config.num_hidden_layers
        if compress:
            for l in range(layers):
                # for other method
                model.model.layers[l].self_attn.config.max_capacity = budget
                model.model.layers[l].self_attn.config.lookahead_window_size = window_size
                model.model.layers[l].self_attn.config.kernel_size = kernel_size
                model.model.layers[l].self_attn.config.lookahead_max_capacity = coarse_budget
                model.model.layers[l].self_attn.config.lookahead_size = lookahead_steps
                model.model.layers[l].self_attn.config.stage2_window_sizes = 8

                # for SegKV-CF
                model.model.layers[l].self_attn.config.window_size = window_size
                model.model.layers[l].self_attn.config.coarse_budget = coarse_budget
                model.model.layers[l].self_attn.config.budget = budget
                model.model.layers[l].self_attn.config.lookahead_steps = lookahead_steps

        if kv_comp == "SegKV" or kv_comp == "SegKV_CF":
            from kv_compress.segkv.segkv_cf_utils import get_segments
            segments = get_segments(tokenized_prompt.input_ids, model_name, tokenizer, window_size)
            for i in range(model_config.num_hidden_layers):
                model.model.layers[i].self_attn.config.segments = segments


        torch.cuda.synchronize()
        start = time.time()

        if dataset_name in ["trec","samsum",]:         # from adakv, prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            eos_token = [tokenizer.eos_token_id,
                         tokenizer.encode("\n", add_special_tokens=False)[-1],]
            generate_ids = model.generate(
                tokenized_prompt.input_ids,
                attention_mask=tokenized_prompt.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=eos_token,
            )
        else:
            generate_ids = model.generate(
                tokenized_prompt.input_ids,
                attention_mask=tokenized_prompt.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,

            )

        torch.cuda.synchronize()
        end = time.time()
        print(f"generation time: {end - start}")

        generation = tokenizer.batch_decode(generate_ids[:, context_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        save_generation(out_path, model_name, generation, data, dataset_name)

        break




if __name__ == '__main__':
    seed_everything(42)
    args=parse_args()

    model_name = args.model
    compress = args.compress

    # model
    model2path = json.load(open("config/model2path.json", "r"))
    model_path = model2path[model_name]

    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    max_length = model2maxlen[model_name]


    data_path = args.datapath

    dataset_names = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    if compress:
        compress_args = {
            "kv_comp": args.kv_comp,
            "coarse_budget": args.coarse_budget,
            "budget": args.budget,
            "lookahead_steps": args.lookahead_steps,
            "window_size": args.window_size,
            "kernel_size": args.kernel_size,
        }
        out_path = f"{args.kv_comp}_{args.budget}"
    else:
        compress_args = None
        out_path = f"FullKV"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for dataset_name in dataset_names:
        print(f"model name: {model_name}")
        print(f"dataset name: {dataset_name}")

        dataset = load_dataset("json", data_files={'test': f"{data_path}/{dataset_name}.jsonl"}, split="test")

        dataset2prompt = json.load(open("config/dataset2prompt.json", "r", encoding="utf-8"))
        dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

        prompt_format = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]

        if compress:
            inference_with_single_gpu(dataset, dataset_name, prompt_format, max_length, max_gen, model_name, model_path, out_path, compress, **compress_args)
        else:
            inference_with_single_gpu(dataset, dataset_name, prompt_format, max_length, max_gen, model_name, model_path, out_path)

