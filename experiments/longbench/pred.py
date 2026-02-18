import torch
import argparse
import json
import os
import time
from datasets import load_dataset
from utils import seed_everything, load_model_and_tokenizer, build_chat, save_generation, clear_cache


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama-3.1-8b-instruct",
                        choices=["llama-3.1-8b-instruct",
                                "mistral-7B-instruct-v0.2"])
    parser.add_argument('--compress', type=bool, default=True, help="Compress args")
    parser.add_argument('--kv_comp', type=str, default="segment_two_stage")
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--max_capacity', type=int, default=128)
    parser.add_argument('--lookahead_steps', type=int, default=2)

    return parser.parse_args(args)

@torch.inference_mode()
def inference_with_single_gpu(dataset, dataset_name, prompt_format,
                              max_length, max_gen, model_name, path, out_path,
                              compress=False, kv_comp=None, window_size=None, max_capacity=None, lookahead_steps=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_config = load_model_and_tokenizer(model_name, path, device, compress, kv_comp, window_size, max_capacity, lookahead_steps)


    idx = -1
    for data in dataset:
        idx += 1
        print(f"Processing {idx}")

        # 格式化 prompt
        prompt = prompt_format.format(**data)

        if dataset_name not in ["trec", "samsum", "lcc", "repobench-p"]: # From ada-kv, chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        tokenized_prompt = tokenizer(prompt, truncation=False, return_offsets_mapping=True, return_tensors='pt').to(device)

        context_length = tokenized_prompt.input_ids.shape[-1]
        print(f"context_length: {context_length}")

        if kv_comp in ("segment", "segment_two_stage"):
            from kv_compress.segmentkv.segmentkv_two_stages_utils import TwoStageSegmentKV
            segments = TwoStageSegmentKV.get_segments(tokenized_prompt.input_ids, model_name, window_size)
            for i in range(model_config.num_hidden_layers):
                model.model.layers[i].self_attn.config.segments = segments

        torch.cuda.synchronize()
        start = time.time()

        if dataset_name in ["trec", "samsum"]:         # from adakv, prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            eos_tokens = [tokenizer.eos_token_id,
                         tokenizer.encode("\n", add_special_tokens=False)[-1],
                         tokenizer.encode(".\n", add_special_tokens=False)[-1],
                         tokenizer.encode(". \n", add_special_tokens=False)[-1]]
            generate_ids = model.generate(
                tokenized_prompt.input_ids,
                attention_mask=tokenized_prompt.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=eos_tokens,
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

        # print(f"generation: {generation}")
        save_generation(out_path, model_name, generation, data, dataset_name)
        # break



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


    data_path = "datasets/"

    dataset_names = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    compress = True
    compress_args = {
            "kv_comp": args.kv_comp,
            "window_size": args.window_size,
            "max_capacity": args.max_capacity,
            "lookahead_steps": args.lookahead_steps,
        }


    if compress:
        out_path = f"{compress_args["kv_comp"]}/budget{args.max_capacity}"
    else:
        out_path = f"vanilla"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for dataset_name in dataset_names:
        print(f"model name: {model_name}")
        print(f"dataset name: {dataset_name}")
        print(f"compress: {compress}")

        dataset = load_dataset("json", data_files={'test': f"{data_path}/{dataset_name}.jsonl"}, split="test")

        dataset2prompt = json.load(open("config/dataset2prompt.json", "r", encoding="utf-8"))
        dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

        prompt_format = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]

        if compress:
            inference_with_single_gpu(dataset, dataset_name, prompt_format, max_length, max_gen, model_name, model_path, out_path, compress, **compress_args)
        else:
            inference_with_single_gpu(dataset, dataset_name, prompt_format, max_length, max_gen, model_name, model_path, out_path, compress)

