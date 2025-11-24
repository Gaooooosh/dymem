# Copyright 2024 the MemoryLLM team.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT Licence

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/wangyu-ustc/MemoryLLM/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import time
import shutil

from dymem.transformer.qwen2_dymem import register_customized_qwen2
register_customized_qwen2()

from dymem.utils import CacheWithMem


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--max_length", default=None, type=int)
    parser.add_argument("--split", default="longbook_qa_eng", type=str)
    parser.add_argument("--split_model", default=False, action="store_true")
    parser.add_argument("--part", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--retrieval", default=None, help="Retrieval method", type=str)
    parser.add_argument("--exclude_or", default=False, action="store_true")
    parser.add_argument("--force_run", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="qwen2")
    parser.add_argument("--attention_sink", type=int, default=128)
    parser.add_argument("--sliding_window", type=int, default=512)
    parser.add_argument("--enable_yarn", default=False, action="store_true")
    parser.add_argument("--fa_layer_ids", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args(args)  # return parser.parse_known_args(args)[0]


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "qwen2" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "qwen3" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    elif "llama2" in model_name and "chat" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"

    return prompt


def post_process(response, model_name):
    """
    Post-process the response.

    Args:
        response (str): Model response.
        model_name (str): Model name.

    Returns:
        str: Post-processed response.
    """
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    retrieval=None,
    exclude_or=False,
):

    preds = []

    count = 0
    for json_obj in tqdm(data):
        count += 1
        if exclude_or:
            if "or" in json_obj["input"]:
                continue

        prompt = prompt_format.format(**json_obj)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        cache = CacheWithMem(model.config)
        with torch.no_grad():
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                past_key_values=cache,
                use_cache=True,
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        try:
            print(f"Prediction: {pred}, Answer: {json_obj['outputs']}")
            preds.append(
                {
                    "pred": pred,
                    "answers": json_obj["outputs"],
                    # "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                }
            )
        except:
            print(f"Prediction: {pred}, Answer: {json_obj['outputs']}")
            preds.append({"pred": pred, "answers": json_obj["outputs"]})
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model = model.eval()
    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    model.config.sliding_window = args.sliding_window
    model.config.num_attn_sinks = args.attention_sink

    datasets = [
        # 'vt',
        # 'fwe',
        # 'cwe',
        # 'niah_single_1',
        # 'niah_single_2',
        # 'niah_single_3',
        # 'niah_multikey_1',
        # 'niah_multikey_2',
        # 'niah_multikey_3',
        # 'niah_multiquery',
        # 'niah_multivalue',
        'qa_1',
        # 'qa_2'
    ]


    for dataset in datasets:

        data = load_dataset(
            "lighteval/RULER-8192-Qwen2.5-Instruct", split=dataset, trust_remote_code=True
        )
        data.save_to_disk(f"ruler8192/data/{dataset}")

        if not os.path.exists(f"ruler8192/pred_seed{args.seed}_e/{model_name}"):
            os.makedirs(f"ruler8192/pred_seed{args.seed}_e/{model_name}")
        out_path = f"ruler8192/pred_seed{args.seed}_e/{model_name}/{dataset}.jsonl"
        # save dataset

        if args.exclude_or:
            out_path = out_path.split("/")
            out_path[-2] += "_exor"
            out_path = "/".join(out_path)

            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))

        if os.path.exists(out_path) and not args.force_run:
            continue

        preds = get_pred(
            model,
            tokenizer,
            data,
            args.max_length,
            100,
            "{input}",
            dataset,
            device,
            model_name,
            args.retrieval,
            exclude_or=args.exclude_or,
        )

        tmp_path = f"/tmp/{dataset}.jsonl"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")

        time.sleep(1)

        shutil.copy(tmp_path, out_path)
