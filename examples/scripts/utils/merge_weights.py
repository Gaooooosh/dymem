# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download

# Register custom Qwen layers that add AHN support.
from dymem.transformer.qwen2_dymem import register_customized_qwen2

register_customized_qwen2()


def parse_args():
    p = argparse.ArgumentParser(description="Load base model + AHN weights and run a sample generation.")
    p.add_argument(
        "--base-model",
        type=str,
        required=True,
        help='HF repo or local path to the base model (e.g., "Qwen/Qwen2.5-3B-Instruct").',
    )
    p.add_argument(
        "--ahn-path",
        type=str,
        default=None,
        help='Path to AHN-only weights (.safetensors or .pt) or a local HF folder or repo id.',
    )
    p.add_argument(
        "--output-path",
        type=str,
        default=None,
        help='Destination to save the merged model weights (base + AHN)',
    )
    return p.parse_args()


def is_local_dir(s: str) -> bool:
    return os.path.isdir(s)

def is_local_file(s: str) -> bool:
    return os.path.isfile(s)


def str_to_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

def is_safetensors_file(path: str) -> bool:
    return path.endswith(".safetensors")

def is_pt_file(path: str) -> bool:
    return path.endswith(".pt") or path.endswith(".bin") or path.endswith(".pth")

def load_ahn_sd_from_file(path: str):
    if is_safetensors_file(path):
        return load_file(path, device="cpu")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        sd = {}
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
    return out

def build_config_for_base(base_model: str):
    cfg = AutoConfig.from_pretrained(base_model)
    cfg._layer_implementation = "Qwen2DyMemDecoderLayer"
    cfg._ahn_implementation = "Mamba2"
    cfg.use_compressor = True
    if getattr(cfg, "sliding_window", None) is None:
        cfg.sliding_window = 1024
    if getattr(cfg, "num_attn_sinks", None) is None:
        cfg.num_attn_sinks = 128
    if getattr(cfg, "mem_max_len", None) is None:
        cfg.mem_max_len = 1024
    return cfg


def load_ahn_overrides(model, ahn_sd):
    model_sd = model.state_dict()
    overlap_keys = [k for k in ahn_sd.keys() if k in model_sd]

    target_dtype = next(model.parameters()).dtype
    filtered = {k: (ahn_sd[k].to(dtype=target_dtype) if ahn_sd[k].dtype != target_dtype else ahn_sd[k])
                for k in overlap_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    return {
        "overridden_count": len(filtered),
        "total_ahn_keys": len(ahn_sd),
        "overlap_example": overlap_keys[:5],
        "missing": missing,       # should be []
        "unexpected": unexpected  # should be []
    }


def merge_and_safe(args):
    if os.path.isdir(args.ahn_path):
        ahn_root = args.ahn_path
        config = AutoConfig.from_pretrained(ahn_root)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=torch.bfloat16)
        ahn_ckpt = os.path.join(ahn_root, "model.safetensors")
        ahn_sd = load_file(ahn_ckpt, device="cpu")
    elif os.path.isfile(args.ahn_path):
        config = build_config_for_base(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=torch.bfloat16)
        if is_safetensors_file(args.ahn_path):
            ahn_sd = load_file(args.ahn_path, device="cpu")
        else:
            ahn_sd = load_ahn_sd_from_file(args.ahn_path)
    else:
        ahn_root = snapshot_download(repo_id=args.ahn_path)
        config = AutoConfig.from_pretrained(ahn_root)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=torch.bfloat16)
        ahn_ckpt = os.path.join(ahn_root, "model.safetensors")
        ahn_sd = load_file(ahn_ckpt, device="cpu")

    report = load_ahn_overrides(model, ahn_sd)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.save_pretrained(args.output_path)
    except Exception as e:
        print(f"[WARN] Tokenizer not saved: {e}")
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(args.output_path)

    model.config.save_ahn_only = False
    model.save_pretrained(
        args.output_path,
        state_dict=model.state_dict(),
        safe_serialization=True,
        max_shard_size="4GB"
    )
    print(f"[INFO] Merged model saved to {args.output_path}")


def main():
    args = parse_args()
    merge_and_safe(args)


if __name__ == "__main__":
    main()