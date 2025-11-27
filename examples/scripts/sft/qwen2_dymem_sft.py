import os
import sys
import json
import argparse
from typing import List, Dict
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from dymem.transformer.qwen2_dymem import register_customized_qwen2
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)
from untils import (
    freeze_backbone_except_compressor,
    set_sliding_window,
    reinit_compressor,
    save_compressor_weights,
    collect_compressor_config,
    load_compressor_weights,
    RandomSlidingWindowCallback,
    ResetMemCallback,
)


"""
CUDA_VISIBLE_DEVICES=0 python examples/scripts/sft/qwen2_dymem_sft.py \
--base_model /home/xiaoyonggao/dymem/qwen2.5-3b-compressor-Instruct\
--data_jsonl /home/xiaoyonggao/dymem/memory_sft_dataset.jsonl --output_dir /home/xiaoyonggao/dymem/outputs_sft_compressor --max_seq_len 8000 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --num_train_epochs 1 --learning_rate 5e-5 --bf16 --init_sliding_window 256 --num_attn_sinks 128 --mem_max_len 1024
"""

class CompressorOnlyTrainer(Trainer):
    def save_model(self, output_dir: str = None, _internal_call: bool = True):
        out_dir = output_dir or self.args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        path = None
        if getattr(self.args, "deepspeed", None):
            try:
                import deepspeed
            except Exception:
                deepspeed = None
            model_to_save = self.model_wrapped.module if hasattr(self, "model_wrapped") and hasattr(self.model_wrapped, "module") else (self.model.module if hasattr(self.model, "module") else self.model)
            if self.args.process_index == 0:
                sd = {}
                for name, p in model_to_save.named_parameters():
                    if ".compressor." in name:
                        if deepspeed is not None:
                            try:
                                with deepspeed.zero.GatheredParameters(p, modifier_rank=0):
                                    if p is not None and p.data is not None and p.data.numel() > 0:
                                        sd[name] = p.data.detach().cpu().clone()
                            except Exception:
                                pass
                        else:
                            if p is not None and p.data is not None and p.data.numel() > 0:
                                sd[name] = p.data.detach().cpu().clone()
                if sd:
                    path = os.path.join(out_dir, "compressor.pt")
                    torch.save(sd, path)
                    try:
                        cfg = collect_compressor_config(model_to_save)
                        with open(os.path.join(out_dir, "compressor_config.json"), "w", encoding="utf-8") as f:
                            json.dump(cfg, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
        else:
            path = save_compressor_weights(self.model, out_dir)
        if self.args.process_index == 0:
            self.state.save_to_json(os.path.join(out_dir, "trainer_state.json"))
            if self.args.should_save:
                torch.save(self.args, os.path.join(out_dir, "training_args.bin"))
            if path:
                self.log({"compressor_ckpt": path})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="/home/xiaoyonggao/dymem/qwen2.5-3b-compressor-Instruct")
    p.add_argument("--data_jsonl", type=str, default="/home/xiaoyonggao/dymem/data/mem_sft/memory_sft_dataset.jsonl")
    p.add_argument("--output_dir", type=str, default="/home/xiaoyonggao/dymem/outputs_sft_compressor")
    p.add_argument("--max_seq_len", type=int, default=20000)
    p.add_argument("--min_train_tokens", type=int, default=4096)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--init_sliding_window", type=int, default=256)
    p.add_argument("--sliding_window_choices", type=str, default="128,256,512,1024")
    p.add_argument("--num_attn_sinks", type=int, default=128)
    p.add_argument("--mem_max_len", type=int, default=1024)
    return p.parse_args()


def build_sft_dataset(tokenizer: AutoTokenizer, jsonl_path: str, max_seq_len: int, min_train_tokens: int) -> Dataset:
    samples: List[Dict[str, List[int]]] = []

    def tok(text: str) -> List[int]:
        out = tokenizer(text, add_special_tokens=False, truncation=False)
        return out["input_ids"]

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            instr = obj.get("instruction", "").strip()
            inp = obj.get("input", "").strip()
            out = obj.get("output", obj.get("response", "")).strip()

            prompt = "Instruction: " + instr
            if inp:
                prompt += "\nInput: " + inp
            prompt += "\nResponse: "

            p_ids = tok(prompt)
            o_ids = tok(out) + [tokenizer.eos_token_id] if out else []

            if not out:
                text_ids = tok((instr + "\n" + inp).strip())
                text_ids = text_ids[: max_seq_len - 1] + [tokenizer.eos_token_id]
                if min_train_tokens <= 0 or len(text_ids) >= min_train_tokens:
                    samples.append({"input_ids": text_ids, "labels": text_ids[:]})
                continue

            total = len(p_ids) + len(o_ids)
            if total > max_seq_len:
                budget_resp = min(len(o_ids), max_seq_len // 2)
                budget_prompt = max_seq_len - budget_resp
                p_ids = p_ids[: max(1, budget_prompt - 1)]
                o_ids = o_ids[: budget_resp]
                total = len(p_ids) + len(o_ids)
                if total > max_seq_len:
                    o_ids = o_ids[: max_seq_len - len(p_ids)]

            input_ids = p_ids + o_ids
            labels = [-100] * len(p_ids) + o_ids[:]
            if min_train_tokens <= 0 or len(input_ids) >= min_train_tokens:
                samples.append({"input_ids": input_ids, "labels": labels})

    return Dataset.from_list(samples)


def _find_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    subs = []
    for d in os.listdir(output_dir):
        p = os.path.join(output_dir, d)
        if os.path.isdir(p) and d.startswith("checkpoint-"):
            try:
                step = int(d.split("-")[-1])
            except Exception:
                step = -1
            if step >= 0:
                subs.append((step, p))
    if not subs:
        return None
    subs.sort(key=lambda x: x[0])
    return subs[-1][1]


def _has_hf_model_files(checkpoint_dir: str) -> bool:
    if not os.path.isdir(checkpoint_dir):
        return False
    names = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
    ]
    for n in names:
        if os.path.isfile(os.path.join(checkpoint_dir, n)):
            return True
    return False


class SFTDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - len(ids)
            if self.tokenizer.padding_side == "right":
                padded_ids = ids + [self.pad_token_id] * pad_len
                padded_labs = labs + [self.label_pad_token_id] * pad_len
                attn = [1] * len(ids) + [0] * pad_len
            else:
                padded_ids = [self.pad_token_id] * pad_len + ids
                padded_labs = [self.label_pad_token_id] * pad_len + labs
                attn = [0] * pad_len + [1] * len(ids)
            batch_input_ids.append(padded_ids)
            batch_labels.append(padded_labs)
            batch_attention_mask.append(attn)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    register_customized_qwen2()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(args.base_model)
    config._layer_implementation = "Qwen2DyMemDecoderLayer"
    config._ahn_implementation = "Mamba2"
    config.use_compressor = True
    config.sliding_window = args.init_sliding_window
    config.num_attn_sinks = args.num_attn_sinks
    config.mem_max_len = args.mem_max_len
    target_dtype = (torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None))
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
        dtype=target_dtype,
        trust_remote_code=False,
    )
    model.to(dtype=(target_dtype if target_dtype is not None else torch.float32))
    model.train()
    freeze_backbone_except_compressor(model)
    set_sliding_window(model, args.init_sliding_window)
    train_dataset = build_sft_dataset(tokenizer, args.data_jsonl, args.max_seq_len, args.min_train_tokens)
    data_collator = SFTDataCollator(tokenizer)
    win_choices = [int(x) for x in args.sliding_window_choices.split(",") if x.strip()]
    win_choices = [w for w in win_choices if w < args.max_seq_len]
    if not win_choices:
        win_choices = [128, 256, 512, 1024,2048,4096]
    win_choices = [128, 256, 512, 1024]
    model.config.windows_choices = win_choices
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=2,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        report_to="wandb",
    )
    callbacks = [RandomSlidingWindowCallback(win_choices), ResetMemCallback()]
    trainer = CompressorOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        processing_class=tokenizer,
    )
    resume_path = None
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
    elif args.auto_resume:
        resume_path = _find_latest_checkpoint(args.output_dir)
    if resume_path:
        cpath = resume_path if os.path.isfile(resume_path) else os.path.join(resume_path, "compressor.pt")
        if os.path.exists(cpath):
            try:
                load_compressor_weights(model, cpath, strict=False)
            except Exception:
                pass
        if os.path.isdir(resume_path) and _has_hf_model_files(resume_path):
            trainer.train(resume_from_checkpoint=resume_path)
        else:
            trainer.train()
    else:
        trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
