# -*- coding: utf-8 -*-
"""
train.py
- 使用 Hugging Face Trainer 在 vllg/loong_c4 上训练你自定义的“记忆压缩器”(compressor)
- 冻结 Qwen2.5-3B backbone，仅训练 compressor
- 每 step 随机采样滑动注意力窗口大小，防过拟合
- 只保存 compressor 权重（自定义 Trainer.save_model）

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc-per-node=2 train.py --base_model Qwen/Qwen2.5-3B-Instruct --dataset_name vllg/loong_c4 --dataset_split train --max_seq_len 4096 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 1 --init_sliding_window 128 --sliding_window_choices 512,1024,2048,3176 --num_attn_sinks 128 --mem_max_len 256 --output_dir /work/xiaoyonggao/dymem3/ --tf32 --bf16 --max_grad_norm 1 --deepspeed /home/xiaoyonggao/dymem/examples/deepspeed/ds_z2_config.json --dataset_limit 50000
"""

from curses import flash
import os
import sys
import math
import argparse
from itertools import chain

import torch
torch.autograd.set_detect_anomaly(True)
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    Trainer,
)
import torch.nn.init as init
# 若 qwen2_dymem.py 与本脚本在同一目录，确保可导入
torch._dynamo.config.allow_unspec_int_on_nn_module = True
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

# --- 导入你提供的自定义实现并注册 ---
from dymem.transformer.qwen2_dymem import register_customized_qwen2,Qwen2ForCausalLM  # noqa: E402
register_customized_qwen2()  # 将 'qwen2' 绑定到你的自定义 Qwen2 类族

# --- 工具函数与回调 ---
from untils import (
    freeze_backbone_except_compressor,
    set_sliding_window,
    save_compressor_weights,
    load_compressor_weights,
    collect_compressor_state_dict,
    RandomSlidingWindowCallback,
    ResetMemCallback,
    LogRecLossToWandbCallback,
    reinit_compressor,
)
from debugger import (
    register_param_sanity_checks,
    debug_one_step,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # 基座模型与数据
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B",
                        help="作为backbone的Qwen2.5-3B（base）模型。")
    parser.add_argument("--dataset_name", type=str, default="vllg/loong_c4",
                        help="训练数据集（HuggingFace Hub id）。")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="数据集split，默认train。")
    parser.add_argument("--dataset_limit", type=int, default=0,
                        help="仅加载前N条原始数据（0表示不限制）。")
    parser.add_argument("--data_jsonl", type=str, default=None,
                        help="本地jsonl数据文件路径，设置后优先使用该数据。")
    parser.add_argument("--streaming", action="store_true",
                        help="使用datasets的流式读取以节省内存。")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="仅用于调试：限制训练样本数量（0表示不限制）。")
    parser.add_argument("--shuffle", action="store_true",
                        help="启用数据随机打乱（map式数据默认由Trainer随机采样；streaming需显式打乱）。")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000,
                        help="streaming 模式下的打乱缓冲区大小。")

    # 序列与批大小
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="拼接后chunk长度（block size）。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # 训练时长与优化
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    # 梯度裁剪
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="全局梯度裁剪阈值（L2范数）；<=0 表示关闭。")

    # 日志与保存
    parser.add_argument("--output_dir", type=str, default="./outputs_dymem")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # 精度与设备
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    # 随机滑窗设置
    parser.add_argument("--init_sliding_window", type=int, default=256,
                        help="模型初始化时的滑动窗口（训练时每step会被随机覆盖）。")
    parser.add_argument("--sliding_window_choices", type=str,
                        default="128,256,512,1024,2048,4096",
                        help="随机采样的滑窗候选，逗号分隔。")
    parser.add_argument("--num_attn_sinks", type=int, default=128)
    parser.add_argument("--mem_max_len", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=-1,help="当使用 --streaming 时必须指定，或设置 --max_train_samples 让脚本自动估算。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_mem_tokens", type=int, default=1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")

    # 调试选项
    parser.add_argument("--debug_one_step", action="store_true", help="开启单步前向+反向调试，捕捉 compressor 引入的 NaN/Inf。")
    parser.add_argument("--debug_batch_size", type=int, default=1, help="调试单步使用的 batch 大小。")
    parser.add_argument("--debug_only", action="store_true", help="仅运行调试单步后退出，不继续正式训练。")
    return parser.parse_args()


def pick_text_column(dataset):
    # vllg/loong_c4 数据列包含 text/timestamp/url（以官方viewer为准）
    # 为兼容其它文本数据，做一层兜底
    candidates = ["evicted_text", "text", "content", "raw_text", "raw_content", "document", "noise_text"]
    cols = list(dataset.features.keys()) if hasattr(dataset, "features") else dataset.column_names
    for c in candidates:
        if c in cols:
            return c
    # 兜底：取首列
    return cols[0]


def build_dataset(tokenizer, args):
    if args.data_jsonl:
        ds_dict = load_dataset("json", data_files={"train": args.data_jsonl}, streaming=args.streaming)
        ds = ds_dict["train"]
        if (not args.streaming) and args.dataset_limit and args.dataset_limit > 0:
            ds = ds.select(range(args.dataset_limit))
        elif args.streaming and args.dataset_limit and args.dataset_limit > 0:
            ds = ds.take(args.dataset_limit)
    else:
        split = args.dataset_split
        if (not args.streaming) and args.dataset_limit and args.dataset_limit > 0:
            if "[" not in split:
                split = f"{split}[:{args.dataset_limit}]"
        ds = load_dataset(args.dataset_name, split=split, streaming=args.streaming)
        if args.streaming and args.dataset_limit and args.dataset_limit > 0:
            ds = ds.take(args.dataset_limit)
    if args.shuffle:
        if args.streaming:
            ds = ds.shuffle(buffer_size=args.shuffle_buffer_size, seed=args.seed)
        else:
            ds = ds.shuffle(seed=args.seed)
    text_col = pick_text_column(ds)

    # tokenization
    def tok_fn(batch):
        if text_col == "evicted_text" or text_col == "noise_text":
            ev = batch.get("evicted_text")
            noise = batch.get("noise_text")
            base = batch.get(text_col)
            n = len(base if base is not None else (ev if ev is not None else noise))
            texts = []
            for i in range(n):
                t = None
                if ev is not None:
                    t = ev[i]
                    if (t is None) or (isinstance(t, str) and t.strip() == ""):
                        t = None
                if t is None and noise is not None:
                    t = noise[i]
                if t is None and base is not None:
                    t = base[i]
                if t is None:
                    t = ""
                texts.append(t if isinstance(t, str) else str(t))
            return tokenizer(texts, add_special_tokens=False, truncation=False)
        else:
            return tokenizer(batch[text_col], add_special_tokens=False, truncation=False)

    # 仅保留分词生成的列，去掉原始文本等其它列，避免后续分组时列长度不一致
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    # 拼接并切 block（Causal LM 常用做法）
    def group_texts(examples):
        # 将一个 batch 里的样本拼接在一起再切块
        concatenated = list(chain(*examples["input_ids"]))
        total_len = len(concatenated)
        block = args.max_seq_len
        if total_len >= block:
            total_len = (total_len // block) * block
        result = {
            "input_ids": [concatenated[i: i + block] for i in range(0, total_len, block)]
        }
        # 标签与输入相同（自回归）
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    # 生成按块切分后的样本，并删除分词阶段的所有旧列，保留新的 input_ids/labels
    ds_tok = ds_tok.map(group_texts, batched=True, remove_columns=ds_tok.column_names)

    # 仅调试用采样
    if args.max_train_samples and args.max_train_samples > 0:
        if args.streaming:
            ds_tok = ds_tok.take(args.max_train_samples)
        else:
            ds_tok = ds_tok.select(range(args.max_train_samples))

    return ds_tok


class CompressorOnlyTrainer(Trainer):
    """覆写保存逻辑：只保存 compressor 权重"""
    def save_model(self, output_dir: str = None, _internal_call: bool = True):
        out_dir = output_dir or self.args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        path = None
        if getattr(self.args, "deepspeed", None):
            try:
                import deepspeed
            except Exception:
                pass
            model_to_save = self.model_wrapped.module if hasattr(self, "model_wrapped") and hasattr(self.model_wrapped, "module") else (self.model.module if hasattr(self.model, "module") else self.model)
            sd = {} if self.args.process_index == 0 else None
            for name, p in model_to_save.named_parameters():
                if (".compressor." in name) or (".self_attn.rec_head." in name):
                    try:
                        with deepspeed.zero.GatheredParameters(p, modifier_rank=0):
                            if self.args.process_index == 0:
                                if p is not None and p.data is not None and p.data.numel() > 0:
                                    sd[name] = p.data.detach().cpu().clone()
                    except Exception:
                        pass
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            if self.args.process_index == 0 and sd:
                path = os.path.join(out_dir, "compressor.pt")
                torch.save(sd, path)
                try:
                    if getattr(self.args, "save_safetensors", True):
                        from safetensors.torch import save_file
                        save_file(sd, os.path.join(out_dir, "model.safetensors"))
                    else:
                        torch.save(sd, os.path.join(out_dir, "pytorch_model.bin"))
                except Exception:
                    pass
                try:
                    from untils import collect_compressor_config
                    cfg = collect_compressor_config(model_to_save)
                    with open(os.path.join(out_dir, "compressor_config.json"), "w", encoding="utf-8") as f:
                        import json
                        json.dump(cfg, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
        else:
            path = save_compressor_weights(self.model, out_dir)
            try:
                sd_local = collect_compressor_state_dict(self.model)
                if getattr(self.args, "save_safetensors", True):
                    from safetensors.torch import save_file
                    save_file(sd_local, os.path.join(out_dir, "model.safetensors"))
                else:
                    torch.save(sd_local, os.path.join(out_dir, "pytorch_model.bin"))
            except Exception:
                pass
        if self.args.process_index == 0:
            self.state.save_to_json(os.path.join(out_dir, "trainer_state.json"))
            if self.args.should_save:
                torch.save(self.args, os.path.join(out_dir, "training_args.bin"))
            if path:
                self.log({"compressor_ckpt": path})
    def log(self, logs: dict, start_time=None):
        # 只在 rank0 注入（避免多卡重复）
        if self.state.is_world_process_zero:
            m = self.model.module if hasattr(self.model, "module") else self.model

            # 你的 rec loss 存在：Qwen2ForCausalLM.model.rec_loss_total
            rec = getattr(getattr(m, "model", None), "rec_loss_total", None)
            if rec is not None:
                logs = dict(logs)  # 避免原地修改
                logs["train/loss_rec"] = float(rec.detach().float().item())

            # （可选）如果你在 Qwen2ForCausalLM.forward 里缓存了 loss_main_last
            loss_main = getattr(m, "loss_main_last", None)
            if loss_main is not None:
                logs = dict(logs)
                logs["loss_main"] = float(loss_main.detach().float().item())

        return super().log(logs, start_time=start_time)

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # Qwen2.5 使用相同的 bos/eos；如无 pad，绑到 eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- config ---
    # 这里会返回你自定义的 Qwen2Config（因为前面 register_customized_qwen2 了）
    config = AutoConfig.from_pretrained(args.base_model)
    # 切换到带记忆压缩器的层实现
    config._layer_implementation = "Qwen2DyMemDecoderLayer"
    config._ahn_implementation = "Mamba2"
    # config.attn_implementation = "flash_attention_2"
    config.use_compressor = True
    config.sliding_window = args.init_sliding_window
    config.num_attn_sinks = args.num_attn_sinks
    config.mem_max_len = args.mem_max_len
    config.num_mem_tokens = args.num_mem_tokens
    config.conv_kernel = 4
    config.state_size = 256
    config.chunk_size = 512
    config.lambda_rec = 0.05
    config.rec_learnable_pos = False
    config.rec_loss_type = "smooth_l1"
    
    # --- model ---
    # 注意：backbone 从 base 权重加载，compressor 为新增参数 -> ignore_mismatched_sizes=True
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        trust_remote_code=False,
    )
    

    # 冻结 backbone，仅训练 compressor
    model.train()
    freeze_backbone_except_compressor(model)
    if not args.resume_from_checkpoint and not args.auto_resume:
        reinit_compressor(model)
    for n, p in model.named_parameters():
        if ".self_attn.rec_head." in n:
            p.requires_grad = True

    model.gradient_checkpointing_enable()
    # 初始化一次滑动窗口（训练中还会被回调随机覆盖）
    set_sliding_window(model, args.init_sliding_window)

    # 在单步调试前，确保模型已迁移到可用设备（避免 Triton 内核收到 CPU 张量）
    if args.debug_one_step:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    if args.streaming:
        if args.max_steps is None or args.max_steps <= 0:
            if args.max_train_samples and args.max_train_samples > 0:
                eff_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps
                args.max_steps = max(1, math.ceil(args.max_train_samples / max(1, eff_bs)))
                print(f"[streaming] auto-estimated max_steps={args.max_steps} from max_train_samples={args.max_train_samples}, eff_bs={eff_bs}")
            else:
                raise ValueError(
                    "Using --streaming requires a known number of steps. "
                    "Pass --max_steps > 0, or set --max_train_samples so we can auto-estimate max_steps."
                )
    else:
        # 非 streaming 情况下，按 epoch 走；确保 max_steps = -1
        if args.max_steps is None or args.max_steps < 0:
            args.max_steps = -1

    # --- dataset & collator ---
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = build_dataset(tokenizer, args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- training args ---
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
        gradient_checkpointing=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # 不要丢掉模型forward可能用到的字段
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_torch",
        report_to="wandb",  # 如需W&B等自行开启
        run_name="dymem_rec",
        deepspeed=args.deepspeed,
    )

    # --- callbacks: 随机滑窗 + 每轮重置记忆状态 ---
    win_choices = [int(x) for x in args.sliding_window_choices.split(",") if x.strip()]
    # 过滤掉 >= max_seq_len 的候选，避免无意义的大窗口
    win_choices = [w for w in win_choices if w < args.max_seq_len]
    if not win_choices:
        win_choices = [128, 256, 512, 1024]

    model.config.windows_choices = win_choices
    # --- sanity check: 确保至少有一个参数可训练 ---
    num_trainable = sum(p.requires_grad for _, p in model.named_parameters())
    if num_trainable == 0:
        raise RuntimeError(
            "No parameters require grad after freezing. "
            "Check freeze_backbone_except_compressor matching rules and your parameter names."
        )

    callbacks = [
        LogRecLossToWandbCallback(log_every_n_steps = 1),
    ]

    trainer = CompressorOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    dl = trainer.get_train_dataloader()

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

    def _has_trainer_state(checkpoint_dir: str) -> bool:
        return os.path.isfile(os.path.join(checkpoint_dir, "trainer_state.json"))

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
        if os.path.isdir(resume_path) and _has_hf_model_files(resume_path) and _has_trainer_state(resume_path):
            trainer.train(resume_from_checkpoint=resume_path)
        else:
            trainer.train()
    else:
        trainer.train() 
    # 结束时再保存一次（依然只保存 compressor）
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
