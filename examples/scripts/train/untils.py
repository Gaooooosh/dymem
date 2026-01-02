# -*- coding: utf-8 -*-
"""
untils.py
- 冻结backbone、仅训练compressor
- 运行时动态设置滑动窗口；重置各层记忆缓存，避免跨batch污染
- 仅保存/加载compressor权重
- 合并导出：将 compressor 权重并入 Qwen 基座并保存为完整可用的HF权重目录
- 提供 Trainer 回调：每 step 随机滑窗；每轮重置记忆
"""

import os
import json
import torch
from typing import Dict, List
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
import torch.nn as nn
import torch.nn.init as init
import math
# ------- 训练相关：冻结与窗口设置 --------

def freeze_backbone_except_compressor(model: torch.nn.Module):
    """
    仅训练“记忆压缩器”相关参数；匹配更鲁棒：
    - 包含 ".compressor." / "compressor." / ".compressor"
    - 或者名字里出现 "compress"（宽松）
    - 或者可能用于 AHN 的关键字（ahn, mamba）
    """
    train_keywords = ("compressor", "compress", "ahn", "mamba", "mem_norm")
    total, trainable = 0, 0
    hit_names = []

    for name, p in model.named_parameters():
        lname = name.lower()
        keep = any(k in lname for k in train_keywords)
        # 常见形式额外兜底
        keep = keep or (".compressor." in name) or ("compressor." in name) or (".compressor" in name)

        if keep:
            p.requires_grad = True
            trainable += p.numel()
            hit_names.append(name)
        else:
            p.requires_grad = False
        total += p.numel()

    print(f"[freeze] total params: {total/1e6:.2f}M, trainable (compressor-like): {trainable/1e6:.2f}M")
    if trainable == 0:
        # 保险兜底：若完全没匹配到，临时放开 lm_head（避免无梯度直接崩）
        if hasattr(model, "lm_head"):
            for n, p in model.lm_head.named_parameters():
                p.requires_grad = True
                trainable += p.numel()
                hit_names.append(f"lm_head.{n}")
            print("[freeze][fallback] No compressor params matched -> temporarily unfreezing lm_head to keep graph trainable.")
        else:
            raise RuntimeError(
                "No trainable params matched compressor.*; please check your module/param names. "
                "Try printing model.named_parameters() to inspect the exact names."
            )

    # 打印前若干个可训练参数名，方便你确认命名是否正确
    preview = "\n  - ".join(hit_names[:20])
    print(f"[freeze] first trainable params:\n  - {preview}{'\\n  - ...' if len(hit_names) > 20 else ''}")



def _iter_attn_modules(model: torch.nn.Module):
    """遍历含有滑窗/记忆的注意力模块（你的自定义注意力里有 sliding_window 字段）。"""
    for _, m in model.named_modules():
        if hasattr(m, "sliding_window") and hasattr(m, "mem_max_len"):
            yield m


def set_sliding_window(model: torch.nn.Module, window_size: int):
    """统一设置模型与所有注意力层的滑动窗口大小。"""
    if hasattr(model, "config"):
        model.config.sliding_window = int(window_size)
    for attn in _iter_attn_modules(model):
        try:
            attn.sliding_window = int(window_size)
        except Exception:
            pass


def reset_dymem_state(model: torch.nn.Module):
    """重置各层动/静态记忆缓存，避免跨样本/step串扰。"""
    for attn in _iter_attn_modules(model):
        for attr in ("sink_k", "sink_v", "mem_k", "mem_v", "active_k", "active_v", "recent_k", "recent_v"):
            if hasattr(attn, attr):
                setattr(attn, attr, None)


# ------- 仅保存/加载压缩器权重 --------

def collect_compressor_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """提取只含 '.compressor.' 的权重子字典。"""
    full_sd = model.state_dict()
    return {k: v for k, v in full_sd.items() if ".compressor." in k}

def collect_mem_projector_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        model = model.module
    mp = getattr(model, "mem_projector", None)
    if mp is None:
        return {}
    sd = mp.state_dict()
    return {f"mem_projector.{k}": v.detach().cpu().clone() for k, v in sd.items()}

def collect_compressor_config(model: torch.nn.Module) -> Dict[str, object]:
    cfg = getattr(model, "config", None)
    out = {}
    if cfg is not None:
        def _get(name, default=None):
            return getattr(cfg, name, default)
        out.update({
            "use_compressor": _get("use_compressor", True),
            "layer_implementation": _get("_layer_implementation", None),
            "ahn_implementation": _get("_ahn_implementation", None),
            "sliding_window": _get("sliding_window", None),
            "num_attn_sinks": _get("num_attn_sinks", None),
            "mem_max_len": _get("mem_max_len", None),
            "conv_kernel": _get("conv_kernel", None),
            "state_size": _get("state_size", None),
            "chunk_size": _get("chunk_size", None),
            "dtype": str(_get("dtype", None)),
            "windows_choices": _get("windows_choices", None),
            "hidden_size": _get("hidden_size", None),
            "num_attention_heads": _get("num_attention_heads", None),
            "num_key_value_heads": _get("num_key_value_heads", None),
            "num_hidden_layers": _get("num_hidden_layers", None),
            "num_mem_tokens": _get("num_mem_tokens", None),
            "lambda_rec": _get("lambda_rec", None),
        })
    return out


def save_compressor_weights(model: torch.nn.Module, path_or_dir: str) -> str:
    """只保存 compressor 权重；返回保存路径。"""
    sd = collect_compressor_state_dict(model)
    if os.path.isdir(path_or_dir) or path_or_dir.endswith(os.sep):
        os.makedirs(path_or_dir, exist_ok=True)
        path = os.path.join(path_or_dir, "compressor.pt")
        cfg_path = os.path.join(path_or_dir, "compressor_config.json")
    else:
        # 允许传入文件路径
        dirn = os.path.dirname(path_or_dir)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        path = path_or_dir
        cfg_path = os.path.join(dirn if dirn else ".", "compressor_config.json")
    torch.save(sd, path)
    print(f"[save] compressor weights -> {path} ({len(sd)} tensors)")
    try:
        cfg = collect_compressor_config(model)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"[save] compressor config -> {cfg_path}")
    except Exception as e:
        print(f"[save][warn] compressor config not saved: {e}")
    return path


def save_mem_projector_weights(model: torch.nn.Module, path_or_dir: str) -> str:
    sd = collect_mem_projector_state_dict(model)
    if os.path.isdir(path_or_dir) or path_or_dir.endswith(os.sep):
        os.makedirs(path_or_dir, exist_ok=True)
        path = os.path.join(path_or_dir, "mem_projector.pt")
    else:
        dirn = os.path.dirname(path_or_dir)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        path = path_or_dir
    torch.save(sd, path)
    print(f"[save] mem_projector weights -> {path} ({len(sd)} tensors)")
    return path


def load_compressor_weights(model: torch.nn.Module, compressor_path: str, strict: bool = False):
    """将 compressor 权重加载到当前模型（strict=False 更稳妥）。"""
    sd = torch.load(compressor_path, map_location="cpu")
    
    # 修复权重名称：将所有 self_attn.mem_norm.* 转换为 self_attn.compressor.mem_norm.*
    new_sd = {}
    renamed_count = 0
    for key, value in sd.items():
        if "self_attn.mem_norm." in key:
            new_key = key.replace("self_attn.mem_norm.", "self_attn.compressor.mem_norm.")
            new_sd[new_key] = value
            renamed_count += 1
        else:
            new_sd[key] = value
    
    if renamed_count > 0:
        print(f"[rename] Renamed {renamed_count} mem_norm keys to compressor.mem_norm")
        sd = new_sd
    
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[load] compressor from {compressor_path} | missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected


def load_mem_projector_weights(model: torch.nn.Module, mem_projector_path: str, strict: bool = True):
    sd = torch.load(mem_projector_path, map_location="cpu")
    if hasattr(model, "module"):
        model = model.module
    mp = getattr(model, "mem_projector", None)
    if mp is None:
        return ["mem_projector"], []
    prefix = "mem_projector."
    sub_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            sub_sd[k[len(prefix):]] = v
        elif k.startswith("module." + prefix):
            sub_sd[k[len("module." + prefix):]] = v
    missing, unexpected = mp.load_state_dict(sub_sd, strict=strict)
    print(f"[load] mem_projector from {mem_projector_path} | missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected


@torch.no_grad()
def reinit_mem_projector(model: torch.nn.Module, w_std: float = None):
    if hasattr(model, "module"):
        model = model.module
    mp = getattr(model, "mem_projector", None)
    if mp is None:
        return 0
    if w_std is None:
        cfg = getattr(model, "config", None)
        w_std = float(getattr(cfg, "initializer_range", 0.002) if cfg is not None else 0.002)
    touched = 0
    for module in mp.modules():
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=w_std)
            if module.bias is not None:
                init.zeros_(module.bias)
            touched += 1
    print(f"[reinit_mem_projector] initialized {touched} Linear layers (std={w_std})")
    return touched


# ------- 合并导出：压缩器 + Qwen 基座 -> 完整可用HF模型 --------

def export_merged_model(
    base_model_name_or_path: str,
    compressor_path: str,
    output_dir: str,
    init_sliding_window: int = 256,
    num_attn_sinks: int = 128,
    mem_max_len: int = 1024,
    dtype: str = "auto",
) -> str:
    """
    将 compressor 权重并入 Qwen 基座，导出一个可直接使用的 HuggingFace 模型目录。
    用法（训练完后）：
        export_merged_model("Qwen/Qwen2.5-3B", "outputs_dymem/compressor.pt", "merged_qwen_dymem")
    """
    # 注册你自定义的Qwen2类型
    from dymem.transformer.qwen2_dymem import register_customized_qwen2
    register_customized_qwen2()

    # 读取并修改配置，切换到自定义层实现（带compressor）
    cfg = AutoConfig.from_pretrained(base_model_name_or_path)
    cfg._layer_implementation = "Qwen2DyMemDecoderLayer"
    cfg._ahn_implementation = "Mamba2"
    cfg.use_compressor = True
    cfg.sliding_window = init_sliding_window
    cfg.num_attn_sinks = num_attn_sinks
    cfg.mem_max_len = mem_max_len

    torch_dtype = None
    if dtype != "auto":
        torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        config=cfg,
        ignore_mismatched_sizes=True,  # compressor为新增参数
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    )

    # 只加载压缩器权重
    load_compressor_weights(model, compressor_path, strict=False)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
        tok.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"[export] merged model saved to: {output_dir}")
    return output_dir


# -------- Trainer 回调：随机滑窗 / 重置记忆 --------

class RandomSlidingWindowCallback(TrainerCallback):
    """
    每个 batch 开始时，根据该 batch 的 seq_len 采样滑窗，保证 window < seq_len，
    从而 overflow > 0，确保 compressor 参与前向、产生梯度。
    """
    def __init__(self, choices: List[int]):
        self.choices = sorted({int(c) for c in choices if int(c) > 0})

    def on_train_batch_begin(self, args, state, control, **kwargs):
        import random
        model = kwargs["model"]
        inputs = kwargs.get("inputs", None)

        # 拿到本 batch 的序列长度（优先 labels，其次 input_ids）
        seq_len = None
        if inputs is not None:
            if isinstance(inputs, dict):
                if "labels" in inputs and inputs["labels"] is not None:
                    seq_len = int(inputs["labels"].shape[1])
                elif "input_ids" in inputs and inputs["input_ids"] is not None:
                    seq_len = int(inputs["input_ids"].shape[1])

        # 兜底：如果拿不到，就用 config 里的 max_position_embeddings 或者 2048
        if not seq_len:
            seq_len = getattr(getattr(model, "config", object()), "max_position_embeddings", 2048)

        cfg = getattr(model, "config", None)
        base_sink = getattr(cfg, "num_attn_sinks", 128) if cfg is not None else 128
        sink_len = int(min(base_sink, seq_len))
        valid = [int(w) for w in self.choices if int(w) < seq_len and (seq_len - int(w)) > sink_len]
        if valid:
            win = random.choice(valid)
        else:
            win = max(1, int(seq_len - sink_len - 1))

        set_sliding_window(model, win)

        target_sink = max(1, min(sink_len, seq_len - win - 1))
        for attn in _iter_attn_modules(model):
            try:
                if hasattr(attn, "num_attn_sinks"):
                    attn.num_attn_sinks = int(target_sink)
            except Exception:
                pass

        if state.global_step % max(1, args.logging_steps) == 0:
            print(f"[rand-win] step={state.global_step} batch_seq_len={seq_len} window={win} sink={target_sink}")

        return control

class LogRecLossToWandbCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=1):
        self.log_every_n_steps = log_every_n_steps

    def on_train_begin(self, args, state, control, **kwargs):
        # 这里 kwargs 里通常会带 trainer
        self._trainer = kwargs.get("trainer", None)
        if state.is_world_process_zero:
            print("[LogRecLossToWandbCallback] on_train_begin called!")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # on_log 一定会被 Trainer 调用（每到 logging_steps）
        if not state.is_world_process_zero:
            return
        if state.global_step % self.log_every_n_steps != 0:
            return

        model = kwargs.get("model", None)
        if model is None:
            print("[LogRecLossToWandbCallback] on_log called but model=None")
            return

        m = model.module if hasattr(model, "module") else model

        # 你存 rec loss 的位置：Qwen2ForCausalLM -> .model.rec_loss_total
        rec = getattr(getattr(m, "model", None), "rec_loss_total", None)
        rec_main = getattr(getattr(m, "model", None), "loss_main", None)
        if rec is None:
            print(f"[LogRecLossToWandbCallback] step={state.global_step} rec_loss_total=None")
            return

        rec_val = float(rec.detach().float().item())
        # print(f"[LogRecLossToWandbCallback] step={state.global_step} loss_rec={rec_val:.6f}")
        logs["train/loss_rec"] = rec_val


torch.no_grad()
def _inv_softplus(y: torch.Tensor):
    # 数值稳定的 softplus 逆：softplus^{-1}(y) = y + log(-expm1(-y))
    # 参考 PyTorch 实现思路；y>0
    return y + torch.log(-torch.expm1(-y))

@torch.no_grad()
def reinit_compressor(model, verbose=True,
                                        a_min=1e-2, a_max=1.0,
                                        dt_min=1e-3, dt_max=1e-1,
                                        w_std=0.02):
    """
    在 transformers 风格的基础上，额外对 Mamba2 的关键参数进行稳定初始化：
    - Linear/Conv/Embedding: N(0, 0.02), bias=0
    - LayerNorm/RMSNorm: weight=1, bias=0, eps>=1e-5
    - Mamba2.A_log: 设定 A = -softplus(A_log)，将 |A| 的初值在 [a_min, a_max] 做几何级数分布
    - Mamba2.dt_proj: 先把 weight=0，仅用 bias 给出目标 dt（softplus^{-1}(dt0)）
    - Mamba2.dt_bias: 全 0（让 dt 初值只由 dt_proj.bias 决定；更可控）
    - Mamba2.g_proj: N(0, 0.02), bias=0
    - Mamba2.o_proj(GroupLinear): N(0, 0.02), bias=0
    - RMSNormGated: weight=1, bias=0（若有）
    额外：确保这些敏感参数为 FP32（不要被混精把参数也降精度）。
    """
    total, touched = 0, 0
    modules_seen = set()

    def is_norm_like(m: nn.Module):
        cname = m.__class__.__name__.lower()
        return isinstance(m, nn.LayerNorm) or "rmsnorm" in cname or "rms_norm" in cname

    for name, module in model.named_modules():
        if "compressor" not in name:
            continue
        total += 1

        # ---- 通用部分：linear/conv/embedding/norm ----
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=w_std)
            if module.bias is not None:
                init.zeros_(module.bias)
            touched += 1; continue

        if isinstance(module, nn.Conv1d):
            init.normal_(module.weight, mean=0.0, std=w_std)
            if module.bias is not None:
                init.zeros_(module.bias)
            touched += 1; continue

        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=w_std)
            touched += 1; continue

        if is_norm_like(module):
            if hasattr(module, "weight") and module.weight is not None:
                init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)
            if hasattr(module, "eps"):
                module.eps = max(getattr(module, "eps", 1e-5), 1e-5)
            touched += 1; continue

        # ---- 特殊：RMSNormGated / GroupLinear 的常规参数名也处理 ----
        cname = module.__class__.__name__.lower()
        if "group" in cname and "linear" in cname:
            # 假设内部有 weight/bias（按组切分没关系，一并初始化）
            if hasattr(module, "weight") and module.weight is not None:
                init.normal_(module.weight, mean=0.0, std=w_std)
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)
            touched += 1; continue

        # ---- Mamba2 特定：通过属性名识别并初始化 ----
        # 下面不依赖类名，只要模块上有这些属性就会生效
        has_heads = hasattr(module, "num_heads") and hasattr(module, "head_dim")
        if has_heads and any(hasattr(module, k) for k in ["A_log", "dt_proj", "g_proj", "o_proj", "dt_bias", "norm"]):
            H = int(module.num_heads)
            D = int(module.head_dim)

            # A_log: 让 A = -softplus(A_log) 的初值在 [a_min, a_max] 做 logspace
            if hasattr(module, "A_log") and isinstance(module.A_log, nn.Parameter):
                # 目标 |A|（正数）做几何级数分布：小头慢衰减，大头快衰减
                # 遵循全局 dtype，不强制使用 FP32
                a_vals = torch.logspace(math.log10(a_min), math.log10(a_max), H, dtype=module.A_log.data.dtype)
                # A = -softplus(A_log) ⇒ A_log = softplus^{-1}(|A|)
                A_log_init = _inv_softplus(a_vals)
                module.A_log.data = A_log_init.to(module.A_log.data.device, dtype=module.A_log.data.dtype)
                # 建议训练时对 A_log 关闭 weight decay；lr 可单独调低（在 optimizer 里做）
                module.A_log._no_weight_decay = True
                touched += 1
            if hasattr(module, "D") and isinstance(module.D, nn.Parameter):
                a_vals = torch.logspace(math.log10(a_min), math.log10(a_max), H, dtype=module.D.data.dtype)
                D_log_init = _inv_softplus(a_vals)
                module.D.data = D_log_init.to(module.D.data.device, dtype=module.D.data.dtype)
                # 建议训练时对 D_log 关闭 weight decay；lr 可单独调低（在 optimizer 里做）
                module.D._no_weight_decay = True
                touched += 1

            # dt_bias: 简单设为 0（把 dt 初值交给 dt_proj.bias）
            if hasattr(module, "dt_bias") and isinstance(module.dt_bias, nn.Parameter):
                module.dt_bias.data.zero_()
                module.dt_bias.data = module.dt_bias.data.to(dtype=module.dt_bias.data.dtype)
                touched += 1

            # dt_proj: 用 bias 给出目标 dt 初值（权重先置 0，确保初期 dt 与上游无关）
            if hasattr(module, "dt_proj") and isinstance(module.dt_proj, nn.Linear):
                # 令目标 dt0 为区间中值（几何中值更合适）
                dt0 = math.exp((math.log(dt_min) + math.log(dt_max)) / 2.0)  # 几何均值
                beta0 = float(_inv_softplus(torch.tensor(dt0)))               # softplus^{-1}(dt0)
                nn.init.zeros_(module.dt_proj.weight)
                nn.init.constant_(module.dt_proj.bias, beta0)
                # 保持当前 dtype，不强制 FP32
                module.dt_proj.weight.data = module.dt_proj.weight.data.to(module.dt_proj.weight.data.dtype)
                module.dt_proj.bias.data   = module.dt_proj.bias.data.to(module.dt_proj.bias.data.dtype)
                touched += 1

            # g_proj: 常规小权重 & 零偏置（门控从“温和”开始）
            if hasattr(module, "g_proj") and isinstance(module.g_proj, nn.Linear):
                init.normal_(module.g_proj.weight, mean=0.0, std=w_std)
                if module.g_proj.bias is not None:
                    init.zeros_(module.g_proj.bias)
                touched += 1

            # o_proj(GroupLinear): 小权重 & 零偏置
            if hasattr(module, "o_proj"):
                op = module.o_proj
                if hasattr(op, "weight") and op.weight is not None:
                    init.normal_(op.weight, mean=0.0, std=w_std)
                if hasattr(op, "bias") and op.bias is not None:
                    init.zeros_(op.bias)
                touched += 1

            # RMSNormGated（若存在）：weight=1, bias=0
            if hasattr(module, "norm") and is_norm_like(module.norm):
                nm = module.norm
                if hasattr(nm, "weight") and nm.weight is not None:
                    init.ones_(nm.weight)
                if hasattr(nm, "bias") and nm.bias is not None:
                    init.zeros_(nm.bias)
                if hasattr(nm, "eps"):
                    nm.eps = max(getattr(nm, "eps", 1e-5), 1e-5)
                touched += 1

            modules_seen.add(name)
            continue

        # 兜底：有 reset_parameters 就调用一次
        if hasattr(module, "reset_parameters"):
            try:
                module.reset_parameters()
                touched += 1
            except Exception:
                pass

    mem_touched = 0
    for name, module in model.named_modules():
        if "mem_norm" in name.lower():
            if hasattr(module, "weight") and module.weight is not None:
                init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)
            if hasattr(module, "eps"):
                module.eps = max(getattr(module, "eps", 1e-5), 1e-5)
            if hasattr(module, "variance_epsilon"):
                module.variance_epsilon = max(getattr(module, "variance_epsilon", 1e-6), 1e-6)
            mem_touched += 1

    rec_touched = 0
    for name, module in model.named_modules():
        if "rec_head" not in name.lower():
            continue

        if isinstance(module, nn.MultiheadAttention):
            if getattr(module, "in_proj_weight", None) is not None:
                init.normal_(module.in_proj_weight, mean=0.0, std=w_std)
            if getattr(module, "in_proj_bias", None) is not None:
                init.zeros_(module.in_proj_bias)
            if getattr(module, "bias_k", None) is not None:
                init.normal_(module.bias_k, mean=0.0, std=w_std)
            if getattr(module, "bias_v", None) is not None:
                init.normal_(module.bias_v, mean=0.0, std=w_std)
            rec_touched += 1
            continue

        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=w_std)
            if module.bias is not None:
                init.zeros_(module.bias)
            rec_touched += 1
            continue

        if isinstance(module, nn.Conv1d):
            init.normal_(module.weight, mean=0.0, std=w_std)
            if module.bias is not None:
                init.zeros_(module.bias)
            rec_touched += 1
            continue

        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=w_std)
            rec_touched += 1
            continue

        if is_norm_like(module):
            if hasattr(module, "weight") and module.weight is not None:
                init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)
            if hasattr(module, "eps"):
                module.eps = max(getattr(module, "eps", 1e-5), 1e-5)
            rec_touched += 1
            continue

    # 不再强制将 compressor 的敏感参数置为 FP32，遵循全局 dtype
    # for n, p in model.named_parameters():
    #     if "compressor" in n and any(k in n for k in ["A_log", "dt_bias"]):
    #         p.data = p.data.float()

    if verbose:
        print(f"[reinit_compressor_with_mamba2_style] 触达模块: {len(modules_seen)}，已初始化 {touched}/{total} 个 compressor 子模块；"
              f"策略: Linear/Conv/Emb ~ N(0,{w_std}), Norm→(1,0), A_log→logspace[{a_min},{a_max}], "
              f"dt_proj.weight=0 & bias=inv_softplus(geo({dt_min},{dt_max})), dt_bias=0")
        print(f"[reinit_mem_norm] 已初始化 {mem_touched} 个 mem_norm 模块")
        print(f"[reinit_rec_head] 已初始化 {rec_touched} 个 rec_head 子模块")


class ResetMemCallback(TrainerCallback):
    """在每个 epoch 开头重置各层记忆缓存。"""
    def on_epoch_begin(self, args, state, control, **kwargs):
        reset_dymem_state(kwargs["model"])
        return control
