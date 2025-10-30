# train_dymem_distill_single.py
# -*- coding: utf-8 -*-
"""
Qwen2.5 + HuggingFace Trainer：记忆压缩自蒸馏（Teacher–Student 双路径）
单文件实现：双路径前向、三项损失（CE/ KL / 层间 MSE）、仅训练 compressor、滑动窗口、可选 L2 归一化、
记忆路径专用 gradient checkpointing、参数统计与仅保存可训练参数。
依赖：transformers>=4.43、torch>=2.3、datasets（如需从文本文件加载）、qwen2_dymem.py（你的实现）

用法（示例）：
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --output_dir ./ckpts/dymem-ts \
  --train_file ./data/train.txt \
  --per_device_train_batch_size 1 \
  --max_seq_len 4096 \
  --sliding_window 4096 \
  --sliding_window_type random \
  --loss_type ce kl distill \
  --alpha_ce 1.0 --alpha_kl 1.0 --alpha_distill 0.5 --temperature 2.0 \
  --only_save_trainable True
"""

from __future__ import annotations

import os, json, math, argparse, random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers import (
    AutoTokenizer, AutoConfig, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.trainer import logger as hf_logger

# —— 你的自定义 Qwen2 动态记忆实现（已上传为 qwen2_dymem.py） ——
# 该文件内含：Qwen2Config / Qwen2ForCausalLM / Qwen2DyMemDecoderLayer / Qwen2AttentionWithMem / register_customized_qwen2 等
from dymem.transformer.qwen2_dymem import (
    Qwen2Config, Qwen2ForCausalLM, Qwen2DyMemDecoderLayer, Qwen2AttentionWithMem,
    register_customized_qwen2
)
# 复用底层工具（线性投影、repeat_kv、RoPE）
from dymem.transformer.qwen2.modeling_qwen2 import (
    repeat_kv, apply_rotary_pos_emb
)

# ========== 蒸馏与训练配置 ==========
@dataclass
class DistillArgs:
    loss_type: List[str] = field(default_factory=lambda: ["ce", "kl", "distill"])
    alpha_ce: float = 1.0
    alpha_kl: float = 1.0
    alpha_distill: float = 0.5
    temperature: float = 2.0
    # 层间蒸馏
    distill_layers: str = "all"          # "all" | "every2" | "0,4,8,12"
    normalize_hidden: bool = True        # 层间蒸馏前做 L2
    # 仅学生记忆路径的数值/效率控制
    normalize_fused: bool = True         # 将融合后的 o 做 L2 再写 ACTIVE
    gc_for_mem: bool = True              # compressor 路径启用 gradient checkpointing
    train_prefill_as_decode: bool = True # 训练期将 prefill 视作 decode（使 MEM/ACTIVE 参与注意力）
    # 仅训练参数匹配前缀
    trainable_prefixes: List[str] = field(default_factory=lambda: ["compressor."])

# ========== 实用函数 ==========
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def human_count(n: int) -> str:
    return f"{n/1e6:.2f}M" if n < 1e9 else f"{n/1e9:.2f}B"

def mark_trainable(model: nn.Module, include_prefixes: List[str], exclude_prefixes: Optional[List[str]]=None):
    exclude_prefixes = exclude_prefixes or []
    total, trainable = 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        ok = any(name.startswith(pref) or (("." + pref) in name) for pref in include_prefixes)
        bad = any(name.startswith(pref) or (("." + pref) in name) for pref in exclude_prefixes)
        if ok and not bad:
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
    print(f"[Trainable Stats] 总参数量: {human_count(total)} / 可训练参数量: {human_count(trainable)} "
          f"({trainable/total*100:.4f}%)")
    return total, trainable

def compute_dy_sliding_window(seq_len: int, base_sw: int, sw_type: str) -> int:
    if sw_type == "fixed":
        return int(base_sw)
    if sw_type == "linear":
        return max(seq_len // 4, 512)
    if sw_type == "random":
        cands = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        v = [c for c in cands if seq_len/8 < c < seq_len]
        return random.choice(v) if v else min(base_sw, seq_len)
    raise ValueError(f"Unknown sliding_window_type={sw_type}")

def shift_ce_loss(logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,S,V]; labels: [B,S]; valid_mask: [B,S]; 仅在 labels != -100 且 valid_mask=True 的位置计入 CE
    右移对齐：使用 logits[:, :-1] 预测 labels[:, 1:]
    """
    B, S, V = logits.shape
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    mask = valid_mask[:, 1:].contiguous() & (labels != -100)
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    loss = F.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1),
        reduction="none", ignore_index=-100
    ).view(B, S-1)
    return (loss * mask.float()).sum() / mask.float().sum()

def masked_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, valid_mask: torch.Tensor, T: float) -> torch.Tensor:
    B, S, V = student_logits.shape
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(-1)  # [B,S]
    mask = valid_mask & torch.ones_like(kl, dtype=torch.bool, device=kl.device)
    if mask.sum() == 0:
        return kl.new_tensor(0.0)
    return (kl * mask.float()).sum() / mask.float().sum() * (T * T)

def pick_layer_indices(n_layers: int, spec: str) -> List[int]:
    if spec == "all":
        return list(range(n_layers + 1))  # 包含最后一层 norm 后输出
    if spec.startswith("every"):
        k = int(spec.replace("every", ""))
        return list(range(0, n_layers + 1, k))
    ids = [int(x) for x in spec.split(",") if x.strip().isdigit()]
    ids = [i for i in ids if 0 <= i <= n_layers]
    return ids or list(range(n_layers + 1))

def layer_mse_loss(h_mem: List[torch.Tensor], h_ref: List[torch.Tensor],
                   valid_mask: torch.Tensor, layers_spec: str, l2: bool=True) -> torch.Tensor:
    """
    h_*: tuple/list of [B,S,H]
    """
    idxs = pick_layer_indices(len(h_mem)-1, layers_spec)
    losses = []
    for i in idxs:
        a, b = h_mem[i], h_ref[i]  # [B,S,H]
        if l2:
            a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
        mse = ((a - b) ** 2).mean(-1)  # [B,S]
        m = valid_mask & torch.ones_like(mse, dtype=torch.bool)
        if m.sum() == 0:
            continue
        losses.append((mse * m.float()).sum() / m.float().sum())
    if not losses:
        return h_mem[0].new_tensor(0.0)
    return torch.stack(losses).mean()

# ========== 学生注意力：TS 版本（prefill 启用 MEM/ACTIVE、记忆路径 GC、融合归一化、动态块掩码） ==========
class Qwen2AttentionWithMemTS(Qwen2AttentionWithMem):
    def forward(
        self,
        prenormed_hidden: torch.Tensor,                                 # [B,T,H]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        compressor: Optional[nn.Module] = None,
        do_solidify: bool = False,
        is_decode: bool = True,
        block_mask: Optional[torch.Tensor] = None,  # 新增：允许外部传 block mask（不强制）
    ) -> torch.Tensor:
        B, T, H = prenormed_hidden.shape
        # 线性投影
        q_states = self.q_proj(prenormed_hidden)
        k_states = self.k_proj(prenormed_hidden)
        v_states = self.v_proj(prenormed_hidden)
        # 分头
        q = q_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B,H,T,D]
        k_kv = k_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v_kv = v_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        k = repeat_kv(k_kv, self.num_kv_groups)  # [B,H,T,D]
        v = repeat_kv(v_kv, self.num_kv_groups)

        # 新 token 的 RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # recent 累积
        if self.recent_k is None:
            self.recent_k, self.recent_v = k, v
            self.recent_q, self.recent_h = q, prenormed_hidden
        else:
            self.recent_k = torch.cat([self.recent_k, k], dim=2)
            self.recent_v = torch.cat([self.recent_v, v], dim=2)
            self.recent_q = torch.cat([self.recent_q, q], dim=2)
            self.recent_h = torch.cat([self.recent_h, prenormed_hidden], dim=1)

        # 计算溢出并触发压缩（仅记忆路径可选 GC）
        overflow = max(0, (0 if self.recent_k is None else self.recent_k.size(2)) - self.sliding_window)
        if overflow > 0 and compressor is not None:
            h_e = self.recent_h[:, :overflow, :].contiguous()
            q_e = self.recent_q[:, :, :overflow, :].contiguous()
            k_e = self.recent_k[:, :, :overflow, :].contiguous()
            v_e = self.recent_v[:, :, :overflow, :].contiguous()

            def _mem_block():
                compressor.update_cache(h_e)
                # 传递 (ab, g) 双通道特征；当前使用同一张量以满足接口
                o, _ = compressor(hidden_states=(h_e, h_e), q_states=q_e, k_states=k_e, v_states=v_e,
                                  use_cache=True, past_key_values=None)
                # L2 归一化避免数值爆炸
                o_last = o[:, -1, :]
                if getattr(self.config, "normalize_fused", True):
                    o_last = F.normalize(o_last, p=2, dim=-1, eps=1e-6)
                self._set_active_from_o(o_last)

            if getattr(self.config, "gc_for_mem", False) and self.training:
                checkpoint(_mem_block)
            else:
                _mem_block()

            # 裁剪 recent
            self.recent_k = self.recent_k[:, :, overflow:, :].contiguous()
            self.recent_v = self.recent_v[:, :, overflow:, :].contiguous()
            self.recent_q = self.recent_q[:, :, overflow:, :].contiguous()
            self.recent_h = self.recent_h[:, overflow:, :].contiguous()

        # 固化 ACTIVE→MEM（可选）
        if do_solidify:
            self._solidify_active()

        # 训练期将 prefill 当作 decode：使 MEM/ACTIVE 参与注意力，并做“查询时对齐”的 RoPE（按最后步角度）
        treat_as_decode = is_decode or (self.training and getattr(self.config, "train_prefill_as_decode", True))
        mem_k, mem_v = self.mem_k, self.mem_v
        act_k, act_v = self.active_k, self.active_v
        if treat_as_decode and position_embeddings is not None:
            cos, sin = position_embeddings
            cos_t = cos[:, -1:, :]
            sin_t = sin[:, -1:, :]
            if mem_k is not None:
                cos_mem = cos_t.expand(-1, mem_k.size(2), -1)
                sin_mem = sin_t.expand(-1, mem_k.size(2), -1)
                mem_k, _ = apply_rotary_pos_emb(mem_k, mem_k, cos_mem, sin_mem, unsqueeze_dim=1)
            if act_k is not None:
                act_k, _ = apply_rotary_pos_emb(act_k, act_k, cos_t, sin_t, unsqueeze_dim=1)
        elif not treat_as_decode:
            # 保持 prefill 也能使用 MEM/ACTIVE（但不做对齐旋转）
            pass

        # 组装 KV
        K_total = self._cat([self.sink_k, mem_k, act_k, self.recent_k])
        V_total = self._cat([self.sink_v, mem_v, act_v, self.recent_v])
        kv_len = 0 if K_total is None else K_total.size(2)

        # 构造块掩码：SINK/MEM/ACTIVE 全可见；WINDOW（recent）按因果 + 窗口截断
        if block_mask is None:
            sink_len = 0 if self.sink_k is None else self.sink_k.size(2)
            mem_len = 0 if mem_k is None else mem_k.size(2)
            act_len = 0 if act_k is None else act_k.size(2)
            recent_len = 0 if self.recent_k is None else self.recent_k.size(2)
            offset = sink_len + mem_len + act_len
            # q_idx ∈ [0, T-1]，recent 的 kv 相对索引 i ∈ [0, recent_len-1] 对应真实步 (T - recent_len + i)
            # 因果条件： (T - recent_len + i) <= q_idx  <=>  i <= q_idx - (T - recent_len)
            def ctx(b, h, q_idx, kv_idx):
                is_sink = kv_idx < sink_len
                is_mem  = (kv_idx >= sink_len) & (kv_idx < sink_len + mem_len)
                is_act  = (kv_idx >= sink_len + mem_len) & (kv_idx < sink_len + mem_len + act_len)
                i = kv_idx - offset
                in_recent = (i >= 0) & (i < recent_len)
                causal_ok = i <= (q_idx - (T - recent_len))
                return is_sink | is_mem | is_act | (in_recent & causal_ok)
            from torch.nn.attention.flex_attention import create_block_mask
            block_mask = create_block_mask(
                ctx, B=1, H=self.num_heads, Q_LEN=T, KV_LEN=kv_len,
                device=q.device, BLOCK_SIZE=128, _compile=False
            )

        out = torch.nn.attention.flex_attention.flex_attention(q, K_total, V_total, block_mask=block_mask)
        out = out[:, :, :T, :].transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)

# —— 将学生模型的注意力层就地替换为 TS 版 ——
def patch_student_attention(student: Qwen2ForCausalLM, distill_args: DistillArgs, dy_sw: int):
    student.config.normalize_fused = distill_args.normalize_fused
    student.config.gc_for_mem = distill_args.gc_for_mem
    student.config.train_prefill_as_decode = distill_args.train_prefill_as_decode
    # 动态窗口写回到各层 self_attn（overflow 计算依赖该值）
    for i, layer in enumerate(student.model.layers):
        if isinstance(layer, Qwen2DyMemDecoderLayer):
            old = layer.self_attn
            ts = Qwen2AttentionWithMemTS(old.config, old.layer_idx)
            ts.load_state_dict(old.state_dict())
            # 用本 batch 动态窗口（prefill 阶段以 overflow 裁 recent）
            ts.sliding_window = dy_sw
            # 保持设备一致，避免 CPU/GPU 混用导致报错
            try:
                dev = next(old.parameters()).device
            except StopIteration:
                dev = next(student.parameters()).device
            ts = ts.to(dev)
            layer.self_attn = ts

# ========== Teacher–Student 容器 ==========
class TSContainer(nn.Module):
    def __init__(self, student: Qwen2ForCausalLM, teacher: Qwen2ForCausalLM, distill_args: DistillArgs):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.distill_args = distill_args

    # 让 Trainer 能找到 config
    @property
    def config(self):
        return self.student.config

    def forward(self, **kwargs):
        # 仅为 Trainer 默认行为兼容；实际损失在 Trainer.compute_loss 中计算
        return self.student(**kwargs)

# ========== 自定义 Trainer：双路径 + 蒸馏三项损失 ==========
class DyMemDistillTrainer(Trainer):
    def compute_loss(self, model: TSContainer, inputs, return_outputs=False,**Kwargs):
        # 兼容 DataParallel/DDP 包裹
        m = model.module if hasattr(model, "module") else model
        student, teacher = m.student, m.teacher
        args: DistillArgs = m.distill_args

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        device = input_ids.device
        B, S = input_ids.shape

        # 动态滑窗（按本 batch 序列长度）
        sw_type = getattr(student.config, "sliding_window_type", "fixed")
        base_sw = getattr(student.config, "sliding_window", 4096)
        dy_sw = compute_dy_sliding_window(S, base_sw, sw_type)
        setattr(student.config, "dy_sliding_window", dy_sw)
        # 将动态窗口写入学生注意力，用于 recent 溢出裁剪与记忆压缩
        patch_student_attention(student, args, dy_sw)

        # 共享 embedding / 位置信息（同一 batch、同一输入）
        pos_ids = torch.arange(0, S, device=device)[None, :]
        inputs_embeds = student.get_input_embeddings()(input_ids)

        # Teacher——全量注意力（no_grad）
        with torch.no_grad():
            out_t = teacher(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids,
                use_cache=False,
                output_hidden_states=("distill" in args.loss_type),
                logits_to_keep=0,
            )

        # Student——记忆增强
        out_s = student(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids,
            use_cache=False,
            output_hidden_states=("distill" in args.loss_type),
            logits_to_keep=0,
        )

        logits_ref: torch.Tensor = out_t.logits     # [B,S,V]
        logits_mem: torch.Tensor = out_s.logits     # [B,S,V]
        hs_ref = list(out_t.hidden_states) if out_t.hidden_states is not None else None
        hs_mem = list(out_s.hidden_states) if out_s.hidden_states is not None else None

        # valid_mask：仅对“滑窗之外”的 token 生效（目标位置）
        ar = torch.arange(S, device=device)[None, :].expand(B, -1)
        valid_mask = ar >= dy_sw

        # 组合损失
        total_loss = logits_mem.new_tensor(0.0)
        logs = {}

        if "ce" in args.loss_type:
            l_ce = shift_ce_loss(logits_mem, labels, valid_mask)
            total_loss = total_loss + args.alpha_ce * l_ce
            logs["loss_ce"] = l_ce.item()

        if "kl" in args.loss_type:
            l_kl = masked_kl(logits_mem[:, :-1, :], logits_ref[:, :-1, :], valid_mask[:, 1:], args.temperature)
            total_loss = total_loss + args.alpha_kl * l_kl
            logs["loss_kl"] = l_kl.item()

        if "distill" in args.loss_type and hs_mem is not None and hs_ref is not None:
            l_mse = layer_mse_loss(hs_mem, hs_ref, valid_mask, args.distill_layers, l2=args.normalize_hidden)
            total_loss = total_loss + args.alpha_distill * l_mse
            logs["loss_distill_mse"] = l_mse.item()

        # 记录动态窗口
        logs["dy_sliding_window"] = int(dy_sw)
        self.log(logs)

        return (total_loss, (out_s if return_outputs else None))

    # 仅保存可训练参数（可选）
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 按需保存
        only_trainable = getattr(self.args, "only_save_trainable", False)
        if only_trainable:
            # 仅保存 student 的可训练参数
            model = self.model.module if hasattr(self.model, "module") else self.model
            student = model.student
            trainable_names = {n for n, p in student.named_parameters() if p.requires_grad}
            sd = {k: v for k, v in student.state_dict().items() if k in trainable_names}
            torch.save(sd, os.path.join(output_dir, "compressor_only.pt"))
            meta = {
                "base_model": getattr(self.args, "model_name_or_path", ""),
                "trainable_prefixes": model.distill_args.trainable_prefixes,
                "loss_type": model.distill_args.loss_type,
                "alpha": {
                    "ce": model.distill_args.alpha_ce,
                    "kl": model.distill_args.alpha_kl,
                    "distill": model.distill_args.alpha_distill
                },
                "temperature": model.distill_args.temperature,
                "sliding_window_type": getattr(model.config, "sliding_window_type", "fixed"),
                "sliding_window": getattr(model.config, "sliding_window", None),
            }
            with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            hf_logger.info(f"Saved trainable weights to {output_dir}/compressor_only.pt")
        else:
            # 常规保存整模型（包含被冻结参数）
            return super().save_model(output_dir, _internal_call)

# ========== 数据集 & collator ==========
def build_tokenized_ds(tokenizer, train_file: Optional[str], max_seq_len: int):
    """
    使用 HuggingFace Hub 上的 vllg/loong_c4 作为训练集；
    保持接口不变（参数与返回类型一致），仍返回 'train' split 的 Dataset。
    """
    from datasets import load_dataset  # 懒导入
    # 直接加载 vllg/loong_c4（包含 'text' 字段），得到 DatasetDict
    ds = load_dataset("vllg/loong_c4")

    # 如果传入的 train_file 是数字字符串，则视为希望使用的样本数量
    limit: Optional[int] = None
    if train_file is not None:
        try:
            limit = int(str(train_file).strip())
        except Exception:
            limit = None

    train = ds["train"]
    if limit is not None and limit > 0:
        n = min(limit, len(train))
        train = train.select(range(n))

    def tok(ex):
        out = tokenizer(
            ex["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors=None,
        )
        out["labels"] = out["input_ids"].copy()
        return out
    train = train.map(tok, batched=False, remove_columns=["text", "timestamp", "url"])
    return train

# ========== 构建 Teacher & Student ==========
def build_models(model_name_or_path: str, sliding_window: int, num_attn_sinks: int,
                 mem_max_len: int, sliding_window_type: str, distill_args: DistillArgs):
    register_customized_qwen2(exist_ok=True)
    # Teacher：全量注意力
    teacher_cfg = Qwen2Config.from_pretrained(model_name_or_path)
    teacher_cfg._layer_implementation = "Qwen2DecoderLayer"
    teacher_cfg.use_cache = False
    teacher = Qwen2ForCausalLM.from_pretrained(model_name_or_path, config=teacher_cfg)

    # Student：DyMem
    student_cfg = Qwen2Config.from_pretrained(model_name_or_path)
    student_cfg._layer_implementation = "Qwen2DyMemDecoderLayer"
    student_cfg.use_cache = False
    student_cfg.sliding_window = sliding_window
    student_cfg.num_attn_sinks = num_attn_sinks
    student_cfg.mem_max_len = mem_max_len
    student_cfg.sliding_window_type = sliding_window_type

    student = Qwen2ForCausalLM.from_pretrained(model_name_or_path, config=student_cfg)

    # 仅训练 compressor 相关
    mark_trainable(student, include_prefixes=distill_args.trainable_prefixes)

    return student, teacher

# ========== Main ==========
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True, help="每行一条样本的纯文本文件")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # 长序列/掩码
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--pad_to_multiple_of", type=int, default=128)
    parser.add_argument("--sliding_window", type=int, default=4096)
    parser.add_argument("--mem_max_len", type=int, default=1024)
    parser.add_argument("--num_attn_sinks", type=int, default=0)
    parser.add_argument("--sliding_window_type", type=str, choices=["fixed", "linear", "random"], default="random")

    # 蒸馏与路径控制
    parser.add_argument("--loss_type", nargs="+", default=["ce", "kl", "distill"])
    parser.add_argument("--alpha_ce", type=float, default=1.0)
    parser.add_argument("--alpha_kl", type=float, default=1.0)
    parser.add_argument("--alpha_distill", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--normalize_hidden", type=bool, default=True)
    parser.add_argument("--normalize_fused", type=bool, default=True)
    parser.add_argument("--gc_for_mem", type=bool, default=True)
    parser.add_argument("--train_prefill_as_decode", type=bool, default=True)
    parser.add_argument("--trainable_prefixes", nargs="+", default=["ahn.", "memory.", "router.", "compressor."])

    # 仅保存可训练参数
    parser.add_argument("--only_save_trainable", type=bool, default=False)

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # 模型
    distill_cfg = DistillArgs(
        loss_type=args.loss_type,
        alpha_ce=args.alpha_ce, alpha_kl=args.alpha_kl, alpha_distill=args.alpha_distill,
        temperature=args.temperature,
        normalize_hidden=args.normalize_hidden,
        normalize_fused=args.normalize_fused,
        gc_for_mem=args.gc_for_mem,
        train_prefill_as_decode=args.train_prefill_as_decode,
        trainable_prefixes=args.trainable_prefixes,
    )
    student, teacher = build_models(
        model_name_or_path=args.model_name_or_path,
        sliding_window=args.sliding_window,
        num_attn_sinks=args.num_attn_sinks,
        mem_max_len=args.mem_max_len,
        sliding_window_type=args.sliding_window_type,
        distill_args=distill_cfg
    )

    # 训练数据
    train_ds = build_tokenized_ds(tokenizer, args.train_file, args.max_seq_len)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=args.pad_to_multiple_of
    )

    # 容器与训练器
    container = TSContainer(student=student, teacher=teacher, distill_args=distill_cfg)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        report_to=[],  # 如需 W&B：["wandb"]
    )
    # 挂一个开关，供 save_model 使用
    setattr(targs, "only_save_trainable", args.only_save_trainable)
    setattr(targs, "model_name_or_path", args.model_name_or_path)

    trainer = DyMemDistillTrainer(
        model=container,
        args=targs,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
