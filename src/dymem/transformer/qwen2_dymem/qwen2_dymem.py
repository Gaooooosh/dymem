from __future__ import annotations
import os
from math import log
import torch
torch.backends.cudnn.conv.fp32_precision = 'tf32'
import wandb
import random
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List, Dict, Any
from dymem.transformer.qwen2 import (
    Qwen2Config as Qwen2Config_,
    Qwen2Model as Qwen2Model_,
    Qwen2ForCausalLM as Qwen2ForCausalLM_,
)
import math
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from dymem.transformer.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
    rotate_half,
    eager_attention_forward,
)

from dymem.utils import CacheWithMem, CausalLMOutputWithPastAndRecLoss
from dymem.rnn.mamba2 import Mamba2
from dataclasses import dataclass
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward, FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    add_start_docstrings,
    can_return_tuple,
    # deprecate_kwarg,
    logging,
)
from torch.nn.functional import scaled_dot_product_attention as sdpa
logger = logging.get_logger(__name__)



@dataclass
class CustomizedCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    mem_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Qwen2Config(Qwen2Config_):
    def __init__(
        self,
        _layer_implementation: Optional[str] = "Qwen2DyMemDecoderLayer",
        _ahn_implementation: Optional[str] = "Mamba2",
        use_compressor: bool = True,
        sliding_window: Optional[int] = 256,
        num_attn_sinks: int = 128,
        mem_max_len: int = 1024,
        fa_layer_ids: str = "",
        conv_kernel: int = 4,
        state_size: int = 256,
        chunk_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._layer_implementation = _layer_implementation
        self._ahn_implementation = _ahn_implementation
        self.use_compressor = use_compressor
        self.sliding_window = sliding_window
        self.num_attn_sinks = num_attn_sinks
        self.mem_max_len = mem_max_len
        self.fa_layer_ids = fa_layer_ids or ""
        self.conv_kernel = conv_kernel
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.dtype = dtype


class ReusedProjReconstructor(nn.Module):
    """
    用“本层已有的 q_proj/k_proj/v_proj/o_proj”做 cross-attn 重建：
      pred = Attn( Q = q_proj(query_hidden),
                   K,V = k_proj(mem_act), v_proj(mem_act) ) -> o_proj
    query_hidden 由位置编码生成（learnable 或 fixed sincos）

    这个模块本身不创建注意力权重，不会新增 Q/K/V/O 参数。
    可选新增一个 rec_pos_emb（learnable），参数量很小。
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scaling: float,
        mem_max_len: int,
        use_learnable_pos: bool = False,
        loss_type: str = "mse",          # "mse" | "smooth_l1"
        smooth_l1_beta: float = 0.1,
        use_layernorm_on_q: bool = False,
        num_kv_heads: int = 16,
        num_kv_groups: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.mem_max_len = mem_max_len
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups
        self.use_learnable_pos = use_learnable_pos
        if use_learnable_pos:
            self.rec_pos_emb = nn.Embedding(mem_max_len, hidden_size)
        else:
            self.rec_pos_emb = None

        self.loss_type = loss_type
        self.smooth_l1_beta = smooth_l1_beta

        self.use_layernorm_on_q = use_layernorm_on_q
        self.q_ln = nn.LayerNorm(hidden_size) if use_layernorm_on_q else nn.Identity()

    @staticmethod
    def _fixed_sincos_pos(E: int, H: int, device, dtype):
        # [E, H] fixed sin/cos positional encoding (no params)
        half = H // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device, dtype=dtype) / max(1, half)
        )
        pos = torch.arange(E, device=device, dtype=dtype).unsqueeze(1)  # [E,1]
        ang = pos * freq.unsqueeze(0)                                  # [E,half]
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)        # [E,2*half]
        if H % 2 == 1:
            pe = F.pad(pe, (0, 1))
        return pe  # [E,H]

    def build_query_hidden(self, B: int, E: int, device, dtype):
        pos = torch.arange(E, device=device, dtype=torch.long)
        pos = pos.clamp(max=self.mem_max_len - 1)

        if self.rec_pos_emb is not None:
            q_hidden = self.rec_pos_emb(pos).to(dtype=dtype)  # [E,H]
        else:
            q_hidden = self._fixed_sincos_pos(E, self.hidden_size, device, dtype)

        q_hidden = q_hidden.unsqueeze(0).expand(B, -1, -1)  # [B,E,H]
        q_hidden = self.q_ln(q_hidden)
        return q_hidden, pos

    def forward(self, mem_act: torch.Tensor, E: int, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module, o_proj: nn.Module):
        """
        mem_act: [B, M, H]
        E: 要重建的长度
        return pred: [B, E, H]
        """
        B, M, H = mem_act.shape
        assert H == self.hidden_size, f"hidden mismatch: mem_act H={H} vs {self.hidden_size}"

        device, dtype = mem_act.device, mem_act.dtype
        q_hidden, _ = self.build_query_hidden(B, E, device, dtype)  # [B,E,H]

        # project -> [B, nh, L, hd]
        B, M, H = mem_act.shape
        q = q_proj(q_hidden).view(B, E, self.num_heads, self.head_dim).transpose(1, 2)  # [B,nh,E,hd]
        k = k_proj(mem_act).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, kvh, M, hd]
        v = v_proj(mem_act).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, kvh, M, hd]

        # 扩展 K/V 到 num_heads（GQA: 每个 kv head 复用给多个 q head）
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)  # [B, num_heads, M, hd]
            v = v.repeat_interleave(self.num_kv_groups, dim=1)  # [B, num_heads, M, hd]
        # 更快：SDPA（Pytorch 2.x）
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=0.5
        )  # [B,nh,E,hd]

        # merge heads -> [B,E,H] and o_proj
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, E, H)
        pred = o_proj(attn_out)  # [B,E,H]
        return pred

    def loss(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred/target: [B,E,H]
        默认建议 target 传入 detach() 后的张量（外部做 detach）
        """
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction="mean", beta=self.smooth_l1_beta)
        return F.mse_loss(pred, target, reduction="mean")

class Qwen2AttentionWithMem(Qwen2Attention):

    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.num_attn_sinks = getattr(config, 'num_attn_sinks', 128)
        self.sliding_window = getattr(config, 'sliding_window', 2048)
        self.use_compress = getattr(config, 'use_compressor', True)
        
        if not self.use_compress:
            logger.warning_once(f"【DyMem】- [Warning] Config does not use compressor")

        self.num_mem_tokens = getattr(config, "num_mem_tokens", 1)
        if self.num_mem_tokens < 1 and self.use_compress:
            self.num_mem_tokens = 1

        self.mem_max_len = getattr(config, 'mem_max_len', 2048)
        self.compressor = Mamba2(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            hidden_size=config.hidden_size,
            state_size = config.state_size,
            expand = 1,
            layer_idx=layer_idx,
            n_groups=self.num_kv_groups,
            chunk_size=config.chunk_size,
        )
        # 段缓存
        self.register_buffer('sink_k', None, persistent=False)
        self.register_buffer('sink_v', None, persistent=False)
        self.register_buffer('mem_k', None, persistent=False)
        self.register_buffer('mem_v', None, persistent=False)
        self.register_buffer('active_k', None, persistent=False)
        self.register_buffer('active_v', None, persistent=False)

        self.recent_k = None  # [B,H,W,D]
        self.recent_v = None
        # 在 Qwen2AttentionWithMem.__init__ 末尾加

        self.rec_head = ReusedProjReconstructor(
            hidden_size=config.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            num_kv_groups=self.num_kv_groups,
            scaling=self.scaling,
            mem_max_len=self.mem_max_len,
            use_learnable_pos=getattr(config, "rec_learnable_pos", False),  # True=新增pos表；False=零参数
            loss_type=getattr(config, "rec_loss_type", "smooth_l1"),
            smooth_l1_beta=getattr(config, "rec_smooth_l1_beta", 0.1),
            use_layernorm_on_q=getattr(config, "rec_ln_q", False),
        )
        self._last_rec_loss = None  # 每次 forward 会覆盖


    @staticmethod
    def apply_rotary_emb_single(
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed

    def _local_update_causal_mask(self, query_states, key_states):
        q_len = query_states.shape[2]
        if q_len==1:
            return None

        k_len = key_states.shape[2]

        dtype = query_states.dtype
        device = query_states.device
        min_val = torch.finfo(dtype).min

        # [Lq, Lk]
        mask = torch.zeros((q_len, k_len), device=device, dtype=dtype)
        # 上三角（严格）置为 -inf，实现因果性
        mask.masked_fill_(torch.ones_like(mask, dtype=torch.bool).triu_(1), min_val)
        # 变成 [1,1,Lq,Lk]
        return mask[None, None, :, :]

    def _sample_indices_include_last(self, E: int, n: int, device, mode: str = "stratified"):
        """
        返回 [n_eff] 的 idx，保证包含 E-1，且排序后单调。
        """
        n_eff = min(n, E)
        if n_eff <= 0:
            return None, 0
        if n_eff == 1:
            return torch.tensor([E - 1], device=device, dtype=torch.long), 1

        # 需要从 [0, E-1) 采样 n_eff-1 个
        k = n_eff - 1
        if mode == "uniform":
            idx_other = torch.randperm(E - 1, device=device)[:k]
        else:
            # 分桶采样覆盖更均匀：把 [0, E-1) 分成 k 个桶，每桶取 1 个
            edges = torch.linspace(0, E - 1, steps=k + 1, device=device)
            picks = []
            for i in range(k):
                start = int(edges[i].item())
                end = int(edges[i + 1].item())
                if end <= start:
                    end = min(start + 1, E - 1)
                j = torch.randint(low=start, high=end, size=(1,), device=device)
                picks.append(j)
            idx_other = torch.cat(picks, dim=0)

        idx = torch.cat([idx_other, torch.tensor([E - 1], device=device, dtype=torch.long)], dim=0)
        idx, _ = torch.sort(idx)
        return idx, n_eff


    def forward(
        self,
        prenormed_hidden: torch.Tensor,                                 # [B,T,H]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin) for new tokens
        past_key_value: Optional[CacheWithMem] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        B, T, H = prenormed_hidden.shape

        self.sliding_window = self.config.sliding_window
        input_shape = prenormed_hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # 预填充阶段可能会更改序列长度，先处理压缩拼接，再计算投影视图形状
        mem_act = None # the mem_tensor that is used to attend to the current token
        mem_out = None # mem_out is the output hidden_state of compressor
        evt_len = 0
        if T > 1: # prefilling
            w = 0 if (self.sliding_window is None or self.sliding_window <= 0) else self.sliding_window
            sink_len = min(self.num_attn_sinks, T)
            keep_window_len = w
            evict_end_index = max(0, T - keep_window_len)
            overflow = max(0, evict_end_index - sink_len)
            if self.compressor is not None:
                if overflow > 0:
                    # 正确的被压缩段：去掉前端 sink 与末端滑窗，中间段作为被驱逐 token
                    evicted_hidden_states = prenormed_hidden[:, sink_len:evict_end_index, :]
                    E = evicted_hidden_states.shape[1]
                    M = self.num_mem_tokens
                    n_eff = 0
                    if self.use_compress:
                        idx, n_eff = self._sample_indices_include_last(E, M, device=evicted_hidden_states.device, mode="stratified")
                        mem_out = self.compressor(
                            evicted_hidden_states,
                            cache_params=past_key_value.mem_cache if (self.config.use_cache and past_key_value is not None and not self.training) else None,
                            cache_position=cache_position,
                        )  # [B, EVC_T, H]
                        mem_act = mem_out.index_select(dim=1, index=idx)  # [B, n_eff, H]
                        # mem_act = mem_out[:, -1:, :] # [B, 1, H]
                        evt_len += mem_out.shape[1]

                        # ====== Rec loss (方案A): 用 mem_act 重建对应位置的 evicted hidden ======
                        self._last_rec_loss = None
                        if self.training and (mem_act is not None):
                            target = evicted_hidden_states.detach()  # [B,E,H]
                            E = target.size(1)

                            pred = self.rec_head(mem_act, E, self.q_proj, self.k_proj, self.v_proj, self.o_proj)    # [B,E,H]
                            with torch.no_grad():
                                target_pred = self.rec_head(target, E, self.q_proj, self.k_proj, self.v_proj, self.o_proj)    # [B,E,H]
                                tgt_n  = F.layer_norm(target_pred, (target_pred.size(-1),))
                            pred_n = F.layer_norm(pred, (pred.size(-1),))
                            self._last_rec_loss = self.rec_head.loss(pred_n, tgt_n)

                    if past_key_value is not None and self.config.use_cache and self.use_compress:
                        past_key_value.hidden_cache.update(prenormed_hidden[:,evict_end_index:, :], layer_idx=self.layer_idx)
                        past_key_value.mem_position_embed.append((position_embeddings[0][:, sink_len, :],position_embeddings[1][:, sink_len, :]))
                        past_key_value.evt_len = evt_len
                        past_key_value.mem_bank[self.layer_idx] = mem_act.detach()
                        past_key_value.mem_valid[self.layer_idx] = n_eff

                    prenormed_hidden = torch.cat([prenormed_hidden[:, :sink_len + n_eff, :], prenormed_hidden[:, evict_end_index:, :],],dim=1,)
                    
                    position_embeddings = (
                        torch.cat([position_embeddings[0][:, :sink_len + n_eff, :], position_embeddings[0][:, evict_end_index:, :]], dim=1),
                        torch.cat([position_embeddings[1][:, :sink_len + n_eff, :], position_embeddings[1][:, evict_end_index:, :]], dim=1),
                    )

                elif past_key_value is not None and T > sink_len and self.config.use_cache:
                        past_key_value.hidden_cache.update(prenormed_hidden[:,sink_len+1:, :], self.layer_idx)
        else: # decode
            mem_act = None
            if past_key_value is not None:
                layer_cache = past_key_value.hidden_cache.layers[self.layer_idx]

                # 2) 若需要，先压缩这些视图；跨回绕时顺序馈入压缩器即可
                if self.compressor is not None and self.use_compress:
                    ev_views = layer_cache.eviction_slices(prenormed_hidden.shape[1])  # T==1 -> 1
                    if ev_views:
                        mem_out_step = None
                        for evt in ev_views:
                            mem_out_step = self.compressor(
                                evt,
                                cache_params=past_key_value.mem_cache,
                                cache_position=cache_position,
                            )
                        new_last = mem_out_step[:, -1:, :]
                        evt_len += mem_out_step.shape[1]
                        past_key_value.evt_len += mem_out_step.shape[1]
                    
                        M = self.num_mem_tokens
                        bank = past_key_value.mem_bank[self.layer_idx]
                        valid = past_key_value.mem_valid[self.layer_idx]
                        if bank is None and new_last is not None:
                            # 这里用 0 填充旧 mem，最后一个是 new_last
                            pad = new_last.new_zeros((B, max(M - 1, 0), H))
                            bank = torch.cat([pad, new_last], dim=1) if M > 1 else new_last
                            valid = bank.shape[1]
                            past_key_value.mem_bank[self.layer_idx] = bank.detach()
                            past_key_value.mem_valid[self.layer_idx] = valid
                            mem_act = bank
                        else:
                            # 只更新最后一个 mem
                            if new_last is not None and valid > 0:
                                # bank = bank.clone()  # 避免原地修改导致 autograd/共享问题
                                bank[:, valid - 1:valid, :] = new_last
                                past_key_value.mem_bank[self.layer_idx] = bank.detach()
                            mem_act = bank

                # 3) 最后再把新 token 写入缓存（O(1) 次数的切片 copy）
                past_key_value.hidden_cache.update(prenormed_hidden, self.layer_idx)


        # 压缩拼接后重新计算形状并保证连续性
        B, T, H = prenormed_hidden.shape # prefilling stage: the len will be sliding_window + 1 when mem_act is not None, the extra one position is pre-alloc for mem_qkv
        input_shape = prenormed_hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 线性投影 
        q_states = self.q_proj(prenormed_hidden).view(hidden_shape).transpose(1, 2) # [B, H, T, D]
        k_states = self.k_proj(prenormed_hidden).view(hidden_shape).transpose(1, 2)
        v_states = self.v_proj(prenormed_hidden).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)
        value_states = v_states
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
        if mem_act is not None:
            sink_len = self.num_attn_sinks
            n_eff = mem_act.shape[1]  # [B, n_eff, H]
            mem_hidden_shape = (B, n_eff, -1, self.head_dim)
            mem_k = self.k_proj(mem_act).view(mem_hidden_shape).transpose(1, 2) # [B, H, 1, D]
            mem_v = self.v_proj(mem_act).view(mem_hidden_shape).transpose(1, 2) # [B, H, 1, D]
            # mem_q,mem_k = apply_rotary_pos_emb(mem_q, mem_k, *past_key_value.mem_position_embed[-1])
            key_states[:, :, sink_len:sink_len + n_eff, :] = mem_k
            value_states[:, :, sink_len:sink_len + n_eff, :] = mem_v
            # if T > 1:
            #     mem_q = self.q_proj(mem_act).view(mem_hidden_shape).transpose(1, 2)
            #     query_states[:, :, sink_len:sink_len + n_eff, :] = mem_q

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                pass
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 生成 Mask
        causal_mask = self._local_update_causal_mask(query_states, key_states)
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        mask_arg = None if self.config._attn_implementation == "flash_attention_2" else causal_mask
        
        if mem_act is not None and mask_arg is not None:
            # attention_mask 形状一般是 [B, 1, Lq, Lk_total]
            bsz, _, q_len, k_len_total = mask_arg.shape
            k_len = key_states.shape[-2]          # 经过 cache/sliding_window 后真正参与注意力的 key 长度
            mem_pos = self.num_attn_sinks                # 假设 mem token 在最后一个 key 位置
            n_eff = mem_act.shape[1]

            # 截取当前会用到的部分
            mask_arg = mask_arg[:, :, :, :k_len]
            evt_len = evt_len if ((self.config.use_cache and past_key_value is not None) or self.training) else past_key_value.evt_len
            evt_len = max(evt_len, 1)
            # 构造一个全 0 bias mask，只在 mem 列加上 bias
            
            Le_fact = evt_len / (k_len + evt_len)
            mem_bias_mask = torch.zeros_like(mask_arg)
            mem_bias_mask[..., mem_pos:mem_pos + n_eff] = 1 * Le_fact / n_eff

            # 叠加：原来的 causal mask + mem bias
            mask_arg = mask_arg + mem_bias_mask

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            mask_arg,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        # 将 [B, T, heads, head_dim] 转为 [B, T, heads*head_dim]
        attn_output = attn_output.reshape(B, T, -1)
        if T > 1:
            if mem_out is not None:    
                attn_output = torch.cat([attn_output[:,:sink_len,:], mem_out[:,:-mem_act.shape[1],:], attn_output[:,sink_len:,:]], dim=1)
            elif evicted_hidden_states is not None:
                attn_output = torch.cat([attn_output[:,:sink_len,:], evicted_hidden_states, attn_output[:,sink_len:,:]], dim=1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen2DyMemDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2AttentionWithMem(config, layer_idx)
        self._do_solidify_next = False

    def mark_solidify(self):
        self._do_solidify_next = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CacheWithMem] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            prenormed_hidden=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs



class Qwen2Model(Qwen2Model_):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        layer_cls = DECODER_LAYER_CLS[config._layer_implementation]
        self.layer_cls = config._layer_implementation
        self.layers = nn.ModuleList(
            [
                layer_cls(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[CacheWithMem] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), CacheWithMem)):
            raise ValueError("The `past_key_values` should be either a `MemCache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = CacheWithMem(self.config)


        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        if self.training:
            w = random.choice(self.config.windows_choices)
            num_mem = random.choice([1,2,4,8,16,24,32,64,128])
            # print(f"sliding_window: {w}")
            for layer in self.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.sliding_window = w
                    layer.self_attn.num_mem_tokens = num_mem
        else:
            self.sliding_window = self.config.sliding_window
            # print(f"sliding_window: {self.sliding_window}")


        rec_loss_total = None  # 用 None 避免无压缩时创建无意义 tensor
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
    
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            # 累加来自 self_attn 的 rec loss
            if self.training and hasattr(decoder_layer, "self_attn"):
                lrec = getattr(decoder_layer.self_attn, "_last_rec_loss", None)
                if lrec is not None:
                    rec_loss_total = lrec if rec_loss_total is None else (rec_loss_total + lrec)


            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 如果没有发生压缩，给一个 0，保持接口稳定
        if rec_loss_total is None:
            rec_loss_total = hidden_states.new_zeros(())
        self.rec_loss_total = rec_loss_total


        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Qwen2ForCausalLM(Qwen2ForCausalLM_):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[CacheWithMem] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPastAndRecLoss:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss_main = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            # 取模型里累加好的 rec loss
            loss_rec = getattr(self.model, "rec_loss_total", None)
            if loss_rec is None:
                loss_rec = hidden_states.new_zeros(())

            lam = getattr(self.config, "lambda_rec", 0.01)  # 你也可以写死或从 config 读
            loss = loss_main + lam * loss_rec
            self.loss_main_last = loss_main.detach()

        return CausalLMOutputWithPastAndRecLoss(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_rec=loss_rec.detach() if self.training else None,         # <= 关键：单独返回，detach 便于记录
            loss_main=loss_main.detach() if self.training else None,       # <= 可选：主 loss 也记录
        )


DECODER_LAYER_CLS = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2DyMemDecoderLayer": Qwen2DyMemDecoderLayer,
}

AHN_CLS = {
    # "GatedDeltaNet": GatedDeltaNet,
    # "DeltaNet": DeltaNet,
    "Mamba2": Mamba2,
}

def register_customized_qwen2(exist_ok=True):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register("qwen2", Qwen2Config, exist_ok=exist_ok)
    AutoModel.register(Qwen2Config, Qwen2Model, exist_ok=exist_ok)
    AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=exist_ok)
