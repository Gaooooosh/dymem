# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE.
#
# This modified file is released under Apache-2.0.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules.layernorm import RMSNorm
from transformers.utils import logging

from ..utils import GroupLinear
from fla.modules.activations import ACT2FN
from fla.modules.layernorm_gated import RMSNormGated

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
    except ImportError:
        selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None, None
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_update, causal_conv1d_fn = None, None
    is_fast_path_available = selective_state_update is not None

if TYPE_CHECKING:

    from fla.models.mamba2.modeling_mamba2 import Mamba2Cache
    from fla.models.utils import Cache
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] ->
        # [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def _ensure_cuda(name, t):
    if t is None: 
        return
    assert t.is_cuda, f"{name} is on {t.device}, expected CUDA"

def _finite_stats(name, t):
    with torch.no_grad():
        print(f"[{name}] shape={tuple(t.shape)} dtype={t.dtype} "
              f"min={float(t.min()) if t.numel() else float('nan'):.4g} "
              f"max={float(t.max()) if t.numel() else float('nan'):.4g} "
              f"mean={float(t.mean()) if t.numel() else float('nan'):.4g} "
              f"NaN={int(torch.isnan(t).sum())} Inf={int(torch.isinf(t).sum())}")

class Mamba2(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 1024,
        state_size: int = 128,
        rms_norm: bool = True,
        chunk_size: int = 256,
        time_step_rank: float = 256,
        time_step_limit: Tuple[float, float] = (0.0, float("inf")),
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        use_bias: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        backend: str = "cuda",
        **kwargs,
    ) -> Mamba2:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size


        self.rms_norm = rms_norm
        self.norm_eps = norm_eps

        self.chunk_size = chunk_size

        self.time_step_rank = int(time_step_rank)
        self.time_step_limit = time_step_limit
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max


        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = RMSNormGated(
            self.num_heads * self.head_dim, eps=self.norm_eps, norm_before_gate=False
        )
        # self.norm = RMSNorm(self.num_heads * self.head_dim, eps=self.norm_eps)
        # self.norm = nn.LayerNorm(self.num_heads * self.head_dim, eps=self.norm_eps)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
        self.layer_idx = layer_idx

        self.dt_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        self.g_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        self.o_proj = GroupLinear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, self.num_heads)
        if self.time_step_limit == (0.0, float("inf")):
            self.time_step_limit = (self.time_step_min, self.time_step_max)
        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of "
                "`(selective_state_update)` is None. "
                "Falling back to the naive implementation. "
                "To install follow https://github.com/state-spaces/mamba/#installation"
            )

    def get_beta_alpha(self, hidden_states: torch.Tensor):
        # 遵循输入 dtype，不强制转换为 FP32
        beta = self.dt_proj(hidden_states)
        return beta, None


    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        Batch, T, _ = hidden_states.shape
        
        # Normalize the hidden states (if needed)
        # Compute dt for state update
        dt = self.dt_proj(hidden_states)  # Compute time step # [B,T,H]

        # Gate projection
        gate = torch.tanh(self.g_proj(hidden_states)).repeat_interleave(self.head_dim, dim=-1)

        A = -torch.exp(self.A_log)  # (nheads,)

        # Repeat time step and bias for the state update
        # dt = dt[:, :, None].expand(-1, -1, self.head_dim)
        # Prepare states for transformation (B, C)
        B = hidden_states.view(Batch, T, self.num_heads, self.head_dim).contiguous()
        C = hidden_states.view(Batch, T, self.num_heads, self.head_dim).contiguous()

        # Single step calculations via cache
        # if cache_params is not None and cache_position is not None and cache_position[0] > 0:
        if T == 1 and recurrent_state is not None: # hardcode here, assuming training seq is always > 1
            output = selective_state_update(
                recurrent_state,
                B,
                dt,
                A,
                C,
                D=self.D,
                dt_bias=self.dt_bias
            ) # out: (batch, dim) or (batch, nheads, dim)
            out = out.unsqueeze(1) # (batch, 1, nheads, dim)
            out = self.norm(out,gate)
        # Fused calculations or step by step if no initialized cache is found
        else:
            output = mamba_chunk_scan_combined(
                hidden_states.view(Batch, T, self.num_heads, self.head_dim).contiguous(),
                dt,
                A,
                B,
                C,
                chunk_size=self.chunk_size,
                D=self.D,
                dt_bias=self.dt_bias,
                return_final_states=True
            ) # out: (batch, seqlen, nheads, headdim)
            out = self.norm(out,gate)
            out = out[:,-1,:,:] # (batch, 1, nheads, headdim)
        # 4. Final linear projection
        out = self.o_proj(out)
        return out # (batch, 1, nheads, headdim)
