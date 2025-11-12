# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from typing import Optional
from transformers.cache_utils import Cache
from fla.models.utils import Cache as MemCache
def register_custom_accelerator():
    from accelerate import Accelerator

    # Store the original method to call it later
    original_backward = Accelerator.backward

    # Define the new method
    def new_backward(self, loss, retain_graph=True, **kwargs):
        """
        Custom backward function with retain_graph=True by default.
        Calls the original backward method.
        """
        kwargs.setdefault("retain_graph", retain_graph)  # Ensure retain_graph is set
        original_backward(self, loss, **kwargs)  # Call the original method

    # Replace the original method
    Accelerator.backward = new_backward


def repeat_memkv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return torch.repeat_interleave(hidden_states, dim=2, repeats=n_rep)



class GroupLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        use_bias: bool = False,
    ):
        super().__init__()
        assert (
            in_features % num_groups == 0
        ), "in_features must be divisible by num_groups"
        assert (
            out_features % num_groups == 0
        ), "out_features must be divisible by num_groups"
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_in = in_features // num_groups
        self.group_out = out_features // num_groups

        self.weight = nn.Parameter(
            torch.Tensor(self.num_groups, self.group_in, self.group_out)
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.num_groups, self.group_out))
            if self.use_bias
            else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.zero_()

    def forward(self, x):
        B, L, _ = x.shape  # (B, L, in_features)
        x = x.view(B, L, self.num_groups, self.group_in)  # (B, L, num_groups, group_in)
        x = x.reshape(
            -1, self.num_groups, self.group_in
        )  # (B * L, num_groups, group_in)
        out = torch.einsum(
            "bgi,gio->bgo", x, self.weight
        )  # (B * L, num_groups, group_out)
        if self.use_bias:
            out = out + self.bias
        out = out.reshape(
            B, L, self.num_groups * self.group_out
        )  # (B, L, out_features)
        return out

class CacheWithMem(Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_cache = MemCache(*args, **kwargs)

class BaseAHN(nn.Module):
    def __init__(
        self,
        ahn_cls: nn.Module,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mode: str = "chunk",
        **kwargs
    ):
        """
        Naive constant-space memory following the recurrence formula M_t = f(M_{t-1}, q).
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.fn = ahn_cls(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mode=mode,
            expand=1,
            **kwargs
        )
        self.cache = None  # Keys: recurrent_states


    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_states: Optional[torch.Tensor] = None,
    ):
        return self.fn(
            hidden_states=hidden_states,
            recurrent_states=recurrent_states,
        )
