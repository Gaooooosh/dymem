# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Optional, Any, override, Union
from transformers import DynamicCache, StaticSlidingWindowLayer
from transformers.cache_utils import Cache,DynamicCache,StaticCache
from fla.models.mamba2.modeling_mamba2 import Mamba2Cache
from transformers.configuration_utils import PretrainedConfig
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

def is_torchdynamo_compiling() -> Union[tuple[bool, str], bool]:
    if not torch.cuda.is_available():
        return False

class StaticSlidingWindowLayerHiddenOnly(StaticSlidingWindowLayer):
    is_sliding = True
    def __init__(self, max_cache_len: int, sliding_window: int):
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len, sliding_window=sliding_window)
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        self.max_batch_size, _, self.hidden_size = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = None
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            # torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def update(
        self,
        hidden_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor]:

        # Hidden_states -> [B,L,H]
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(hidden_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-21 == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position
            if cache_position is not None
            else torch.arange(hidden_states.shape[1], device=self.device)
        )

        cumulative_length = self.cumulative_length
        is_full = cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += hidden_states.shape[1]

        if is_full:
            # In general, we should use a much simpler `cat` here as well, independently of the states size. However,
            # dynamo is currently bugged when doing it - see https://github.com/pytorch/pytorch/issues/159855 for more details
            if hidden_states.shape[1] == 1:
                # Roll all values to the left by 1 position
                new_hidden_states = self.keys.roll(-1, dims=1)
                # Overwrite the last position with new states
                # (note: very important to use a tensor to index here, see https://github.com/pytorch/pytorch/issues/159855)
                index = torch.tensor([-1], dtype=int, device=self.device)
                new_hidden_states[:, index, :] = hidden_states

                # Copy back into `self` (do not just assign again) in order to keep the static dynamo address
                self.keys.copy_(new_hidden_states)
                # Very important to return the `self` tensors here, as they have the static dynamo address
                return self.keys
            # Already full but using more than 1 new token (e.g. prefill caching, chat continuation, etc...)
            else:
                full_hidden_states = torch.cat((self.keys[:, 1:, :], hidden_states), dim=1)
        # Not yet full, but becoming full on this update
        elif cumulative_length + hidden_states.shape[1] > self.max_cache_len:
            # Fast prefill path, no need to cat() in this case, as the cache is currently empty
            if cumulative_length == 0:
                full_hidden_states = hidden_states
            else:
                full_hidden_states = torch.cat((self.keys[:, :cumulative_length, :], hidden_states), dim=1)
        else:
            try:
                self.keys.index_copy_(1, cache_position, hidden_states)
            except NotImplementedError:
                self.keys[:, cache_position, :] = hidden_states

            # Very important to return the `self` tensors here, as they have the static dynamo address
            return self.keys

        # We only cache the last `sliding_window` tokens
        self.keys.copy_(full_hidden_states[:, -self.max_cache_len :, :])
        # we should return the whole states instead of `self.keys` here, as otherwise we lose some context
        return full_hidden_states

class StaticHiddenCache(StaticCache):
    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(
        self,
        config: PretrainedConfig,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        max_cache_len = config.sliding_window
        layer_types = getattr(config, "layer_types", None)
        super().__init__(config,max_cache_len=max_cache_len, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        self.layers = []
        for layer_type in range(config.num_hidden_layers):
            layer = StaticSlidingWindowLayerHiddenOnly(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
            self.layers.append(layer)



    def update(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(hidden_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)
            
        keys = self.layers[layer_idx].update(hidden_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys

class CacheWithMem(DynamicCache):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from fla.models.mamba2.configuration_mamba2 import Mamba2Config
        
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_groups = num_heads // config.num_key_value_heads
        mamba_config = Mamba2Config(
            dtype=config.dtype,
            conv_kernel=config.conv_kernel,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            state_size = config.state_size,
            expand = 1,
            n_groups=num_kv_groups,
            chunk_size=config.chunk_size,
            num_hidden_layers = config.num_hidden_layers,
        )

        self.mem_cache = Mamba2Cache(mamba_config,batch_size=1,dtype=config.dtype,*args, **kwargs)
        self.hidden_cache = StaticHiddenCache(config, *args, **kwargs)
