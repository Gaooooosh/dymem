# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Optional, Any, override, Union
from transformers import DynamicCache, StaticSlidingWindowLayer
from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from transformers.cache_utils import Cache,DynamicCache,StaticCache,StaticLayer
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
        self.write_pos = 0      # 下次写入位置（等价于“队尾”）
        self.valid_len = 0      # 已填充长度（<= max_cache_len）

    def lazy_initialization(self, key_states: torch.Tensor):
        self.max_batch_size, _, self.hidden_size = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        # 用 empty 避免初始化填零的显著开销；我们只会读取到已经被写入的区域
        self.keys = torch.empty(
            (self.max_batch_size, self.max_cache_len, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = None

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)

        self.is_initialized = True

    @property
    def is_full(self) -> bool:
        return self.valid_len >= self.max_cache_len

    def eviction_slices(self, incoming_len: int) -> list[torch.Tensor]:
        """
        返回“将被覆盖”的旧 token 视图（1 或 2 段），在写入前调用。
        仅在环已满、或 incoming_len 超过剩余空间时非空。
        """
        if incoming_len <= 0 or self.max_cache_len <= 0:
            return []

        cap = self.max_cache_len
        # 需要驱逐的数量：当已有长度 + 新写入 > 容量时
        need_evict = max(0, self.valid_len + incoming_len - cap)
        if need_evict == 0:
            return []

        start = self.write_pos  # 最早被覆盖的位置
        end = start + need_evict
        if end <= cap:
            # 不回绕：一段视图
            return [self.keys.narrow(1, start, need_evict)]
        else:
            # 回绕：两段视图
            first = cap - start
            second = need_evict - first
            return [self.keys.narrow(1, start, first),
                    self.keys.narrow(1, 0, second)]

    def update(
        self,
        hidden_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        # hidden_states: [B, L, H]
        if not self.is_initialized:
            self.lazy_initialization(hidden_states)

        B, L, H = hidden_states.shape
        assert H == self.hidden_size, "hidden_size mismatch"

        cap = self.max_cache_len
        pos = self.write_pos
        end = pos + L

        # 写入：最多两段（若跨回绕）
        if end <= cap:
            # 单段写入
            self.keys.narrow(1, pos, L).copy_(hidden_states)
        else:
            # 两段写入：先 [pos:cap)，再 [0:end-cap)
            first = cap - pos
            second = L - first

            # 如果 L > cap，只保留最后 cap 个 token
            if L > cap:
                # 丢弃最前面的 L-cap；只写最后 cap 个
                offset = L - cap
                # 写到 [pos:cap)
                part1 = min(first, cap)
                if part1 > 0:
                    self.keys.narrow(1, pos, part1).copy_(hidden_states.narrow(1, offset, part1))
                # 写到 [0:cap-part1)
                part2 = cap - part1
                if part2 > 0:
                    self.keys.narrow(1, 0, part2).copy_(hidden_states.narrow(1, offset + part1, part2))
                L = cap  # 实际有效写入 cap
                end = (pos + L)  # 仅用于下方指针更新
            else:
                # 正常两段
                self.keys.narrow(1, pos, first).copy_(hidden_states.narrow(1, 0, first))
                self.keys.narrow(1, 0, second).copy_(hidden_states.narrow(1, first, second))

        # 指针/长度/统计更新
        self.write_pos = (pos + L) % cap
        self.valid_len = min(cap, self.valid_len + L)
        self.cumulative_length += L

        # 返回拥有静态地址的张量本体
        return self.keys

class StaticHiddenCache(StaticCache):
    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(
        self,
        config: PretrainedConfig,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        max_cache_len = config.sliding_window
        config = config.get_text_config(decoder=True)
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

class Mamba2Cache:
    def __init__(
        self,
        config: Mamba2Config,
        batch_size: int,
    ):
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)
        self.num_hidden_layers = config.num_hidden_layers
        self.batch_size = batch_size
        self.is_initialized = False

    def lazy_initialization(self, state: torch.Tensor):
        self.dtype, self.device = state.dtype, state.device

        # 用 empty 避免初始化填零的显著开销；我们只会读取到已经被写入的区域
        self.conv_states = torch.empty(
            self.num_hidden_layers,
            self.batch_size,
            self.intermediate_size + 2 * self.n_groups * self.state_size,
            self.conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.ssm_states = torch.empty(
            self.num_hidden_layers,
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=self.device,
            dtype=self.dtype,
        )

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.conv_states)
            torch._dynamo.mark_static_address(self.ssm_states)

        self.is_initialized = True

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_init: bool = False
    ) -> torch.Tensor:
        if not self.is_initialized:
            self.lazy_initialization(new_conv_state)
        
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        if not self.is_initialized:
            self.lazy_initialization(new_ssm_state)

        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        if not self.is_initialized:
            self.lazy_initialization(state)
        self.conv_states.zero_()
        self.ssm_states.zero_()

class StaticSlidingWindowLayerWithSink(StaticLayer):
    def __init__(self, sink: int, sliding_window: int):
        self.sink = sink
        self.sliding_windows = sliding_window
        super().__init__(sink + sliding_window)

    def lazy_initialization(self, key_states: torch.Tensor):
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # 用于跟踪“已经写了多少个 token”（逻辑长度），不依赖 cache_position
        self.current_seq_len = 0

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,  # 兼容旧接口，但不使用 cache_position
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # 本次要写入的序列长度（逻辑上是连续追加）
        seq_len = key_states.shape[-2]

        # 这批 token 的逻辑位置 [start, start + seq_len)
        start = self.current_seq_len
        logical_positions = torch.arange(
            start,
            start + seq_len,
            device=self.device,
        )  # [seq_len]

        # 映射到物理 cache 位置
        mapped_positions = logical_positions.clone()

        # sink 段: pos < self.sink -> 物理位置 pos
        # sliding 段: pos >= self.sink -> 环形缓冲区
        if self.sliding_windows > 0:
            sliding_mask = mapped_positions >= self.sink
            if sliding_mask.any():
                sliding_pos = mapped_positions[sliding_mask]
                mapped_positions[sliding_mask] = (
                    self.sink + (sliding_pos - self.sink) % self.sliding_windows
                )

        # 更新 cache（注意用 mapped_positions，而不是 cache_position）
        try:
            self.keys.index_copy_(2, mapped_positions, key_states)
            self.values.index_copy_(2, mapped_positions, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ 可能不支持
            self.keys[:, :, mapped_positions] = key_states
            self.values[:, :, mapped_positions] = value_states

        # 逻辑长度增加
        self.current_seq_len += seq_len

        return self.keys, self.values


class CacheWithMem(StaticCache):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        
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
        self.mem_position_embed = []
        self.mem_cache = Mamba2Cache(mamba_config,batch_size=1,*args, **kwargs)
        self.hidden_cache = StaticHiddenCache(config, *args, **kwargs)

        super().__init__(config,max_cache_len=config.sliding_window + 1 + config.num_attn_sinks, *args, **kwargs)
        config = config.get_text_config(decoder=True)
        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            layer = StaticSlidingWindowLayerWithSink(config.num_attn_sinks + 1, config.sliding_window)
            self.layers.append(layer)
