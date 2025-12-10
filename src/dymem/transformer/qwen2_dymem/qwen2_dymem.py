from __future__ import annotations
from math import log
import torch
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

from dymem.utils import CacheWithMem
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
class CustomizedModelOutputWithPast(ModelOutput):
    last_hidden_states: torch.FloatTensor = None
    last_mem_hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    mem_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_distill_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


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

# ------------------------------
# 简易决策头（MVP 未启用在线切分，但保留接口）
# ------------------------------
class ChunkDecisionHead(nn.Module):
    def __init__(self, hidden_size: int, feat_mul: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * feat_mul, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,S,H] -> 池化特征 [B, 2H]
        mean = x.mean(dim=1)
        mx = x.amax(dim=1)
        feat = torch.cat([mean, mx], dim=-1)
        return self.proj(feat)  # [B,1]

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
        if self.sliding_window:
            logger.warning_once(f"【DyMem】- [Warning] sliding_window set to {self.sliding_window}")
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
        self.use_compress = getattr(config, 'use_compressor', True)
        if not self.use_compress:
            logger.warning_once(f"【DyMem】- [Warning] Config does not use compressor")
        # 段缓存
        self.register_buffer('sink_k', None, persistent=False)
        self.register_buffer('sink_v', None, persistent=False)
        self.register_buffer('mem_k', None, persistent=False)
        self.register_buffer('mem_v', None, persistent=False)
        self.register_buffer('active_k', None, persistent=False)
        self.register_buffer('active_v', None, persistent=False)

        self.recent_k = None  # [B,H,W,D]
        self.recent_v = None

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
        k_len = key_states.shape[2]
    
        dtype = query_states.dtype
        device = query_states.device
        
        # 获取配置参数
        sliding_window = getattr(self.config, 'sliding_window', None)
        num_attn_sinks = getattr(self.config, 'num_attn_sinks', 0)
        
        # 确定“受保护区域”长度 (Sink + Memory)
        # 假设 Memory 紧跟在 Sink 之后。如果 k_len 很短说明还没产生 Memory，mem_len=0
        mem_len = 1 if (self.use_compress and k_len > num_attn_sinks) else 0
        protected_len = num_attn_sinks + mem_len

        # =====================================================
        # Case 1: Decoding (q_len == 1)
        # =====================================================
        if q_len == 1:
            # 如果没有定义滑窗，或者滑窗甚至比当前总长度还长，就不需要 Mask
            if sliding_window is None or sliding_window <= 0 or k_len <= (protected_len + sliding_window):
                return None
            
            # 初始化 Mask [1, 1, 1, k_len]
            # 注意：Decoding 时 Q 只有一个，所以 dim=2 是 1
            mask = torch.zeros((1, 1, 1, k_len), device=device, dtype=dtype)
            
            # 我们只需要判断 Key 的位置 (col_idx)
            col_idx = torch.arange(k_len, device=device)[None, None, :] # [1, 1, k_len]
            
            # Mask 逻辑：
            # 我们要 Mask 掉的是中间那段“被遗忘的区域”。
            # 条件 A: 它不在左边的受保护区 (col >= protected_len)
            # 条件 B: 它不在右边的滑动窗口内 (col < k_len - sliding_window)
            # 注意：最新的 token 索引是 k_len-1，所以窗口起始点是 k_len - sliding_window
            
            mask_cond = (col_idx >= protected_len) & (col_idx < (k_len - sliding_window))
            
            min_val = torch.finfo(dtype).min
            mask.masked_fill_(mask_cond, min_val)
            
            return mask

        # =====================================================
        # Case 2: Prefilling (q_len > 1)
        # =====================================================
        else:
            # 基础全 0 Mask
            mask = torch.full((q_len, k_len), 0, device=device, dtype=dtype)
            min_val = torch.finfo(dtype).min

            if sliding_window is not None and sliding_window > 0:
                row_idx = torch.arange(q_len, device=device)[:, None]
                col_idx = torch.arange(k_len, device=device)[None, :]
                
                # 1. 因果性 (不能看未来)
                causal_mask = col_idx > row_idx
                
                # 2. 滑窗限制 (不能看太旧)
                # 只有当它既不是 Sink 也不是 Memory 时，才受窗口限制
                window_mask = (col_idx < (row_idx - sliding_window)) & (col_idx >= protected_len)
                
                final_mask_cond = causal_mask | window_mask
                mask.masked_fill_(final_mask_cond, min_val)
            else:
                # 标准因果 Mask
                mask.masked_fill_(torch.ones_like(mask, dtype=torch.bool).triu_(1), min_val)
            
            return mask[None, None, :, :]

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
        if T > 1: # prefilling
            w = 0 if (self.sliding_window is None or self.sliding_window <= 0) else self.sliding_window
            sink_len = min(self.num_attn_sinks, T)
            keep_window_len = w
            evict_end_index = max(0, T - keep_window_len)
            overflow = max(0, evict_end_index - sink_len)
            if self.compressor is not None and self.use_compress:
                if overflow > 0:
                    # 正确的被压缩段：去掉前端 sink 与末端滑窗，中间段作为被驱逐 token
                    evicted_hidden_states = prenormed_hidden[:, sink_len:evict_end_index, :]
                    mem_out = self.compressor(
                        evicted_hidden_states if self.training else evicted_hidden_states,
                        cache_params=past_key_value.mem_cache if past_key_value is not None else None,
                        cache_position=cache_position,
                    )  # [B, EVC_T, H]
                    mem_act = mem_out[:, -1:, :] # [B, 1, H]
                    if past_key_value is not None:
                        past_key_value.hidden_cache.update(prenormed_hidden[:,evict_end_index:, :], layer_idx=self.layer_idx)
                        # past_key_value.mem_position_embed.append((position_embeddings[0][:, sink_len, :],position_embeddings[1][:, sink_len, :]))

                elif past_key_value is not None and T > sink_len:
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
                                    cache_params=past_key_value.mem_cache if past_key_value is not None else None,
                                    cache_position=cache_position,
                                )
                            mem_act = mem_out_step[:,-1:,:]

                # 3) 最后再把新 token 写入缓存（O(1) 次数的切片 copy）
                past_key_value.hidden_cache.update(prenormed_hidden, self.layer_idx)

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
            mem_hidden_shape = (B, 1, -1, self.head_dim)
            mem_k = self.k_proj(mem_act).view(mem_hidden_shape).transpose(1, 2) # [B, H, 1, D]
            mem_v = self.v_proj(mem_act).view(mem_hidden_shape).transpose(1, 2) # [B, H, 1, D]

            key_states[:,:,sink_len,:] = mem_k[:,:,0,:]
            value_states[:,:,sink_len,:] = mem_v[:,:,0,:]
            if T > 1:
                mem_q = self.q_proj(mem_act).view(mem_hidden_shape).transpose(1, 2)
                query_states[:,:,sink_len,:] = mem_q[:,:,0,:]

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

            # 截取当前会用到的部分
            mask_arg = mask_arg[:, :, :, :k_len]

            # 构造一个全 0 bias mask，只在 mem 列加上 bias
            Le_fact = (key_states.shape[-2] - self.num_attn_sinks - self.sliding_window) / key_states.shape[-2]
            mem_bias_mask = torch.zeros_like(mask_arg)
            mem_bias_mask[..., mem_pos] = 0.5 * Le_fact

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

        attn_output = attn_output.reshape(B, T, -1)
        if mem_out is not None:
            if T > 1 and not self.training:
                attn_output[:,sink_len+1:T-w,:] = mem_out[:,:-1,:]
            else :
                # 1. 取出 Sink 部分 (保留 Attention 结果)
                out_sink = attn_output[:, :sink_len, :]
                
                # 2. 中间部分直接用 mem_out (Mamba 输出)
                # 注意：需确保 mem_out 形状与被替换部分一致
                out_mem = mem_out 
                
                # 3. 取出 Window 部分 (保留 Attention 结果)
                # T-w 就是 evict_end_index
                out_window = attn_output[:, T-w:, :]
                
                # 4. 拼接 (Out-of-place 操作，安全)
                attn_output = torch.cat([out_sink, out_mem, out_window], dim=1)
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
            # print(f"sliding_window: {w}")
            for layer in self.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.sliding_window = w
        else:
            self.sliding_window = self.config.sliding_window
            # print(f"sliding_window: {self.sliding_window}")
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

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

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
    ) -> CausalLMOutputWithPast:
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
