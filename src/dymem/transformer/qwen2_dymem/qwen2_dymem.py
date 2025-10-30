from __future__ import annotations
import torch
import wandb
import random
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)
from typing import Union, Optional, Tuple, List, Dict, Any
from dymem.transformer.qwen2 import (
    Qwen2Config as Qwen2Config_,
    Qwen2Model as Qwen2Model_,
    Qwen2ForCausalLM as Qwen2ForCausalLM_,
)
from dymem.transformer.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
    rotate_half,
)

from dymem.utils import BaseAHN, AHNRouter
from dymem.rnn import Mamba2
from dataclasses import dataclass
from fla.models.utils import Cache as MemCache
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward, FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    add_start_docstrings,
    can_return_tuple,
    # deprecate_kwarg,
    logging,
)

logger = logging.get_logger(__name__)

# FlexAttention configuration
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
flex_attention = torch.compile(flex_attention)

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
    """在 Qwen2Attention 基础上，加入四段式 KV 与 SSM-ActiveMem 逻辑，并以 flex_attention 计算注意力。"""
    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.num_attn_sinks = getattr(config, 'num_attn_sinks', 128)
        self.sliding_window = getattr(config, 'sliding_window', 256)
        self.mem_max_len = getattr(config, 'mem_max_len', 2048)

        # 段缓存
        self.register_buffer('sink_k', None, persistent=False)
        self.register_buffer('sink_v', None, persistent=False)
        self.register_buffer('mem_k', None, persistent=False)
        self.register_buffer('mem_v', None, persistent=False)
        self.register_buffer('active_k', None, persistent=False)
        self.register_buffer('active_v', None, persistent=False)

        self.recent_k = None  # [B,H,W,D]
        self.recent_v = None
        self.recent_h = None  # [B,W,H]
        self.recent_q = None  # [B,H,W,D]

    # --------- 工具 ---------
    @staticmethod
    def _cat(parts: List[Optional[torch.Tensor]]):
        parts = [p for p in parts if p is not None]
        return torch.cat(parts, dim=2) if parts else None

    def _set_active_from_o(self, o_last: torch.Tensor):
        # o_last: [B,H]
        x = o_last.unsqueeze(1)  # [B,1,H]
        k_states = self.k_proj(x)
        v_states = self.v_proj(x)
        B, L1, _ = k_states.shape
        # reshape to KV heads then repeat to attn heads
        k_kv = k_states.view(B, L1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v_kv = v_states.view(B, L1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        k = repeat_kv(k_kv, self.num_kv_groups)
        v = repeat_kv(v_kv, self.num_kv_groups)
        self.active_k, self.active_v = k, v

    def _solidify_active(self):
        if self.active_k is None:
            return
        if self.mem_k is None:
            self.mem_k, self.mem_v = self.active_k, self.active_v
        else:
            self.mem_k = torch.cat([self.mem_k, self.active_k], dim=2)
            self.mem_v = torch.cat([self.mem_v, self.active_v], dim=2)
        # 截断 mem 左端
        if self.mem_k.size(2) > self.mem_max_len:
            cut = self.mem_k.size(2) - self.mem_max_len
            self.mem_k = self.mem_k[:, :, cut:, :].contiguous()
            self.mem_v = self.mem_v[:, :, cut:, :].contiguous()
        # 新建空 active
        B, Hh, _, D = self.mem_k.shape
        zeros_k = torch.zeros(B, Hh, 1, D, device=self.mem_k.device, dtype=self.mem_k.dtype)
        zeros_v = torch.zeros(B, Hh, 1, D, device=self.mem_v.device, dtype=self.mem_v.dtype)
        self.active_k, self.active_v = zeros_k, zeros_v

    def _build_block_mask(self, kv_len: int, q_len: int, device: torch.device):
        # 简化版：S/M/A 全可见；WINDOW 段长度受 recent 限制（不再对窗口做 q 相对因果限制，decode=1 安全）
        def ctx(b, h, q_idx, kv_idx):
            # 必须返回 Tensor，不能返回 Python bool，否则 vmap 报错
            return torch.ones((), dtype=torch.bool, device=device)
        # 使用实际长度创建 block mask，避免后续再裁剪
        mask = create_block_mask(
            ctx,
            B=1,
            H=self.num_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
            # BLOCK_SIZE=128,
            _compile=False,
        )
        return mask

    # --------- 前向（推理） ---------
    def forward(
        self,
        prenormed_hidden: torch.Tensor,                                 # [B,T,H]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin) for new tokens
        compressor: Optional[BaseAHN] = None,
        do_solidify: bool = False,
        is_decode: bool = True,
    ) -> torch.Tensor:
        B, T, H = prenormed_hidden.shape
        # 线性投影
        q_states = self.q_proj(prenormed_hidden)  # [B,T,H]
        k_states = self.k_proj(prenormed_hidden)
        v_states = self.v_proj(prenormed_hidden)
        # 分头
        q = q_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B,H,T,D]
        # reshape k/v to KV heads then repeat to attn heads
        k_kv = k_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()  # [B,K,T,D]
        v_kv = v_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()  # [B,K,T,D]
        k = repeat_kv(k_kv, self.num_kv_groups)  # [B,H,T,D]
        v = repeat_kv(v_kv, self.num_kv_groups)  # [B,H,T,D]
        # 对“新来的窗口段”应用 RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # 维护 recent（k/q 已是旋转后的；v 不旋）
        if self.recent_k is None:
            self.recent_k, self.recent_v = k, v
            self.recent_h, self.recent_q = prenormed_hidden, q
        else:
            self.recent_k = torch.cat([self.recent_k, k], dim=2)
            self.recent_v = torch.cat([self.recent_v, v], dim=2)
            self.recent_h = torch.cat([self.recent_h, prenormed_hidden], dim=1)
            self.recent_q = torch.cat([self.recent_q, q], dim=2)
        # 计算驱逐长度并喂给 AHN
        overflow = max(0, (0 if self.recent_k is None else self.recent_k.size(2)) - self.sliding_window)
        # print(f"overflow:{overflow}, Compressor:{compressor is not None}")
        if overflow > 0 and compressor is not None:
            print("use comporessor")
            h_e = self.recent_h[:, :overflow, :].contiguous()
            q_e = self.recent_q[:, :, :overflow, :].contiguous()
            k_e = self.recent_k[:, :, :overflow, :].contiguous()
            v_e = self.recent_v[:, :, :overflow, :].contiguous()
            compressor.update_cache(h_e)
            o, _ = compressor(hidden_states=h_e, q_states=q_e, k_states=k_e, v_states=v_e, use_cache=True, past_key_values=None)
            self._set_active_from_o(o[:, -1, :])
            # recent 左裁剪
            self.recent_k = self.recent_k[:, :, overflow:, :].contiguous()
            self.recent_v = self.recent_v[:, :, overflow:, :].contiguous()
            self.recent_q = self.recent_q[:, :, overflow:, :].contiguous()
            self.recent_h = self.recent_h[:, overflow:, :].contiguous()
        # 固化 ACTIVE
        if do_solidify:
            self._solidify_active()
        # 组装 K/V（prefill 屏蔽 MEM/ACTIVE；decode 启用）
        mem_k = self.mem_k
        mem_v = self.mem_v
        act_k = self.active_k
        act_v = self.active_v
        if is_decode and position_embeddings is not None:
            # 对 MEM/ACTIVE 的 K 做“查询时对齐”的 RoPE（以当前步角度广播到 mem 长度）
            cos, sin = position_embeddings  # [B, T, D]
            cos_t = cos[:, -1:, :]  # [B,1,D]
            sin_t = sin[:, -1:, :]  # [B,1,D]
            if mem_k is not None:
                cos_mem = cos_t.expand(-1, mem_k.size(2), -1)  # [B,mem_len,D]
                sin_mem = sin_t.expand(-1, mem_k.size(2), -1)
                mem_k, _ = apply_rotary_pos_emb(mem_k, mem_k, cos_mem, sin_mem, unsqueeze_dim=1)
            if act_k is not None:
                act_k, _ = apply_rotary_pos_emb(act_k, act_k, cos_t, sin_t, unsqueeze_dim=1)
        else:
            mem_k = mem_v = None
            act_k = act_v = None
        K_total = self._cat([self.sink_k, mem_k, act_k, self.recent_k])
        V_total = self._cat([self.sink_v, mem_v, act_v, self.recent_v])
        KV_LEN = 0 if K_total is None else K_total.size(2)
        # flex block mask（这里简化为全 True；recent 长度本身受限于窗口）
        block_mask = self._build_block_mask(KV_LEN, T, device=q.device)
        out = flex_attention(q, K_total, V_total, block_mask=block_mask)
        out = out[:, :, :T, :].transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)


class Qwen2DyMemDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = Qwen2AttentionWithMem(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ahn_impl = getattr(config, '_ahn_implementation', None)
        if ahn_impl is not None:
            self.compressor = BaseAHN(ahn_cls=AHN_CLS[ahn_impl], layer_idx=layer_idx,
                                    hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
                                    head_dim=config.hidden_size // config.num_attention_heads)
        else:
            self.compressor = None
        self._do_solidify_next = False

    def mark_solidify(self):
        self._do_solidify_next = True

    def forward(
        self,
        hidden_states: torch.Tensor,                 # [B,T,H]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[object] = None,
        use_cache: Optional[bool] = True,
    ):
        x = self.input_layernorm(hidden_states)
        is_decode = True if (hidden_states.size(1) == 1) else False
        attn_out = self.self_attn(
            prenormed_hidden=x,
            position_embeddings=position_embeddings,
            compressor=self.compressor,
            do_solidify=self._do_solidify_next,
            is_decode=is_decode,
        )
        self._do_solidify_next = False
        h = hidden_states + attn_out
        h2 = self.post_attention_layernorm(h)
        h2 = self.mlp(h2)
        out = h + h2
        return (out,)



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

    @staticmethod
    def create_sparse_mask(sliding_window: int = 2048, num_attn_sinks: int = 0):

        def ctx_sliding_window_mask(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            sliding = (q_idx - kv_idx) < sliding_window
            sink = kv_idx < num_attn_sinks
            return causal & (sliding | sink)

        return ctx_sliding_window_mask

    def pre_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None
    ):
        """
        Build FlexAttention mask with custom sliding window and attention sink.
        """
        flex_attention_mask = None
        if self.training and self.layer_cls == "Qwen2DyMemDecoderLayer":
            B, L = input_ids.shape
            sliding_window_type = getattr(self.config, "sliding_window_type", "fixed")
            sliding_window = getattr(self.config, "sliding_window", None)
            assert (
                sliding_window > 0
            ), "Please provide 'sliding_window' for 'Qwen2DyMemDecoderLayer' forward pass."

            # Get sliding window
            if sliding_window_type == "fixed":
                self.config.dy_sliding_window = sliding_window
            elif sliding_window_type == "linear":
                # Grow with sequence length but lower-bounded
                self.config.dy_sliding_window = max(L // 4, 512)
            elif sliding_window_type == "random":
                def sample_window_size(
                    seq_len: str,
                    candidates: list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                ):
                    valid_sizes = [c for c in candidates if seq_len / 8 < c < seq_len]
                    return random.choice(valid_sizes)
                random_size = sample_window_size(L)
                self.config.dy_sliding_window = random_size
            else:
                raise NotImplementedError

            # Get attention sink
            sliding_window = getattr(
                self.config, "dy_sliding_window", self.config.sliding_window
            )
            ahn_position = getattr(self.config, "ahn_position", "prefix")

            if ahn_position == "prefix":
                num_attn_sinks = 0
            elif ahn_position == "random":
                def sample_num_attn_sinks(
                    sliding_window: int,
                    candidates: list = [0, 32, 64, 128, 512, 2048, 4096],
                ):
                    valid_sizes = [c for c in candidates if c <= 0.5 * sliding_window]
                    return random.choice(valid_sizes)
                num_attn_sinks = sample_num_attn_sinks(sliding_window)
                self.config.dy_num_attn_sinks = num_attn_sinks
                sliding_window = sliding_window - num_attn_sinks
            else:
                raise NotImplementedError
        else:
            sliding_window = getattr(self.config, "sliding_window", None)
            num_attn_sinks = getattr(self.config, "num_attn_sinks", 0)
            assert (
                sliding_window > 0
            ), "Please provide 'sliding_window' for 'Qwen2MemDecoderLayer' forward pass."

        # Generate FlexAttention mask
        # TODO(XIAO):添加与mem_token数量匹配的sliding_window数量
        sparse_mask = self.create_sparse_mask(
            sliding_window=sliding_window, num_attn_sinks=num_attn_sinks
        )
        
        seq_len = input_ids.shape[1]
        padded_lens = [2048, 4096, 8192, 16384, 24576, 32768]
        padded_seq_len = next(
            (l for l in padded_lens if l >= seq_len), (seq_len // 128 + 1) * 128
        )

        flex_attention_mask = create_block_mask(
            sparse_mask,
            B=1,
            H=self.config.num_attention_heads,
            Q_LEN=padded_seq_len,
            KV_LEN=padded_seq_len,
            device=input_ids.device,
            BLOCK_SIZE=128,
            _compile=True,
        )

        return flex_attention_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

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

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
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

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

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
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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