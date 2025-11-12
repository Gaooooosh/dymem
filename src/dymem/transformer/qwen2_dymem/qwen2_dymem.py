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

from dymem.utils import BaseAHN,repeat_memkv
# from dymem.rnn import Mamba2
from fla.layers.mamba2 import Mamba2
from dataclasses import dataclass
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
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

        ahn_impl = getattr(config, '_ahn_implementation', None)
        if ahn_impl is not None:
            self.compressor = Mamba2(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                hidden_size=config.hidden_size,
                state_size = 256,
                expand = 1,
                layer_idx=layer_idx,
                n_groups=self.num_kv_groups,
                chunk_size=64,
                )
        else:
            self.compressor = None
        # 段缓存
        self.register_buffer('sink_k', None, persistent=False)
        self.register_buffer('sink_v', None, persistent=False)
        self.register_buffer('mem_k', None, persistent=False)
        self.register_buffer('mem_v', None, persistent=False)
        self.register_buffer('active_k', None, persistent=False)
        self.register_buffer('active_v', None, persistent=False)

        self.recent_k = None  # [B,H,W,D]
        self.recent_v = None

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

    def forward(
        self,
        prenormed_hidden: torch.Tensor,                                 # [B,T,H]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin) for new tokens
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        B, T, H = prenormed_hidden.shape
        # 预填充阶段可能会更改序列长度，先处理压缩拼接，再计算投影视图形状
        mem_out = None
        if T > 1:
            # prefilling
            w = 0 if (self.sliding_window is None or self.sliding_window <= 0) else self.sliding_window
            sink_len = min(self.num_attn_sinks, T)
            keep_window_len = w
            evict_end_index = max(0, T - keep_window_len)
            overflow = max(0, evict_end_index - sink_len)
            if self.compressor is not None and overflow > 0:
                # 正确的被压缩段：去掉前端 sink 与末端滑窗，中间段作为被驱逐 token
                evicted_hidden_states = prenormed_hidden[:, sink_len:evict_end_index, :]
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    mem_out = self.compressor(
                        evicted_hidden_states.float().contiguous(),
                        cache_params=(getattr(past_key_value, "mem_cache", None) if past_key_value is not None else None),
                        cache_position=cache_position,
                    )  # [B, EVC_T, H]
                mem_out = mem_out.to(dtype=prenormed_hidden.dtype, device=prenormed_hidden.device)
                mem_act = mem_out[:, -1, :].unsqueeze(1)  # [B, 1, H]

                prenormed_hidden = torch.cat(
                    [
                        prenormed_hidden[:, :sink_len, :],
                        mem_act.to(prenormed_hidden.device),
                        prenormed_hidden[:, evict_end_index:, :],  # 保留末端滑窗
                    ],
                    dim=1,
                )
                position_embeddings = (
                    torch.cat([position_embeddings[0][:, :sink_len+1, :], position_embeddings[0][:, evict_end_index:, :]], dim=1),
                    torch.cat([position_embeddings[1][:, :sink_len+1, :], position_embeddings[1][:, evict_end_index:, :]], dim=1),
                )

        # 压缩拼接后重新计算形状并保证连续性
        B, T, H = prenormed_hidden.shape
        input_shape = prenormed_hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        prenormed_hidden = prenormed_hidden.contiguous()

        # 线性投影
        q_states = self.q_proj(prenormed_hidden).view(hidden_shape).transpose(1, 2)
        k_states = self.k_proj(prenormed_hidden).view(hidden_shape).transpose(1, 2)
        v_states = self.v_proj(prenormed_hidden).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            q_states.to(dtype=cos.dtype),
            k_states.to(dtype=cos.dtype),
            cos,
            sin,
        )
        value_states = v_states

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        # 将 [B, T, heads, head_dim] 转为 [B, T, heads*head_dim]
        attn_output = attn_output.reshape(B, T, -1).contiguous()
        if mem_out is not None:
            attn_output = torch.cat([
                attn_output[:, :sink_len, :],
                mem_out.to(dtype=attn_output.dtype, device=attn_output.device),
                attn_output[:, sink_len + 1 :, :]
            ], dim=1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen2DyMemDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = Qwen2AttentionWithMem(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._do_solidify_next = False

    def mark_solidify(self):
        self._do_solidify_next = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
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
            # Attach a memory cache for compressor usage if available
            try:
                from fla.models.mamba2.modeling_mamba2 import Mamba2Cache
                if not hasattr(past_key_values, "mem_cache"):
                    past_key_values.mem_cache = Mamba2Cache()
            except Exception:
                # Fallback: ensure attribute exists to avoid AttributeError; compressor will ignore None
                if not hasattr(past_key_values, "mem_cache"):
                    past_key_values.mem_cache = None

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
        if self.training:
            self.sliding_window = random.choice(self.config.windows_choices)
        else:
            self.sliding_window = self.config.sliding_window
            # print(f"sliding_window: {self.sliding_window}")
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
