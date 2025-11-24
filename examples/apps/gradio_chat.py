import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from dymem.transformer.qwen2_dymem import register_customized_qwen2
from dymem.utils import CacheWithMem
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.train.untils import set_sliding_window
USE_DYMEM_CACHE = True
USE_GENERATION_CACHE = os.environ.get("USE_GENERATION_CACHE", "1") == "1"
MAX_HISTORY_MSGS = int(os.environ.get("MAX_HISTORY_MSGS", "12"))

# runtime safety flags
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass
try:
    import torch._inductor as _inductor
    _inductor.config.cudagraphs = False
    _inductor.config.triton.cudagraphs = False
    _inductor.config.triton.cudagraph_skip_dynamic_graphs = True
except Exception:
    pass

register_customized_qwen2(exist_ok=True)

MODEL_DIR = "/home/xiaoyonggao/dymem/qwen2.5-3b-compressor-Instruct"
DEVICE = torch.device(os.environ.get("CUDA_DEVICE", "cuda:4") if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and DEVICE.type == "cuda":
    torch.cuda.set_device(DEVICE)

_tokenizer = None
_model = None
_global_sessions = {}


def _normalize_messages(history):
    msgs = []
    for item in history or []:
        if isinstance(item, dict) and "role" in item and "content" in item:
            msgs.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            msgs.append({"role": "user", "content": item[0]})
            msgs.append({"role": "assistant", "content": item[1]})
    return msgs


def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and DEVICE.type == "cuda") else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=dtype,
            attn_implementation="eager",
        ).to(DEVICE)
        try:
            if hasattr(_model, "model") and hasattr(_model.model, "layers"):
                for lyr in _model.model.layers:
                    sa = getattr(lyr, "self_attn", None)
                    if sa is not None and hasattr(sa, "compressor"):
                        sa.compressor = None
        except Exception:
            pass
        try:
            _model.config._ahn_implementation = None
            _model.config.use_compressor = False
        except Exception:
            pass
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, padding_side="right")
        _model.eval()
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            _tokenizer.pad_token_id = _tokenizer.eos_token_id
    return _tokenizer, _model


def get_session(session_id: str, sliding_window: int):
    if session_id not in _global_sessions:
        tok, model = load_model()
        model.config.sliding_window = sliding_window
        try:
            set_sliding_window(model, sliding_window)
        except Exception:
            pass
        _global_sessions[session_id] = {
            "history": [],
            "cache": CacheWithMem(model.config),
        }
    return _global_sessions[session_id]


def reset_session(session_id: str):
    if session_id in _global_sessions:
        del _global_sessions[session_id]
    return []


def generate_reply(session_id: str, system_prompt: str, user_text: str, max_new_tokens: int, temperature: float, top_p: float, sliding_window: int):
    tok, model = load_model()
    session = get_session(session_id, sliding_window)
    history = session["history"]
    normalized_history = _normalize_messages(history)[-MAX_HISTORY_MSGS:]
    cache = session["cache"]

    messages_for_model = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + normalized_history + [
        {"role": "user", "content": user_text}
    ]

    prompt_str = tok.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_str, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            num_beams=1,
            do_sample=bool(temperature and temperature > 0),
            temperature=float(temperature),
            top_p=float(top_p),
            use_cache=USE_GENERATION_CACHE,
            past_key_values=cache,
        )[0]
    pred = tok.decode(output[input_len:], skip_special_tokens=True)

    normalized_history.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": pred},
    ])
    session["history"] = normalized_history
    return normalized_history


def generate_reply_stream(session_id: str, system_prompt: str, user_text: str, max_new_tokens: int, temperature: float, top_p: float, sliding_window: int):
    tok, model = load_model()
    session = get_session(session_id, sliding_window)
    history = session["history"]
    normalized_history = _normalize_messages(history)[-MAX_HISTORY_MSGS:]
    cache = session["cache"]

    messages_for_model = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + normalized_history + [
        {"role": "user", "content": user_text}
    ]

    prompt_str = tok.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_str, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def _decode_loop():
        acc = ""
        for piece in streamer:
            acc += piece
            yield [
                *normalized_history,
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": acc},
            ]

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            num_beams=1,
            do_sample=bool(temperature and temperature > 0),
            temperature=float(temperature),
            top_p=float(top_p),
            use_cache=USE_GENERATION_CACHE,
            past_key_values=cache,
            streamer=streamer,
        )

    final = tok.decode(streamer.final_token_sequence[input_len:], skip_special_tokens=True) if hasattr(streamer, "final_token_sequence") else None
    if final is not None:
        normalized_history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": final},
        ])

    session["history"] = normalized_history
    return _decode_loop()


with gr.Blocks(title="Qwen2.5-3B Compressor Chat") as demo:
    gr.Markdown("# Qwen2.5-3B-Compressor-Instruct 对话系统")
    with gr.Row():
        session_id = gr.Textbox(value="default", label="Session ID")
        system_prompt = gr.Textbox(value="You are Qwen, a helpful assistant.", label="System Prompt")
    with gr.Row():
        max_new_tokens = gr.Slider(32, 2048, value=512, step=32, label="max_new_tokens")
        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
        sliding_window = gr.Slider(64, 4096, value=128, step=64, label="sliding_window")

    chat = gr.Chatbot(type="messages", height=600)
    user_in = gr.Textbox(label="输入", placeholder="输入问题...", autofocus=True)
    with gr.Row():
        submit = gr.Button("发送")
        submit_stream = gr.Button("流式发送")
        reset = gr.Button("清空会话")

    def _submit(u, sid, sp, mnt, t, p, sw, history):
        if not u:
            return history
        return generate_reply(sid, sp, u, mnt, t, p, sw)

    def _submit_stream(u, sid, sp, mnt, t, p, sw, history):
        if not u:
            yield history
            return
        for h in generate_reply_stream(sid, sp, u, mnt, t, p, sw):
            yield h

    submit.click(_submit,
                 inputs=[user_in, session_id, system_prompt, max_new_tokens, temperature, top_p, sliding_window, chat],
                 outputs=[chat])
    submit_stream.click(_submit_stream,
                        inputs=[user_in, session_id, system_prompt, max_new_tokens, temperature, top_p, sliding_window, chat],
                        outputs=[chat])
    reset.click(lambda sid: reset_session(sid), inputs=[session_id], outputs=[chat])

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860)))
    try:
        demo.launch(server_name="0.0.0.0", server_port=port)
    except OSError:
        for p in range(port + 1, port + 20):
            try:
                demo.launch(server_name="0.0.0.0", server_port=p)
                break
            except OSError:
                continue
