# ==== NaN / Inf debugging helpers ============================================
import re
import math
import contextlib
from collections import defaultdict
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    Trainer,
)
def tensor_stats(x, name="", max_print=5):
    with torch.no_grad():
        # 使用 detach().clone() 的副本进行统计，不更改原始张量设备/图
        flat = x.detach().clone().to(dtype=torch.float32).view(-1)
        s = {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "min": float(flat.min().item()) if flat.numel() else float("nan"),
            "max": float(flat.max().item()) if flat.numel() else float("nan"),
            "mean": float(flat.mean().item()) if flat.numel() else float("nan"),
            "std": float(flat.std().item()) if flat.numel() else float("nan"),
            "n_nan": int(torch.isnan(flat).sum().item()),
            "n_inf": int(torch.isinf(flat).sum().item()),
        }
    tag = f"[{name}] " if name else ""
    print(f"{tag}shape={s['shape']} dtype={s['dtype']} "
          f"min={s['min']:.6g} max={s['max']:.6g} mean={s['mean']:.6g} std={s['std']:.6g} "
          f"NaN={s['n_nan']} Inf={s['n_inf']}")
    return s

def check_ids_and_labels(batch, tokenizer, vocab_size=None):
    problems = []
    ids = batch["input_ids"]
    labels = batch.get("labels", None)
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    if ids.dtype not in (torch.long, torch.int64):
        problems.append(f"input_ids dtype is {ids.dtype}, expected torch.long")
    if labels is not None and labels.dtype not in (torch.long, torch.int64):
        problems.append(f"labels dtype is {labels.dtype}, expected torch.long")

    # out-of-range token ids?
    with torch.no_grad():
        below0 = (ids < 0).any().item()
        aboveV = (ids >= vocab_size).any().item()
        if below0: problems.append("input_ids contain negative values")
        if aboveV: problems.append("input_ids contain values >= vocab_size")
        if labels is not None:
            below0_l = (labels < -100).any().item()  # -100 is allowed as ignore_index
            aboveV_l = (labels >= vocab_size).any().item()
            if below0_l: problems.append("labels contain values < -100")
            if aboveV_l: problems.append("labels contain values >= vocab_size")

    if tokenizer.pad_token_id is None:
        problems.append("tokenizer.pad_token_id is None; may cause attention/mask issues")
    elif tokenizer.pad_token_id != tokenizer.eos_token_id:
        print(f"[warn] pad_token_id({tokenizer.pad_token_id}) != eos_token_id({tokenizer.eos_token_id})")

    if problems:
        print("⚠️  ID sanity issues:", problems)
    else:
        print("✓ ID/label sanity check passed")

def register_activation_nan_hooks(model, atol=1e-5, rtol=1e4, big=1e4):
    """
    Returns a list of hook handles. Scans every module output for NaN/Inf and huge magnitudes.
    """
    handles = []
    tripped = {"hit": False}

    def make_hook(mod_name):
        def hook(mod, inp, out):
            # some modules return tuple; we only check Tensors
            outs = []
            if isinstance(out, torch.Tensor):
                outs = [out]
            elif isinstance(out, (tuple, list)):
                outs = [x for x in out if isinstance(x, torch.Tensor)]
            for i, t in enumerate(outs):
                if t is None: 
                    continue
                if torch.isnan(t).any() or torch.isinf(t).any():
                    print(f"❌ NaN/Inf in ACTIVATION at {mod_name}[{i}]")
                    tensor_stats(t, name=f"{mod_name}.out[{i}]")
                    tripped["hit"] = True
                    # Raise to break the forward right here:
                    raise RuntimeError(f"NaN/Inf activation detected at {mod_name}")
                with torch.no_grad():
                    mx = t.detach().abs().max().item()
                    if not math.isfinite(mx) or mx > big:
                        print(f"⚠️ Large activation (max={mx:.3g}) at {mod_name}[{i}]")
                        tensor_stats(t, name=f"{mod_name}.out[{i}]")
        return hook

    for name, module in model.named_modules():
        # avoid duplicating very small utility modules if noisy; you can filter here if needed
        handles.append(module.register_forward_hook(make_hook(name)))

    return handles, tripped

def register_grad_nan_hooks(model, big=1e4):
    """
    Attach hooks to parameters to catch NaN/Inf/huge grads right after backward.
    """
    handles = []
    def make_hook(p_name):
        def hook(grad):
            if grad is None:
                return
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"❌ NaN/Inf in GRAD for {p_name}")
                tensor_stats(grad, name=f"{p_name}.grad")
                # Raise to stop immediately:
                raise RuntimeError(f"NaN/Inf grad at {p_name}")
            with torch.no_grad():
                gmax = grad.detach().abs().max().item()
                if not math.isfinite(gmax) or gmax > big:
                    print(f"⚠️ Large grad (max={gmax:.3g}) at {p_name}")
        return hook

    for n, p in model.named_parameters():
        if p.requires_grad:
            handles.append(p.register_hook(make_hook(n)))
    return handles

@contextlib.contextmanager
def anomaly_guard(enabled=True):
    if enabled:
        with torch.autograd.set_detect_anomaly(True):
            yield
    else:
        yield

def print_first_tokens(batch, tokenizer, k=1, n=128):
    ids = batch["input_ids"][:k]
    for i, row in enumerate(ids):
        # 拷贝 detach().clone() 的 CPU 版本以进行 decode，不影响原始训练张量
        row_cpu = row.detach().clone().cpu()
        text = tokenizer.decode(row_cpu[:n].tolist(), skip_special_tokens=False)
        print(f"[sample {i}] first {n} tokens decoded:\n{text[:500]}...\n")

def safe_forward_backward(model, batch, scaler=None, use_amp=False):
    """
    Does a single forward+backward to trigger hooks and report shapes/stats.
    """
    model.train()
    # Show logits shape agreement with labels
    out = model(**batch)
    loss = out.loss
    logits = out.logits
    print(f"forward: loss={float(loss.item()) if torch.isfinite(loss) else loss}, logits={tuple(logits.shape)}, logits.dtype={logits.dtype}")
    tensor_stats(logits, name="logits")

    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # optional: global grad norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # 使用 detach().clone() 的副本计算范数，避免 .data 和图污染
            param_norm = p.grad.detach().clone().to(dtype=torch.float32).norm(2)
            total_norm += float(param_norm.item()) ** 2
    total_norm = total_norm ** 0.5
    print(f"global grad norm: {total_norm:.6g}")
    return out


def pre_hook_print(name):
    def _hook(mod, inputs):
        xs = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                xs.append(x)
            elif isinstance(x, (tuple, list)):
                xs += [t for t in x if isinstance(t, torch.Tensor)]
        for i, t in enumerate(xs):
            print(f"[PRE {name}[{i}]]")
            tensor_stats(t, name=f"{name}.pre[{i}]")  # 用你前面加的 tensor_stats
            if torch.isinf(t).any() or torch.isnan(t).any():
                print(f"❌ non-finite BEFORE {name}")
    return _hook


def trip_once_on_nan_in_subtree(model, subtree_key="compressor", big=1e4):
    handles = []
    tripped = {"hit": False}

    for name, module in model.named_modules():
        if subtree_key not in name:
            continue

        def make_hook(mod_name):
            def hook(mod, inp, out):
                outs = []
                if isinstance(out, torch.Tensor):
                    outs = [out]
                elif isinstance(out, (tuple, list)):
                    outs = [x for x in out if isinstance(x, torch.Tensor)]
                for i, t in enumerate(outs):
                    if t is None:
                        continue
                    if torch.isnan(t).any() or torch.isinf(t).any():
                        print(f"❌ NaN/Inf at {mod_name}[{i}]")
                        tensor_stats(t, f"{mod_name}.out[{i}]")
                        tripped["hit"] = True
                        raise RuntimeError(f"NaN/Inf at {mod_name}")

                    # ---- 修正点在这里 ----
                    with torch.no_grad():
                        mx = t.detach().abs().max()
                        # mx 是 Tensor，用 item() 转成 float 仅用于打印，不用于 torch.isfinite
                        if not torch.isfinite(mx).all() or mx.item() > big:
                            print(f"⚠️ Large activation {mx.item():.3g} at {mod_name}[{i}]")
            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    return handles, tripped
# ============================================================================

# ----------------------------------------------------------------------------
# Compressor/Mamba2 定向调试辅助
# ----------------------------------------------------------------------------

def register_compressor_pre_hooks(model, subtree_key="compressor"):
    """为名包含 subtree_key 的所有子模块注册 forward_pre_hook，输出输入张量统计。

    返回: handles(list)
    """
    handles = []
    for name, module in model.named_modules():
        if subtree_key in name:
            try:
                handles.append(module.register_forward_pre_hook(pre_hook_print(name)))
            except Exception:
                pass
    return handles


def register_param_sanity_checks(model, subtree_key="compressor"):
    """调试开始前打印 compressor 相关参数的数值统计（如 A_log, dt_bias 等）。"""
    print(f"[param-sanity] scanning parameters under subtree '{subtree_key}'")
    for name, p in model.named_parameters():
        if subtree_key in name:
            try:
                tensor_stats(p.detach().clone(), name=name)
            except Exception:
                pass


def cleanup_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def make_one_batch(dataset, collator, batch_size=1):
    """从 HF Dataset 构造一个 DataLoader 并取第一个 batch。"""
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return next(iter(dl))


def debug_one_step(model, dataset, collator, tokenizer, batch_size=1, use_amp=False):
    """在 compressor 子树挂钩，跑一个 step 的 forward+backward，捕捉 NaN/Inf 并打印梯度范数。"""
    device = next(model.parameters()).device
    batch = make_one_batch(dataset, collator, batch_size=batch_size)
    # 设备与 dtype 对齐
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 文本采样与 ID 合法性检查
    try:
        print_first_tokens(batch, tokenizer, k=1, n=128)
        check_ids_and_labels(batch, tokenizer)
    except Exception:
        pass

    # 注册钩子
    pre_handles = register_compressor_pre_hooks(model, subtree_key="compressor")
    act_handles, act_tripped = trip_once_on_nan_in_subtree(model, subtree_key="compressor")
    grad_handles = register_grad_nan_hooks(model)

    error = None
    try:
        with anomaly_guard(enabled=True):
            safe_forward_backward(model, batch, scaler=None, use_amp=use_amp)
    except Exception as e:
        error = e
        print(f"[debug-one-step] caught exception: {e}")
    finally:
        cleanup_hooks(pre_handles)
        cleanup_hooks(act_handles)
        cleanup_hooks(grad_handles)

    if error is not None:
        print("[debug-one-step] likely NaN/Inf source located above; see hooks output.")
    else:
        print("[debug-one-step] completed without NaN/Inf.")
    return error
