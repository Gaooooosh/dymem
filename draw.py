# -*- coding: utf-8 -*-
"""
KV 内存与 FLOPs 随上下文增长的对比（含 >W 后按 k 倍压缩）
- 左图：权重(常数) + KV 内存（k ∈ K 多曲线），交点用“顶部刻度 + 表格”展示
- 右图：Prefill FLOPs（k ∈ K 多曲线）
- 交叉点避免重叠：不逐点写文本，改顶部刻度和表格

仅用于论文趋势展示（理论估算），无需实际跑模型。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ========= 可调参数（按需修改） =========
model_name   = "Qwen2.5-3B (approx.)"
# 模型结构（近似即可，用于趋势）
n_params     = 3_000_000_000   # 3B 参数
n_layers     = 28
d_model      = 2560
n_heads      = 32
head_dim     = d_model // n_heads
n_kv_heads   = n_heads         # 若做 GQA/MQA，可设 < n_heads
# 精度设定
weight_bytes = 2               # bf16/fp16
kv_bytes     = 2               # KV 精度（1 表示更激进量化 FP8/INT8 等）
# 横轴范围与窗口压缩
seq_min, seq_max, num_points = 128, 100_000, 600
window_size = 32_768          # 注意力窗口 W
K_list = [1, 2, 5, 10]        # 压缩倍率集合；k=1 即无压缩
# 可选：在交点位置画浅灰竖线
DRAW_VLINES_AT_CROSS = True
# 可选：同时画“总显存=权重+KV_k(L)”（线会变多，默认False）
PLOT_TOTAL_MEMORY_PER_K = False
# ====================================

# -------- 基础准备 --------
seq_lens = np.linspace(seq_min, seq_max, num_points, dtype=int).astype(np.int64)
L = seq_lens.astype(float)
to_gb = 1e-9
to_tflops = 1e-12

def effective_len(Lval, W, k):
    """超过窗口 W 的部分按 k 倍压缩后的有效长度"""
    return np.where(Lval <= W, Lval, W + (Lval - W) / k)

def prefill_flops_from_L(Lval):
    """Prefill FLOPs 近似： per-layer ≈ 16*L*d^2 + 4*L^2*d ，总层数乘 n_layers"""
    d = float(d_model)
    return n_layers * (16.0 * Lval * d**2 + 4.0 * (Lval**2) * d)

# -------- 权重与 KV 内存 --------
weights_mem_bytes = n_params * weight_bytes
weights_mem_gb = weights_mem_bytes * to_gb

kv_per_token_per_layer_bytes = (2 * n_kv_heads * head_dim) * kv_bytes  # (K+V)
per_token_all_layers_bytes = n_layers * kv_per_token_per_layer_bytes

# 计算各 k 的 KV 内存与 FLOPs，以及交叉点
kv_mem_curves = {}     # k -> GB 数组
flops_curves = {}      # k -> TFLOPs 数组
cross_points = {}      # k -> (L_cross, weights_mem_gb)

# L_eff 在交点处与 k 无关（由等式 KV(L_eff)=Weights 决定）
L_eff_cross = weights_mem_bytes / per_token_all_layers_bytes

for k in K_list:
    L_eff = effective_len(L, window_size, k)
    # KV 内存 (GB)
    kv_mem_curves[k] = (L_eff * per_token_all_layers_bytes) * to_gb
    # FLOPs (TFLOPs)
    flops_curves[k] = prefill_flops_from_L(L_eff) * to_tflops

    # 把 L_eff 的交点反解回原始 L（分段）
    if L_eff_cross <= window_size:
        L_cross = L_eff_cross
    else:
        L_cross = window_size + k * (L_eff_cross - window_size)
    if seq_min <= L_cross <= seq_max:
        cross_points[k] = (L_cross, weights_mem_gb)

# 可选：总显存曲线
total_mem_curves = {}
if PLOT_TOTAL_MEMORY_PER_K:
    for k in K_list:
        total_mem_curves[k] = weights_mem_gb + kv_mem_curves[k]

# -------- 绘图 --------
colors = cm.viridis(np.linspace(0.1, 0.9, len(K_list)))
fig, (ax_mem, ax_flops) = plt.subplots(1, 2, figsize=(12.8, 5.8), dpi=160)

# 左图：KV 内存 + 权重(常数)
ax_mem.plot(seq_lens, np.full_like(L, weights_mem_gb), color='gray', lw=2, label="Weights (constant)")
for idx, k in enumerate(K_list):
    style = '-' if k == 1 else '--'
    ax_mem.plot(seq_lens, kv_mem_curves[k], linestyle=style, color=colors[idx], label=f"KV memory (k={k})")

if PLOT_TOTAL_MEMORY_PER_K:
    for idx, k in enumerate(K_list):
        ax_mem.plot(seq_lens, total_mem_curves[k], color=colors[idx], alpha=0.45,
                    label=f"Total memory (k={k})")

# 顶部 x 轴刻度 + 右下角表格：展示各 k 的交点位置（KV=Weights）
ticks_x, tick_labels, table_rows = [], [], []
for k in K_list:
    if k in cross_points:
        Lx, mem_gb = cross_points[k]
        # 在曲线上放小点（不写文字）
        ax_mem.scatter([Lx], [mem_gb], s=22, color=colors[K_list.index(k)], zorder=5)
        if DRAW_VLINES_AT_CROSS:
            ax_mem.axvline(Lx, ls=":", lw=0.8, color="gray", alpha=0.5)

        ticks_x.append(Lx)
        tick_labels.append(f"k={k}")
        table_rows.append([str(k), f"{int(round(Lx))}"])

# 顶部 twin x-axis 显示“k 对应的交点”
ax_top = ax_mem.twiny()
ax_top.set_xlim(ax_mem.get_xlim())
if ticks_x:
    ax_top.set_xticks(ticks_x)
    ax_top.set_xticklabels(tick_labels, fontsize=9)
ax_top.set_xlabel("Break-even positions (KV = Weights) by k", fontsize=9, labelpad=6)

# 小表格（k 与 L_cross）
if table_rows:
    the_table = ax_mem.table(
        cellText=table_rows,
        colLabels=["k", "L@KV=Weights"],
        loc="lower right",
        colWidths=[0.12, 0.28],
        cellLoc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1.0, 1.05)

# 窗口竖线
ax_mem.axvline(window_size, linestyle="--", linewidth=1, alpha=0.75, color='gray')
ax_mem.text(window_size * 1.01, ax_mem.get_ylim()[1] * 0.08, f"W = {window_size}",
            rotation=90, va="bottom", fontsize=9)
ax_mem.set_xlabel("Context length L (tokens)")
ax_mem.set_ylabel("Memory (GB)")
ax_mem.set_title("KV Cache Memory vs. Context (with >W compressed)")
ax_mem.grid(True, linestyle="--", alpha=0.6)
ax_mem.legend(loc="upper left", ncol=1)

# 右图：Prefill FLOPs 多曲线
for idx, k in enumerate(K_list):
    style = '-' if k == 1 else '--'
    ax_flops.plot(seq_lens, flops_curves[k], linestyle=style, color=colors[idx], label=f"Prefill FLOPs (k={k})")

ax_flops.axvline(window_size, linestyle="--", linewidth=1, alpha=0.75, color='gray')
ax_flops.text(window_size * 1.01, ax_flops.get_ylim()[1] * 0.08, f"W = {window_size}",
              rotation=90, va="bottom", fontsize=9)
ax_flops.set_xlabel("Context length L (tokens)")
ax_flops.set_ylabel("FLOPs (TFLOPs)")
ax_flops.set_title("Compute (Prefill) vs. Context (with >W compressed)")
ax_flops.grid(True, linestyle="--", alpha=0.6)
ax_flops.legend(loc="upper left", ncol=1)

# 总标题与布局
fig.suptitle(
    f"{model_name}: KV Memory & Prefill Compute Scaling\n"
    f"Window W={window_size}, Compression K={K_list}, "
    f"n_layers={n_layers}, d_model={d_model}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, "
    f"weights={weight_bytes}B, KV={kv_bytes}B",
    fontsize=12.5
)
plt.tight_layout(rect=[0, 0, 1, 0.92])

# 导出高分辨率
plt.savefig("kv_memory_flops_window_compression_clean.png", bbox_inches="tight", dpi=300)
# plt.savefig("kv_memory_flops_window_compression_clean.pdf", bbox_inches="tight")
plt.show()

# -------- 关键数字（便于论文引用） --------
print("==== Key Numbers ====")
print(f"Model: {model_name}")
print(f"Weights memory (constant): {weights_mem_gb:.2f} GB")
print(f"KV per token per layer: {(2*n_kv_heads*head_dim*kv_bytes)/1024:.1f} kB (K+V, {kv_bytes}B/elt)")
print(f"Window size W: {window_size}")
print(f"Effective-length crossing (L_eff) where KV=Weights: ~{int(round(L_eff_cross))} tokens")
for k in K_list:
    if k in cross_points:
        Lx, _ = cross_points[k]
        print(f"k={k}: Break-even original L where KV=Weights ≈ {int(round(Lx))} tokens")
for Lp in [8192, 32768, 65536, 100000]:
    vals = []
    for k in K_list:
        L_eff = effective_len(Lp, window_size, k)
        f = prefill_flops_from_L(L_eff) * to_tflops
        vals.append(f"k={k}:{f:.1f}")
    print(f"L={Lp}: Prefill FLOPs -> " + ", ".join(vals))
