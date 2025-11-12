# test_qwen2_5_dymem_infer.py
import os
os.environ.setdefault("TRITON_DISABLE_AUTOTUNING", "1")
os.environ.setdefault("TRITON_DISABLE_TUNING_CACHE", "1")
os.environ.setdefault("TRITON_MAX_CL_NUM_WARPS", "4")
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from dymem.transformer.qwen2_dymem import register_customized_qwen2, Qwen2Config, Qwen2ForCausalLM

register_customized_qwen2(exist_ok=True)
BASE_ID = "Qwen/Qwen2.5-3B-Instruct"  # 预训练基座
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda:1")
# 保证 Triton 在正确的 CUDA 设备上运行
if torch.cuda.is_available():
    torch.cuda.set_device(device)
# ==== 2) 取预训练配置，并构造你的自定义配置 ====
base_cfg = AutoConfig.from_pretrained(BASE_ID)
cfg_dict = base_cfg.to_dict()
cfg_dict['sliding_window']=512
# 关键：切换到你的解码层 & 压缩器实现（默认 Mamba2），并开启滑窗/记忆等参数
cfg = Qwen2Config(
    **cfg_dict,
)
# ==== 3) 构建你的模型骨架 ==== 
model = Qwen2ForCausalLM(cfg).to(device)
model.eval()


# base_model = AutoModelForCausalLM.from_pretrained(BASE_ID).to(device)
# missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
# del base_model  # 释放显存/内存

# print("[load_state_dict] missing:", len(missing), "unexpected:", len(unexpected))
# 预期：missing 主要是你新增的 DyMem / AHN / flex-attn 相关权重；unexpected 通常为 0 或少量命名差异  :contentReference[oaicite:5]{index=5}

# ==== 6) Tokenizer ====
tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(cfg)

# ==== 7) 推理测试（单条生成）====
prompt = """非常好，这一步是理解并复用「记忆增强长上下文模型」的关键。下面我给出一份**系统的思路总结与实现要点清单**，帮助你掌握这类模型（如 Qwen2+AHN、MemGPT、RetNet、Mamba 等）在架构与训练实现上的核心逻辑，从而能**在你自己的模型中复现和改造**。

---

# 🧠 一、核心理念：分离“表达能力”和“记忆能力”

普通 Transformer 的问题：

* 注意力是**全局依赖** → 计算和显存随上下文平方增长；
* 过长序列导致旧 token 的信息在梯度中被稀释；
* 没有「状态压缩」或「长程记忆」。

因此，“记忆增强模型”的基本思想是：

> **保留 Transformer 的表达能力（局部建模） + 增加一个可训练的状态压缩器（记忆建模）**。

这个压缩器（compressor / memory module）是核心创新模块，它负责：

* 接收被丢弃的历史 token；
* 将它们编码成固定长度的「长期状态」；
* 在后续 token 的注意力中重新注入。

---

# 🧩 二、总体架构拆解

一类“带记忆的 Transformer”模型一般由以下组件组成：

| 模块                       | 功能             | 本实现中的对应部分                                       |
| ------------------------ | -------------- | ----------------------------------------------- |
| **Backbone Transformer** | 短期语义建模，负责语言流畅性 | Qwen2 主干（冻结）                                    |
| **Memory Compressor**    | 负责状态压缩，存储长程信息  | `BaseAHN`（可选 DeltaNet / GatedDeltaNet / Mamba2） |
| **Router / Gating**      | 控制何时使用记忆、如何融合  | `AHNRouter`                                     |
| **Sparse Attention**     | 控制注意力范围（减少显存）  | `FlexAttention`                                 |
| **Dual-path Training**   | 教师–学生对齐，提供监督   | Full-attn vs Mem-attn 路径                        |
| **Multi-loss Objective** | 蒸馏、KL、CE       | `loss_type = ["ce", "kl", "distill"]`           |
| **Cache Manager**        | 维护动态记忆状态       | `MemCache` + `trim_cache`                       |

---

# ⚙️ 三、实现要点与关键路径

### 1️⃣ 模型结构设计（架构层面）

1. **模块解耦：**

   * 把“记忆逻辑”独立封装成一个类（如 `BaseAHN`），只暴露 forward 接口；
   * 不改动 Transformer 层接口，这样可与任意 backbone 兼容。

2. **混合路径执行：**

   * 每层可通过 `enable_ahn` 决定是否启用记忆模块；
   * 在训练阶段走“双路径”（teacher + student）；
   * 推理阶段仅走 memory 路径。

3. **Router 设计：**

   * 对应 `h_out = α * h_mem + β * h_local`；
   * α, β 可为全局参数或随层、随位置动态生成；
   * 可加入 softmax 归一化防止数值发散。

---

### 2️⃣ 注意力实现（高效计算）

1. **稀疏 Attention：**

   * 使用 FlexAttention / FlashAttention；
   * 控制滑动窗口（`sliding_window`）；
   * 对齐 padding 长度为 128 的倍数（保证 CUDA kernel 友好）。

2. **Sliding + Sink Mask：**

   * 定义滑动窗口内局部注意力；
   * 保留少量 sink token（如序列开头）供全局信息流通；
   * 通过 `create_block_mask()` 构造 mask。

---

### 3️⃣ 训练机制设计（蒸馏与梯度）

1. **双路径对齐：**

   * 同一输入经过：

     * Full Attention（teacher，无梯度）
     * Memory Attention（student，可反传）
   * Teacher 输出作为监督信号。

2. **三种损失组合：**

   * CE（token 级预测）
   * KL（概率分布蒸馏）
   * Distill（隐层特征蒸馏）

3. **梯度流控制：**

   * 主干参数冻结；
   * 仅 AHN / Router 有梯度；
   * teacher 路径使用 `torch.no_grad()`；
   * 避免梯度穿透两条路径。

---

### 4️⃣ 记忆压缩实现（RNN / 状态模型）

1. **核心接口：**

   ```python
   def forward(self, hidden_states, q_states, k_states, v_states, past_mem=None):
       ...
       return mem_output, new_mem_cache
   ```

2. **压缩策略：**

   * 类似 RNN 状态更新：`m_t = f(m_{t-1}, new_info)`
   * 可以用 DeltaNet（线性递推）或 Mamba2（S6滤波器）结构；
   * 保证梯度能在短序列内反传，不爆炸。

3. **缓存维护：**

   * `update_cache()` 更新长期记忆；
   * `trim_cache()` 清理被压缩的旧 token；
   * 推理时每层维护独立 `mem_past_key_values`。

---

### 5️⃣ 训练稳定性与收敛技巧

| 技巧                      | 说明                                           |
| ----------------------- | -------------------------------------------- |
| **L2 Normalization**    | 对记忆输出做单位化，防梯度爆炸                              |
| **渐进启用 AHN**            | 训练前几 epoch 不启用记忆，稳定后再开启                      |
| **Distillation Warmup** | 先训练 CE loss，再逐步增加 KL 和 Distill 权重            |
| **随机滑动窗口**              | `sliding_window_type=random` 提升泛化性           |
| **Checkpointing**       | 节省显存，仅在 AHN 路径保留梯度                           |
| **混合精度 + ZeRO**         | 支持大模型训练                                      |
| **多卡同步损失**              | 使用 `torch.distributed.all_gather()` 聚合 CE/KL |

---

# 🔁 四、推理路径优化逻辑

推理时，只启用 memory 路径：

1. **Prefill 阶段（首段）**

   * 使用 FlexAttention 处理长上下文；
   * 填充缓存。

2. **Decode 阶段（增量）**

   * 使用 FlashAttention；
   * 从缓存中读取历史压缩记忆；
   * 动态更新 KV + Memory。

3. **缓存修剪**

   * 调用 `trim_cache()` 去除已压缩的 KV；
   * 保留 num_attn_sinks 头部 token。

---

# 🧮 五、你在改造模型时应关注的关键环节

| 关键点           | 你需要做的                                                      |
| ------------- | ---------------------------------------------------------- |
| **主干冻结**      | 加载 pretrained Transformer，冻结所有非记忆参数                        |
| **记忆模块接口**    | 设计 `forward(hidden_states, mem_cache)`                     |
| **注意力替换**     | 将普通 Self-Attention 替换为 `MemFlexAttn` 或类 FlashAttention 的变体 |
| **loss 对齐**   | 构建 teacher/student 双 forward，定义 KL + MSE                   |
| **梯度隔离**      | 在 teacher forward 外包裹 `torch.no_grad()`                    |
| **Router 融合** | 实现简单线性门控或自适应加权                                             |
| **mask 生成**   | 实现动态 `sliding_window` + `sink token` mask                  |
| **缓存更新**      | 在每层维护独立的 `mem_past_key_values`                             |
| **蒸馏层数控制**    | 根据任务需求选取部分层蒸馏（如高层）                                         |

---

# ⚠️ 六、常见坑点

| 问题        | 原因                            | 解决                                      |
| --------- | ----------------------------- | --------------------------------------- |
| loss 恒为 0 | 输入长度 ≤ sliding_window         | 提高序列长度或调小窗口                             |
| loss 爆炸   | 未做 L2 normalize 或 Router 未归一化 | 加入 `F.normalize` 或 sigmoid 门控           |
| 模型无法收敛    | teacher 路径未 `no_grad()` 导致冲突  | 明确包裹 `with torch.no_grad():`            |
| 显存过高      | 两路径都反向传播                      | teacher 路径仅前向                           |
| 推理失效      | 没正确维护 `mem_past_key_values`   | 检查 `update_cache()` 与 `trim_cache()` 调用 |
| 数值抖动      | AHN 状态未裁剪或未初始化                | 控制序列长度，正确 reset cache                   |

---

# 📚 七、总结：实现一类“可学习压缩记忆 Transformer”的思路框架

```text
Input → Embedding
      ↘ Full-Attention (teacher, no_grad)
        ↘ logits_ref
      ↘ Memory-Attention (student, trainable)
        ↘ AHN (DeltaNet/Mamba2)
        ↘ logits_mem
Loss = CE(logits_mem) + KL(logits_ref, logits_mem) + Distill(hidden_ref, hidden_mem)
Only AHN / Router get gradients
```

关键目标：

> **在不破坏主干语义的前提下，让压缩模块学习替代全注意力的长程建模功能。**

---

# ✅ 最后给你的建议

如果你要把这个思路迁移到自己的模型：

1. **先保证 teacher 路径可独立运行**（即 baseline 模型稳定）；
2. **实现 AHN 结构的“状态缓存 + 更新”接口**；
3. **设计一个最小可训练版本（仅一层 AHN）来验证梯度流正确**；
4. **逐步增加多层记忆 + 蒸馏**；
5. **最终冻结 backbone，只训练 memory 模块。**

---

如果你愿意，我可以帮你：

* 🧩 画出模块结构图（包括梯度方向）；
* 或者 ✍️ 给出一份最小 PyTorch 实现骨架（backbone + memory forward + dual-path loss）。
  你希望我下一步帮你做哪个？

"""
inputs = tok(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
    )
out = tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
print("=== GENERATION ===\n", out)
