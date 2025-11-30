import argparse
import json
import os
import time
import random
import uuid
from collections import deque
from typing import Dict, Any


CONFIG_DEFAULTS = {
    # 这些参数保留只是为了接口统一，实际不会再去访问远程数据集
    "DATASET_NAME": "synthetic_niah",
    "DATASET_CONFIG": "default",
    "SPLIT": "train",

    "OUTPUT_FILE": "niah_probe_dataset.jsonl",

    # 现在：必须按 token 数控制
    "MIN_CHARS": 0,
    "MIN_INPUT_TOKENS": 4096,
    "TOKENIZER_MODEL": "Qwen/Qwen2.5-3B-Instruct",
    "MAX_INPUT_TOKENS": 9000,

    "TARGET_COUNT": 5000,

    # 去重相关
    "DEDUP_SIM_THRESHOLD": 1.0,
    "DEDUP_WINDOW": 200,

    "ERROR_LOG": os.path.join("examples", "scripts", "gen_data_errors.log"),

    # probe ID 类型比例
    "MAGIC_RATIO": 0.5,          # 0.5 = 一半 magic number，一半 uuid

    # 中文样本比例
    "ZH_PROB": 0.3,              # 每条样本是中文的概率
}


# ------------------------ 工具函数 ------------------------


def ensure_error_log_path(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def log_error(path: str, msg: str) -> None:
    ensure_error_log_path(path)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(path, "a", encoding="utf-8") as ef:
        ef.write(f"[{ts}] {msg}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate OFFLINE synthetic NIAH-style probe retrieval dataset "
        "with unified format: instruction / input / output, length controlled by token counts."
    )

    # 这些参数只是为了接口兼容，实际上不会访问远程
    p.add_argument("--dataset-name", default=CONFIG_DEFAULTS["DATASET_NAME"])
    p.add_argument("--dataset-config", default=CONFIG_DEFAULTS["DATASET_CONFIG"])
    p.add_argument("--split", default=CONFIG_DEFAULTS["SPLIT"])

    p.add_argument("--output-file", default=CONFIG_DEFAULTS["OUTPUT_FILE"])

    # 严格按 token 数控制
    p.add_argument("--min-chars", type=int, default=CONFIG_DEFAULTS["MIN_CHARS"])
    p.add_argument("--min-input-tokens", type=int, default=CONFIG_DEFAULTS["MIN_INPUT_TOKENS"])
    p.add_argument("--tokenizer-model", default=CONFIG_DEFAULTS["TOKENIZER_MODEL"])
    p.add_argument("--max-input-tokens", type=int, default=CONFIG_DEFAULTS["MAX_INPUT_TOKENS"])

    # 数据规模 & 去重
    p.add_argument("--target-count", type=int, default=CONFIG_DEFAULTS["TARGET_COUNT"])
    p.add_argument("--dedup-sim-threshold", type=float, default=CONFIG_DEFAULTS["DEDUP_SIM_THRESHOLD"])
    p.add_argument("--dedup-window", type=int, default=CONFIG_DEFAULTS["DEDUP_WINDOW"])

    # 探针 ID 类型比例
    p.add_argument(
        "--magic-ratio",
        type=float,
        default=CONFIG_DEFAULTS["MAGIC_RATIO"],
        help="Probability of using magic-number style probe IDs vs UUID (0.0-1.0).",
    )

    # 中文样本比例
    p.add_argument(
        "--zh-prob",
        type=float,
        default=CONFIG_DEFAULTS["ZH_PROB"],
        help="Probability that a sample is in Chinese (0.0-1.0).",
    )

    # 其他
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42, help="Random seed for probe insertion and IDs")

    return p.parse_args()


# ------------------------ 去重相关 ------------------------


def should_skip_by_similarity(window: deque, text: str, threshold: float) -> bool:
    if threshold >= 1.0 or len(window) == 0:
        return False
    import difflib

    for prev in window:
        if difflib.SequenceMatcher(a=prev, b=text).ratio() >= threshold:
            return True
    return False


# ------------------------ 探针相关 ------------------------


def generate_magic_number() -> str:
    """生成一个类似 C 里 'magic number' 的 32-bit 十六进制值, 例如 0x1A2B3C4D"""
    return f"0x{random.getrandbits(32):08X}"


def generate_uuid() -> str:
    """标准 UUID 字符串"""
    return str(uuid.uuid4())


def build_probe_templates(lang_is_zh: bool, probe_id: str) -> Dict[str, str]:
    """
    probe_id: 已经是完整 ID，例如 'NIAH-0xDEADBEEF' 或 'NIAH-550e8400-e29b-...'
    """
    if lang_is_zh:
        instruction = "请在阅读完整输入后，找出并复述文中出现的唯一探针编号。"
        question_prefix = "问题：文中唯一的探针编号是什么？\n"
        needle_sentence = f"【探针句】在这段文本中，唯一的探针编号是：{probe_id}。"
        answer = probe_id
    else:
        instruction = "After reading the entire input, identify and repeat the unique probe ID mentioned in the text."
        question_prefix = "Question: What is the unique probe ID mentioned in the text?\n"
        needle_sentence = f"[PROBE SENTENCE] In this passage, the unique probe ID is: {probe_id}."
        answer = probe_id

    return {
        "instruction": instruction,
        "question_prefix": question_prefix,
        "needle_sentence": needle_sentence,
        "answer": answer,
    }


def inject_needle(context: str, needle_sentence: str) -> str:
    """
    将探针句插入到长文本中间某个段落之间。
    按行切分，随机在 1/3~2/3 之间插入。
    """
    lines = context.split("\n")
    if not lines:
        return needle_sentence

    n = len(lines)
    if n <= 2:
        insert_idx = 1
    else:
        low = max(1, n // 3)
        high = min(n - 1, (2 * n) // 3)
        if low >= high:
            insert_idx = low
        else:
            insert_idx = random.randint(low, high)

    new_lines = lines[:insert_idx] + [needle_sentence] + lines[insert_idx:]
    return "\n".join(new_lines)


# ------------------------ 合成英文 / 中文“干草堆” ------------------------


EN_WORDS = [
    "system", "network", "memory", "representation", "analysis", "dataset", "embedding", "attention",
    "sequence", "optimization", "performance", "evaluation", "context", "architecture", "iteration",
    "feature", "semantic", "symbolic", "probability", "gradient", "inference", "learning", "baseline",
    "token", "alignment", "robustness", "generalization", "signal", "structure", "transformer",
    "scale", "computation", "objective", "regularization", "module", "intervention", "probe",
    "experiment", "interpretation", "behavior", "pattern", "distribution", "configuration",
    "automatic", "generation", "search", "index", "retrieval", "projection"
]


def generate_english_line() -> str:
    """生成一行英文句子（20-60 个词），用于逐步堆长上下文。"""
    line_len = random.randint(20, 60)
    words = [random.choice(EN_WORDS) for _ in range(line_len)]
    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:] + "."
    return sentence


ZH_SEGMENTS = [
    "在这一部分的讨论中，我们重点关注模型在长上下文场景下的表现",
    "实验结果表明，在不同的数据规模下，系统的收敛速度存在明显差异",
    "当输入序列长度逐渐增加时，注意力机制的计算开销也会随之升高",
    "为了更准确地评估模型的记忆能力，我们设计了一系列对照实验",
    "这些实验不仅考虑了理论上的复杂度分析，也关注了实际推理中的延迟",
    "在多轮迭代之后，模型逐渐学会了如何在噪声中识别关键信息",
    "我们还观察到，在特定维度上的表示会呈现出高度结构化的几何形态",
    "这提示我们，内部表征可能蕴含一种近似符号化的编码方式",
    "在构建评测基准时，我们刻意加入了一些较为极端的数据分布",
    "这样可以检验模型在超出训练分布时的泛化与稳健性",
    "针对不同的探针任务，我们分别设计了检索类与推理类两种配置",
    "在某些情况下，模型能够在极长的上下文中准确定位到特定信息片段",
    "这说明合适的训练目标可以显著提升长程记忆与检索能力",
    "然而，当噪声比例进一步增加时，性能开始出现明显波动",
    "因此，我们认为仅靠参数规模的堆叠并不能解决所有问题",
    "更有效的方法是精心构造数据集和任务目标，从而引导模型学会内在结构",
]


def generate_chinese_line() -> str:
    """生成一行中文句子，用于逐步堆长上下文。"""
    seg = random.choice(ZH_SEGMENTS)
    if random.random() < 0.3:
        seg = "此外，" + seg
    if random.random() < 0.3:
        seg = seg + "。总体而言，这一趋势相当稳定。"
    return seg


def build_context_with_token_bounds(
    lang_is_zh: bool,
    tok,
    min_tokens: int,
    max_tokens: int,
) -> str:
    """
    不断追加行（英文/中文），每次 tokenize 检查长度，
    直到达到 [min_tokens, max_tokens] 范围。如果超过 max_tokens，就按 token 截断。
    """
    assert max_tokens > 0 and min_tokens > 0 and max_tokens >= min_tokens

    lines = []
    # 安全保护，最多追加这么多行，防止极端情况死循环
    max_lines = 10000

    for _ in range(max_lines):
        # 先追加一行
        if lang_is_zh:
            lines.append(generate_chinese_line())
        else:
            lines.append(generate_english_line())

        context = "\n".join(lines)
        # token 统计
        ids = tok(context, add_special_tokens=False, truncation=False)["input_ids"]
        n_tok = len(ids)

        if n_tok >= max_tokens:
            # 直接截断到 max_tokens，然后 decode
            ids = ids[:max_tokens]
            context = tok.decode(ids)
            return context

        if n_tok >= min_tokens:
            # 已经在区间内，直接返回
            return context

    # 如果意外跑满 max_lines 还不够，就直接按当前 context 截断到 min_tokens（尽力而为）
    context = "\n".join(lines)
    ids = tok(context, add_special_tokens=False, truncation=False)["input_ids"]
    if len(ids) >= min_tokens:
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        context = tok.decode(ids)
        return context

    # 实在不行就返回当前 context（极少发生）
    return context


# ------------------------ 主逻辑（完全离线，按 token 控制） ------------------------


def generate_probe_dataset(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_file": args.output_file,
                    "target_count": args.target_count,
                    "magic_ratio": args.magic_ratio,
                    "zh_prob": args.zh_prob,
                    "min_input_tokens": args.min_input_tokens,
                    "max_input_tokens": args.max_input_tokens,
                    "tokenizer_model": args.tokenizer_model,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    random.seed(args.seed)

    # 必须有 token 限制
    if args.min_input_tokens <= 0 or args.max_input_tokens <= 0 or args.max_input_tokens < args.min_input_tokens:
        print("参数错误：必须提供有效的 --min-input-tokens 和 --max-input-tokens（>0 且 max>=min）。")
        return

    # 必须有 tokenizer
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
    except Exception as e:
        print(f"错误：加载 tokenizer 失败（{repr(e)}），无法按 token 数控制长度。")
        return

    # 已有数据量（支持断点续写）
    existing = 0
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as rf:
                for _ in rf:
                    existing += 1
        except Exception as e:
            log_error(CONFIG_DEFAULTS["ERROR_LOG"], f"read_existing_failed: {e}")

    dedup_set = set()
    sim_window = deque(maxlen=max(1, args.dedup_window))

    # 进度条（可选）
    use_tqdm = False
    tqdm = None
    try:
        from tqdm import tqdm as _tqdm  # type: ignore

        use_tqdm = True
        tqdm = _tqdm
    except Exception:
        pass

    with open(args.output_file, "a", encoding="utf-8") as f:
        pbar = None
        if use_tqdm and tqdm is not None:
            pbar = tqdm(
                total=args.target_count,
                initial=min(existing, args.target_count),
                desc="written",
            )

        count = existing
        seen = existing  # 这里只是计数用

        while count < args.target_count:
            seen += 1

            # 随机选择中文 / 英文样本
            lang_is_zh = random.random() < args.zh_prob

            # 构造满足 token 数的上下文
            context = build_context_with_token_bounds(
                lang_is_zh=lang_is_zh,
                tok=tok,
                min_tokens=args.min_input_tokens,
                max_tokens=args.max_input_tokens,
            )

            # 简单字符长度过滤（可选）
            if args.min_chars and len(context) < args.min_chars:
                continue

            # 去重（基于前 1024 字符）
            h = hash(context[:1024])
            if h in dedup_set:
                continue
            if should_skip_by_similarity(sim_window, context[:2000], args.dedup_sim_threshold):
                continue

            # 生成 probe ID
            if random.random() < args.magic_ratio:
                raw_id = generate_magic_number()
            else:
                raw_id = generate_uuid()
            full_probe_id = f"NIAH-{raw_id}"

            tmpl = build_probe_templates(lang_is_zh=lang_is_zh, probe_id=full_probe_id)

            # 插针
            context_with_probe = inject_needle(context, tmpl["needle_sentence"])

            # instruction / input / output 统一格式
            full_input = (
                tmpl["question_prefix"]
                + "\n"
                + "===== Context Start =====\n"
                + context_with_probe
                + "\n===== Context End ====="
            )

            row = {
                "instruction": tmpl["instruction"],
                "input": full_input,
                "output": tmpl["answer"],
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            dedup_set.add(h)
            sim_window.append(context[:2000])

            count += 1
            if pbar is not None:
                pbar.update(1)
            elif count % 10 == 0:
                print(f"progress: {count}/{args.target_count}")

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    print(f"生成完成: {args.output_file}")
    print(f"统计: seen={seen}, written={count - existing}")


def main():
    args = parse_args()
    generate_probe_dataset(args)


if __name__ == "__main__":
    main()
