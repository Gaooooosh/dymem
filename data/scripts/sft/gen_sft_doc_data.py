import argparse
import json
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from collections import deque
from typing import Optional, Iterable, Dict, Any


CONFIG_DEFAULTS = {
    "API_KEY": "",
    "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "MODEL_NAME": "qwen3-max",
    "DATASET_NAME": "HuggingFaceFW/fineweb-edu",
    "DATASET_CONFIG": "default",
    "SPLIT": "train",
    "OUTPUT_FILE": "memory_sft_dataset.jsonl",
    "MIN_CHARS": 0,
    "MIN_INPUT_TOKENS": 4096,
    "TOKENIZER_MODEL": "Qwen/Qwen2.5-3B-Instruct",
    "MAX_INPUT_TOKENS": 9000,
    "TARGET_COUNT": 1000,
    "TEMPERATURE": 0.7,
    "RPM": 600,
    "MIN_CHINESE_RATIO": 0.0,
    "DEDUP_SIM_THRESHOLD": 1.0,
    "DEDUP_WINDOW": 200,
    "ERROR_LOG": os.path.join("examples", "scripts", "gen_data_errors.log"),
}


SYSTEM_PROMPT = """
# Role
你是一个高精度长文本数据构建专家。你的任务是阅读长文本，构建“基于原文的精准复原”训练数据。

# Goal
请扫描输入文本的【中间区域】（30%-70%），锁定一段语义完整、信息密度高的连续内容，生成微调数据。

# Critical Constraints
1. **语言一致性**：JSON 中的 instruction 和 output 必须与输入文本的语言保持严格一致（英文对英文，中文对中文）。
2. **原文优先**：output 必须高度忠实于原文。尽量直接摘录或拼接原文中的句子。严禁大幅度改写。目标是让模型学会“复现”原文。

# Instruction Templates
- [CN] 请尽可能还原文中关于[主题]的详细描述。
- [CN] 请根据原文，复述[事件]的具体经过与细节。
- [EN] Please restore the details regarding [Topic] as described in the text.
- [EN] Please reproduce the original text regarding the process of [Event].

# Output Format (JSON Only)
{
  "instruction": "清晰的指令，明确要求复原特定的中间内容。",
  "output": "一段由原文句子组成的、逻辑连贯的文本。保留原文的数据和修辞。"
}
"""


# ------------------------ 工具函数 ------------------------


def get_api_key() -> str:
    env_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    # 当前目录 .env
    try:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as ef:
                for line in ef:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("DASHSCOPE_API_KEY=") or line.startswith("OPENAI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

    # 家目录 key 文件
    home = os.path.expanduser("~")
    for name in (".dashscope_api_key", ".openai_api_key"):
        try:
            p = os.path.join(home, name)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
        except Exception:
            pass
    return ""


def chinese_ratio(text: str) -> float:
    total = 0
    zh = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if "\u4e00" <= ch <= "\u9fff":
            zh += 1
    if total == 0:
        return 0.0
    return zh / total


def extract_json_object(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    stack = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            stack += 1
        elif c == "}":
            stack -= 1
            if stack == 0:
                return s[start : i + 1]
    return None


def valid_row_fields(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    if "instruction" not in d or "output" not in d:
        return False
    if not isinstance(d["instruction"], str) or not isinstance(d["output"], str):
        return False
    if len(d["instruction"].strip()) == 0 or len(d["output"].strip()) == 0:
        return False
    return True


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
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default="")
    p.add_argument("--base-url", default=CONFIG_DEFAULTS["BASE_URL"])
    p.add_argument("--model-name", default=CONFIG_DEFAULTS["MODEL_NAME"])

    p.add_argument("--dataset-name", default=CONFIG_DEFAULTS["DATASET_NAME"])
    p.add_argument("--dataset-config", default=CONFIG_DEFAULTS["DATASET_CONFIG"])
    p.add_argument("--split", default=CONFIG_DEFAULTS["SPLIT"])
    p.add_argument("--output-file", default=CONFIG_DEFAULTS["OUTPUT_FILE"])

    p.add_argument("--min-chars", type=int, default=CONFIG_DEFAULTS["MIN_CHARS"])
    p.add_argument("--min-input-tokens", type=int, default=CONFIG_DEFAULTS["MIN_INPUT_TOKENS"])
    p.add_argument("--tokenizer-model", default=CONFIG_DEFAULTS["TOKENIZER_MODEL"])
    p.add_argument("--max-input-tokens", type=int, default=CONFIG_DEFAULTS["MAX_INPUT_TOKENS"])

    p.add_argument("--target-count", type=int, default=CONFIG_DEFAULTS["TARGET_COUNT"])
    p.add_argument("--temperature", type=float, default=CONFIG_DEFAULTS["TEMPERATURE"])
    p.add_argument("--rpm", type=int, default=CONFIG_DEFAULTS["RPM"])
    p.add_argument("--min-chinese-ratio", type=float, default=CONFIG_DEFAULTS["MIN_CHINESE_RATIO"])
    p.add_argument("--dedup-sim-threshold", type=float, default=CONFIG_DEFAULTS["DEDUP_SIM_THRESHOLD"])
    p.add_argument("--dedup-window", type=int, default=CONFIG_DEFAULTS["DEDUP_WINDOW"])

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ------------------------ 模型调用 ------------------------


def build_client(api_key: str, base_url: str):
    """
    简化版：优先走 openai SDK（兼容 DashScope 模式），失败就用 HTTP。
    """
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url)
        return {"type": "openai", "client": client}
    except Exception:
        # 直接 HTTP
        return {"type": "http", "api_key": api_key, "base_url": base_url.rstrip("/")}


def make_completion(client, model: str, system_prompt: str, user_content: str, temperature: float):
    if client["type"] == "openai":
        c = client["client"]
        return c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )

    # HTTP 模式
    api_key = client["api_key"]
    base_url = client["base_url"]
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    def _post_with_retry() -> Dict[str, Any]:
        backoff = 1.0
        for _ in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
                    continue
                raise
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
        raise RuntimeError("http_post_failed_after_retries")

    result = _post_with_retry()
    return {
        "choices": [
            {
                "message": {
                    "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                }
            }
        ]
    }


def get_message_content(completion) -> str:
    try:
        return completion.choices[0].message.content
    except Exception:
        try:
            return completion["choices"][0]["message"]["content"]
        except Exception:
            return ""


# ------------------------ 数据集加载（简化重点） ------------------------


def hf_rows(dataset: str, config: str, split: str, start: int = 0, batch: int = 100) -> Iterable[Dict[str, Any]]:
    """
    原来的 HTTP fallback，保留但代码很简单。
    """
    offset = start
    base = "https://datasets-server.huggingface.co/rows"
    while True:
        q = (
            f"{base}?dataset={urllib.parse.quote(dataset)}"
            f"&config={urllib.parse.quote(config)}"
            f"&split={urllib.parse.quote(split)}"
            f"&offset={offset}&length={batch}"
        )
        try:
            with urllib.request.urlopen(q, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
        except Exception:
            break
        rows = data.get("rows", [])
        if not rows:
            break
        for r in rows:
            yield r.get("row", {})
        offset += len(rows)


def iter_dataset(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    """
    极简策略：
    1. 尝试使用 datasets.load_dataset(streaming=True)
    2. 失败则使用 hf_rows HTTP 接口
    """
    try:
        from datasets import load_dataset  # type: ignore

        print(f"Loading dataset via datasets.load_dataset: {args.dataset_name} ({args.dataset_config}) [{args.split}]")
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split, streaming=True)
        for row in ds:
            yield row
        return
    except Exception as e:
        print(f"[Warning] datasets.load_dataset 失败，使用 hf_rows 备用: {e}")

    print(f"Loading dataset via hf_rows API: {args.dataset_name} ({args.dataset_config}) [{args.split}]")
    yield from hf_rows(args.dataset_name, args.dataset_config, args.split, start=0, batch=100)


# ------------------------ 去重相关 ------------------------


def should_skip_by_similarity(window: deque, text: str, threshold: float) -> bool:
    if threshold >= 1.0 or len(window) == 0:
        return False
    import difflib

    for prev in window:
        if difflib.SequenceMatcher(a=prev, b=text).ratio() >= threshold:
            return True
    return False


# ------------------------ 主逻辑 ------------------------


def generate_dataset(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "output_file": args.output_file,
                    "min_input_tokens": args.min_input_tokens,
                    "tokenizer_model": args.tokenizer_model,
                    "max_input_tokens": args.max_input_tokens,
                    "target_count": args.target_count,
                    "temperature": args.temperature,
                    "rpm": args.rpm,
                    "min_chinese_ratio": args.min_chinese_ratio,
                    "dedup_sim_threshold": args.dedup_sim_threshold,
                    "dedup_window": args.dedup_window,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    # tokenizer 可选
    tok = None
    if (args.min_input_tokens and args.min_input_tokens > 0) or (
        args.max_input_tokens and args.max_input_tokens > 0
    ):
        if not args.tokenizer_model:
            print("参数错误: 使用 token 过滤时必须提供 --tokenizer-model")
            return
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
        except Exception as e:
            print(f"参数错误: 加载 tokenizer 失败（{repr(e)}），无法执行 token 过滤")
            return

    # 模型 client
    try:
        client = build_client(args.api_key, args.base_url)
    except Exception as e:
        raise RuntimeError(f"client_init_failed: {e}")

    # 进度条（可选）
    use_tqdm = False
    tqdm = None
    try:
        from tqdm import tqdm as _tqdm  # type: ignore

        use_tqdm = True
        tqdm = _tqdm
    except Exception:
        pass

    print(f"Loading stream from {args.dataset_name}...")
    ds = iter_dataset(args)

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

    with open(args.output_file, "a", encoding="utf-8") as f:
        scan_pbar = None
        pbar = None
        if use_tqdm and tqdm is not None:
            try:
                scan_pbar = tqdm(total=None, desc="scan", mininterval=0.5)
            except Exception:
                scan_pbar = None
            pbar = tqdm(
                total=args.target_count,
                initial=min(existing, args.target_count),
                desc="written",
            )

        count = existing
        last_ts = 0.0

        seen = 0
        passed = 0
        skipped_short_raw = 0
        skipped_lang = 0
        skipped_few_tokens = 0
        skipped_dup = 0
        skipped_sim = 0

        for item in ds:
            if count >= args.target_count:
                break
            if scan_pbar is not None:
                scan_pbar.update(1)

            raw_text = item.get("text", "")
            seen += 1

            # 长度过滤（字符数）
            if args.min_chars and len(raw_text) < args.min_chars:
                skipped_short_raw += 1
                continue

            # 中文比例过滤
            if args.min_chinese_ratio > 0.0 and chinese_ratio(raw_text) < args.min_chinese_ratio:
                skipped_lang += 1
                continue

            prompt_context = raw_text

            # token 截断 / 最小 token 限制
            if tok is not None:
                try:
                    ids = tok(prompt_context, add_special_tokens=False, truncation=False)["input_ids"]

                    if args.max_input_tokens and args.max_input_tokens > 0:
                        ids = ids[: args.max_input_tokens]
                        prompt_context = tok.decode(ids)

                    if args.min_input_tokens and args.min_input_tokens > 0:
                        if len(ids) < args.min_input_tokens:
                            skipped_few_tokens += 1
                            continue
                except Exception:
                    # tokenizer 出问题就直接用原文（不再强制 token 过滤）
                    pass

            # 简单去重 + 相似度过滤
            h = hash(prompt_context[:1024])
            if h in dedup_set:
                skipped_dup += 1
                continue
            if should_skip_by_similarity(sim_window, prompt_context[:2000], args.dedup_sim_threshold):
                skipped_sim += 1
                continue

            passed += 1

            # RPM 控制
            delay = max(0.0, 60.0 / float(max(1, args.rpm)))
            now = time.monotonic()
            if now - last_ts < delay:
                time.sleep(delay - (now - last_ts))

            # 调接口
            try:
                completion = make_completion(
                    client=client,
                    model=args.model_name,
                    system_prompt=SYSTEM_PROMPT,
                    user_content=f"长文本内容：\n\n{prompt_context}",
                    temperature=args.temperature,
                )
                last_ts = time.monotonic()
            except Exception as e:
                log_error(CONFIG_DEFAULTS["ERROR_LOG"], f"api_error: {repr(e)}")
                continue

            # 解析结果并写入
            try:
                content = get_message_content(completion)
                try:
                    data = json.loads(content)
                except Exception:
                    obj = extract_json_object(content)
                    if not obj:
                        raise ValueError("json_extract_failed")
                    data = json.loads(obj)

                if not valid_row_fields(data):
                    continue

                row = {
                    "instruction": data["instruction"],
                    "input": prompt_context,
                    "output": data["output"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                dedup_set.add(h)
                sim_window.append(prompt_context[:2000])

                count += 1
                if pbar is not None:
                    pbar.update(1)
                elif count % 10 == 0:
                    print(f"progress: {count}/{args.target_count}")
            except Exception as e:
                log_error(CONFIG_DEFAULTS["ERROR_LOG"], f"parse_or_write_error: {repr(e)}")
                continue

            if pbar is None and (seen % 50 == 0):
                print(
                    f"scan: seen={seen}, passed={passed}, written={count - existing}, "
                    f"skip_raw={skipped_short_raw}, skip_lang={skipped_lang}, "
                    f"skip_tokens={skipped_few_tokens}, skip_dup={skipped_dup}, skip_sim={skipped_sim}"
                )

        if scan_pbar is not None:
            try:
                scan_pbar.close()
            except Exception:
                pass
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    print(f"生成完成: {args.output_file}")
    print(f"统计: seen={seen}, written={count - existing}")


def main():
    args = parse_args()
    if not args.api_key:
        args.api_key = get_api_key()
    if not args.api_key and not args.dry_run:
        print("缺少 API Key: 请设置环境变量 DASHSCOPE_API_KEY 或使用 --api-key")
        return
    generate_dataset(args)


if __name__ == "__main__":
    main()
