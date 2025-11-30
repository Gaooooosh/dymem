import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import json
import argparse
import time
import shutil
import random
import statistics
import re
import string


def seed_everything(seed: int):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(tokenizer, content: str, model_name: str) -> str:
    if "qwen2" in model_name:
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return content


def load_model_and_tokenizer(path: str, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dymem.transformer.qwen2_dymem import register_customized_qwen2
    import torch

    register_customized_qwen2()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    return model, tok


def generate_one(
    model,
    tok,
    prompt: str,
    device,
    max_new_tokens: int,
    model_name: str,
):
    from dymem.utils import CacheWithMem
    import torch

    text = build_chat(tok, prompt, model_name)
    inputs = tok(text, return_tensors="pt").to(device)
    ctx_len = inputs["input_ids"].shape[-1]
    cache = CacheWithMem(model.config, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )[0]
    pred = tok.decode(out[ctx_len:], skip_special_tokens=True)
    return pred


def score_longbench(name: str, pred: str, label: str, all_classes=None, zh: bool = False):
    from eval.metrics import (
        qa_f1_score,
        qa_f1_zh_score,
        rouge_score,
        rouge_zh_score,
        classification_score,
        count_score,
        retrieval_score,
        retrieval_zh_score,
        code_sim_score,
    )

    if name in {
        "hotpotqa",
        "2wikimqa",
        "musique",
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "triviaqa",
        "nq",
    }:
        return qa_f1_score(pred, label)
    if name in {"multifieldqa_zh"}:
        return qa_f1_zh_score(pred, label)
    if name in {"gov_report", "qmsum", "multi_news", "samsum"}:
        return rouge_score(pred, label)
    if name in {"vcsum", "dureader"}:
        return rouge_zh_score(pred, label)
    if name in {"trec", "lsht"}:
        return classification_score(pred, label, all_classes=all_classes)
    if name in {"passage_count"}:
        return count_score(pred, label)
    if name in {"passage_retrieval_en"}:
        return retrieval_score(pred, label)
    if name in {"passage_retrieval_zh"}:
        return retrieval_zh_score(pred, label)
    if name in {"lcc", "repobench-p"}:
        return code_sim_score(pred, label)
    return qa_f1_score(pred, label)


def run_ruler(args, model, tok, device):
    from datasets import load_dataset
    from tqdm import tqdm

    splits = args.datasets.split(",") if args.datasets else ["qa_1", "qa_2","fwe","cwe","vt","niah_single_1","niah_single_2","niah_single_3","niah_multikey_1","niah_multikey_2","niah_multikey_3","niah_multivalue","niah_multiquery"]
    root = args.results_dir or os.path.join("ruler8192")
    os.makedirs(root, exist_ok=True)
    for split in splits:
        data = load_dataset(
            "lighteval/RULER-8192-Qwen2.5-Instruct",
            split=split
        )
        save_dir = os.path.join(root, f"data/{split}")
        try:
            data.save_to_disk(save_dir)
        except Exception:
            pass
        out_dir = os.path.join(root, f"pred_seed{args.seed}_e/{args.model_name}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        if args.exclude_or:
            out_dir = out_dir + "_exor"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{split}.jsonl")
        # smart resume/skip logic
        existing_lines = 0
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as rf:
                    existing_lines = sum(1 for _ in rf)
            except Exception:
                existing_lines = 0
        if args.exclude_or:
            existing_lines = 0  # disable resume for exclude_or to avoid misalignment
        dataset_len = len(data)
        if existing_lines >= dataset_len and not args.force_run:
            print(f"{split} already has {existing_lines}/{dataset_len} results, skipping")
            continue
        sum_scores, correct, n = 0.0, 0, 0
        mode = "a" if (existing_lines > 0 and not args.force_run) else "w"
        with open(out_path, mode, encoding="utf-8") as f:
            for i, ex in enumerate(tqdm(data, desc=f"RULER-{split}")):
                if i < existing_lines and not args.force_run:
                    continue
                if args.exclude_or and ("or" in ex.get("input", "")):
                    continue
                prompt = ex["input"]
                pred = generate_one(
                    model, tok, prompt, device, args.max_new_tokens, args.model_name
                )
                answers = ex.get("outputs")
                length = ex.get("length")
                s, acc = score_ruler(split, pred, answers)
                sum_scores += float(s)
                correct += int(acc == 1.0)
                n += 1
                json.dump({"pred": pred, "answers": answers, "length": length, "score": s}, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
        avg = float(sum_scores / n) if n > 0 else 0.0
        acc = float(correct / n) if n > 0 else 0.0
        print(f"{split} avg_f1={avg:.4f} accuracy={acc:.4f}")


def run_longbench(args, model, tok, device):
    from datasets import load_dataset
    from tqdm import tqdm

    ds = (
        args.datasets.split(",")
        if args.datasets
        else [
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]
    )
    out_root = os.path.join("longbench", f"pred_seed{args.seed}", args.model_name)
    os.makedirs(out_root, exist_ok=True)
    for name in ds:
        split_name = f"{name}_e" if args.e else name
        data = load_dataset("THUDM/LongBench", split_name, split="test")
        out_path = os.path.join(out_root, f"{name}.jsonl")
        sum_scores, n = 0.0, 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in tqdm(data, desc=f"LongBench-{name}"):
                lang = ex.get("language", "EN").upper()
                if lang.startswith("ZH"):
                    prompt = (
                        f"请根据给定上下文回答问题。\n\n上下文：{ex['context']}\n\n问题：{ex['input']}\n\n请直接给出答案。"
                    )
                else:
                    prompt = (
                        f"Answer the question based on the given context.\n\nContext:\n{ex['context']}\n\nQuestion:\n{ex['input']}\n\nAnswer:"
                    )
                pred = generate_one(
                    model, tok, prompt, device, args.max_new_tokens, args.model_name
                )
                label = ex.get("answers", [])
                label_str = (
                    label[0] if isinstance(label, list) and len(label) > 0 else str(label)
                )
                score = score_longbench(
                    name, pred, label_str, all_classes=ex.get("all_classes"), zh=lang.startswith("ZH")
                )
                sum_scores += float(score)
                n += 1
                json.dump({"pred": pred, "answers": label, "score": score}, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
        avg = float(sum_scores / n) if n > 0 else 0.0
        print(f"{name} avg_score={avg:.4f}")


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["ruler", "longbench"], required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--e", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--attention_sink", type=int, default=128)
    p.add_argument("--sliding_window", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--force_run", action="store_true")
    p.add_argument("--exclude_or", action="store_true")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device(
        os.environ.get("CUDA_DEVICE", "cuda:6") if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    model, tok = load_model_and_tokenizer(args.model_path, device)
    model.config.sliding_window = args.sliding_window
    model.config.num_attn_sinks = args.attention_sink
    if args.bench == "ruler":
        run_ruler(args, model, tok, device)
    else:
        run_longbench(args, model, tok, device)

def clean_prediction(pred: str) -> str:
    return (
        pred.split(".assistant")[0]
        .split("\n\nQuestion")[0]
        .split("</s>")[0]
        .split("(Document")[0]
        .split("\n\nQuestion")[0]
        .split("\n\nAnswer")[0]
        .split("(Passage")[0]
        .strip()
    )


def extract_words(pred: str, limit: int = None):
    pred = clean_prediction(pred)
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", pred)
    words, seen = [], set()
    stop = {"the", "and", "of", "to", "in", "a", "is", "it", "for", "on"}
    for t in tokens:
        w = t.lower()
        if len(w) < 2:
            continue
        if w in stop:
            continue
        if w not in seen:
            seen.add(w)
            words.append(w)
        if limit is not None and len(words) >= limit:
            break
    return words


def extract_upper_vars(pred: str, limit: int = None):
    pred = clean_prediction(pred)
    tokens = re.findall(r"\b[A-Z]{5}\b", pred)
    out, seen = [], set()
    for v in tokens:
        if v not in seen:
            seen.add(v)
            out.append(v)
        if limit is not None and len(out) >= limit:
            break
    return out


def extract_numbers(pred: str, limit: int = None):
    pred = clean_prediction(pred)
    tokens = re.findall(r"\d+", pred)
    out, seen = [], set()
    for n in tokens:
        if n not in seen:
            seen.add(n)
            out.append(n)
        if limit is not None and len(out) >= limit:
            break
    return out


def extract_uuids(pred: str, limit: int = None, prefer_last: bool = True):
    pred = clean_prediction(pred)
    uuid_regex = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    tokens = re.findall(uuid_regex, pred)
    out, seen = [], set()
    for u in tokens:
        ul = u.lower()
        if ul not in seen:
            seen.add(ul)
            out.append(ul)
    if limit is not None and limit > 0:
        if prefer_last and len(out) > limit:
            return out[-limit:]
        return out[:limit]
    return out


def prf1(pred_set, gold_set):
    pset, gset = set(pred_set), set(gold_set)
    if len(pset) == 0 and len(gset) == 0:
        return 1.0, 1.0, 1.0
    if len(pset) == 0:
        return 0.0, 0.0, 0.0
    inter = pset & gset
    tp = len(inter)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(gset) if gset else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


RULER_SET_CONF = {
    "vt": {"type": "vars", "limit": 5},
    "fwe": {"type": "words", "limit": 3},
    "cwe": {"type": "words", "limit": 10},
    "niah_single_1": {"type": "numbers"},
    "niah_single_2": {"type": "numbers"},
    "niah_single_3": {"type": "uuids"},
    "niah_multikey_1": {"type": "numbers", "limit": 1},
    "niah_multikey_2": {"type": "numbers", "limit": 1},
    "niah_multikey_3": {"type": "uuids", "limit": 1},
    "niah_multiquery": {"type": "numbers"},
    "niah_multivalue": {"type": "numbers"},
}


def normalize_answer_local(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score_local(prediction: str, ground_truth: str) -> float:
    normalized_prediction = normalize_answer_local(prediction)
    normalized_ground_truth = normalize_answer_local(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = sum(min(prediction_tokens.count(w), ground_truth_tokens.count(w)) for w in common)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_ruler(dataset, prediction, ground_truths, all_classes=None):
    pred_clean = clean_prediction(prediction)
    if dataset in RULER_SET_CONF:
        conf = RULER_SET_CONF[dataset]
        t = conf["type"]
        if t == "words":
            limit = conf.get("limit", len(ground_truths))
            pred_items = extract_words(pred_clean, limit=limit)
            gold_items = [str(a).lower() for a in ground_truths]
            _, _, f1 = prf1(pred_items, gold_items)
            acc = 1.0 if set(pred_items) == set(gold_items) else 0.0
            return f1, acc
        if t == "vars":
            limit = conf.get("limit", len(ground_truths))
            pred_vars = extract_upper_vars(pred_clean, limit=limit)
            gold_vars = [str(a).upper() for a in ground_truths]
            _, _, f1 = prf1(pred_vars, gold_vars)
            acc = 1.0 if set(pred_vars) == set(gold_vars) else 0.0
            return f1, acc
        if t == "numbers":
            pred_nums = extract_numbers(pred_clean)
            gold_nums = [re.findall(r"\d+", str(a))[0] if re.findall(r"\d+", str(a)) else str(a) for a in ground_truths]
            _, _, f1 = prf1(pred_nums, gold_nums)
            acc = 1.0 if set(pred_nums) == set(gold_nums) else 0.0
            return f1, acc
        if t == "uuids":
            limit = conf.get("limit", len(ground_truths))
            pred_uuids = extract_uuids(pred_clean)
            if limit == 1 and len(pred_uuids) > 1:
                pred_uuids = [pred_uuids[-1]]
            else:
                pred_uuids = pred_uuids[:limit]
            gold_uuids = [str(a).lower() for a in ground_truths]
            _, _, f1 = prf1(pred_uuids, gold_uuids)
            acc = 1.0 if set(pred_uuids) == set(gold_uuids) else 0.0
            return f1, acc
    score = 0.0
    em = 0.0
    for gt in ground_truths:
        s = qa_f1_score_local(pred_clean, gt)
        if s > score:
            score = s
        if normalize_answer_local(pred_clean) == normalize_answer_local(gt):
            em = 1.0
    return score, em


if __name__ == "__main__":
    main()
