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
from collections import Counter
import torch
from tqdm import tqdm
from datasets import load_dataset

# ==========================================
#          Metrics Import & Fallback
# ==========================================
try:
    from metrics import (
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
    HAS_METRICS = True
except ImportError:
    print("Warning: 'metrics.py' not found. LongBench scores will rely on local fallbacks or fail.")
    HAS_METRICS = False

# ==========================================
#          Setup & Loading
# ==========================================

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(tokenizer, content: str, model_name: str) -> str:
    if "qwen2" in model_name.lower():
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return content

def load_model_and_tokenizer(path: str, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from dymem.transformer.qwen2_dymem import register_customized_qwen2
        register_customized_qwen2()
    except ImportError:
        pass # dymem not available, standard load

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    return model, tok

def generate_one(model, tok, prompt: str, device, max_new_tokens: int, model_name: str):
    try:
        from dymem.utils import CacheWithMem
        use_custom_cache = True
    except ImportError:
        use_custom_cache = False
    
    text = build_chat(tok, prompt, model_name)
    inputs = tok(text, return_tensors="pt").to(device)
    ctx_len = inputs["input_ids"].shape[-1]
    
    cache = None
    if use_custom_cache:
        cache = CacheWithMem(model.config, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        if use_custom_cache:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=1,
                do_sample=False, use_cache=True, past_key_values=cache
            )[0]
        else:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=1,
                do_sample=False, use_cache=True
            )[0]
            
    pred = tok.decode(out[ctx_len:], skip_special_tokens=True)
    return pred

# ==========================================
#          Cleaning & Normalization
# ==========================================

def clean_prediction(pred: str) -> str:
    """清理生成结果，移除特殊的 EOS 标记，但保留内容。"""
    stops = ["<|endoftext|>", "<|im_end|>", "</s>"]
    for s in stops:
        pred = pred.split(s)[0]
    return pred.strip()

def normalize_text_loose(text: str) -> str:
    """宽松标准化：转小写，移除非字母数字字符（保留连字符），合并空格。"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return " ".join(text.split())

def normalize_answer_local(s: str) -> str:
    """标准的 QA 标准化 (SQuAD 风格)。"""
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

# ==========================================
#          Core Scoring Strategies
# ==========================================

def substring_match_score(pred: str, ground_truths: list) -> float:
    """
    【关键策略】Loose Match (子串匹配)
    只要正确答案作为独立单词出现在预测中，即判定为正确。
    解决模型输出 "The answer is Rollo" 导致 F1 低的问题。
    """
    pred_norm = normalize_text_loose(pred)
    
    for gt in ground_truths:
        gt_norm = normalize_text_loose(str(gt))
        if not gt_norm:
            continue
            
        # 1. 精确全等
        if pred_norm == gt_norm:
            return 1.0
            
        # 2. 词边界子串匹配 (避免 "1" 匹配到 "1999")
        # pattern: (开头或空格) + 答案 + (空格或结尾)
        pattern = r"(^|\s)" + re.escape(gt_norm) + r"(\s|$)"
        if re.search(pattern, pred_norm):
            return 1.0
            
    return 0.0

def qa_f1_score_local_calc(pred: str, gt: str) -> float:
    """计算单对文本的 F1 Score (基于 Counter)"""
    pred_norm = normalize_answer_local(pred)
    gt_norm = normalize_answer_local(gt)
    
    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def smart_qa_match(pred: str, ground_truths: list) -> float:
    """
    混合策略：先尝试 Loose Match，失败则回退到 F1。
    """
    # 1. 尝试 Loose Match (针对 NIAH/实体检索)
    if substring_match_score(pred, ground_truths) == 1.0:
        return 1.0
        
    # 2. 回退到标准 F1 (针对 QA 描述性答案)
    best_f1 = 0.0
    for gt in ground_truths:
        best_f1 = max(best_f1, qa_f1_score_local_calc(pred, str(gt)))
    return best_f1

def prf1(pred_list, gold_list):
    """计算集合级别的 Precision, Recall, F1"""
    pset, gset = set(pred_list), set(gold_list)
    if not pset and not gset: return 1.0, 1.0, 1.0
    if not pset: return 0.0, 0.0, 0.0
    
    inter = pset & gset
    tp = len(inter)
    p = tp / len(pset) if pset else 0.0
    r = tp / len(gset) if gset else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

# ==========================================
#          Extractors
# ==========================================

def extract_words(pred: str, limit: int = None):
    text = normalize_text_loose(clean_prediction(pred))
    tokens = text.split()
    words, seen = [], set()
    stop = {"the", "and", "of", "to", "in", "a", "is", "it", "for", "on", "at", "with"}
    for w in tokens:
        if len(w) < 2 or w in stop: continue
        if w not in seen:
            seen.add(w)
            words.append(w)
        if limit and len(words) >= limit: break
    return words

def extract_upper_vars(pred: str, limit: int = None):
    pred = clean_prediction(pred)
    tokens = re.findall(r"\b[A-Z]{5,}\b", pred) or re.findall(r"\b[A-Z]{5}\b", pred)
    out, seen = [], set()
    for v in tokens:
        if v not in seen:
            seen.add(v)
            out.append(v)
        if limit and len(out) >= limit: break
    return out

def extract_numbers(pred: str, limit: int = None):
    pred = clean_prediction(pred)
    tokens = re.findall(r"\d+", pred)
    out, seen = [], set()
    for n in tokens:
        if n not in seen:
            seen.add(n)
            out.append(n)
        if limit and len(out) >= limit: break
    return out

def extract_uuids(pred: str, limit: int = None, prefer_last: bool = True):
    pred = clean_prediction(pred)
    # UUID regex
    tokens = re.findall(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", pred)
    out, seen = [], set()
    for u in tokens:
        ul = u.lower()
        if ul not in seen:
            seen.add(ul)
            out.append(ul)
    if not out: return []
    if limit and limit > 0:
        return out[-limit:] if prefer_last and len(out) > limit else out[:limit]
    return out

# ==========================================
#          RULER & LongBench Routers
# ==========================================

RULER_SET_CONF = {
    # 变量追踪：严格提取
    "vt": {"type": "vars", "limit": 5},
    # 常见词：严格提取
    "fwe": {"type": "words", "limit": 3},
    "cwe": {"type": "words", "limit": 10},
    
    # NIAH (大海捞针)：使用 loose_match (Substring Match)
    "niah_single_1": {"type": "loose_match"}, 
    "niah_single_2": {"type": "loose_match"},
    "niah_single_3": {"type": "loose_match"},
    "niah_multikey_1": {"type": "loose_match"},
    "niah_multikey_2": {"type": "loose_match"},
    "niah_multikey_3": {"type": "loose_match"},
    "niah_multiquery": {"type": "loose_match"},
    "niah_multivalue": {"type": "loose_match"},
}

def score_ruler(dataset, prediction, ground_truths):
    pred_clean = clean_prediction(prediction)
    
    if dataset in RULER_SET_CONF:
        conf = RULER_SET_CONF[dataset]
        t = conf["type"]
        
        # === 核心优化: Loose Match ===
        if t == "loose_match":
            score = substring_match_score(pred_clean, ground_truths)
            return score, 1.0 if score == 1.0 else 0.0

        if t == "words":
            limit = conf.get("limit", len(ground_truths))
            pred_items = extract_words(pred_clean, limit=limit)
            gold_items = [str(a).lower() for a in ground_truths]
            _, _, f1 = prf1(pred_items, gold_items)
            return f1, 1.0 if set(pred_items) == set(gold_items) else 0.0
            
        if t == "vars":
            limit = conf.get("limit", len(ground_truths))
            pred_vars = extract_upper_vars(pred_clean, limit=limit)
            gold_vars = [str(a).upper() for a in ground_truths]
            _, _, f1 = prf1(pred_vars, gold_vars)
            return f1, 1.0 if set(pred_vars) == set(gold_vars) else 0.0
            
        if t == "numbers":
            pred_nums = extract_numbers(pred_clean)
            gold_nums = [re.findall(r"\d+", str(a))[0] if re.findall(r"\d+", str(a)) else str(a) for a in ground_truths]
            _, _, f1 = prf1(pred_nums, gold_nums)
            return f1, 1.0 if set(pred_nums) == set(gold_nums) else 0.0
            
        if t == "uuids":
            limit = conf.get("limit", len(ground_truths))
            pred_uuids = extract_uuids(pred_clean)
            if limit == 1 and len(pred_uuids) > 1:
                pred_uuids = [pred_uuids[-1]]
            else:
                pred_uuids = pred_uuids[:limit]
            gold_uuids = [str(a).lower() for a in ground_truths]
            _, _, f1 = prf1(pred_uuids, gold_uuids)
            return f1, 1.0 if set(pred_uuids) == set(gold_uuids) else 0.0

    # Fallback to Smart QA
    score = smart_qa_match(pred_clean, ground_truths)
    return score, 1.0 if score == 1.0 else 0.0

def score_longbench(name: str, pred: str, label: list, all_classes=None, zh: bool = False):
    """
    LongBench 评分分发。
    必须传入 label 为 list 类型。
    """
    if not isinstance(label, list):
        label = [label]

    # 如果有官方 metrics.py，优先使用
    if HAS_METRICS:
        if name in {"hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", 
                    "multifieldqa_en", "triviaqa", "nq", "dureader"}:
            if zh or name == "dureader": return qa_f1_zh_score(pred, label)
            return qa_f1_score(pred, label)
        if name in {"multifieldqa_zh"}: return qa_f1_zh_score(pred, label)
        if name in {"gov_report", "qmsum", "multi_news", "samsum"}: return rouge_score(pred, label)
        if name in {"vcsum"}: return rouge_zh_score(pred, label)
        if name in {"trec", "lsht"}: return classification_score(pred, label, all_classes=all_classes)
        if name in {"passage_count"}: return count_score(pred, label)
        if name in {"passage_retrieval_en"}: return retrieval_score(pred, label)
        if name in {"passage_retrieval_zh"}: return retrieval_zh_score(pred, label)
        if name in {"lcc", "repobench-p"}: return code_sim_score(pred, label)
    
    # 本地 Fallback (如果没有 metrics.py)
    return smart_qa_match(pred, label)

# ==========================================
#          Main Runners
# ==========================================

def run_ruler(args, model, tok, device):
    splits = args.datasets.split(",") if args.datasets else ["qa_1", "qa_2","fwe","cwe","vt","niah_single_1","niah_single_2","niah_single_3","niah_multikey_1","niah_multikey_2","niah_multikey_3","niah_multivalue","niah_multiquery"]
    root = args.results_dir or os.path.join("ruler8192")
    os.makedirs(root, exist_ok=True)
    
    for split in splits:
        print(f"Loading RULER split: {split}")
        try:
            data = load_dataset("lighteval/RULER-8192-Qwen2.5-Instruct", split=split)
        except Exception as e:
            print(f"Failed to load {split}: {e}")
            continue

        out_dir = os.path.join(root, f"pred_seed{args.seed}_e/{args.model_name}")
        if args.exclude_or: out_dir += "_exor"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split}.jsonl")

        existing_lines = 0
        if os.path.exists(out_path) and not args.force_run and not args.exclude_or:
            with open(out_path, "r", encoding="utf-8") as rf:
                existing_lines = sum(1 for _ in rf)
        
        if existing_lines >= len(data):
            print(f"{split} finished ({existing_lines}), skipping.")
            continue

        sum_scores, correct, n = 0.0, 0, 0
        mode = "a" if existing_lines > 0 else "w"
        
        with open(out_path, mode, encoding="utf-8") as f:
            for i, ex in enumerate(tqdm(data, desc=f"RULER-{split}")):
                if i < existing_lines: continue
                if args.exclude_or and "or" in ex.get("input", ""): continue

                prompt = ex["input"]
                pred = generate_one(model, tok, prompt, device, args.max_new_tokens, args.model_name)
                answers = ex.get("outputs", [])
                
                s, acc = score_ruler(split, pred, answers)
                
                sum_scores += float(s)
                correct += int(acc == 1.0)
                n += 1
                
                json.dump({
                    "pred": pred, 
                    "answers": answers, 
                    "score": s, 
                    "accuracy": acc
                }, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
        
        avg = sum_scores/n if n else 0.0
        acc = correct/n if n else 0.0
        print(f"Result {split}: avg_f1={avg:.4f} accuracy={acc:.4f}")

def run_longbench(args, model, tok, device):
    ds_list = (
        args.datasets.split(",") if args.datasets else 
        ["hotpotqa", "2wikimqa", "musique", "dureader", "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    )
    
    out_root = os.path.join("longbench", f"pred_seed{args.seed}", args.model_name)
    os.makedirs(out_root, exist_ok=True)
    
    for name in ds_list:
        split_name = f"{name}_e" if args.e else name
        try:
            print(f"Loading LongBench: {split_name}")
            data = load_dataset("THUDM/LongBench", split_name, split="test")
        except Exception:
            continue

        out_path = os.path.join(out_root, f"{name}.jsonl")
        sum_scores, n = 0.0, 0
        
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in tqdm(data, desc=f"LongBench-{name}"):
                lang = ex.get("language", "EN").upper()
                if lang.startswith("ZH"):
                    prompt = f"请根据给定上下文回答问题。\n\n上下文：{ex['context']}\n\n问题：{ex['input']}\n\n请直接给出答案。"
                else:
                    prompt = f"Answer the question based on the given context.\n\nContext:\n{ex['context']}\n\nQuestion:\n{ex['input']}\n\nAnswer:"
                
                pred = generate_one(model, tok, prompt, device, args.max_new_tokens, args.model_name)
                label = ex.get("answers", []) # 确保这里是 list
                
                score = score_longbench(name, pred, label, ex.get("all_classes"), lang.startswith("ZH"))
                
                sum_scores += float(score)
                n += 1
                
                json.dump({"pred": pred, "answers": label, "score": score}, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
        
        avg = sum_scores / n if n else 0.0
        print(f"Result {name}: avg_score={avg:.4f}")

# ==========================================
#          Entry Point
# ==========================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["ruler", "longbench"], required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--e", action="store_true", help="Use _e split for LongBench (even longer context)")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--attention_sink", type=int, default=128)
    p.add_argument("--sliding_window", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--force_run", action="store_true")
    p.add_argument("--exclude_or", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, tok = load_model_and_tokenizer(args.model_path, device)
    
    # Configure custom model params if available
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = args.sliding_window
    if hasattr(model.config, "num_attn_sinks"):
        model.config.num_attn_sinks = args.attention_sink

    if args.bench == "ruler":
        run_ruler(args, model, tok, device)
    else:
        run_longbench(args, model, tok, device)

if __name__ == "__main__":
    main()