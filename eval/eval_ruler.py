import os
import re
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any
from metrics import qa_f1_score  # reuse existing QA F1 from longbench metrics


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--e", action="store_true", help="Evaluate on ruler")
    parser.add_argument("--results_path", type=str, default=None)
    return parser.parse_args(args)


# --------------------------
# QA metric is imported from metrics.py to keep consistency
# --------------------------


# --------------------------
# RULER-specific extractors
# --------------------------
STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "a",
    "is",
    "it",
    "for",
    "on",
}


def clean_prediction(pred: str) -> str:
    """Mimic eval.py style of cleaning noisy suffixes/prefixes."""
    if pred is None:
        return ""
    cleaned = (
        pred.split(".assistant")[0]
        .split("\n\nQuestion")[0]
        .split("</s>")[0]
        .split("(Document")[0]
        .split("\n\nQuestion")[0]
        .split("\n\nAnswer")[0]
        .split("(Passage")[0]
        .strip()
    )
    return cleaned


def extract_words(pred: str, limit: int = None) -> List[str]:
    """Extract lowercase words from prediction, excluding common stopwords.

    - Keeps alphabetic tokens and hyphenated words.
    - Deduplicates while preserving order.
    - Optionally returns first `limit` unique tokens.
    """
    pred = clean_prediction(pred)
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", pred)
    words = []
    seen = set()
    for t in tokens:
        w = t.lower()
        if len(w) < 2:
            continue
        if w in STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            words.append(w)
        if limit is not None and len(words) >= limit:
            break
    return words


def extract_upper_vars(pred: str, limit: int = None) -> List[str]:
    """Extract 5-letter uppercase variable names from prediction.

    - Matches tokens like ABCDE
    - Deduplicates while preserving order.
    """
    pred = clean_prediction(pred)
    tokens = re.findall(r"\b[A-Z]{5}\b", pred)
    vars_ = []
    seen = set()
    for v in tokens:
        if v not in seen:
            seen.add(v)
            vars_.append(v)
        if limit is not None and len(vars_) >= limit:
            break
    return vars_


def extract_numbers(pred: str, limit: int = None) -> List[str]:
    """Extract integer numbers from prediction as strings; deduplicate preserving order."""
    pred = clean_prediction(pred)
    tokens = re.findall(r"\d+", pred)
    nums = []
    seen = set()
    for n in tokens:
        if n not in seen:
            seen.add(n)
            nums.append(n)
        if limit is not None and len(nums) >= limit:
            break
    return nums


def extract_uuids(pred: str, limit: int = None, prefer_last: bool = True) -> List[str]:
    """Extract UUIDs (hex format with hyphens) from prediction.

    - Matches canonical UUID pattern: 8-4-4-4-12 hex digits
    - Normalizes to lowercase
    - Deduplicates while preserving order
    - If `limit` is provided and prefer_last=True, returns the last `limit` uuids
    """
    pred = clean_prediction(pred)
    # canonical UUID regex
    uuid_regex = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    tokens = re.findall(uuid_regex, pred)
    uuids = []
    seen = set()
    for u in tokens:
        ul = u.lower()
        if ul not in seen:
            seen.add(ul)
            uuids.append(ul)
    if limit is not None and limit > 0:
        if prefer_last and len(uuids) > limit:
            return uuids[-limit:]
        return uuids[:limit]
    return uuids


# --------------------------
# Set-based metrics (precision/recall/F1)
# --------------------------
def prf1(pred_set: List[str], gold_set: List[str]) -> Tuple[float, float, float]:
    pset = set(pred_set)
    gset = set(gold_set)
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


dataset2metric = {
    # General QA tasks within RULER
    "qa_1": qa_f1_score,
    "qa_2": qa_f1_score,
}

# RULER set-comparison tasks configuration (handled specially inside scorer)
RULER_SET_CONF = {
    # Variable Tracking: extract uppercase 5-letter vars, compare to gold set
    "vt": {"type": "vars", "limit": 5},
    # Frequent Word Extraction: extract top-3 words (unordered set)
    "fwe": {"type": "words", "limit": 3},
    # Common Word Extraction: extract top-10 words
    "cwe": {"type": "words", "limit": 10},
    # Needle In A Haystack variants: numbers
    "niah_single_1": {"type": "numbers"},
    "niah_single_2": {"type": "numbers"},
    "niah_single_3": {"type": "uuids"},
    # Multikey variants use UUIDs; prediction often contains both query and answer UUIDs
    # We score using the last extracted UUID (the answer UUID) with limit=1
    "niah_multikey_1": {"type": "numbers", "limit": 1},
    "niah_multikey_2": {"type": "numbers", "limit": 1},
    "niah_multikey_3": {"type": "uuids", "limit": 1},
    "niah_multiquery": {"type": "numbers"},
    "niah_multivalue": {"type": "numbers"},
}


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    """Length-bucketed scoring, aligned with eval.py. For RULER set tasks, use F1; for QA, use qa_f1."""
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        # Clean prediction following eval.py style
        prediction = (
            prediction.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )

        if dataset in RULER_SET_CONF:
            # One-shot set-based score against full ground_truths list
            dtype = RULER_SET_CONF[dataset]["type"]
            if dtype == "words":
                limit = RULER_SET_CONF[dataset].get("limit", len(ground_truths))
                pred_items = extract_words(prediction, limit=limit)
                gold_items = [str(a).lower() for a in ground_truths]
                _, _, f1 = prf1(pred_items, gold_items)
                score = f1
            elif dtype == "vars":
                limit = RULER_SET_CONF[dataset].get("limit", len(ground_truths))
                pred_vars = extract_upper_vars(prediction, limit=limit)
                gold_vars = [str(a).upper() for a in ground_truths]
                _, _, f1 = prf1(pred_vars, gold_vars)
                score = f1
            elif dtype == "numbers":
                pred_nums = extract_numbers(prediction)
                gold_nums = [re.findall(r"\d+", str(a))[0] if re.findall(r"\d+", str(a)) else str(a) for a in ground_truths]
                _, _, f1 = prf1(pred_nums, gold_nums)
                score = f1
            elif dtype == "uuids":
                limit = RULER_SET_CONF[dataset].get("limit", len(ground_truths))
                pred_uuids = extract_uuids(prediction)
                # Prefer the answer UUID which appears last in the sentence
                if limit == 1 and len(pred_uuids) > 1:
                    pred_uuids = [pred_uuids[-1]]
                else:
                    pred_uuids = pred_uuids[:limit]
                gold_uuids = [str(a).lower() for a in ground_truths]
                _, _, f1 = prf1(pred_uuids, gold_uuids)
                score = f1
        else:
            # QA and other metric functions (if any) use dataset2metric
            for ground_truth in ground_truths:
                try:
                    score = max(
                        score,
                        dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes),
                    )
                except Exception:
                    continue

        if length is None:
            bucket = "0-4k"
        elif length < 4000:
            bucket = "0-4k"
        elif length < 8000:
            bucket = "4-8k"
        else:
            bucket = "8k+"
        scores[bucket].append(score)
    for key in scores.keys():
        scores[key] = round(100 * (np.mean(scores[key]) if len(scores[key]) > 0 else 0.0), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    """Overall average score, aligned with eval.py. Handles RULER set tasks specially."""
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        prediction = (
            prediction.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )

        if dataset in RULER_SET_CONF:
            dtype = RULER_SET_CONF[dataset]["type"]
            if dtype == "words":
                limit = RULER_SET_CONF[dataset].get("limit", len(ground_truths))
                pred_items = extract_words(prediction, limit=limit)
                gold_items = [str(a).lower() for a in ground_truths]
                _, _, f1 = prf1(pred_items, gold_items)
                score = f1
            elif dtype == "vars":
                limit = RULER_SET_CONF[dataset].get("limit", len(ground_truths))
                pred_vars = extract_upper_vars(prediction, limit=limit)
                gold_vars = [str(a).upper() for a in ground_truths]
                _, _, f1 = prf1(pred_vars, gold_vars)
                score = f1
            elif dtype == "numbers":
                pred_nums = extract_numbers(prediction)
                gold_nums = [re.findall(r"\d+", str(a))[0] if re.findall(r"\d+", str(a)) else str(a) for a in ground_truths]
                _, _, f1 = prf1(pred_nums, gold_nums)
                score = f1
            elif dtype == "uuids":
                limit = RULER_SET_CONF[dataset].get("limit", len(ground_truths))
                pred_uuids = extract_uuids(prediction)
                if limit == 1 and len(pred_uuids) > 1:
                    pred_uuids = [pred_uuids[-1]]
                else:
                    pred_uuids = pred_uuids[:limit]
                gold_uuids = [str(a).lower() for a in ground_truths]
                _, _, f1 = prf1(pred_uuids, gold_uuids)
                score = f1
        else:
            for ground_truth in ground_truths:
                try:
                    score = max(
                        score,
                        dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes),
                    )
                except Exception:
                    continue
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == "__main__":
    args = parse_args()
    scores = dict()
    if args.results_path:
        path = args.results_path
    else:
        path = f"./ruler8192/pred_seed{args.seed}_e/{args.model}/"
    all_files = os.listdir(path)
    all_files.sort()
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split(".")[0]

        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])  # list
                    all_classes = data.get("all_classes", None)
                    if "length" in data:
                        lengths.append(data["length"])
                    else:
                        lengths.append(None)
                except Exception:
                    continue
        if len(predictions) == 0:
            continue
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[filename] = score

        print(f"{filename}: {score}")

    if args.results_path:
        out_path = os.path.join(args.results_path, "result.json")
    else:
        out_path = f"./ruler8192/pred_seed{args.seed}_e/{args.model}/result.json"

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
