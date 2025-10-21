#!/usr/bin/env python3
"""
Evaluate an AHN-augmented HuggingFace Transformers causal LM on the RULER dataset
using the lm-evaluation-harness (lm-eval) with the transformers backend.

This script mirrors the custom layer registration used in the local inference template
so that AHN-enabled Qwen variants load correctly via AutoModelForCausalLM.
"""

import argparse
import json
import numpy as np
from pathlib import Path

from lm_eval import evaluator
from lm_eval.tasks import TaskManager
# Register custom Qwen layers that add AHN support (same as in inference.py)
from ahn.transformer.qwen2_ahn import register_customized_qwen2
from ahn.transformer.qwen3_ahn import register_customized_qwen3


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on RULER with lm-eval (HF backend)")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the HF model to evaluate (pretrained=...)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used by the HF backend",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for evaluation, e.g., cuda:0 or cpu",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default="ruler",
        help="Comma-separated task list (default: ruler)",
    )
    p.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size for evaluation (use 'auto' to let lm-eval choose)",
    )
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for tasks that support it",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (useful for quick smoke tests)",
    )
    p.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply HF chat template for instruct/chat models (auto if None)",
    )
    p.add_argument(
        "--fewshot-as-multiturn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render few-shot as multi-turn chat when chat template is applied",
    )
    p.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to HF from_pretrained",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=32640,
        help="Annotate metadata: model's supported max sequence length",
    )
    p.add_argument(
        "--output",
        type=str,
        default="ruler_eval_results.json",
        help="Path to save evaluation results JSON",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure AHN customizations for Qwen models are registered in this Python process
    register_customized_qwen2()
    register_customized_qwen3()

    # Auto-apply chat template for instruct/chat models unless overridden
    is_instruct_like = any(x in args.model.lower() for x in ["instruct", "chat"]) 
    apply_chat = args.apply_chat_template if args.apply_chat_template is not None else is_instruct_like
    print(f"Apply chat template: {apply_chat}")
    # Compose lm-eval model args for the HF backend (only args intended for HF from_pretrained)
    model_args = (
        f"pretrained={args.model},tokenizer={args.model},dtype={args.dtype},"
        f"trust_remote_code={'true' if args.trust_remote_code else 'false'}"
    )
    # Run evaluation via the Python API (equivalent to CLI `lm-eval`)
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        device=args.device,
        limit=args.limit,
        apply_chat_template=apply_chat,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        task_manager=TaskManager(metadata={"max_seq_lengths": [8000], "tokenizer": args.model}),
    )

    # Persist results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # Handle numpy and other non-JSON-serializable objects gracefully
        def _json_default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            if isinstance(o, (np.dtype,)):
                return str(o)
            # Fallback to string representation to avoid crashes
            return str(o)

        json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)

    # Print a short summary to stdout
    print(f"Saved results to: {out_path}")
    if "versions" in results:
        print("lm-eval versions:", results["versions"]) 
    if "results" in results and "ruler" in results["results"]:
        print("RULER metrics:")
        for k, v in results["results"]["ruler"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()