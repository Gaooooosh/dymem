import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1" 
import torch
import argparse
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# === lm_eval imports ===
from lm_eval import simple_evaluate
# 尝试兼容不同的 HFLM 导入路径
try:
    from lm_eval.models.huggingface import HFLM
except ImportError:
    try:
        from lm_eval.models.gpt2 import HFLM
    except ImportError:
        print("Error: Could not import HFLM. Please ensure lm_eval is installed (pip install lm_eval).")
        exit(1)

# === 自定义模型 imports ===
from dymem.transformer.qwen2_dymem import register_customized_qwen2
from dymem.utils import CacheWithMem

# 设置标准日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_dymem")

# 注册模型
register_customized_qwen2(exist_ok=True)

class DyMemWrapper(HFLM):
    """
    自定义的 Wrapper，用于在 generate 时注入 CacheWithMem
    """
    def __init__(self, pretrained, **kwargs):
        # 初始化父类
        super().__init__(pretrained=pretrained, backend="causal", **kwargs)

    def _model_generate(self, context, max_length, stop, **kwargs):
        """
        重写生成逻辑，核心目的是在每次生成前初始化 CacheWithMem
        """
        # 1. 显式初始化 CacheWithMem
        # 确保 device 和 dtype 正确
        past_key_values = CacheWithMem(
            self.model.config, 
            dtype=self.model.dtype, 
            device=self.device
        )

        # 2. 准备参数
        generation_kwargs = kwargs.copy()
        generation_kwargs["use_cache"] = True
        generation_kwargs["past_key_values"] = past_key_values
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        
        # 处理 stop token
        # lm_eval 传入的 stop 是一个 string list
        # HF generate 接受 stop_strings (新版) 或需要外部处理
        # 这里我们直接透传，如果 HF 版本较旧可能不支持 stop_strings，会报 warning
        
        return self.model.generate(
            context,
            max_length=max_length,
            stop_strings=stop, 
            tokenizer=self.tokenizer,
            **generation_kwargs
        )

def main():
    parser = argparse.ArgumentParser(description="Evaluate DyMem model with CacheWithMem")
    parser.add_argument("--model_path", type=str, default="/home/xiaoyonggao/dymem/qwen2.5-3b-Ins-comp-5811")
    parser.add_argument("--tasks", type=str, default="longbench", help="Comma separated list of tasks")
    parser.add_argument("--batch_size", type=str, default="1", help="Batch size (e.g., '1', 'auto')")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()

    # 设置 device
    if ":" in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    
    logger.info(f"Loading model from: {args.model_path}")
    
    # 初始化自定义 Wrapper
    lm = DyMemWrapper(
        pretrained=args.model_path,
        device=args.device,
        batch_size=args.batch_size, # 注意：batch_size 类型如果是字符串 'auto' 可能需要特殊处理，这里建议传 int 或让 HFLM 处理
        dtype="bfloat16", 
        trust_remote_code=True,
    )

    # 处理 Tokenizer pad_token
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
        lm.tokenizer.pad_token_id = lm.tokenizer.eos_token_id

    # 处理任务列表
    task_list = args.tasks.split(",")
    logger.info(f"Starting evaluation on tasks: {task_list}")
    
    # 运行评测
    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=args.batch_size,
        device=args.device,
        log_samples=False,
    )

    # 打印简要结果
    if results is not None:
        print(json.dumps(results.get("results", {}), indent=2))
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "result.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()