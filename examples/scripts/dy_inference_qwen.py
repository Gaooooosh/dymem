# test_qwen2_5_dymem_infer.py
import os
import torch
# import torch._dynamo as dynamo
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
device = torch.device(os.environ.get("CUDA_DEVICE", "cuda:4") if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

MODEL_DIR = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,attn_implementation="sdpa").to(device)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, padding_side="right")
model.eval()
# ==== 6) Tokenizer ====
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
# tok.padding_side = "left"

# ==== 7) 推理测试（单条生成）====
# Guy de Lusignan -> QA1
# "629c2ae3-1d9a-4659-82ec-9f2dfbf6e16f" -> S3
# ["1432519","3211291","1249314","7010308"] -> MV
with open('/home/xiaoyonggao/dymem/examples/scripts/S3.txt', 'r') as f:
    prompt = f.read()

messages = [
  {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
  {"role": "user", "content": f"{prompt}"}
]
inputs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(inputs, return_tensors="pt").to(device)
input_len = inputs["input_ids"].shape[-1]
with torch.no_grad():
    model.config.sliding_window = 128
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=1,
        use_cache=True,
        do_sample=False,
    )[0]
    pred = tok.decode(output[input_len:], skip_special_tokens=True)
print(f"\n=== GENERATION ===\n{pred}\n")
print(f"\n输入长度:{input_len}")

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# 也可以导出给 TensorBoard 看火焰图
# prof.export_chrome_trace("trace.json")