"""非交互式测试 v2 模型对话。"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from jiuan.common import ROOT, load_config

model_id = "qwen0.5b-sft-20260823-122722"
base_path = str(ROOT / "data" / "registry" / "base" / "qwen2.5-0.5b-instruct")
adapter_path = str(ROOT / "data" / "models" / model_id / "weights")

print(f"base: {base_path}")
print(f"adapter: {adapter_path}")
print("loading...")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_path, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("loaded!\n")

SYSTEM_PROMPT = "你是经验丰富的高考志愿填报指导老师，帮助高中毕业生根据分数、兴趣和就业前景选择适合的大学和专业。回答要具体、实用、可操作。"

def generate(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# 两轮对话测试
q1 = "我理科考了580分，想学计算机，有什么好的大学推荐？"
q2 = "那如果我想去北京读呢？有哪些学校可以考虑？"

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for i, q in enumerate([q1, q2], 1):
    messages.append({"role": "user", "content": q})
    active = messages[:1] + messages[1:][-12:]  # system + 最近6轮
    t0 = time.time()
    reply = generate(active)
    elapsed = time.time() - t0
    messages.append({"role": "assistant", "content": reply})
    print(f"第{i}轮 ({elapsed:.1f}s):")
    print(f"  Q: {q}")
    print(f"  A: {reply}")
    print()

print("测试通过! 可启动交互式对话: python _chat_v2.py")
