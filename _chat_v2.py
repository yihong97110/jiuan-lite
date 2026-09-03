"""用 v2 模型做交互式多轮对话（高考志愿填报指导老师）。

直接用 transformers + LoRA adapter，手动维护 messages + 滑动窗口截断。
不依赖 langchain/langgraph。
"""
from __future__ import annotations
import os, sys, json, time
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

print(f"模型: {model_id}")
print(f"  base: {base_path}")
print(f"  adapter: {adapter_path}")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = model_meta.get("base_model_path") or model_meta.get("base")
adapter_path = str(ROOT / "data" / "models" / model_id / "weights")

print(f"  adapter: {adapter_path}")
print("加载中...")

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("模型加载完成!\n")

SYSTEM_PROMPT = "你是经验丰富的高考志愿填报指导老师，帮助高中毕业生根据分数、兴趣和就业前景选择适合的大学和专业。回答要具体、实用、可操作。"
MEMORY_WINDOW = 6  # 保留最近 6 轮

def generate(messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# --- 交互式对话 ---
session_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("=" * 60)
print("高考志愿填报指导老师 (v2 模型)")
print("输入 'quit' 退出，'reset' 重置对话")
print("=" * 60)

while True:
    try:
        user_input = input("\n[你] ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见!")
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("再见!")
        break
    if user_input.lower() == "reset":
        session_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("[系统] 对话已重置")
        continue

    session_messages.append({"role": "user", "content": user_input})

    # 滑动窗口：system 永远保留，对话只留最近 MEMORY_WINDOW 轮
    system_msgs = [m for m in session_messages if m["role"] == "system"]
    conv_msgs = [m for m in session_messages if m["role"] != "system"]
    if len(conv_msgs) > MEMORY_WINDOW * 2:
        conv_msgs = conv_msgs[-(MEMORY_WINDOW * 2):]
    active_messages = system_msgs + conv_msgs

    t0 = time.time()
    reply = generate(active_messages)
    elapsed = time.time() - t0

    session_messages.append({"role": "assistant", "content": reply})
    print(f"\n[老师] ({elapsed:.1f}s)\n{reply}")
