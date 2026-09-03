"""实测单条加载 vs 批量加载的耗时差。"""
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

os.environ["JIUAN_CONFIG"] = str(ROOT / "configs" / "qwen0.5b.yaml")

from jiuan.pipeline import infer

# 用刚训出来的 v2 模型，3 条 prompt 测单条 vs 批量
prompts_3 = [
    "厨房油锅起火如何扑救？",
    "AED 自动体外除颤器如何使用？",
    "台风预警信号有哪几种颜色？",
]

def silent_log(m): pass

print("=== 单条加载（3 次独立加载模型）===")
t0 = time.time()
for p in prompts_3:
    r = infer.generate("qwen0.5b-sft-20260820-122837", p, log=silent_log, backend="transformers")
t_single = time.time() - t0
print(f"单条加载耗时: {t_single:.1f}s（3 条）")

print("\n=== 批量加载（1 次加载评 3 条）===")
t0 = time.time()
results = infer.generate_batch("qwen0.5b-sft-20260820-122837", prompts_3, log=silent_log, backend="transformers")
t_batch = time.time() - t0
print(f"批量加载耗时: {t_batch:.1f}s（3 条）")

print(f"\n加速比: {t_single/t_batch:.1f}x")
print(f"外推到 16 条 valid 集：")
print(f"  单条: {t_single*16/3:.1f}s ≈ {t_single*16/3/60:.1f} 分钟")
print(f"  批量: {t_batch*16/3:.1f}s ≈ {t_batch*16/3/60:.1f} 分钟")
