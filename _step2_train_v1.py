"""第 2 步：CPU 训练 v1 模型（高考志愿填报）。"""
from __future__ import annotations
import os, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ["JIUAN_CONFIG"] = str(ROOT / "configs" / "qwen0.5b.yaml")

from jiuan.pipeline import train

params = {
    "dataset_id": "gaokao-v1-20260820",
    "base_model": "0.5b",
    "epochs": 10,
    "lr": 2.0e-4,
    "batch_size": 1,
    "grad_accum": 1,
    "precision": "fp32",
    "device": "cpu",
}

def log(msg):
    print(f"[train] {msg}", flush=True)

def progress(p):
    print(f"[progress] {p}", flush=True)

print("=" * 60)
print("v1 训练配置:")
for k, v in params.items():
    print(f"  {k} = {v}")
print("=" * 60, flush=True)

t0 = time.time()
try:
    result = train.run(params, log=log, progress=progress)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"v1 训练完成，耗时 {elapsed:.1f}s")
    print(f"结果: {result}")
except Exception as e:
    print(f"\n[FAIL] v1 训练失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
