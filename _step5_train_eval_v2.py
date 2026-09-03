"""第 5 步：训练 v2 + 评测。"""
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

from jiuan.pipeline import train, evaluate

def log(msg):
    print(f"[{msg.split(':')[0]}] {msg.split(':',1)[1].strip() if ':' in msg else ''}", flush=True)

def progress(p):
    print(f"[progress] {p}", flush=True)

# --- 训练 v2 ---
params = {
    "dataset_id": "gaokao-v2-20260820",
    "base_model": "0.5b",
    "epochs": 10,
    "lr": 2.0e-4,
    "batch_size": 1,
    "grad_accum": 1,
    "precision": "fp32",
    "device": "cpu",
}

print("=" * 60)
print(f"v2 训练: dataset=gaokao-v2-20260820 (20条)")
print("=" * 60, flush=True)

t0 = time.time()
try:
    result = train.run(params, log=lambda m: print(f"[train] {m}", flush=True),
                       progress=progress)
    elapsed = time.time() - t0
    v2_model_id = result.get("model_id", "")
    print(f"\nv2 训练完成，耗时 {elapsed:.1f}s")
    print(f"model_id: {v2_model_id}")
    print(f"train_loss: {result.get('train_loss', 'N/A')}")
except Exception as e:
    print(f"\n[FAIL] v2 训练失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- 评测 v2 ---
print("\n" + "=" * 60)
print(f"v2 评测: model={v2_model_id}")
print("=" * 60, flush=True)

eval_params = {
    "model_id": v2_model_id,
    "dataset_id": "gaokao-v2-20260820",
    "use_judge": True,
}

t0 = time.time()
try:
    result = evaluate.run(eval_params,
                          log=lambda m: print(f"[eval] {m}", flush=True),
                          progress=progress)
    elapsed = time.time() - t0
    print(f"\nv2 评测完成，耗时 {elapsed:.1f}s")
    print(f"报告: {result.get('report_path', 'N/A')}")
    print(f"gaps: {len(result.get('gaps', []))}")
    if result.get("metrics"):
        print(f"指标: {result['metrics']}")
except Exception as e:
    print(f"\n[FAIL] v2 评测失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
