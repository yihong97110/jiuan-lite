"""第 3 步：v1 评测（含 Judge + gap 分析）。"""
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

from jiuan.pipeline import evaluate

params = {
    "model_id": "qwen0.5b-sft-20260823-122019",
    "dataset_id": "gaokao-v1-20260820",
    "use_judge": True,
}

def log(msg):
    print(f"[eval] {msg}", flush=True)

def progress(p):
    print(f"[progress] {p}", flush=True)

print("=" * 60)
print(f"v1 评测: model={params['model_id']}")
print(f"  dataset={params['dataset_id']}, judge=True")
print("=" * 60, flush=True)

t0 = time.time()
try:
    result = evaluate.run(params, log=log, progress=progress)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"v1 评测完成，耗时 {elapsed:.1f}s")
    print(f"报告路径: {result.get('report_path', '未知')}")
    print(f"gaps 数: {len(result.get('gaps', []))}")
    if result.get("metrics"):
        print(f"指标: {result['metrics']}")
except Exception as e:
    print(f"\n[FAIL] v1 评测失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
