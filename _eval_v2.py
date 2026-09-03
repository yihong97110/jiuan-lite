"""评测 v2 模型，对比 v1 bad case 是否改善。"""
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

# 用 v1 同一 valid 集，评测 v2 训练出的模型
params = {
    "model_id": "qwen0.5b-sft-20260820-122837",  # 刚训出的 v2
    "dataset_id": "iter-v2-badcase-fix-20260820",  # 同一数据集，valid 集 = v1 的 16 条
    "use_judge": True,
}

def log(msg):
    print(f"[eval] {msg}", flush=True)

def progress(p):
    print(f"[progress] {p}", flush=True)

print("=" * 60)
print(f"评测模型: {params['model_id']}")
print(f"评测数据集: {params['dataset_id']}")
print(f"启用 Judge: {params['use_judge']}")
print("=" * 60, flush=True)

t0 = time.time()
try:
    result = evaluate.run(params, log=log, progress=progress)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"评测完成，耗时 {elapsed:.1f}s")
    print(f"报告路径: {result.get('report_path', '未知')}")
    print(f"指标: {result.get('metrics', {})}")
    print(f"gaps 数: {len(result.get('gaps', []))}")
except Exception as e:
    print(f"\n❌ 评测失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
