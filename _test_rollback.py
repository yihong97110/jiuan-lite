"""用 v1/v2 真实报告跑通自动回滚判定 + 逐 prompt 分类 + 构造 v3。"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from jiuan.pipeline import rollback

v1_report_path = ROOT / "data" / "reports" / "eval-iter-v1-20260714-123622-20260714-124824.json"
v2_report_path = ROOT / "data" / "reports" / "eval-qwen0.5b-sft-20260820-122837-20260820-124201.json"

v1 = rollback.load_report(v1_report_path)
v2 = rollback.load_report(v2_report_path)

print("=" * 70)
print("【1】指标对比判定 should_rollback")
print("=" * 70)
decision = rollback.should_rollback(v1, v2)
print(f"  action: {decision['action']}")
print(f"  severity: {decision['severity']}")
print(f"  reason: {decision['reason']}")
print(f"  deltas: {json.dumps(decision['deltas'], ensure_ascii=False, indent=2)}")

print()
print("=" * 70)
print("【2】逐 prompt 分类 analyze_prompt_deltas")
print("=" * 70)
analysis = rollback.analyze_prompt_deltas(v1, v2)
for cat in ("improved", "regressed", "stable_good", "stable_bad"):
    items = analysis[cat]
    print(f"\n  [{cat}] ({len(items)} 条):")
    for item in items:
        print(f"    Q: {item['prompt'][:50]}")
        print(f"       {item['parent_score']:.0f} -> {item['child_score']:.0f} (Δ{item['delta']:+.0f})")
        print(f"       v2答: {item['child_prediction'][:60]}...")
        if item['child_hallucination']:
            print(f"       [HALLUC]")

print()
print("=" * 70)
print("【3】安全阀 check_regression_ratio")
print("=" * 70)
safety = rollback.check_regression_ratio(analysis, len(v2.get("details", [])))
print(f"  triggered: {safety['triggered']}")
print(f"  ratio: {safety['ratio']:.1%} (regressed {safety['regressed_count']}/{safety['total']})")
if safety['triggered']:
    print(f"  ⚠ {safety['recommendation']}")

print()
print("=" * 70)
print("【4】构造回滚数据集 build_rollback_dataset")
print("=" * 70)
if not safety['triggered'] or decision['action'] == 'rollback':
    result = rollback.build_rollback_dataset(
        parent_dataset_id="iterv-20260714-123547",      # v1 训练集
        child_dataset_id="iter-v2-badcase-fix-20260820", # v2 训练集
        analysis=analysis,
        output_dataset_id="rollback-v3-20260820",
    )
    print(f"  {json.dumps(result, ensure_ascii=False, indent=2)}")
else:
    print("  安全阀触发，跳过自动构造，建议人工介入")

print()
print("=" * 70)
print("完成")
print("=" * 70)
