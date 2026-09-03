"""第 6 步：自动回滚判定 v1->v2 + 逐 prompt 分类。"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from jiuan.pipeline import rollback

v1_report = rollback.load_report("data/reports/eval-qwen0.5b-sft-20260823-122019-20260823-122529.json")
v2_report = rollback.load_report("data/reports/eval-qwen0.5b-sft-20260823-122722-20260823-123458.json")

print("=" * 60)
print("[1] should_rollback")
print("=" * 60)
decision = rollback.should_rollback(v1_report, v2_report)
print(f"  action: {decision['action']}")
print(f"  severity: {decision['severity']}")
print(f"  reason: {decision['reason']}")
print(f"  deltas: {json.dumps(decision['deltas'], indent=2)}")

print("\n" + "=" * 60)
print("[2] analyze_prompt_deltas")
print("=" * 60)
analysis = rollback.analyze_prompt_deltas(v1_report, v2_report)
for cat in ("improved", "regressed", "stable_good", "stable_bad"):
    items = analysis[cat]
    print(f"\n  [{cat}] ({len(items)}):")
    for item in items:
        print(f"    Q: {item['prompt'][:50]}")
        print(f"       {item['parent_score']:.0f} -> {item['child_score']:.0f} (d={item['delta']:+.0f})")

print("\n" + "=" * 60)
print("[3] check_regression_ratio")
print("=" * 60)
safety = rollback.check_regression_ratio(analysis, len(v2_report.get("details", [])))
print(f"  triggered: {safety['triggered']}")
print(f"  ratio: {safety['ratio']:.1%} ({safety['regressed_count']}/{safety['total']})")

print("\n" + "=" * 60)
print("DECISION: " + decision['action'])
print("=" * 60)
