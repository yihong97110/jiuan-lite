import json
r = json.load(open("data/reports/eval-qwen0.5b-sft-20260823-122019-20260823-122529.json", encoding="utf-8"))
for i, g in enumerate(r["gaps"]):
    print(f"[{i+1}] Q: {g['prompt'][:60]}")
    print(f"    type: {g['gap_type']}")
    print(f"    suggestion: {g.get('suggestion', '')[:80]}")
    print()
