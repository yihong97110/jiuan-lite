"""查看前端迭代看板里 v1 和 v2 的状态。"""
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from jiuan import registry

result = registry.iterations()

print("=" * 70)
print("前端 /iterations 接口返回的迭代看板数据")
print("=" * 70)
for chain in result:
    print(f"\n血缘链: {' -> '.join(chain.get('chain', []))}")
    for v in chain.get("versions", []):
        print(f"\n  [{v.get('dataset_id', '?')}]")
        print(f"    model_id:       {v.get('model_id', '-')}")
        print(f"    sample_count:   {v.get('sample_count', '-')}")
        print(f"    parent_dataset: {v.get('parent_dataset', '-')}")
        print(f"    train_loss:     {v.get('train_loss', '-')}")
        print(f"    rouge_l_f:      {v.get('rouge_l_f', '-')}")
        print(f"    bleu_1:         {v.get('bleu_1', '-')}")
        print(f"    judge_avg:      {v.get('judge_avg', '-')}")
        print(f"    bad_case_rate:  {v.get('bad_case_rate', '-')}")
        print(f"    version:        {v.get('version', '-')}")
