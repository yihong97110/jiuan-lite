"""一键跑通 标->训->推->评 全链路（直接调用 pipeline，无需启动服务）。

用法:
    python scripts/quickstart.py
真实模型:
    $env:JIUAN_MOCK=0; python scripts/quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jiuan.common import mock_mode  # noqa: E402
from jiuan.pipeline import dataprep, evaluate, infer, train  # noqa: E402


def log(msg: str) -> None:
    print("   " + msg)


def step(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    is_mock = mock_mode()
    print(f"运行模式: {'MOCK (离线兜底，指标仅演示)' if is_mock else 'REAL (Qwen2.5-0.5B)'}")

    step("① 标：数据准备（清洗 + train/valid 切分）")
    dp = dataprep.run({"source": "data/samples.jsonl", "name": "emergency-sft"}, log)
    print(f"-> 数据集 {dp['dataset_id']}  train={dp['train_count']} valid={dp['valid_count']}")

    step("② 训：SFT/LoRA 微调（仅用 train 切分）")
    tr = train.run({"dataset_id": dp["dataset_id"], "name": "qwen0.5b-sft"}, log)
    print(f"-> 模型 {tr['model_id']} (mode={tr['mode']})")

    step("③ 推：推理（含 token 计量）")
    res = infer.run({"model_id": tr["model_id"], "prompt": "地震发生时应如何避险？"}, log)
    print(f"-> 回答: {res['answer']}")
    print(f"-> 计量({res['usage']['metering']}): {res['usage']}  延迟 {res['latency_s']}s")

    step("④ 评：评测（在 valid 切分上）")
    ev = evaluate.run(
        {"model_id": tr["model_id"], "dataset_id": dp["dataset_id"], "split": "valid"}, log
    )
    reliable = "可参考" if ev["reliable"] else "仅演示(mock)"
    print(f"-> 指标[{reliable}] {ev['metrics']}  报告 {ev['report_path']}")

    print("\n全链路完成 [OK]")
    if is_mock:
        print("提示: 当前为 MOCK 模式，指标不代表真实效果；启用真实模型请设 JIUAN_MOCK=0 并安装 requirements-train.txt")


if __name__ == "__main__":
    main()
