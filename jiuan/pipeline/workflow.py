"""工作流：一键串起 标->训->推->评 全链路，逐步记录，避免重复踩坑。

对标久安「任务管理/任务编排」：把四个环节固化成一条可审计的流水线，
每步产物回填到下一步，任一步失败即整体失败并保留已完成步骤的结果。
"""
from __future__ import annotations

from typing import Callable

from . import dataprep, evaluate, infer, train


def run(params: dict, log: Callable[[str], None]) -> dict:
    steps: list[dict] = []

    def record(name: str, result: dict) -> None:
        steps.append({"step": name, "result": result})

    # ① 标：数据准备（清洗 + train/valid 切分）
    log("① 标 dataprep 开始")
    dp = dataprep.run(
        {
            "source": params.get("source", "data/samples.jsonl"),
            "name": params.get("name", "workflow"),
            "valid_ratio": params.get("valid_ratio", 0.2),
            "seed": params.get("seed", 42),
        },
        log,
    )
    dataset_id = dp["dataset_id"]
    record("dataprep", dp)
    log(f"① 标 完成：dataset={dataset_id} train={dp['train_count']} valid={dp['valid_count']}")

    # ② 训：SFT/LoRA（后端可选 auto/hf/llamafactory/mock）
    log("② 训 train 开始")
    tr = train.run(
        {
            "dataset_id": dataset_id,
            "backend": params.get("backend"),
            "method": params.get("method"),
            "epochs": params.get("epochs"),
            "device": params.get("device"),
            "name": params.get("model_name", "qwen0.5b-sft"),
        },
        log,
    )
    model_id = tr["model_id"]
    record("train", tr)
    log(f"② 训 完成：model={model_id} backend={tr.get('backend')} loss={tr.get('train_loss')}")

    # ③ 推：抽样推理验证（可通过 probes 自定义问题）
    probes = params.get("probes") or [
        "火灾逃生的基本原则是什么？",
        "台风来临前应做哪些准备？",
    ]
    log("③ 推 infer 开始")
    infer_results = []
    for q in probes:
        res = infer.generate(model_id, q, log=lambda _m: None)
        infer_results.append({"prompt": q, "answer": res["answer"], "usage": res["usage"]})
        log(f"③ 推 [{q}] -> {res['answer'][:40]}…")
    record("infer", {"probes": infer_results})

    # ④ 评：在 valid 切分上评测
    log("④ 评 eval 开始")
    ev = evaluate.run(
        {"model_id": model_id, "dataset_id": dataset_id, "split": "valid"}, log
    )
    record("eval", ev)
    log(f"④ 评 完成：reliable={ev['reliable']} metrics={ev['metrics']}")

    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "backend": tr.get("backend"),
        "train_loss": tr.get("train_loss"),
        "metrics": ev["metrics"],
        "reliable": ev["reliable"],
        "steps": steps,
    }
