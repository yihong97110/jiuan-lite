"""Impact validation for annotation backfill.

回灌样本不能原样进入 held-out 评测集，否则会训练/评测泄漏。
本模块为每次回灌生成“同主题、不同问法”的 impact 验证集，
用回灌前模型和回灌后模型做对比，并把未改善项生成补漏标注任务。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from .. import registry
from ..common import DATA, DATASETS, REPORTS, ROOT
from . import evaluate

IMPACT_DIR = DATA / "impact_eval"
ANNOTATIONS = DATA / "annotations"
IMPACT_DIR.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _resolved_logs() -> list[dict]:
    path = ANNOTATIONS / "resolved_log.jsonl"
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: r.get("committed_at", 0), reverse=True)
    return rows


def _select_log(dataset_id: str | None = None) -> dict:
    logs = _resolved_logs()
    if not logs:
        raise ValueError("暂无回灌记录，无法生成回灌效果验证集")
    if dataset_id:
        for row in logs:
            if row.get("dataset_id") == dataset_id:
                return row
        raise ValueError(f"未找到回灌记录: {dataset_id}")
    return logs[0]


def _committed_rows(task_id: str) -> list[dict]:
    path = ANNOTATIONS / f"{task_id}-committed.jsonl"
    rows = _read_jsonl(path)
    by_prompt = {(r.get("instruction") or "").strip(): r for r in rows}
    return list(by_prompt.values())


def _topic(prompt: str) -> str:
    for pat in (r"解释[:：](.+?)[。.?？]", r"「(.+?)」", r"关于[「\"]?(.+?)[」\"]?[，,。]"):
        m = re.search(pat, prompt)
        if m:
            return m.group(1).strip()
    text = re.sub(r"\s+", " ", prompt).strip()
    return text[:32] or "该知识点"


def _impact_prompt(prompt: str, idx: int) -> str:
    topic = _topic(prompt)
    templates = [
        "回灌效果验证题：围绕「{topic}」，请换一个真实研究或应用场景说明定义、关键机制、证据或判断要点，并指出一个常见误区。",
        "迁移验证题：如果学生把「{topic}」和相近概念混淆，你会如何纠正？请给出定义、机制和应用判断。",
        "查缺验证题：请用专业生物学语言解释「{topic}」的核心含义，并说明一个可验证证据或实验/应用判断。",
    ]
    return templates[idx % len(templates)].format(topic=topic)


def _patch_prompt(impact_prompt: str, idx: int) -> str:
    topic = _topic(impact_prompt)
    templates = [
        "从专业生物学角度补充说明：{topic}。要求包含定义、关键机制、判断证据和常见误区。",
        "针对薄弱点补漏：{topic}在真实研究或应用中应如何专业解释？请给出机制和一个判断要点。",
    ]
    return templates[idx % len(templates)].format(topic=topic)


def _to_chat(row: dict) -> dict:
    system_prompt = (row.get("system_prompt") or "你是专业生物学知识助手，回答要准确区分概念、机制、证据与应用。").strip()
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["instruction"].strip()},
            {"role": "assistant", "content": row["output"].strip()},
        ]
    }


def _create_eval_dataset(source_rows: list[dict], name: str, source_path: Path) -> dict:
    dataset_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = DATASETS / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_rows = [_to_chat(r) for r in source_rows]
    _write_jsonl(out_dir / "train.jsonl", [])
    _write_jsonl(out_dir / "valid.jsonl", valid_rows)
    registry.register_dataset(
        dataset_id,
        parent=None,
        sample_count=len(valid_rows),
        source=_rel(source_path),
        extra={
            "role": "eval",
            "eval_type": "impact",
            "valid_count": len(valid_rows),
            "train_count": 0,
        },
    )
    registry.mark_eval_dataset(dataset_id)
    return {
        "dataset_id": dataset_id,
        "valid_count": len(valid_rows),
        "valid_path": str(out_dir / "valid.jsonl"),
    }


def _latest_model_for_dataset(dataset_id: str | None) -> str | None:
    if not dataset_id:
        return None
    rows = [
        r for r in registry.list_models()
        if r.get("type") != "eval" and r.get("dataset_id") == dataset_id and r.get("model_id")
    ]
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("registered_at", 0), reverse=True)
    return rows[0]["model_id"]


def _avg(values: list[float]) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def _pair_details(before: dict, after: dict) -> list[dict]:
    before_by_prompt = {(d.get("prompt") or "").strip(): d for d in before.get("details", [])}
    pairs = []
    for after_d in after.get("details", []):
        prompt = (after_d.get("prompt") or "").strip()
        b = before_by_prompt.get(prompt, {})
        b_judge = b.get("judge_score")
        a_judge = after_d.get("judge_score")
        b_rouge = b.get("rouge_l_f")
        a_rouge = after_d.get("rouge_l_f")
        judge_delta = round(a_judge - b_judge, 4) if isinstance(a_judge, (int, float)) and isinstance(b_judge, (int, float)) else None
        rouge_delta = round(a_rouge - b_rouge, 4) if isinstance(a_rouge, (int, float)) and isinstance(b_rouge, (int, float)) else None
        improved = (
            (judge_delta is not None and judge_delta > 0)
            or (judge_delta is None and rouge_delta is not None and rouge_delta > 0.02)
        )
        weak_after = (
            (isinstance(a_judge, (int, float)) and a_judge < 3.5)
            or (isinstance(a_rouge, (int, float)) and a_rouge < 0.2)
            or not improved
        )
        pairs.append({
            "prompt": prompt,
            "reference": after_d.get("reference", ""),
            "before_prediction": b.get("prediction", ""),
            "after_prediction": after_d.get("prediction", ""),
            "before": {"judge": b_judge, "rouge_l_f": b_rouge, "report_id": before.get("report_id")},
            "after": {"judge": a_judge, "rouge_l_f": a_rouge, "report_id": after.get("report_id")},
            "judge_delta": judge_delta,
            "rouge_delta": rouge_delta,
            "improved": improved,
            "weak_after": weak_after,
        })
    return pairs


def _with_report_details(summary: dict) -> dict:
    path = summary.get("report_path")
    if not path:
        return summary
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return summary
    return {**report, **summary, "details": report.get("details", [])}


def _create_patch_task(report_id: str, pairs: list[dict], name: str) -> dict:
    weak = [p for p in pairs if p.get("weak_after")]
    if not weak:
        return {"task_id": None, "count": 0}
    task_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = ANNOTATIONS / f"{task_id}.jsonl"
    rows = []
    held_out = registry.held_out_prompts()
    for idx, item in enumerate(weak):
        prompt = _patch_prompt(item["prompt"], idx)
        if prompt in held_out:
            continue
        rows.append({
            "prompt": prompt,
            "reference": item.get("reference", ""),
            "annotation": item.get("reference", ""),
            "status": "annotated",
            "gap_type": "回灌效果验证未改善/查缺补漏",
            "iteration": "impact",
            "source_report_id": report_id,
            "source_impact_prompt": item.get("prompt", ""),
        })
    _write_jsonl(path, rows)
    return {"task_id": task_id, "count": len(rows), "rel_path": _rel(path)}


def run(params: dict, log: Callable[[str], None]) -> dict:
    dataset_id = params.get("dataset_id") or None
    max_items = int(params.get("max_items") or 12)
    variants = max(1, min(3, int(params.get("variants_per_prompt") or 1)))
    use_judge = params.get("use_judge") or "auto"
    create_patch = bool(params.get("create_patch_task", True))
    patch_name = params.get("patch_task_name") or "impact-gap"

    lg = _select_log(dataset_id)
    dataset_id = lg["dataset_id"]
    parent_dataset = lg.get("parent_dataset")
    committed = _committed_rows(lg["task_id"])
    if not committed:
        raise ValueError(f"未找到回灌 source: {lg['task_id']}-committed.jsonl")

    source_rows: list[dict] = []
    for row in committed:
        original = (row.get("instruction") or "").strip()
        answer = (row.get("output") or "").strip()
        if not original or not answer:
            continue
        for i in range(variants):
            source_rows.append({
                "instruction": _impact_prompt(original, i),
                "input": "",
                "output": answer,
                "system_prompt": row.get("system_prompt", ""),
                "source_prompt": original,
                "impact_variant": i,
            })
            if len(source_rows) >= max_items:
                break
        if len(source_rows) >= max_items:
            break
    if not source_rows:
        raise ValueError("没有可用于生成 impact 验证集的回灌样本")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    safe_dataset = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", dataset_id)
    source_path = IMPACT_DIR / f"impact-{safe_dataset}-{run_id}.jsonl"
    _write_jsonl(source_path, source_rows)
    ev_ds = _create_eval_dataset(source_rows, f"impact-{safe_dataset}", source_path)
    eval_dataset_id = ev_ds["dataset_id"]
    before_model = params.get("before_model_id") or _latest_model_for_dataset(parent_dataset) or "base"
    after_model = params.get("after_model_id") or _latest_model_for_dataset(dataset_id)
    if not after_model:
        raise ValueError(f"未找到回灌后模型，dataset_id={dataset_id}")

    log(f"impact 验证集 {eval_dataset_id}: {len(source_rows)} 条，before={before_model}, after={after_model}")
    eval_common = {
        "dataset_id": eval_dataset_id,
        "split": "valid",
        "use_judge": use_judge,
        "max_samples": len(source_rows),
        "system_prompt": source_rows[0].get("system_prompt") or None,
        "attach_eval": False,
    }
    before = _with_report_details(evaluate.run({**eval_common, "model_id": before_model}, log))
    after = _with_report_details(evaluate.run({**eval_common, "model_id": after_model}, log))
    pairs = _pair_details(before, after)
    improved_count = sum(1 for p in pairs if p.get("improved"))
    weak_count = sum(1 for p in pairs if p.get("weak_after"))
    report_id = f"impact-{safe_dataset}-{run_id}"
    patch = _create_patch_task(report_id, pairs, patch_name) if create_patch else {"task_id": None, "count": 0}

    report = {
        "report_id": report_id,
        "dataset_id": dataset_id,
        "parent_dataset": parent_dataset,
        "annotation_task_id": lg.get("task_id"),
        "impact_dataset_id": eval_dataset_id,
        "impact_source": _rel(source_path),
        "before_model_id": before_model,
        "after_model_id": after_model,
        "samples": len(source_rows),
        "use_judge": use_judge,
        "before_report_id": before.get("report_id"),
        "after_report_id": after.get("report_id"),
        "before_metrics": before.get("metrics", {}),
        "after_metrics": after.get("metrics", {}),
        "judge_delta": (
            round((after.get("metrics", {}).get("judge_avg") or 0) - (before.get("metrics", {}).get("judge_avg") or 0), 4)
            if ("judge_avg" in after.get("metrics", {}) or "judge_avg" in before.get("metrics", {})) else None
        ),
        "rouge_delta": round((after.get("metrics", {}).get("rouge_l_f") or 0) - (before.get("metrics", {}).get("rouge_l_f") or 0), 4),
        "improved_count": improved_count,
        "weak_count": weak_count,
        "pairs": pairs,
        "patch_task": patch,
        "created_at": time.time(),
        "method_note": (
            "impact 验证集由回灌样本生成同主题变体题，不使用训练原题；"
            "未改善项会生成新的补漏标注任务，继续进入回灌闭环。"
        ),
    }
    path = REPORTS / f"{report_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"impact 验证完成：改善 {improved_count}/{len(pairs)}，待补漏 {weak_count}，报告 {report_id}")
    return {**report, "report_path": str(path)}


def list_reports(limit: int = 20) -> list[dict]:
    rows: list[dict] = []
    for path in REPORTS.glob("impact-*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "report_id": report.get("report_id", path.stem),
            "dataset_id": report.get("dataset_id"),
            "parent_dataset": report.get("parent_dataset"),
            "impact_dataset_id": report.get("impact_dataset_id"),
            "before_model_id": report.get("before_model_id"),
            "after_model_id": report.get("after_model_id"),
            "samples": report.get("samples"),
            "before_metrics": report.get("before_metrics", {}),
            "after_metrics": report.get("after_metrics", {}),
            "judge_delta": report.get("judge_delta"),
            "rouge_delta": report.get("rouge_delta"),
            "improved_count": report.get("improved_count"),
            "weak_count": report.get("weak_count"),
            "patch_task": report.get("patch_task", {}),
            "pairs": (report.get("pairs") or [])[:50],
            "created_at": report.get("created_at", 0),
            "report_path": str(path),
            "method_note": report.get("method_note", ""),
        })
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows[: max(1, int(limit or 20))]
