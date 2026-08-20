"""Breadth validation across domain categories.

This module is intentionally small: it reads a category-labeled JSONL source,
samples across categories in a round-robin order, runs model inference, then
optionally asks the configured LLM-as-Judge for per-sample quality scores.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ..common import REPORTS, ROOT, load_config
from . import infer, judge
from .evaluate import _score


def _source_path(source: str) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    path.relative_to(ROOT.resolve())
    return path


def _read_rows(source: str) -> list[dict]:
    path = _source_path(source)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prompt(row: dict) -> str:
    prompt = (row.get("instruction") or row.get("prompt") or "").strip()
    if row.get("input"):
        prompt += "\n" + str(row["input"]).strip()
    return prompt


def _reference(row: dict) -> str:
    return (row.get("output") or row.get("reference") or row.get("annotation") or "").strip()


def _category(row: dict) -> str:
    return (row.get("category") or row.get("domain") or "未分类").strip() or "未分类"


def _select_breadth(rows: list[dict], max_samples: int | None) -> list[dict]:
    if not max_samples or len(rows) <= max_samples:
        return rows
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_category(row)].append(row)
    ordered = sorted(buckets)
    selected: list[dict] = []
    idx = 0
    while len(selected) < max_samples and ordered:
        next_ordered = []
        for cat in ordered:
            items = buckets[cat]
            if idx < len(items):
                selected.append(items[idx])
                if len(selected) >= max_samples:
                    break
            if idx + 1 < len(items):
                next_ordered.append(cat)
        ordered = next_ordered
        idx += 1
    return selected


def _judge_mode(cfg: dict, want: str | None) -> tuple[bool, bool, str]:
    value = str(want or "auto").lower()
    if value in ("false", "0", "no"):
        return False, False, "disabled"
    available = judge.available(cfg)
    required = value in ("true", "1", "yes")
    if required and not available:
        raise RuntimeError("已明确开启 LLM-as-Judge，但 Judge 配置、密钥或服务当前不可用")
    return available, required, "pending" if available else "unavailable"


def _safe_report_name(model_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", model_id).strip("_") or "model"


def list_reports(limit: int = 20) -> list[dict]:
    """Return recent breadth reports for UI panels."""
    rows: list[dict] = []
    for path in REPORTS.glob("breadth-*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        issues = report.get("issues") or []
        category_scores = report.get("category_scores") or []
        details = report.get("details") or []
        legacy_errors = [str(d.get("judge_error")) for d in details if d.get("judge_error")]
        legacy_scores = [d.get("judge_score") for d in details if d.get("judge_score") is not None]
        judge_status = report.get("judge_status")
        if not judge_status and legacy_errors:
            judge_status = "failed"
        elif not judge_status and legacy_scores:
            judge_status = "succeeded"
        elif not judge_status:
            judge_status = "pending" if report.get("judge_enabled") else "disabled"
        judge_error = report.get("judge_error") or (legacy_errors[0] if legacy_errors else None)
        rows.append({
            "report_id": path.stem,
            "model_id": report.get("model_id"),
            "source": report.get("source"),
            "samples": report.get("samples"),
            "judge_enabled": report.get("judge_enabled"),
            "judge_status": judge_status,
            "judge_error": judge_error,
            "judge_scored_samples": report.get("judge_scored_samples", len(legacy_scores)),
            "category_count": len(category_scores),
            "issue_count": len(issues),
            "issues": issues[:20],
            "category_scores": category_scores[:50],
            "created_at": report.get("created_at", 0),
            "report_path": str(path),
            "expansion_note": (
                "验证集由 Agent/数据源生成或扩展；judge 负责按类别评分与识别薄弱广度，"
                "不会把训练样本泄漏进 held-out 评测集。"
            ),
        })
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows[: max(1, int(limit or 20))]


def run(params: dict, log: Callable[[str], None], progress: "Callable[[str], None] | None" = None) -> dict:
    cfg = load_config()
    rows = _select_breadth(_read_rows(params["source"]), params.get("max_samples"))
    model_id = params["model_id"]
    system_prompt = params.get("system_prompt")
    backend = params.get("backend")
    use_judge, judge_required, judge_status = _judge_mode(cfg, params.get("use_judge"))

    log(f"广度验证读取 {len(rows)} 条样本，类别 {len({_category(r) for r in rows})} 个")
    prompts = [_prompt(r) for r in rows]
    refs = [_reference(r) for r in rows]
    cats = [_category(r) for r in rows]
    gen_params = {"system_prompt": system_prompt, "max_new_tokens": params.get("max_new_tokens", 192)}
    preds = infer.generate_batch(
        model_id,
        prompts,
        log=lambda _m: None,
        backend=backend,
        params=gen_params,
        on_item=(lambda i, total: progress(f"infer {i}/{total}") if progress else None),
    )

    details: list[dict] = []
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"rouge_l_f": [], "bleu_1": [], "bleu_2": [], "judge": []})
    for i, (cat, prompt, ref, pred) in enumerate(zip(cats, prompts, refs, preds), 1):
        answer = pred["answer"]
        sc = _score(answer, ref)
        detail = {
            "category": cat,
            "prompt": prompt,
            "reference": ref,
            "prediction": answer,
            **sc,
        }
        for k in ("rouge_l_f", "bleu_1", "bleu_2"):
            by_cat[cat][k].append(sc[k])
        details.append(detail)
        if progress:
            progress(f"score {i}/{len(rows)}")

    judge_error = None
    if use_judge:
        if progress:
            progress(f"judge batch 0/{len(rows)}")
        try:
            scored = judge.score_batch(cfg, details, log=log)
            for detail, jr in zip(details, scored):
                detail["judge_score"] = jr.get("overall", 0)
                detail["judge_reason"] = jr.get("reason", "")
                by_cat[detail["category"]]["judge"].append(float(jr.get("overall", 0)))
            judge_status = "succeeded"
            if progress:
                progress(f"judge batch {len(rows)}/{len(rows)}")
        except Exception as exc:  # noqa: BLE001
            judge_status = "failed"
            judge_error = str(exc)[:300]
            for detail in details:
                detail["judge_error"] = judge_error
            log(f"广度 Judge 批量评分失败（已跳过，不阻断流程）：{judge_error}")

    def avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    category_scores = []
    for cat in sorted(by_cat):
        item = {
            "category": cat,
            "samples": len(by_cat[cat]["rouge_l_f"]),
            "rouge_l_f": avg(by_cat[cat]["rouge_l_f"]),
            "bleu_1": avg(by_cat[cat]["bleu_1"]),
            "bleu_2": avg(by_cat[cat]["bleu_2"]),
            "judge_avg": avg(by_cat[cat]["judge"]),
        }
        category_scores.append(item)

    issues = [
        row for row in category_scores
        if (row.get("judge_avg") is not None and row["judge_avg"] < 3.5)
        or (row.get("rouge_l_f") is not None and row["rouge_l_f"] < 0.2)
    ]
    issues.sort(key=lambda r: ((r.get("judge_avg") if r.get("judge_avg") is not None else 5), r.get("rouge_l_f") or 0))

    report = {
        "model_id": model_id,
        "source": params["source"],
        "samples": len(rows),
        "judge_enabled": use_judge,
        "judge_status": judge_status,
        "judge_error": judge_error,
        "judge_scored_samples": sum(len(v["judge"]) for v in by_cat.values()),
        "category_scores": category_scores,
        "issues": issues,
        "details": details,
        "created_at": time.time(),
    }
    report_id = f"breadth-{_safe_report_name(model_id)}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = REPORTS / f"{report_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"广度分析完成：{len(category_scores)} 个类别，薄弱类别 {len(issues)} 个，报告 {report_id}.json")
    return {**report, "report_id": report_id, "report_path": str(path)}
