"""标注后端 · jsonl 实现（默认后端，单机零依赖）。

把评测薄弱点(gap)转成可人工填空的标注任务文件，
业务人员在 jsonl 里编辑 answer(annotation) 列即可，无需额外工具(如 Label Studio)。

闭环：
  gap 分析 → 生成 gap-<日期>.jsonl（标注任务）
  → 人工在 annotation 列填空、status 置 annotated
  → commit：把已标注项追加为训练样本(instruction/output)，
     经 dataprep(parent 增量继承) 生成新数据集，供触发训练。

标注任务行格式：
  {"prompt": "...", "reference": "", "annotation": "", "status": "pending",
   "gap_type": "...", "iteration": "vN"}
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..common import DATA, ROOT
from ..pipeline import evaluate
from .. import registry

ANNOTATIONS = DATA / "annotations"
ANNOTATIONS.mkdir(parents=True, exist_ok=True)

SYS_INSTRUCTION = "你是应急管理领域的专业助手，回答需准确、简洁、可执行。"


def _read_jsonl(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def create_task(name: str = "gap", max_items: Optional[int] = None, log=None) -> dict:
    """从当前所有评测薄弱点生成一个标注任务文件（按 prompt 去重）。"""
    log = log or (lambda _m: None)
    gaps = evaluate.collect_all_gaps()
    held_out = registry.held_out_prompts()  # 固定评测集 prompt，禁止进入训练→标注闭环
    seen: set = set()
    skipped_eval = 0
    items = []
    for g in gaps:
        p = (g.get("prompt") or "").strip()
        if not p or p in seen:
            continue
        if p in held_out:
            skipped_eval += 1
            continue
        seen.add(p)
        items.append({
            "prompt": p,
            "reference": g.get("reference", ""),
            "annotation": "",
            "status": "pending",
            "gap_type": g.get("gap_type", ""),
            "iteration": g.get("iteration", "-"),
        })
    if max_items:
        items = items[:max_items]

    task_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = ANNOTATIONS / f"{task_id}.jsonl"
    _write_jsonl(path, items)
    log(f"生成标注任务 {task_id}：{len(items)} 条待标注（按 prompt 去重）"
        + (f"，已排除 {skipped_eval} 条固定评测集 prompt(防泄漏)" if skipped_eval else ""))
    return {
        "task_id": task_id,
        "path": str(path),
        "rel_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "count": len(items),
        "pending": len(items),
    }


def list_tasks() -> list:
    """列出所有标注任务文件及其完成进度。"""
    out = []
    for p in sorted(ANNOTATIONS.glob("*.jsonl")):
        if p.stem.endswith("-committed") or p.stem == "resolved_log" or p.name.startswith("."):
            continue  # 回灌 source / 追踪日志 / 隐藏文件 不是标注任务
        rows = _read_jsonl(p)
        done = sum(1 for r in rows if _is_annotated(r))
        out.append({
            "task_id": p.stem,
            "rel_path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "count": len(rows),
            "annotated": done,
            "pending": len(rows) - done,
        })
    out.sort(key=lambda x: x["task_id"], reverse=True)
    return out


def _is_annotated(row: dict) -> bool:
    """已标注判定：annotation 非空即视为已标注（status 可选辅助）。"""
    ann = (row.get("annotation") or "").strip()
    status = (row.get("status") or "").strip().lower()
    if status in ("annotated", "done", "ok"):
        return bool(ann)
    return bool(ann)  # 只要填了 annotation 就算标注完成


def save_annotations(task_id: str, annotations: dict, log=None) -> dict:
    """浏览器内直接标注：把 {行号: 答案} 回写到标注任务 jsonl，无需手动编辑文件。

    annotations: {"0": "答案A", "3": "答案B"}，key 为行索引(字符串或数字)。
    """
    log = log or (lambda _m: None)
    path = ANNOTATIONS / f"{task_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"标注任务不存在: {task_id}")
    rows = _read_jsonl(path)
    updated = 0
    for k, v in (annotations or {}).items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(rows):
            ans = (v or "").strip()
            rows[idx]["annotation"] = ans
            rows[idx]["status"] = "annotated" if ans else "pending"
            updated += 1
    _write_jsonl(path, rows)
    done = sum(1 for r in rows if _is_annotated(r))
    log(f"保存标注 {updated} 条，当前已标注 {done}/{len(rows)}")
    return {"task_id": task_id, "updated": updated, "count": len(rows),
            "annotated": done, "pending": len(rows) - done}


def coverage() -> dict:
    """标注覆盖度：跨所有标注任务，按 gap_type / iteration 统计已标注/待标注，
    一眼看出哪些类别还没补完。"""
    by_type: dict = {}
    by_iter: dict = {}
    total = done = 0
    for p in sorted(ANNOTATIONS.glob("*.jsonl")):
        if p.stem.endswith("-committed") or p.stem == "resolved_log" or p.name.startswith("."):
            continue
        for r in _read_jsonl(p):
            total += 1
            annotated = _is_annotated(r)
            done += 1 if annotated else 0
            gt = (r.get("gap_type") or "未分类").strip() or "未分类"
            it = (r.get("iteration") or "-").strip() or "-"
            for bucket, key in ((by_type, gt), (by_iter, it)):
                b = bucket.setdefault(key, {"total": 0, "annotated": 0})
                b["total"] += 1
                b["annotated"] += 1 if annotated else 0
    def _fmt(bucket):
        out = []
        for k, v in bucket.items():
            pct = round(v["annotated"] / v["total"] * 100, 1) if v["total"] else 0.0
            out.append({"key": k, "total": v["total"], "annotated": v["annotated"],
                        "pending": v["total"] - v["annotated"], "pct": pct})
        out.sort(key=lambda x: (x["pct"], -x["total"]))  # 覆盖率低的排前面
        return out
    return {
        "total": total, "annotated": done, "pending": total - done,
        "overall_pct": round(done / total * 100, 1) if total else 0.0,
        "by_gap_type": _fmt(by_type),
        "by_iteration": _fmt(by_iter),
    }


def preview(task_id: str) -> dict:
    path = ANNOTATIONS / f"{task_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"标注任务不存在: {task_id}")
    rows = _read_jsonl(path)
    return {
        "task_id": task_id,
        "count": len(rows),
        "annotated": sum(1 for r in rows if _is_annotated(r)),
        "rows": rows,
    }


def commit(task_id: str, name: str = "annotated", parent: Optional[str] = None,
           hard_weight: int = 1, log=None) -> dict:
    """把已标注项(annotation 非空)转成训练样本，经 dataprep(parent 增量)生成新数据集。"""
    log = log or (lambda _m: None)
    from ..pipeline import dataprep  # 延迟导入避免循环

    path = ANNOTATIONS / f"{task_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"标注任务不存在: {task_id}")
    rows = _read_jsonl(path)
    annotated = [r for r in rows if _is_annotated(r)]
    if not annotated:
        raise ValueError("没有已标注的样本（请先在 annotation 列填写答案）")

    # 防泄漏：固定评测集里的 prompt 不得回灌进训练集
    held_out = registry.held_out_prompts()
    before = len(annotated)
    annotated = [r for r in annotated if (r.get("prompt") or "").strip() not in held_out]
    if before != len(annotated):
        log(f"防泄漏：已剔除 {before - len(annotated)} 条与固定评测集重叠的标注")
    if not annotated:
        raise ValueError("已标注样本全部与固定评测集重叠，已跳过(防泄漏)")

    # 转 instruction/output 格式，写成新 source 文件
    hw = max(1, int(hard_weight or 1))
    src_records = []
    hard_n = 0
    for r in annotated:
        gt = (r.get("gap_type") or "")
        is_hard = "知识缺失" in gt or "质量薄弱" in gt
        w = hw if (is_hard and hw > 1) else 1
        if w > 1:
            hard_n += 1
        system_prompt = (r.get("system_prompt") or r.get("system") or SYS_INSTRUCTION).strip()
        src_records.append({
            "instruction": r["prompt"].strip(), "input": "",
            "output": r["annotation"].strip(), "weight": w,
            "system_prompt": system_prompt,
        })
    if hard_n:
        log(f"硬样本加权：{hard_n} 条知识缺失类样本在训练集内按 {hw}x 复制")
    src_path = ANNOTATIONS / f"{task_id}-committed.jsonl"
    _write_jsonl(src_path, src_records)
    log(f"已标注 {len(annotated)}/{len(rows)} 条 → 写入 source {src_path.name}")

    # 走 dataprep（支持 parent 增量继承），生成新数据集
    rel_src = str(src_path.relative_to(ROOT)).replace("\\", "/")
    dp = dataprep.run(
        {"source": rel_src, "name": name, "valid_ratio": 0.2, "seed": 42, "parent": parent},
        log,
    )
    # P2: 记录本次回灌解决了哪些 gap prompt(供跨版本追踪)
    resolved_prompts = [r["prompt"].strip() for r in annotated]
    _append_resolved_log({
        "dataset_id": dp["dataset_id"],
        "parent_dataset": dp.get("parent_dataset"),
        "task_id": task_id,
        "prompts": resolved_prompts,
        "hard_weight": hw,
        "committed_at": time.time(),
    })
    return {
        "task_id": task_id,
        "annotated": len(annotated),
        "total": len(rows),
        "committed_source": rel_src,
        "dataset_id": dp["dataset_id"],
        "parent_dataset": dp.get("parent_dataset"),
        "dataset_count": dp["count"],
        "added_count": dp.get("added_count"),
        "resolved_prompts": resolved_prompts,
    }


RESOLVED_LOG = ANNOTATIONS / "resolved_log.jsonl"


def _append_resolved_log(record: dict) -> None:
    with open(RESOLVED_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolved_tracking() -> list:
    """P2 跨版本 gap 解决追踪：每次回灌解决了哪些薄弱点 prompt，
    并对比该 prompt 在"回灌前最近一次"与"回灌后最近一次"评测中的 judge/rouge 变化。"""
    if not RESOLVED_LOG.exists():
        return []
    logs = []
    with open(RESOLVED_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except Exception:
                    continue

    # 收集每条评测报告里每个 prompt 的分数，带报告时间
    from ..common import REPORTS
    per_prompt = {}  # prompt -> [(created_at, rouge, judge, report_id)]
    for rp in REPORTS.glob("*.json"):
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        ts = rep.get("created_at", 0)
        for d in rep.get("details", []):
            pr = (d.get("prompt") or "").strip()
            if not pr:
                continue
            per_prompt.setdefault(pr, []).append(
                (ts, d.get("rouge_l_f"), d.get("judge_score"), rp.stem)
            )
    for pr in per_prompt:
        per_prompt[pr].sort(key=lambda x: x[0])

    out = []
    for lg in logs:
        cut = lg.get("committed_at", 0)
        items = []
        for pr in lg.get("prompts", []):
            hist = per_prompt.get(pr.strip(), [])
            before = [h for h in hist if h[0] <= cut]
            after = [h for h in hist if h[0] > cut]
            b = before[-1] if before else None
            a = after[0] if after else None
            if a and b:
                if (a[2] or 0) > (b[2] or 0):
                    status = "improved"
                    note = "该问题在回灌前后都被评测命中，可对比逐题变化。"
                else:
                    status = "flat_or_down"
                    note = "该问题在回灌前后都被评测命中，但 judge 未提升。"
            elif hist:
                status = "pending_eval" if not a else "after_only"
                note = "该问题只在单侧评测报告中出现，缺少完整前后对照。"
            else:
                status = "not_in_eval"
                note = (
                    "该回灌样本未出现在固定评测集，因此没有逐题 judge/ROUGE。"
                    "这是防止训练样本泄漏到 held-out 评测集的正常现象；请看迭代级评测指标。"
                )
            items.append({
                "prompt": pr,
                "before": {"rouge_l_f": b[1], "judge": b[2], "report_id": b[3]} if b else None,
                "after": {"rouge_l_f": a[1], "judge": a[2], "report_id": a[3]} if a else None,
                "status": status,
                "eval_note": note,
            })
        out.append({
            "dataset_id": lg.get("dataset_id"),
            "parent_dataset": lg.get("parent_dataset"),
            "committed_at": lg.get("committed_at"),
            "hard_weight": lg.get("hard_weight", 1),
            "resolved_count": len(lg.get("prompts", [])),
            "items": items,
        })
    out.sort(key=lambda x: x.get("committed_at", 0), reverse=True)
    return out
