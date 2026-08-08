"""自动迭代 Agent：推进 标注 -> 回灌 -> 训练 -> 评测 -> 再标注 的闭环。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .. import annotation, registry, store
from ..common import DATA, ROOT
from ..schemas import Stage, TaskStatus
from . import runner

_lock = threading.RLock()
_thread: threading.Thread | None = None
_agent_task_id: str | None = None
_state: dict[str, Any] = {
    "active": False,
    "status": "idle",
    "message": "未启动",
    "logs": [],
    "summaries": [],
}


def _snapshot() -> dict:
    with _lock:
        out = dict(_state)
        out["logs"] = list(_state.get("logs", []))
        out["summaries"] = list(_state.get("summaries", []))
        return out


def status() -> dict:
    return _snapshot()


def _set(**kw: Any) -> None:
    with _lock:
        _state.update(kw)


def _log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    with _lock:
        logs = _state.setdefault("logs", [])
        logs.append(line)
        del logs[:-120]
        task_id = _agent_task_id
    if task_id:
        store.update_task(task_id, log=message)


def _progress(text: str) -> None:
    _set(progress=text, message=text)
    if _agent_task_id:
        store.update_task(_agent_task_id, progress=text)


def _resolved_task_ids() -> set[str]:
    path = DATA / "annotations" / "resolved_log.jsonl"
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("task_id"):
                done.add(str(rec["task_id"]))
    return done


def _ready_annotation_tasks() -> list[dict]:
    committed = _resolved_task_ids()
    tasks = annotation.list_tasks()
    return [
        t for t in tasks
        if t.get("count", 0) > 0
        and t.get("pending", 0) == 0
        and t.get("task_id") not in committed
    ]


def _pending_annotation_tasks() -> list[dict]:
    return [t for t in annotation.list_tasks() if t.get("pending", 0) > 0]


def _latest_parent_dataset() -> str:
    rows = [d for d in registry.list_datasets() if d.get("role") != "eval"]
    if not rows:
        raise RuntimeError("没有可用父数据集，请先完成一次数据准备")
    rows.sort(key=lambda d: d.get("registered_at", 0))
    return rows[-1]["dataset_id"]


def _latest_eval_dataset() -> str:
    evals = registry.eval_dataset_ids()
    if not evals:
        raise RuntimeError("没有固定 held-out 评测集(role=eval)")
    return evals[-1]


def _wait_task(task_id: str, label: str, poll_interval: float) -> dict:
    last = ""
    while True:
        task = store.get_task(task_id)
        if not task:
            raise RuntimeError(f"{label} 子任务不存在: {task_id}")
        msg = f"{label}: {task.status.value}" + (f" {task.progress}" if task.progress else "")
        _progress(msg)
        if msg != last:
            _log(f"{label} 子任务 {task_id} -> {task.status.value}" + (f" ({task.progress})" if task.progress else ""))
            last = msg
        if task.status == TaskStatus.SUCCEEDED:
            return task.result
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"{label} 子任务失败 {task_id}: {task.error[:500]}")
        time.sleep(poll_interval)


def _create_task_from_gaps(gaps: list[dict], name: str, max_items: int | None = None) -> dict:
    held_out = registry.held_out_prompts()
    seen: set[str] = set()
    rows = []
    skipped_eval = 0
    for gap in gaps:
        prompt = (gap.get("prompt") or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        if prompt in held_out:
            skipped_eval += 1
            continue
        rows.append({
            "prompt": prompt,
            "reference": gap.get("reference", ""),
            "annotation": "",
            "status": "pending",
            "gap_type": gap.get("gap_type", ""),
            "iteration": gap.get("iteration", "-"),
        })
        if max_items and len(rows) >= max_items:
            break
    if not rows:
        return {"task_id": None, "count": 0, "skipped_eval": skipped_eval}
    ann_dir = DATA / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = ann_dir / f"{task_id}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "task_id": task_id,
        "count": len(rows),
        "skipped_eval": skipped_eval,
        "rel_path": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


def start(params: dict) -> dict:
    global _thread, _agent_task_id
    with _lock:
        if _thread and _thread.is_alive():
            return _snapshot()
        task = store.create_task(Stage.AGENT, params)
        _agent_task_id = task.id
        _state.clear()
        _state.update({
            "active": True,
            "status": "running",
            "message": "启动中",
            "progress": "starting",
            "task_id": task.id,
            "params": params,
            "started_at": time.time(),
            "logs": [],
            "summaries": [],
        })
        store.update_task(task.id, status=TaskStatus.RUNNING, log="Agent 自动迭代启动")
        _thread = threading.Thread(target=_run, args=(task.id, params), name="jiuan-agent", daemon=True)
        _thread.start()
        return _snapshot()


def _run(task_id: str, params: dict) -> None:
    try:
        poll_interval = float(params.get("poll_interval") or 5.0)
        max_iterations = int(params.get("max_iterations") or 5)
        parent_dataset = params.get("parent_dataset") or _latest_parent_dataset()
        eval_dataset = params.get("eval_dataset_id") or _latest_eval_dataset()
        _set(parent_dataset=parent_dataset, eval_dataset_id=eval_dataset)
        _log(f"起始父数据集: {parent_dataset}")
        _log(f"固定评测集: {eval_dataset}")

        for idx in range(1, max_iterations + 1):
            _set(iteration=idx)
            _progress(f"iter {idx}: 检查标注任务")
            while True:
                ready = _ready_annotation_tasks()
                if ready:
                    break
                pending = _pending_annotation_tasks()
                if not pending:
                    _log(f"iter {idx}: 无已完成且未回灌的标注任务，结束")
                    _finish("converged", "没有待推进标注任务")
                    return
                names = ", ".join(f"{t['task_id']}({t['annotated']}/{t['count']})" for t in pending[:5])
                _progress(f"iter {idx}: 等待标注完成 {names}")
                time.sleep(poll_interval)

            task_info = ready[0]
            ann_task = task_info["task_id"]
            _log(f"iter {idx}: 回灌标注任务 {ann_task}")
            ds = annotation.commit(
                ann_task,
                params.get("dataset_name") or "agent-iter",
                parent_dataset,
                hard_weight=int(params.get("hard_weight") or 1),
                log=_log,
            )
            parent_dataset = ds["dataset_id"]
            _set(parent_dataset=parent_dataset, last_dataset_id=parent_dataset)

            train_params = {
                "dataset_id": parent_dataset,
                "name": params.get("model_name") or "agent-sft",
                "backend": params.get("train_backend") or None,
                "method": params.get("train_method") or None,
                "device": params.get("train_device") or None,
            }
            _log(f"iter {idx}: 提交训练 {parent_dataset}")
            train_task_id = runner.submit(Stage.TRAIN, train_params)
            _set(train_task_id=train_task_id)
            model = _wait_task(train_task_id, f"iter {idx} 训练", poll_interval)
            model_id = model["model_id"]
            _set(last_model_id=model_id)

            eval_params = {
                "model_id": model_id,
                "dataset_id": eval_dataset,
                "split": params.get("eval_split") or "valid",
                "use_judge": params.get("use_judge") or "auto",
                "baseline": params.get("baseline") or None,
                "max_samples": params.get("max_samples"),
            }
            _log(f"iter {idx}: 提交评测 {model_id}")
            eval_task_id = runner.submit(Stage.EVAL, eval_params)
            _set(eval_task_id=eval_task_id)
            ev = _wait_task(eval_task_id, f"iter {idx} 评测", poll_interval)
            gaps = ev.get("gaps") or []
            summary = {
                "iteration": idx,
                "annotation_task": ann_task,
                "dataset_id": parent_dataset,
                "model_id": model_id,
                "metrics": ev.get("metrics", {}),
                "gap_count": len(gaps),
                "train_task_id": train_task_id,
                "eval_task_id": eval_task_id,
            }
            _state.setdefault("summaries", []).append(summary)
            _set(last_eval=ev)
            _log(f"iter {idx}: 完成，gap={len(gaps)} metrics={ev.get('metrics', {})}")

            if not gaps:
                _finish("converged", f"iter {idx}: 无新 gap，迭代收敛")
                return

            created = _create_task_from_gaps(
                gaps,
                params.get("annotation_name") or "agent-gap",
                params.get("max_annotation_items"),
            )
            if not created.get("count"):
                _finish(
                    "blocked",
                    f"iter {idx}: 新 gap 均为 held-out 或无可标注项，未创建任务",
                )
                return
            _log(f"iter {idx}: 已创建新标注任务 {created['task_id']}，待标注 {created['count']} 条")

        _finish("limit_reached", f"达到 max_iterations={max_iterations}")
    except Exception as exc:  # noqa: BLE001
        _set(active=False, status="failed", message=str(exc), finished_at=time.time())
        store.update_task(task_id, status=TaskStatus.FAILED, error=str(exc), log=f"Agent 失败: {exc}")


def _finish(status_text: str, message: str) -> None:
    _set(active=False, status=status_text, message=message, progress="done", finished_at=time.time())
    _log(message)
    if _agent_task_id:
        store.update_task(_agent_task_id, status=TaskStatus.SUCCEEDED, progress="done", result=_snapshot(), log="Agent 自动迭代结束")
