"""One-click biology package: material/KB completion -> E2E checks -> report."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .. import registry, store
from ..common import DATA, DATASETS, MODELS, REPORTS, ROOT
from ..pipeline import rag_backend
from ..schemas import Stage, TaskStatus
from . import biology_agent, runner

_lock = threading.RLock()
_thread: threading.Thread | None = None
_agent_task_id: str | None = None
_state: dict[str, Any] = {
    "active": False,
    "status": "idle",
    "message": "未启动",
    "logs": [],
    "steps": [],
    "checks": [],
    "problems": [],
}


def status() -> dict:
    with _lock:
        if not _state.get("active") and not _state.get("task_id"):
            latest = _latest_persisted_state()
            if latest:
                return latest
        out = dict(_state)
        for key in ("logs", "steps", "checks", "problems"):
            out[key] = list(_state.get(key, []))
        return out


def _latest_persisted_state() -> dict | None:
    try:
        for task in store.list_tasks(Stage.AGENT):
            if task.params.get("mode") == "oneclick" and task.result:
                return task.result
    except Exception:
        return None
    return None


def _set(**kw: Any) -> None:
    with _lock:
        _state.update(kw)


def _log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    with _lock:
        logs = _state.setdefault("logs", [])
        logs.append(line)
        del logs[:-180]
        task_id = _agent_task_id
    if task_id:
        store.update_task(task_id, log=message)


def _progress(text: str) -> None:
    _set(progress=text, message=text)
    if _agent_task_id:
        store.update_task(_agent_task_id, progress=text)


def _rel(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _add_step(name: str, status_text: str, summary: str, artifacts: dict | None = None) -> None:
    step = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": name,
        "status": status_text,
        "summary": summary,
        "artifacts": artifacts or {},
    }
    with _lock:
        _state.setdefault("steps", []).append(step)


def _add_check(name: str, ok: bool, detail: str, artifact: str = "") -> None:
    item = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "artifact": artifact,
    }
    with _lock:
        _state.setdefault("checks", []).append(item)


def _add_problem(kind: str, detail: str, suggestion: str = "") -> None:
    item = {"kind": kind, "detail": detail, "suggestion": suggestion}
    with _lock:
        _state.setdefault("problems", []).append(item)


def _read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                total += 1
    return total


def _latest_dataset() -> dict | None:
    rows = [
        r for r in registry.list_datasets()
        if r.get("role") != "eval" and "生物学知识" in str(r.get("dataset_id", ""))
    ]
    if rows:
        rows.sort(key=lambda r: r.get("registered_at", 0), reverse=True)
        return rows[0]
    dirs = [p for p in DATASETS.glob("*生物学知识*") if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {"dataset_id": dirs[0].name, "registered_at": dirs[0].stat().st_mtime}


def _latest_model() -> dict | None:
    rows = [
        r for r in registry.list_models()
        if r.get("type") != "eval" and "生物专家" in str(r.get("model_id", ""))
    ]
    if rows:
        rows.sort(key=lambda r: r.get("registered_at", 0), reverse=True)
        row = rows[0]
        model_id = row.get("model_id")
        meta = _model_meta(model_id) if model_id else {}
        return {**row, "meta": meta}
    dirs = [p for p in MODELS.glob("*生物专家*") if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {"model_id": dirs[0].name, "registered_at": dirs[0].stat().st_mtime, "meta": _model_meta(dirs[0].name)}


def _model_meta(model_id: str | None) -> dict:
    if not model_id:
        return {}
    meta_path = MODELS / model_id / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_report(prefix: str, model_id: str | None) -> dict | None:
    rows: list[dict] = []
    for path in REPORTS.glob(f"{prefix}-*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if model_id and report.get("model_id") != model_id:
            continue
        report["report_id"] = path.stem
        report["report_path"] = _rel(path)
        rows.append(report)
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows[0]


def _latest_eval_dataset_id() -> str:
    ids = registry.eval_dataset_ids()
    if ids:
        return ids[-1]
    dirs = [p for p in DATASETS.glob("*广度验证*") if p.is_dir()]
    if not dirs:
        return ""
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0].name


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
            raise RuntimeError(f"{label} 子任务失败 {task_id}: {task.error[:800]}")
        time.sleep(poll_interval)


def _run_biology_if_needed(params: dict, current_model: dict | None, current_dataset: dict | None) -> dict | None:
    force = bool(params.get("force_biology_run"))
    run_if_missing = bool(params.get("run_biology_if_missing", True))
    if current_model and current_dataset and not force:
        _add_step(
            "自动训练 Agent",
            "reused",
            f"已发现最新生物模型 {current_model.get('model_id')} 和数据集 {current_dataset.get('dataset_id')}，本次复用既有三轮产物。",
            {"model_id": current_model.get("model_id"), "dataset_id": current_dataset.get("dataset_id")},
        )
        return None
    if not force and not run_if_missing:
        _add_step("自动训练 Agent", "skipped", "未发现完整产物，但参数要求不自动训练。")
        return None

    _progress("触发生物专家三轮自动训练")
    body = {
        "iteration_prefix": params.get("iteration_prefix") or "bio-v",
        "source_name": params.get("source_name") or "生物知识",
        "max_iterations": 3,
        "train_backend": params.get("train_backend") or "hf",
        "train_device": params.get("train_device") or "cpu",
        "train_epochs": int(params.get("train_epochs") or 1),
        "train_max_seq_len": int(params.get("train_max_seq_len") or 512),
        "eval_max_samples": int(params.get("eval_max_samples") or 11),
        "breadth_max_samples": int(params.get("breadth_max_samples") or 11),
        "use_judge": params.get("use_judge") or "auto",
        "poll_interval": float(params.get("poll_interval") or 5.0),
    }
    bio = biology_agent.start(body)
    _add_step(
        "自动训练 Agent",
        "started",
        "已调用 /agent/biology/start 进入标注回灌、训练、推理、评测、广度分析三轮闭环。",
        {"biology_task_id": bio.get("task_id", "")},
    )
    poll = float(params.get("poll_interval") or 5.0)
    while bio.get("active"):
        _progress("等待生物三轮 Agent 完成")
        _log(bio.get("message") or "生物三轮 Agent 运行中")
        time.sleep(poll)
        bio = biology_agent.status()
    if bio.get("status") != "succeeded":
        raise RuntimeError(f"生物三轮 Agent 未成功完成: {bio.get('message')}")
    _add_step(
        "自动训练 Agent",
        "succeeded",
        f"生物三轮完成，最终模型 {bio.get('final_model_id')}，最终数据集 {bio.get('final_dataset_id')}。",
        {"model_id": bio.get("final_model_id", ""), "dataset_id": bio.get("final_dataset_id", "")},
    )
    return bio


def _build_analysis(eval_report: dict | None, breadth_report: dict | None, rag_stats: dict, rag_hits: list[dict]) -> None:
    metrics = (eval_report or {}).get("metrics") or {}
    judge_avg = metrics.get("judge_avg")
    rouge = metrics.get("rouge_l_f")
    if judge_avg is not None and judge_avg < 3.5:
        _add_problem(
            "评测质量",
            f"最新 judge_avg={judge_avg}，仍低于 3.5，说明专业回答完整性还不稳定。",
            "下一轮优先扩充低分 prompt 的同领域变体，并提高机制/证据/应用三段式答案比例。",
        )
    if rouge is not None and rouge < 0.3:
        _add_problem(
            "字面指标",
            f"最新 ROUGE-L={rouge}，与参考答案的覆盖重合仍偏低。",
            "继续补充标准答案风格样本；评测时保留 LLM-as-Judge，避免只按字面重合判断。",
        )
    issues = (breadth_report or {}).get("issues") or []
    if issues:
        cats = "、".join(str(x.get("category", "")) for x in issues[:8] if x.get("category"))
        _add_problem(
            "验证集广度",
            f"最新广度分析仍有 {len(issues)} 个薄弱类别" + (f"：{cats}" if cats else "。"),
            "让 judge 继续按类别扩展 held-out 验证集，并把低分类别转成下一轮标注任务。",
        )
    if rag_stats.get("total_chunks", 0) <= 0:
        _add_problem("知识库", "biology 知识库为空。", "重新运行一键保通或手动调用 /rag/ingest 入库。")
    if not rag_hits:
        _add_problem("RAG 检索", "质子梯度样例没有检索到任何知识块。", "检查 KB 文件是否存在、collection 是否为 biology。")
    elif max(float(x.get("score") or 0) for x in rag_hits) < rag_backend.HIT_THRESHOLD:
        _add_problem(
            "RAG 检索",
            f"质子梯度样例最高命中分低于阈值 {rag_backend.HIT_THRESHOLD}。",
            "当前 TF-IDF 对中文语义仍偏浅，可切换到 bge/sentence-transformers 语义嵌入。",
        )


def _save_report(run_id: str) -> dict:
    step_dir = DATA / "agent_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    snap = status()
    json_path = step_dir / f"oneclick-{run_id}.json"
    md_path = step_dir / f"oneclick-{run_id}.md"
    json_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = snap.get("checks", [])
    problems = snap.get("problems", [])
    lines = [
        f"# 一键保通打包报告 {run_id}",
        "",
        f"- 状态：{snap.get('status')}",
        f"- 结论：{'通过' if snap.get('overall_ok') else '完成但存在问题'}",
        f"- 最终模型：{snap.get('final_model_id', '')}",
        f"- 最终数据集：{snap.get('final_dataset_id', '')}",
        f"- 知识库 collection：{snap.get('kb_collection', '')}",
        f"- 一键命令：`powershell -ExecutionPolicy Bypass -File scripts/oneclick_biology.ps1`",
        "",
        "## 自动步骤",
    ]
    for i, step in enumerate(snap.get("steps", []), 1):
        lines.append(f"{i}. {step.get('step')} [{step.get('status')}]：{step.get('summary')}")
        for k, v in (step.get("artifacts") or {}).items():
            lines.append(f"   - {k}: {v}")
    lines.extend(["", "## 保通检查"])
    for c in checks:
        mark = "PASS" if c.get("ok") else "FAIL"
        suffix = f"（{c.get('artifact')}）" if c.get("artifact") else ""
        lines.append(f"- {mark} {c.get('name')}：{c.get('detail')}{suffix}")
    lines.extend(["", "## 问题分析"])
    if problems:
        for p in problems:
            lines.append(f"- {p.get('kind')}：{p.get('detail')} 建议：{p.get('suggestion')}")
    else:
        lines.append("- 未发现阻断性问题。")
    pkg = snap.get("package") or {}
    lines.extend(["", "## 关键产物"])
    for k, v in pkg.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            lines.append(f"- {k}: {v}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": _rel(json_path), "markdown": _rel(md_path)}


def start(params: dict) -> dict:
    global _thread, _agent_task_id
    with _lock:
        if _thread and _thread.is_alive():
            return status()
        task = store.create_task(Stage.AGENT, {"mode": "oneclick", **params})
        _agent_task_id = task.id
        _state.clear()
        _state.update({
            "active": True,
            "status": "running",
            "message": "一键保通打包启动中",
            "progress": "starting",
            "task_id": task.id,
            "params": params,
            "started_at": time.time(),
            "logs": [],
            "steps": [],
            "checks": [],
            "problems": [],
            "kb_collection": params.get("collection") or "biology",
        })
        store.update_task(task.id, status=TaskStatus.RUNNING, log="一键保通打包流程启动")
        _thread = threading.Thread(target=_run, args=(task.id, params), name="jiuan-oneclick-agent", daemon=True)
        _thread.start()
        return status()


def _run(task_id: str, params: dict) -> None:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    collection = params.get("collection") or "biology"
    source_name = params.get("source_name") or "生物知识"
    poll = float(params.get("poll_interval") or 5.0)
    try:
        _progress("生成并补全生物场景材料")
        mat = biology_agent._materials(source_name, run_id)  # Reuse the canonical 110-row biology material builder.
        source_count = _read_jsonl_count(mat["source_path"])
        breadth_count = _read_jsonl_count(mat["eval_path"])
        kb_chars = len(mat["kb_path"].read_text(encoding="utf-8")) if mat["kb_path"].exists() else 0
        _set(source=_rel(mat["source_path"]), breadth_source=_rel(mat["eval_path"]), kb_source=_rel(mat["kb_path"]))
        _add_check("标注数据源", source_count == 110, f"{source_count}/110 条，source={source_name}", _rel(mat["source_path"]))
        _add_check("广度验证源", breadth_count >= 30, f"{breadth_count} 条 held-out 广度样本", _rel(mat["eval_path"]))
        _add_check("知识库 Markdown", kb_chars > 5000, f"{kb_chars} 字符", _rel(mat["kb_path"]))
        _add_step(
            "生成/补全场景材料",
            "succeeded",
            f"生成 {source_count} 条标注源、{breadth_count} 条广度验证源，并重写配套知识库 Markdown。",
            {"source": _rel(mat["source_path"]), "breadth_source": _rel(mat["eval_path"]), "kb": _rel(mat["kb_path"])},
        )

        _progress("补全并入库 biology 知识库")
        ing = rag_backend.ingest_document(_rel(mat["kb_path"]), chunk_size=int(params.get("kb_chunk_size") or 700), collection=collection, log=_log)
        stats = rag_backend.stats(collection)
        _add_check("RAG 知识库入库", stats.get("total_chunks", 0) > 0, f"{stats.get('total_chunks', 0)} 块，embedder={stats.get('embedder')}", collection)
        _add_step(
            "补全知识库",
            "succeeded",
            f"collection={collection} 已入库 {ing.get('new_chunks')} 块，总块数 {stats.get('total_chunks')}。",
            {"collection": collection, "source": ing.get("source"), "total_chunks": stats.get("total_chunks")},
        )

        _progress("RAG 检索保通")
        rag_prompt = params.get("rag_probe") or "线粒体氧化磷酸化为什么需要内膜质子梯度？"
        hits = rag_backend.retrieve(rag_prompt, top_k=3, collection=collection)
        max_score = max((float(x.get("score") or 0) for x in hits), default=0.0)
        _add_check("RAG 检索样例", bool(hits), f"命中 {len(hits)} 条，最高分 {round(max_score, 4)}", collection)
        _add_step(
            "RAG 检索保通",
            "succeeded" if hits else "warning",
            f"检索问题“{rag_prompt}”，命中 {len(hits)} 条，最高分 {round(max_score, 4)}。",
            {"top_source": hits[0].get("source") if hits else "", "top_score": max_score},
        )

        model = _latest_model()
        dataset = _latest_dataset()
        _run_biology_if_needed(params, model, dataset)
        model = _latest_model()
        dataset = _latest_dataset()
        model_id = (model or {}).get("model_id")
        dataset_id = (dataset or {}).get("dataset_id")
        eval_dataset_id = _latest_eval_dataset_id()
        _set(final_model_id=model_id or "", final_dataset_id=dataset_id or "", eval_dataset_id=eval_dataset_id)
        _add_check("最终训练数据集", bool(dataset_id and (DATASETS / dataset_id).exists()), dataset_id or "未找到")
        _add_check("最终模型产物", bool(model_id and (MODELS / model_id).exists()), model_id or "未找到")
        if not model_id:
            _add_problem("模型产物", "未找到“生物专家”模型。", "勾选“缺失时自动训练”或“强制重跑三轮”后重新执行一键保通。")
        if not dataset_id:
            _add_problem("数据集产物", "未找到“生物学知识”训练数据集。", "重新执行生物三轮 Agent，确保标注回灌成功。")

        eval_report = _latest_report("eval", model_id)
        breadth_report = _latest_report("breadth", model_id)
        metrics = (eval_report or {}).get("metrics") or {}
        _set(
            eval_report_id=(eval_report or {}).get("report_id", ""),
            breadth_report_id=(breadth_report or {}).get("report_id", ""),
            metrics=metrics,
        )
        _add_check(
            "固定评测报告",
            bool(eval_report and metrics),
            f"report={((eval_report or {}).get('report_id') or '未找到')}，judge={metrics.get('judge_avg', '—')}，ROUGE-L={metrics.get('rouge_l_f', '—')}",
            (eval_report or {}).get("report_path", ""),
        )
        breadth_issues = (breadth_report or {}).get("issues") or []
        _add_check(
            "广度 Judge 报告",
            bool(breadth_report),
            f"report={((breadth_report or {}).get('report_id') or '未找到')}，薄弱类别={len(breadth_issues)}",
            (breadth_report or {}).get("report_path", ""),
        )
        _add_step(
            "评测与广度报告汇总",
            "succeeded" if eval_report and breadth_report else "warning",
            f"最新评测 judge={metrics.get('judge_avg', '—')}，ROUGE-L={metrics.get('rouge_l_f', '—')}；广度薄弱类别 {len(breadth_issues)} 个。",
            {"eval_report": (eval_report or {}).get("report_id"), "breadth_report": (breadth_report or {}).get("report_id")},
        )

        infer_result = None
        if model_id and bool(params.get("run_inference_probe", True)):
            _progress("模型推理抽检保通")
            try:
                infer_task_id = runner.submit(
                    Stage.INFER,
                    {
                        "model_id": model_id,
                        "prompt": params.get("infer_probe") or biology_agent.BIO_PROBE,
                        "backend": params.get("infer_backend"),
                        "system_prompt": biology_agent.BIO_SYSTEM,
                        "max_new_tokens": int(params.get("infer_max_new_tokens") or 80),
                        "do_sample": False,
                    },
                )
                infer_result = _wait_task(infer_task_id, "一键推理抽检", poll)
                answer = infer_result.get("answer") or ""
                _add_check("模型推理抽检", bool(answer), f"answer_len={len(answer)}，task={infer_task_id}")
                _add_step(
                    "模型推理抽检",
                    "succeeded" if answer else "warning",
                    f"完成一次生物学问题推理，回答长度 {len(answer)}。",
                    {"infer_task_id": infer_task_id, "answer_preview": answer[:160]},
                )
            except Exception as exc:  # noqa: BLE001
                _add_check("模型推理抽检", False, str(exc)[:200])
                _add_problem(
                    "模型推理",
                    f"最终模型推理抽检未通过：{str(exc)[:180]}",
                    "先查看 ③推理任务日志；若是 CPU 资源或模型加载问题，可重启服务后只跑推理抽检。",
                )
                _add_step("模型推理抽检", "warning", f"推理抽检失败：{str(exc)[:180]}")
        elif model_id:
            _add_step("模型推理抽检", "skipped", "本次参数关闭推理抽检；已有生物三轮 Agent 历史中包含每轮推理任务。")

        _build_analysis(eval_report, breadth_report, stats, hits)
        problems = status().get("problems", [])
        checks = status().get("checks", [])
        overall_ok = bool(checks) and all(c.get("ok") for c in checks) and not problems
        package = {
            "source": _rel(mat["source_path"]),
            "breadth_source": _rel(mat["eval_path"]),
            "kb_markdown": _rel(mat["kb_path"]),
            "kb_collection": collection,
            "kb_chunks": stats.get("total_chunks"),
            "final_dataset_id": dataset_id,
            "final_model_id": model_id,
            "eval_dataset_id": eval_dataset_id,
            "eval_report_id": (eval_report or {}).get("report_id"),
            "breadth_report_id": (breadth_report or {}).get("report_id"),
            "judge_avg": metrics.get("judge_avg"),
            "rouge_l_f": metrics.get("rouge_l_f"),
            "breadth_issue_count": len(breadth_issues),
            "rag_probe_score": round(max_score, 4),
            "inference_mode": (infer_result or {}).get("mode"),
            "one_click_command": "powershell -ExecutionPolicy Bypass -File scripts/oneclick_biology.ps1",
        }
        _set(package=package, overall_ok=overall_ok)
        message = "一键保通完成" if overall_ok else f"一键保通完成，发现 {len(problems)} 个需要继续迭代的问题"
        _finish("succeeded", message, run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        _add_step("异常终止", "failed", str(exc))
        _set(active=False, status="failed", message=str(exc), finished_at=time.time(), progress="done")
        paths = _save_report(run_id)
        _set(package_report_json=paths.get("json"), package_report_md=paths.get("markdown"))
        store.update_task(task_id, status=TaskStatus.FAILED, error=str(exc), result=status(), log=f"一键保通失败: {exc}")


def _finish(status_text: str, message: str, run_id: str | None = None, paths: dict | None = None) -> None:
    _set(active=False, status=status_text, message=message, progress="done", finished_at=time.time())
    if run_id:
        paths = _save_report(run_id)
    if paths:
        _set(package_report_json=paths.get("json"), package_report_md=paths.get("markdown"))
    _log(message)
    if _agent_task_id:
        store.update_task(_agent_task_id, status=TaskStatus.SUCCEEDED, progress="done", result=status(), log="一键保通打包流程结束")
