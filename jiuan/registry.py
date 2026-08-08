"""轻量模型仓库/血缘登记（对标久安「模型仓库」最小实现）。

集中记录：数据集 -> base 模型 -> 训练参数 -> 产物路径 -> 评测报告，
便于查询与回滚。PoC 用 JSONL 索引 + 目录 artifact；
P2 可替换为 MLflow / HuggingFace Hub。
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from .common import DATA, DATASETS, MODELS, REGISTRY, ROOT

_INDEX = REGISTRY / "index.jsonl"
_DATASET_INDEX = REGISTRY / "datasets.jsonl"
_lock = threading.Lock()


def register_model(meta: dict) -> None:
    """训练完成后登记一条不可变血缘记录。"""
    record = {
        "model_id": meta.get("model_id"),
        "base": meta.get("base"),
        "dataset_id": meta.get("dataset_id"),
        "mode": meta.get("mode"),
        "method": meta.get("method"),
        "artifact_path": str(MODELS / meta.get("model_id", "")),
        "registered_at": time.time(),
    }
    _INDEX.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(_INDEX, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def attach_eval(model_id: str, report_id: str, metrics: dict) -> None:
    """把评测报告回挂到模型血缘（追加一条 eval 记录）。"""
    record = {
        "model_id": model_id,
        "type": "eval",
        "report_id": report_id,
        "metrics": metrics,
        "registered_at": time.time(),
    }
    with _lock, open(_INDEX, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_models() -> list[dict]:
    if not _INDEX.exists():
        return []
    out = []
    with open(_INDEX, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def selectable_models() -> list[dict]:
    """Return unique trained models enriched for resume-model selectors."""
    latest: dict[str, dict] = {}
    for record in list_models():
        model_id = record.get("model_id")
        if not model_id or record.get("type") == "eval":
            continue
        latest[model_id] = record
    rows = []
    for model_id, record in latest.items():
        meta_path = MODELS / model_id / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            meta = {}
        identity_status = meta.get("identity_validation") or "not_checked"
        validation_path = MODELS / model_id / "identity_validation.json"
        if validation_path.exists():
            try:
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                answers = "\n".join(str(item.get("answer") or "") for item in validation.get("results", []))
                if any(marker in answers for marker in ("来自阿里云", "来自OpenAI", "中国应急管理部")):
                    identity_status = "warning"
            except Exception:
                identity_status = "not_checked"
        rows.append({
            "model_id": model_id,
            "dataset_id": meta.get("dataset_id") or record.get("dataset_id"),
            "parent_model_id": meta.get("parent_model_id"),
            "backend": meta.get("backend") or record.get("mode"),
            "mode": meta.get("mode") or record.get("mode"),
            "identity_validation": identity_status,
            "domain_direction": meta.get("domain_direction") or "",
            "created_at": meta.get("created_at") or record.get("registered_at", 0),
        })
    rows.sort(key=lambda row: row.get("created_at", 0), reverse=True)
    return rows


def lineage(model_id: str) -> dict:
    """聚合某模型的血缘：训练记录 + meta + 关联评测。"""
    records = [r for r in list_models() if r.get("model_id") == model_id]
    meta_path = MODELS / model_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "model_id": model_id,
        "meta": meta,
        "train_record": next((r for r in records if r.get("type") != "eval"), None),
        "evals": [r for r in records if r.get("type") == "eval"],
    }


def register_dataset(
    dataset_id: str,
    parent: "str | None",
    sample_count: int,
    source: str = "",
    extra: "dict | None" = None,
) -> None:
    """登记数据集血缘：记录本数据集从哪个数据集扩展而来，用于迭代追溯。

    parent 为空表示 v1 根数据集；有 parent 则为增量迭代版本。
    """
    record = {
        "type": "dataset",
        "dataset_id": dataset_id,
        "parent_dataset": parent or None,
        "sample_count": int(sample_count),
        "source": source,
        "registered_at": time.time(),
    }
    if extra:
        record.update(extra)
    _DATASET_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(_DATASET_INDEX, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_datasets() -> list[dict]:
    """返回全部数据集血缘记录（每个 dataset_id 取最新一条）。"""
    if not _DATASET_INDEX.exists():
        return []
    latest: dict[str, dict] = {}
    with open(_DATASET_INDEX, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                did = r["dataset_id"]
                if did in latest:
                    latest[did].update(r)  # 合并(如 role=eval 标记)，不丢失原有字段
                else:
                    latest[did] = r
    return sorted(latest.values(), key=lambda r: r.get("registered_at", 0))


def mark_eval_dataset(dataset_id: str) -> None:
    """将数据集标记为固定 held-out 评测集(role=eval)，其 prompt 不得进入训练数据。"""
    _DATASET_INDEX.parent.mkdir(parents=True, exist_ok=True)
    record = {"type": "dataset", "dataset_id": dataset_id, "role": "eval",
              "registered_at": time.time()}
    with _lock, open(_DATASET_INDEX, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def eval_dataset_ids() -> list:
    """返回所有被标记为 held-out 评测集的 dataset_id。"""
    out = []
    for r in list_datasets():
        if r.get("role") == "eval":
            out.append(r["dataset_id"])
    return out


def held_out_prompts() -> set:
    """汇总所有 held-out 评测集里的 user prompt(去空格)，用于训练数据泄漏防护。"""
    from .common import DATASETS
    prompts: set = set()
    for ds in eval_dataset_ids():
        for split in ("train.jsonl", "valid.jsonl"):
            fp = DATASETS / ds / split
            if not fp.exists():
                continue
            with open(fp, "r", encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = __import__("json").loads(line)
                    except Exception:
                        continue
                    for m in rec.get("messages", []):
                        if m.get("role") == "user":
                            prompts.add((m.get("content") or "").strip())
    return prompts


def dataset_chain(dataset_id: str) -> list[str]:
    """沿 parent_dataset 上溯，返回从根到当前的血缘链（v1→…→当前）。"""
    by_id = {r["dataset_id"]: r for r in list_datasets()}
    chain: list[str] = []
    seen: set[str] = set()
    cur = dataset_id
    while cur and cur in by_id and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = by_id[cur].get("parent_dataset")
    chain.reverse()
    return chain


def _model_train_loss(model_id: str) -> "float | None":
    meta_path = MODELS / model_id / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta.get("train_loss")


def _is_impact_eval_report(report_id: str | None) -> bool:
    """Impact validation reports should not replace the main iteration metric."""
    if not report_id:
        return False
    from .common import REPORTS

    path = REPORTS / f"{report_id}.json"
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    dataset_id = str(report.get("dataset_id") or "")
    return dataset_id.startswith("impact-") or dataset_id.startswith("bio-impact-")


def iterations() -> list[dict]:
    """按数据集血缘分组，展示每一轮迭代的 train->eval 指标对比。

    每个血缘链(v1->v2->...)为一组，组内按版本顺序给出：
    数据量 / 新增数 / loss / ROUGE / BLEU / judge / 相对上一版提升。
    """
    datasets = [d for d in list_datasets() if d.get("role") != "eval"]  # 固定评测集不算迭代版本
    if not datasets:
        return []
    by_id = {d["dataset_id"]: d for d in datasets}
    records = list_models()

    # 数据集 -> 该集上训练的最新模型
    train_recs = [r for r in records if r.get("type") != "eval" and r.get("dataset_id")]
    latest_model_of: dict[str, str] = {}
    for r in sorted(train_recs, key=lambda x: x.get("registered_at", 0)):
        latest_model_of[r["dataset_id"]] = r.get("model_id")

    # 模型 -> 最新评测 metrics
    eval_recs = [r for r in records if r.get("type") == "eval" and not _is_impact_eval_report(r.get("report_id"))]
    latest_eval_of: dict[str, dict] = {}
    for r in sorted(eval_recs, key=lambda x: x.get("registered_at", 0)):
        if r.get("model_id"):
            latest_eval_of[r["model_id"]] = r.get("metrics", {})

    def _version_row(ds: dict) -> dict:
        did = ds["dataset_id"]
        model_id = latest_model_of.get(did)
        metrics = latest_eval_of.get(model_id, {}) if model_id else {}
        return {
            "dataset_id": did,
            "parent_dataset": ds.get("parent_dataset"),
            "sample_count": ds.get("sample_count"),
            "added_count": ds.get("added_count"),
            "registered_at": ds.get("registered_at"),
            "model_id": model_id,
            "train_loss": _model_train_loss(model_id) if model_id else None,
            "rouge_l_f": metrics.get("rouge_l_f"),
            "bleu_1": metrics.get("bleu_1"),
            "bleu_2": metrics.get("bleu_2"),
            "judge_avg": metrics.get("judge_avg"),
        }

    # 找出根节点(无 parent 或 parent 不在当前集合)，向下展开子链
    children: dict[str, list[dict]] = {}
    roots: list[dict] = []
    for d in datasets:
        p = d.get("parent_dataset")
        if p and p in by_id:
            children.setdefault(p, []).append(d)
        else:
            roots.append(d)

    def _paths_from(cur: dict, prefix: list[dict], seen: set[str]) -> list[list[dict]]:
        did = cur["dataset_id"]
        if did in seen:
            return [prefix]
        path = [*prefix, cur]
        kids = sorted(children.get(did, []), key=lambda d: d.get("registered_at", 0))
        if not kids:
            return [path]
        paths: list[list[dict]] = []
        for kid in kids:
            paths.extend(_paths_from(kid, path, seen | {did}))
        return paths or [path]

    groups: list[dict] = []
    for root in sorted(roots, key=lambda d: d.get("registered_at", 0)):
        # 多个子分支都保留为独立链路，避免新迭代分支在看板中隐身。
        for path in _paths_from(root, [], set()):
            chain = [_version_row(ds) for ds in path]
            # 标版本号 + 算相对上一版提升
            prev = None
            for i, row in enumerate(chain, 1):
                row["version"] = f"v{i}"
                if prev and prev.get("rouge_l_f") is not None and row.get("rouge_l_f") is not None:
                    base = prev["rouge_l_f"] or 0.0
                    row["rouge_delta"] = round(row["rouge_l_f"] - base, 4)
                    row["rouge_delta_pct"] = round((row["rouge_delta"] / base * 100), 1) if base else None
                else:
                    row["rouge_delta"] = None
                    row["rouge_delta_pct"] = None
                prev = row
            groups.append({
                "root": root["dataset_id"],
                "leaf": path[-1]["dataset_id"] if path else root["dataset_id"],
                "versions": chain,
            })
    return groups


def dataset_version(dataset_id: str) -> "int | None":
    """返回数据集在其血缘链中的迭代序号(1-based，即 v1/v2/…)。

    无血缘登记时返回 None。
    """
    chain = dataset_chain(dataset_id)
    if dataset_id in chain:
        return chain.index(dataset_id) + 1
    return None


def model_dataset(model_id: str) -> "str | None":
    """查模型训练所用数据集 id。"""
    for r in list_models():
        if r.get("type") != "eval" and r.get("model_id") == model_id:
            return r.get("dataset_id")
    return None


def _first_dataset_system_prompt(dataset_id: str) -> str:
    for split in ("train.jsonl", "valid.jsonl"):
        path = DATASETS / dataset_id / split
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                for message in row.get("messages", []):
                    if message.get("role") == "system" and message.get("content"):
                        return str(message["content"]).strip()
    return ""


def _agent_snapshot_for_model(model_id: str) -> dict:
    """Find the newest persisted Agent snapshot that produced model_id."""
    candidates: list[tuple[int, float, dict]] = []
    steps_dir = DATA / "agent_steps"
    if not steps_dir.exists():
        return {}
    for path in steps_dir.glob("*.json"):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary_models = {str(item.get("model_id") or "") for item in snapshot.get("summaries", [])}
        if model_id not in summary_models and model_id not in {
            str(snapshot.get("final_model_id") or ""),
            str(snapshot.get("last_model_id") or ""),
        }:
            continue
        quality = 0
        if model_id in summary_models:
            quality += 10
        quality += sum(bool(snapshot.get(key)) for key in ("eval_dataset_id", "breadth_source", "kb_collection", "source"))
        candidates.append((quality, path.stat().st_mtime, snapshot))
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else {}


def _name_without_version(value: str, prefix: str) -> str:
    text = value or ""
    if prefix:
        text = re.sub(rf"^{re.escape(prefix)}\d+-", "", text)
    text = re.sub(r"-\d{8}-\d{6}$", "", text)
    return text.strip("-")


def model_resume_profile(model_id: str) -> dict:
    """Resolve the complete project context needed to safely continue a model."""
    meta_path = MODELS / model_id / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"模型产物不存在：{model_id}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"模型元数据无法读取：{model_id}") from exc
    dataset_id = str(meta.get("dataset_id") or model_dataset(model_id) or "")
    if not dataset_id or not (DATASETS / dataset_id).exists():
        raise ValueError(f"模型没有可恢复的训练数据集：{model_id}")

    snapshot = _agent_snapshot_for_model(model_id)
    saved_params = dict(snapshot.get("params") or {})
    system_prompt = str(meta.get("system_prompt") or _first_dataset_system_prompt(dataset_id)).strip()
    direction = str(meta.get("domain_direction") or snapshot.get("direction") or saved_params.get("direction") or "").strip()
    if not direction and "生物" in (system_prompt + dataset_id + model_id):
        direction = "专业生物学知识"
    if not direction:
        match = re.search(r"你是(.+?)助手", system_prompt)
        direction = match.group(1).strip("《》“”") if match else _name_without_version(dataset_id, "")

    iteration_prefix = str(saved_params.get("iteration_prefix") or "").strip()
    if not iteration_prefix:
        match = re.match(r"(.+?v)\d+-", model_id)
        iteration_prefix = match.group(1) if match else "custom-v"
    dataset_name = str(saved_params.get("dataset_name") or _name_without_version(dataset_id, iteration_prefix) or "领域知识")
    model_name = str(saved_params.get("model_name") or _name_without_version(model_id, iteration_prefix) or "领域专家")
    source_name = str(saved_params.get("source_name") or dataset_name)
    scenario_name = str(saved_params.get("scenario_name") or re.sub(r"(学)?知识$", "", source_name) or direction)

    eval_dataset_id = str(snapshot.get("eval_dataset_id") or "")
    breadth_source = str(snapshot.get("breadth_source") or "")
    kb_collection = str(snapshot.get("kb_collection") or saved_params.get("collection") or meta.get("knowledge_collection") or "")
    if not breadth_source and eval_dataset_id:
        eval_record = next((r for r in list_datasets() if r.get("dataset_id") == eval_dataset_id), {})
        breadth_source = str(eval_record.get("source") or "")
    breadth_path = Path(breadth_source) if breadth_source else None
    if breadth_path and not breadth_path.is_absolute():
        breadth_path = ROOT / breadth_path
    missing = []
    if not eval_dataset_id or not (DATASETS / eval_dataset_id).exists():
        missing.append("held-out 验证集")
    if not breadth_path or not breadth_path.exists():
        missing.append("广度验证源文件")
    if not kb_collection:
        missing.append("知识库 collection")
    if missing:
        raise ValueError(f"模型恢复档案不完整（{', '.join(missing)}），已拒绝跨项目续训：{model_id}")

    source = str(snapshot.get("source") or "")
    output_root = str(saved_params.get("output_root") or "")
    if not output_root and source:
        source_path = Path(source)
        output_root = str(source_path.parent).replace("\\", "/")
    profile = {
        **saved_params,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "direction": direction,
        "system_prompt": system_prompt,
        "scenario_name": scenario_name,
        "source_name": source_name,
        "iteration_prefix": iteration_prefix,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "base_model_path": meta.get("base") or saved_params.get("base_model_path") or "",
        "output_root": output_root or "data/custom_agents",
        "eval_dataset_id": eval_dataset_id,
        "breadth_source": breadth_source.replace("\\", "/"),
        "kb_collection": kb_collection,
        "collection": kb_collection,
        "source": source,
        "summaries": [item for item in snapshot.get("summaries", []) if item.get("model_id") == model_id or int(item.get("iteration") or 0) <= (dataset_version(dataset_id) or 0)],
        "project_steps": list(snapshot.get("steps") or []),
        "profile_source": "agent_steps+model_meta+dataset",
    }
    return profile
