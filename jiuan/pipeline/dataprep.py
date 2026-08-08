"""标：数据准备 —— 清洗 jsonl 指令数据并登记为 SFT 数据集。

对标久安「模型标注系统 + 数据预处理」的最小实现：
去重、去空、字段校验、统一为 {messages:[...]} chat 格式，并切分 train/valid。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

from ..common import DATASETS, ROOT
from .. import registry

SYSTEM_PROMPT = "你是应急管理领域的专业助手，回答需准确、简洁、可执行。"


def _to_chat(rec: dict) -> dict:
    user = rec["instruction"].strip()
    if rec.get("input"):
        user += "\n" + rec["input"].strip()
    system_prompt = (rec.get("system_prompt") or rec.get("system") or SYSTEM_PROMPT).strip()
    chat = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": rec["output"].strip()},
        ]
    }
    try:
        w = int(rec.get("weight", 1) or 1)
    except (TypeError, ValueError):
        w = 1
    if w > 1:
        chat["_weight"] = min(w, 10)  # 硬样本加权：训练集内复制次数(上限10)
    return chat


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_dataset_rows(dataset_id: str) -> list[dict]:
    ddir = DATASETS / dataset_id
    if not (ddir / "train.jsonl").exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_id}")
    rows: list[dict] = []
    for split in ("train.jsonl", "valid.jsonl"):
        p = ddir / split
        if p.exists():
            rows.extend(_read_jsonl(p))
    return rows


def run(params: dict, log: Callable[[str], None]) -> dict:
    name = params.get("name", "dataset")
    valid_ratio = float(params.get("valid_ratio", 0.2))
    seed = int(params.get("seed", 42))
    parent = params.get("parent") or None
    source_dataset = params.get("source_dataset") or None

    raw: list[dict] = []
    source_label = ""
    if source_dataset:
        raw = _read_dataset_rows(source_dataset)
        source_label = f"dataset:{source_dataset}"
        log(f"读取 source 数据集 {source_dataset}：{len(raw)} 条(train+valid)")
    else:
        source = Path(params.get("source") or "data/samples.jsonl")
        if not source.is_absolute():
            source = ROOT / source
        source_label = str(params.get("source") or "data/samples.jsonl")
        log(f"读取源文件 {source}")
        raw = _read_jsonl(source)
        log(f"原始样本 {len(raw)} 条")

    # 增量迭代：若指定 parent，把父数据集(train+valid)合并进来一起去重
    inherited: list[dict] = []
    if parent:
        inherited = _read_dataset_rows(parent)
        log(f"继承父数据集 {parent}：{len(inherited)} 条，与新增 {len(raw)} 条合并")
    log(f"开始清洗（共 {len(inherited) + len(raw)} 条待处理）")

    seen: set[str] = set()
    cleaned: list[dict] = []
    dropped = 0

    def _chat_key(chat: dict) -> "str | None":
        """从 chat 格式提取去重 key(user||assistant)。"""
        msgs = chat.get("messages", [])
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if not user.strip() or not asst.strip():
            return None
        return user.strip() + "||" + asst.strip()

    # 先合并继承自父数据集的样本(已是 chat 格式)
    for chat in inherited:
        key = _chat_key(chat)
        if not key or key in seen:
            dropped += 1
            continue
        seen.add(key)
        cleaned.append(chat)

    # 再处理新增 source。支持原始 instruction/output，也支持已有数据集里的 chat 格式。
    for rec in raw:
        if rec.get("messages"):
            chat = rec
        elif rec.get("instruction") and rec.get("output"):
            chat = _to_chat(rec)
        else:
            dropped += 1
            continue
        key = _chat_key(chat)
        if not key or key in seen:
            dropped += 1
            continue
        seen.add(key)
        cleaned.append(chat)

    # 切分 train/valid：评测应在 valid 上进行，避免在训练集上"作弊"
    rng = random.Random(seed)
    rng.shuffle(cleaned)
    n_valid = int(len(cleaned) * valid_ratio)
    if len(cleaned) >= 2:
        n_valid = max(1, n_valid)
    valid_rows = cleaned[:n_valid]
    train_rows_raw = cleaned[n_valid:]

    def _strip(chat: dict) -> dict:
        return {"messages": chat["messages"]}

    # 硬样本加权：训练集内按 _weight 复制；验证集保持 1x(评测口径干净)
    weighted = 0
    train_rows = []
    for chat in train_rows_raw:
        w = int(chat.get("_weight", 1))
        train_rows.append(_strip(chat))
        for _ in range(max(0, w - 1)):
            train_rows.append(_strip(chat))
            weighted += 1
    valid_rows = [_strip(c) for c in valid_rows]

    dataset_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = DATASETS / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "valid.jsonl", valid_rows)

    # 登记数据集血缘（记录 parent，支撑迭代追溯）
    added = len(cleaned) - len(inherited) if parent else len(cleaned)
    registry.register_dataset(
        dataset_id,
        parent,
        len(cleaned),
        source=source_label,
        extra={
            "train_count": len(train_rows),
            "valid_count": len(valid_rows),
            "added_count": max(0, added),
            "source_dataset": source_dataset,
        },
    )
    log(
        f"清洗完成：保留 {len(cleaned)} 条(丢弃 {dropped})，"
        f"切分 train {len(train_rows)} / valid {len(valid_rows)}"
        + (f"，硬样本加权复制 {weighted} 条" if weighted else "")
        + (f"，继承自 {parent} (新增 {max(0, added)} 条)" if parent else "")
    )
    return {
        "dataset_id": dataset_id,
        "parent_dataset": parent,
        "source": source_label,
        "source_dataset": source_dataset,
        "train_path": str(out_dir / "train.jsonl"),
        "valid_path": str(out_dir / "valid.jsonl"),
        "count": len(cleaned),
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "added_count": max(0, added),
        "dropped": dropped,
    }
