"""RAG 知识库：应急法规/预案文档检索增强生成。

定位：弥补 SFT “硬记”的不足——最新法规条文用检索注入上下文更实用。

设计（适配当前 CPU/离线环境）：
- 默认嵌入器 = TF-IDF(scikit-learn)，零下载纯离线，开箱即用；
- 向量库 = numpy 余弦相似度，持久化到 data/vector_db；
- 可插拔升级：装了 sentence-transformers+bge 用环境变量 JIUAN_RAG_EMBED=bge 切语义嵌入。

流程：上传文档→分段(chunk)→嵌入(embed)→入库；提问→检索→注入 prompt→模型生成。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..common import DATA, ROOT
from . import infer

VECTOR_DIR = DATA / "vector_db"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
_STORE_PATH = VECTOR_DIR / "store.pkl"
_GAPS_PATH = VECTOR_DIR / "rag_gaps.jsonl"          # 检索未命中缺口台账
USER_KB = DATA / "_kb_user.txt"                       # 人工一键补充的知识落盘文件
DEFAULT_COLLECTION = "default"
# 检索命中阈值：最高余弦相似度低于此值视为"知识库缺相关内容"
HIT_THRESHOLD = float(os.environ.get("JIUAN_RAG_HIT_THRESHOLD", "0.15"))

SYSTEM_PROMPT = "你是应急管理领域的专业助手，优先依据检索到的知识回答，如知识不足再结合常识。"

_EMBEDDER = None


def _collection_name(collection: str | None = None) -> str:
    """Normalize a knowledge-base collection name into a safe local namespace."""
    raw = (collection or DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
    safe = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", raw).strip("._-")
    return (safe or DEFAULT_COLLECTION)[:80]


def _collection_dir(collection: str | None = None) -> Path:
    name = _collection_name(collection)
    if name == DEFAULT_COLLECTION:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        return VECTOR_DIR
    path = VECTOR_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _store_path(collection: str | None = None) -> Path:
    name = _collection_name(collection)
    return _STORE_PATH if name == DEFAULT_COLLECTION else _collection_dir(name) / "store.pkl"


def _gaps_path(collection: str | None = None) -> Path:
    name = _collection_name(collection)
    return _GAPS_PATH if name == DEFAULT_COLLECTION else _collection_dir(name) / "rag_gaps.jsonl"


def _user_kb_path(collection: str | None = None) -> Path:
    name = _collection_name(collection)
    return USER_KB if name == DEFAULT_COLLECTION else DATA / f"_kb_user_{name}.txt"


class _TfidfEmbedder:
    """纯离线 TF-IDF 嵌入器（字符 n-gram，适配中文无分词）。"""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
        self._fitted = False

    def fit(self, corpus: list):
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: list):
        if not self._fitted:
            self._vectorizer.fit(texts)
            self._fitted = True
        return self._vectorizer.transform(texts).toarray().astype("float32")


class _BgeEmbedder:
    """可选语义嵌入（需装 sentence-transformers + 本地 bge 模型）。"""

    name = "bge"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: list):
        pass

    def encode(self, texts: list):
        return np.asarray(self._model.encode(texts), dtype="float32")


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        want = os.environ.get("JIUAN_RAG_EMBED", "auto").lower()
        _EMBEDDER = _BgeEmbedder() if want == "bge" else _TfidfEmbedder()
    return _EMBEDDER


def _chunk(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """按段落切块，超长段落带重叠滑窗，避免硬截断语义。"""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for para in paras:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                chunks.append(para[start:start + chunk_size])
                start += chunk_size - overlap
    return chunks or ([text] if text else [])


def _load_store(collection: str | None = None) -> dict:
    path = _store_path(collection)
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return {"docs": [], "sources": [], "matrix": None, "embedder": None}


def _save_store(store: dict, collection: str | None = None) -> None:
    path = _store_path(collection)
    with open(path, "wb") as fh:
        pickle.dump(store, fh)


def _safe_source(source: str) -> Path:
    p = Path(source)
    resolved = (p if p.is_absolute() else (ROOT / p)).resolve()
    resolved.relative_to(ROOT.resolve())  # 防路径遍历，越界抛 ValueError
    return resolved


def ingest_document(
    source: str,
    chunk_size: int = 400,
    log: Optional[Callable] = None,
    collection: str = DEFAULT_COLLECTION,
) -> dict:
    """上传文档，分块、嵌入、入库（重建全量索引以保证 TF-IDF 词表一致）。"""
    log = log or (lambda _m: None)
    path = _safe_source(source)
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {source}")
    text = path.read_text(encoding="utf-8-sig")
    new_chunks = _chunk(text, chunk_size)
    log(f"文档 {path.name} 切分为 {len(new_chunks)} 块")

    collection = _collection_name(collection)
    store = _load_store(collection)
    # 防重复入库：同名 source 再次入库时，先移除其旧 chunk 再追加(幂等)
    replaced = 0
    if path.name in store["sources"]:
        keep_docs, keep_srcs = [], []
        for d, src in zip(store["docs"], store["sources"]):
            if src == path.name:
                replaced += 1
                continue
            keep_docs.append(d)
            keep_srcs.append(src)
        store["docs"], store["sources"] = keep_docs, keep_srcs
        log(f"检测到同名文档，已移除旧 {replaced} 块后重新入库")
    store["docs"].extend(new_chunks)
    store["sources"].extend([path.name] * len(new_chunks))

    embedder = get_embedder()
    if hasattr(embedder, "fit"):
        embedder.fit(store["docs"])
    store["matrix"] = embedder.encode(store["docs"])
    store["embedder"] = embedder.name
    _save_store(store, collection)
    log(f"知识库[{collection}]现有 {len(store['docs'])} 块（嵌入器 {embedder.name}）")
    return {
        "collection": collection,
        "source": path.name,
        "new_chunks": len(new_chunks),
        "replaced_chunks": replaced,
        "total_chunks": len(store["docs"]),
        "embedder": embedder.name,
    }


def _cosine(query_vec, matrix):
    q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return (m @ q.T).ravel()


def retrieve(query: str, top_k: int = 3, collection: str = DEFAULT_COLLECTION) -> list:
    """检索最相关知识段落，返回 [{text, source, score}]。"""
    collection = _collection_name(collection)
    store = _load_store(collection)
    if not store["docs"] or store["matrix"] is None:
        return []
    embedder = get_embedder()
    if hasattr(embedder, "fit") and getattr(embedder, "name", "") == "tfidf":
        embedder.fit(store["docs"])
    q_emb = embedder.encode([query])
    matrix = np.asarray(store["matrix"], dtype="float32")
    if q_emb.shape[1] != matrix.shape[1]:
        matrix = embedder.encode(store["docs"])
    scores = _cosine(q_emb, matrix)
    idx = np.argsort(-scores)[:top_k]
    return [
        {"text": store["docs"][i], "source": store["sources"][i], "score": round(float(scores[i]), 4)}
        for i in idx
    ]


def generate_with_rag(
    prompt: str,
    model_id: str = "base",
    top_k: int = 3,
    log: Optional[Callable] = None,
    collection: str = DEFAULT_COLLECTION,
) -> dict:
    """RAG 推理：先检索知识，再拼入 prompt，再调 infer。

    附带缺口检测：若最高命中分低于 HIT_THRESHOLD，判定为"知识库缺相关内容"，
    记录到 rag_gaps.jsonl，供 UI 通知业务人员一键补充知识。
    """
    log = log or (lambda _m: None)
    collection = _collection_name(collection)
    docs = retrieve(prompt, top_k, collection=collection)
    max_score = max((d["score"] for d in docs), default=0.0)
    hit_ok = bool(docs) and max_score >= HIT_THRESHOLD
    if hit_ok:
        context = "\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
        augmented = f"以下知识供参考：\n{context}\n\n问题：{prompt}"
        log(f"检索到 {len(docs)} 条相关知识(最高分 {max_score})，注入上下文")
    else:
        augmented = prompt
        log(f"知识库无相关内容(最高分 {max_score} < 阈值 {HIT_THRESHOLD})，记录缺口并退化为普通推理")
        record_gap(prompt, max_score, collection=collection)
    res = infer.generate(model_id, augmented, log=lambda _m: None)
    res["rag"] = {
        "retrieved": docs,
        "collection": collection,
        "top_k": top_k,
        "hit": len(docs),
        "max_score": round(float(max_score), 4),
        "hit_ok": hit_ok,
        "threshold": HIT_THRESHOLD,
    }
    res["prompt"] = prompt
    return res


def _read_gaps(collection: str | None = None) -> list:
    path = _gaps_path(collection)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _write_gaps(rows: list, collection: str | None = None) -> None:
    path = _gaps_path(collection)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def record_gap(prompt: str, max_score: float, collection: str = DEFAULT_COLLECTION) -> None:
    """记录一条 RAG 检索缺口(按 prompt 去重，pending 状态)。"""
    collection = _collection_name(collection)
    prompt = (prompt or "").strip()
    if not prompt:
        return
    rows = _read_gaps(collection)
    for r in rows:
        if r.get("prompt") == prompt and r.get("status") == "pending":
            r["hits"] = r.get("hits", 1) + 1        # 累计被问次数
            r["max_score"] = round(float(max_score), 4)
            r["last_at"] = time.time()
            r["collection"] = collection
            _write_gaps(rows, collection)
            return
    rows.append({
        "collection": collection,
        "prompt": prompt,
        "max_score": round(float(max_score), 4),
        "status": "pending",
        "hits": 1,
        "created_at": time.time(),
        "last_at": time.time(),
    })
    _write_gaps(rows, collection)


def list_gaps(status: Optional[str] = None, collection: str = DEFAULT_COLLECTION) -> list:
    """列出 RAG 缺口，默认按最近提问时间倒序。"""
    collection = _collection_name(collection)
    rows = _read_gaps(collection)
    if status:
        rows = [r for r in rows if r.get("status") == status]
    rows.sort(key=lambda r: r.get("last_at", 0), reverse=True)
    return rows


def resolve_gap(
    prompt: str,
    knowledge: str,
    log: Optional[Callable] = None,
    collection: str = DEFAULT_COLLECTION,
) -> dict:
    """业务人员一键补充知识：把 <prompt + knowledge> 追加到用户知识库并重新入库，
    随后把该缺口标记为 resolved。补充后下次提问即可命中。
    """
    log = log or (lambda _m: None)
    collection = _collection_name(collection)
    prompt = (prompt or "").strip()
    knowledge = (knowledge or "").strip()
    if not knowledge:
        raise ValueError("补充的知识内容不能为空")

    # 追加为一个带标题的知识段落(标题用问题，便于检索命中)
    block = f"{prompt}\n{knowledge}\n" if prompt else f"{knowledge}\n"
    user_kb = _user_kb_path(collection)
    with open(user_kb, "a", encoding="utf-8") as fh:
        fh.write("\n" + block)
    log(f"知识已追加到 {user_kb.name}")

    # 重新入库(ingest_document 会重建全量索引，保证 TF-IDF 词表一致)
    rel = str(user_kb.relative_to(ROOT)).replace("\\", "/")
    ing = ingest_document(rel, log=log, collection=collection)

    # 标记缺口 resolved
    rows = _read_gaps(collection)
    changed = 0
    for r in rows:
        if r.get("prompt") == prompt and r.get("status") == "pending":
            r["status"] = "resolved"
            r["resolved_at"] = time.time()
            changed += 1
    _write_gaps(rows, collection)
    return {
        "collection": collection,
        "prompt": prompt,
        "resolved": changed,
        "kb_file": user_kb.name,
        "total_chunks": ing["total_chunks"],
        "pending_left": len([r for r in rows if r.get("status") == "pending"]),
    }


def stats(collection: str = DEFAULT_COLLECTION) -> dict:
    from collections import Counter

    collection = _collection_name(collection)
    store = _load_store(collection)
    pending = len([r for r in _read_gaps(collection) if r.get("status") == "pending"])
    return {
        "collection": collection,
        "total_chunks": len(store["docs"]),
        "sources": dict(Counter(store["sources"])),
        "embedder": store.get("embedder"),
        "pending_gaps": pending,
    }


def collections() -> list[dict]:
    """List available knowledge-base collections with lightweight stats."""
    names: set[str] = set()
    if _STORE_PATH.exists() or _GAPS_PATH.exists():
        names.add(DEFAULT_COLLECTION)
    for p in VECTOR_DIR.iterdir():
        if p.is_dir() and ((p / "store.pkl").exists() or (p / "rag_gaps.jsonl").exists()):
            names.add(p.name)
    if not names:
        names.add(DEFAULT_COLLECTION)
    return [stats(name) for name in sorted(names, key=lambda x: (x != DEFAULT_COLLECTION, x))]


def run(params: dict, log: Callable) -> dict:
    """任务入口（预留，供后续接入任务调度）。"""
    action = params.get("action", "ingest")
    if action == "ingest":
        return ingest_document(
            params["source"],
            params.get("chunk_size", 400),
            log,
            collection=params.get("collection", DEFAULT_COLLECTION),
        )
    if action == "query":
        return generate_with_rag(
            params["prompt"],
            params.get("model_id", "base"),
            params.get("top_k", 3),
            log,
            collection=params.get("collection", DEFAULT_COLLECTION),
        )
    raise ValueError(f"未知 RAG action: {action}")
