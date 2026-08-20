"""RAG 知识库：文档检索增强生成（LangChain 标准组件版）。

定位：弥补 SFT "硬记"的不足——最新法规条文用检索注入上下文更实用。

设计（LangChain 标准组件 + 自定义 RRF 融合）：
- 嵌入器 = HuggingFaceEmbeddings(BAAI/bge-small-zh-v1.5, 384维)；
  · 回退：环境变量 JIUAN_RAG_EMBED=tfidf 切回纯离线 TF-IDF(无需下载模型)。
- 向量库 = langchain_community.FAISS(IndexFlatIP, 归一化后等价余弦相似度)；
  · 持久化到 data/vector_db/{collection}/lc_index.faiss + lc_index.pkl
  · LC FAISS 内部通过 docstore_id 自动映射，天然修复 indices 顺序 bug
- BM25 = langchain_community.BM25Retriever + 自定义中文 bigram preprocess_func
- 混合检索 = 自定义 HybridRRFRetriever(BaseRetriever)：
  · 向量 + BM25 两路检索 → RRF(倒数排名融合) + tiebreaker
  · w_bm25=0.7 > w_vec=0.3（领域QA关键词精确命中信号强于泛化语义）
  · RRF 平局时用 α=0.4 加权归一化分数做 tiebreaker

LangChain 组件映射：
  HuggingFaceEmbeddings  -> Embeddings 抽象（embed_documents/embed_query）
  FAISS(VectorStore)     -> 向量存储（from_texts/similarity_search/save_local）
  BM25Retriever          -> 关键词检索（from_documents/get_relevant_documents）
  BaseRetriever          -> 自定义 HybridRRFRetriever（保留 RRF 融合算法）
  Document               -> 统一数据对象（page_content + metadata）

流程：上传文档→分段(chunk)→嵌入(embed)→入库FAISS；提问→HybridRRFRetriever混合检索→注入prompt→模型生成。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..common import DATA, ROOT
from . import infer

VECTOR_DIR = DATA / "vector_db"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
_STORE_PATH = VECTOR_DIR / "store.pkl"
_GAPS_PATH = VECTOR_DIR / "rag_gaps.jsonl"
USER_KB = DATA / "_kb_user.txt"
DEFAULT_COLLECTION = "default"
HIT_THRESHOLD = float(os.environ.get("JIUAN_RAG_HIT_THRESHOLD", "0.15"))

SYSTEM_PROMPT = "你是应急管理领域的专业助手，优先依据检索到的知识回答，如知识不足再结合常识。"

_EMBEDDER = None

# LangChain 可用性检测
try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    # DistanceStrategy 在不同版本位置不同
    try:
        from langchain_community.vectorstores.base import DistanceStrategy
    except ImportError:
        try:
            from langchain_community.vectorstores.utils import DistanceStrategy
        except ImportError:
            from langchain_core.vectorstores.base import DistanceStrategy
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    Embeddings = object  # type: ignore
    Document = dict  # type: ignore
    BaseRetriever = object  # type: ignore
    DistanceStrategy = None  # type: ignore


# ---------------------------------------------------------------------------
# 路径管理
# ---------------------------------------------------------------------------

def _collection_name(collection: str | None = None) -> str:
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


def _lc_index_path(collection: str | None = None) -> Path:
    """LangChain FAISS 索引路径（不含后缀，LC 会自动加 .faiss/.pkl）。"""
    return _collection_dir(collection) / "lc_index"


def _bm25_path(collection: str | None = None) -> Path:
    """BM25 检索器 pickle 路径。"""
    return _collection_dir(collection) / "bm25.pkl"


# ---------------------------------------------------------------------------
# 嵌入器
# ---------------------------------------------------------------------------

class _TfidfEmbedder:
    """纯离线 TF-IDF 嵌入器（字符 n-gram，适配中文无分词）。

    不实现 LangChain Embeddings 接口（TF-IDF 需要有状态 fit，和 Embeddings 无状态设计冲突）。
    TF-IDF 模式下 retrieve/ingest 走旧的 numpy 路径。
    """

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


class _BgeEmbedder(Embeddings):
    """BGE 语义嵌入器，实现 LangChain Embeddings 接口。

    用 HuggingFaceEmbeddings 包装 BAAI/bge-small-zh-v1.5（384维）。
    可直接传给 FAISS.from_texts(embedding=...) 和 similarity_search。
    """

    name = "bge"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from langchain_huggingface import HuggingFaceEmbeddings
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # 归一化后内积=余弦相似度
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def encode(self, texts: list):
        """兼容旧接口：返回 numpy 数组。"""
        return np.asarray(self.embed_documents(texts), dtype="float32")

    def fit(self, corpus: list):
        pass


# ---------------------------------------------------------------------------
# 中文分词（BM25 用）
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    """中文按 2-gram 滑窗切分；英文按空格。供 BM25Retriever 的 preprocess_func 使用。"""
    text = re.sub(r"\s+", " ", text.strip())
    toks = []
    for w in text.split():
        if re.match(r"^[A-Za-z0-9]", w):
            toks.append(w.lower())
    cn = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(len(cn) - 1):
        toks.append(cn[i:i + 2])
    return toks


# ---------------------------------------------------------------------------
# 融合算法（保留不变）
# ---------------------------------------------------------------------------

_HYBRID_ALPHA = float(os.environ.get("JIUAN_RAG_HYBRID_ALPHA", "0.4"))
_RRF_K = int(os.environ.get("JIUAN_RAG_RRF_K", "60"))
_RRF_W_VEC = float(os.environ.get("JIUAN_RAG_RRF_W_VEC", "0.3"))
_RRF_W_BM25 = float(os.environ.get("JIUAN_RAG_RRF_W_BM25", "0.7"))
_FUSION_METHOD = os.environ.get("JIUAN_RAG_FUSION", "rrf").lower()


def _normalize(scores: list) -> list:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _compute_ranks(scores: list) -> list:
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -scores[i])
    ranks = [0] * n
    current_rank = 1
    prev = None
    for pos, idx in enumerate(order):
        if prev is not None and abs(scores[idx] - prev) > 1e-12:
            current_rank = pos + 1
        ranks[idx] = current_rank
        prev = scores[idx]
    return ranks


def _rrf_fusion(vec_scores: list, bm25_scores: list, k: int = _RRF_K,
                w_vec: float = _RRF_W_VEC, w_bm25: float = _RRF_W_BM25) -> list:
    n = len(vec_scores)
    if n == 0:
        return []
    vec_rank = _compute_ranks(vec_scores)
    bm25_rank = _compute_ranks(bm25_scores)
    return [
        w_vec / (k + vec_rank[i]) + w_bm25 / (k + bm25_rank[i])
        for i in range(n)
    ]


def _cosine(query_vec, matrix):
    q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return (m @ q.T).ravel()


# ---------------------------------------------------------------------------
# 嵌入器工厂
# ---------------------------------------------------------------------------

def get_embedder():
    """获取嵌入器。默认 BGE 语义向量；环境变量 JIUAN_RAG_EMBED=tfidf 回退纯离线。

    BGE 首次加载会从 HuggingFace 下载 ~95MB 模型；若下载失败自动回退 TF-IDF。
    国内服务器自动走 hf-mirror.com 镜像。
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        want = os.environ.get("JIUAN_RAG_EMBED", "bge").lower()
        if want == "tfidf":
            _EMBEDDER = _TfidfEmbedder()
        else:
            if not os.environ.get("HF_ENDPOINT") and os.environ.get("JIUAN_HF_MIRROR", "yes").lower() != "no":
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            try:
                _EMBEDDER = _BgeEmbedder()
            except Exception as e:
                print(f"[rag] BGE 嵌入器加载失败({str(e)[:80]})，回退到 TF-IDF")
                _EMBEDDER = _TfidfEmbedder()
    return _EMBEDDER


# ---------------------------------------------------------------------------
# 文档分块
# ---------------------------------------------------------------------------

def _chunk(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
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


# ---------------------------------------------------------------------------
# 旧 store 管理（保留用于迁移和 TF-IDF 回退）
# ---------------------------------------------------------------------------

def _load_store(collection: str | None = None) -> dict:
    path = _store_path(collection)
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return {"docs": [], "sources": [], "matrix": None, "embedder": None, "bm25_ready": False}


def _save_store(store: dict, collection: str | None = None) -> None:
    path = _store_path(collection)
    with open(path, "wb") as fh:
        pickle.dump(store, fh)


# ---------------------------------------------------------------------------
# LangChain 组件加载/持久化
# ---------------------------------------------------------------------------

def _load_faiss_db(collection: str | None = None):
    """加载 LangChain FAISS VectorStore；不存在返回 None。"""
    if not LANGCHAIN_AVAILABLE:
        return None
    try:
        from langchain_community.vectorstores import FAISS as LCFAISS
        index_path = _lc_index_path(collection)
        if index_path.with_suffix(".faiss").exists() and index_path.with_suffix(".pkl").exists():
            embedder = get_embedder()
            if embedder.name != "bge":
                return None
            return LCFAISS.load_local(
                str(_collection_dir(collection)), embedder, "lc_index",
                allow_dangerous_deserialization=True,
            )
    except Exception as e:
        print(f"[rag] FAISS 加载失败: {str(e)[:80]}")
    return None


def _save_faiss_db(db, collection: str | None = None) -> None:
    """持久化 LangChain FAISS VectorStore 到磁盘。"""
    try:
        db.save_local(str(_collection_dir(collection)), "lc_index")
    except Exception as e:
        print(f"[rag] FAISS 保存失败: {str(e)[:80]}")


def _load_bm25_okapi(collection: str | None = None):
    """加载 BM25Okapi 对象；不存在返回 None。"""
    path = _bm25_path(collection)
    if path.exists():
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as e:
            print(f"[rag] BM25 加载失败: {str(e)[:80]}")
    return None


def _save_bm25_okapi(bm25, collection: str | None = None) -> None:
    """持久化 BM25Okapi 对象到磁盘。"""
    path = _bm25_path(collection)
    with open(path, "wb") as fh:
        pickle.dump(bm25, fh)


def _build_bm25_okapi(docs: list):
    """从文档文本列表构建 BM25Okapi（直接用 rank_bm25，不依赖 BM25Retriever）。

    文档顺序和传入的 docs 列表一致，保证和 FAISS docstore 对齐。
    """
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(d) for d in docs]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# 自定义 HybridRRFRetriever（保留 RRF 融合算法）
# ---------------------------------------------------------------------------

if LANGCHAIN_AVAILABLE:

    class HybridRRFRetriever(BaseRetriever):
        """混合检索器：FAISS 向量检索 + BM25 关键词检索 → RRF 融合。

        继承 LangChain BaseRetriever，实现 _get_relevant_documents 接口。
        内部保留 RRF(倒数排名融合) + tiebreaker 算法不变。

        设计要点：
        - faiss_db: LangChain FAISS VectorStore（MAX_INNER_PRODUCT + normalize → 余弦相似度）
        - bm25: 直接持有 rank_bm25.BM25Okapi 对象（不依赖 BM25Retriever 内部结构）
        - docs/sources: 文档文本和来源列表，顺序和 FAISS docstore 一致
        - 向量分数和 BM25 分数都按 self.docs 顺序对齐，保证 RRF 融合正确
        """
        faiss_db: object = None  # langchain_community.FAISS
        bm25: object = None  # rank_bm25.BM25Okapi（直接持有，不依赖 BM25Retriever）
        docs: list = []  # 文档文本列表（顺序和 FAISS docstore 一致）
        sources: list = []  # 来源列表
        w_vec: float = _RRF_W_VEC
        w_bm25: float = _RRF_W_BM25
        alpha: float = _HYBRID_ALPHA
        k: int = _RRF_K
        top_k: int = 3

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str) -> List[Document]:
            n = len(self.docs)
            if n == 0:
                return []

            # --- 向量检索：FAISS similarity_search_with_score 返回 (Document, score) ---
            # MAX_INNER_PRODUCT 策略下，score 越大越相似（归一化后=余弦相似度）
            vec_scores = [0.0] * n
            if self.faiss_db is not None:
                k_search = min(n, max(self.top_k * 4, 20))
                results = self.faiss_db.similarity_search_with_score(query, k=k_search)
                # 用 docstore_id 对齐：FAISS 返回的 Document 带 id，可以精确映射
                for doc, score in results:
                    text = doc.page_content
                    for i, d in enumerate(self.docs):
                        if d == text:
                            vec_scores[i] = float(score)
                            break

            # --- BM25 检索：直接用 BM25Okapi.get_scores() 获取真实分数 ---
            bm25_scores = [0.0] * n
            if self.bm25 is not None:
                try:
                    q_toks = _tokenize(query)
                    if q_toks:
                        raw_scores = self.bm25.get_scores(q_toks)
                        # BM25Okapi 的文档顺序和 self.docs 一致（构建时保证）
                        for i in range(min(n, len(raw_scores))):
                            bm25_scores[i] = float(raw_scores[i])
                except Exception:
                    pass

            # --- RRF 融合 ---
            vec_norm = _normalize(vec_scores)
            bm25_norm = _normalize(bm25_scores)

            if _FUSION_METHOD == "weighted":
                primary = [self.alpha * v + (1 - self.alpha) * b
                           for v, b in zip(vec_norm, bm25_norm)]
                tiebreak = [0.0] * n
            else:
                primary = _rrf_fusion(vec_scores, bm25_scores)
                tiebreak = [self.alpha * v + (1 - self.alpha) * b
                            for v, b in zip(vec_norm, bm25_norm)]

            idx = sorted(range(n), key=lambda i: (-primary[i], -tiebreak[i]))[:self.top_k]
            return [
                Document(
                    page_content=self.docs[i],
                    metadata={
                        "source": self.sources[i],
                        "score": round(float(primary[i]), 4),
                        "vec_score": round(float(vec_scores[i]), 4),
                        "bm25_score": round(float(bm25_scores[i]), 4),
                        "fusion": _FUSION_METHOD,
                    },
                )
                for i in idx
            ]

else:
    HybridRRFRetriever = None  # type: ignore


# ---------------------------------------------------------------------------
# 数据迁移
# ---------------------------------------------------------------------------

def migrate_collection(collection: str = DEFAULT_COLLECTION) -> dict:
    """把旧格式 store.pkl 迁移到 LangChain FAISS 格式。

    旧格式：store.pkl (dict: docs/sources/matrix) + store.faiss
    新格式：lc_index.faiss + lc_index.pkl (LC FAISS) + bm25.pkl (BM25Retriever)
    """
    collection = _collection_name(collection)
    store = _load_store(collection)
    if not store["docs"]:
        return {"collection": collection, "migrated": False, "reason": "empty store"}

    embedder = get_embedder()
    if embedder.name != "bge":
        return {"collection": collection, "migrated": False, "reason": "tfidf mode, skip"}

    try:
        from langchain_community.vectorstores import FAISS as LCFAISS
        # 用 HuggingFaceEmbeddings 重新编码，distance_strategy=MAX_INNER_PRODUCT（归一化后=余弦）
        lc_docs = [Document(page_content=d, metadata={"source": s})
                   for d, s in zip(store["docs"], store["sources"])]
        db = LCFAISS.from_documents(lc_docs, embedder,
                                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        _save_faiss_db(db, collection)

        # 迁移 BM25（直接用 BM25Okapi，不依赖 BM25Retriever）
        bm25 = _build_bm25_okapi(store["docs"])
        _save_bm25_okapi(bm25, collection)

        # 写迁移标记
        (_collection_dir(collection) / "migrated.flag").write_text(
            f"migrated at {time.time()}", encoding="utf-8"
        )
        return {
            "collection": collection,
            "migrated": True,
            "chunks": len(store["docs"]),
            "embedder": embedder.name,
        }
    except Exception as e:
        return {"collection": collection, "migrated": False, "reason": str(e)[:200]}


# ---------------------------------------------------------------------------
# 安全
# ---------------------------------------------------------------------------

def _safe_source(source: str) -> Path:
    p = Path(source)
    resolved = (p if p.is_absolute() else (ROOT / p)).resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------

def ingest_document(
    source: str,
    chunk_size: int = 400,
    log: Optional[Callable] = None,
    collection: str = DEFAULT_COLLECTION,
) -> dict:
    """上传文档，分块、嵌入、入库。

    BGE 模式：用 LangChain FAISS.from_texts 构建索引 + BM25Retriever
    TF-IDF 模式：回退旧的 numpy store 方式
    """
    log = log or (lambda _m: None)
    path = _safe_source(source)
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {source}")
    text = path.read_text(encoding="utf-8-sig")
    new_chunks = _chunk(text, chunk_size)
    log(f"文档 {path.name} 切分为 {len(new_chunks)} 块")

    collection = _collection_name(collection)
    embedder = get_embedder()

    # --- BGE 模式：LangChain FAISS + BM25 ---
    if embedder.name == "bge" and LANGCHAIN_AVAILABLE:
        from langchain_community.vectorstores import FAISS as LCFAISS

        # 加载已有 FAISS db（如果有）
        db = _load_faiss_db(collection)
        # 防重复入库：同名 source 先删除旧 chunk
        replaced = 0
        if db is not None:
            # 删除同名 source 的旧文档
            old_ids = []
            for doc_id, doc in db.docstore._dict.items():
                if doc.metadata.get("source") == path.name:
                    old_ids.append(doc_id)
            if old_ids:
                db.delete(old_ids)
                replaced = len(old_ids)
                log(f"检测到同名文档，已移除旧 {replaced} 块后重新入库")

        # 添加新 chunk
        new_lc_docs = [Document(page_content=c, metadata={"source": path.name}) for c in new_chunks]
        if db is None:
            db = LCFAISS.from_documents(new_lc_docs, embedder,
                                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        else:
            db.add_documents(new_lc_docs)
        _save_faiss_db(db, collection)

        # 重建 BM25（需要全量文档）
        all_docs = [d.page_content for d in db.docstore._dict.values()]
        all_sources = [d.metadata.get("source", "?") for d in db.docstore._dict.values()]
        bm25 = _build_bm25_okapi(all_docs)
        _save_bm25_okapi(bm25, collection)

        total = len(all_docs)
        log(f"知识库[{collection}]现有 {total} 块（嵌入器 {embedder.name}，FAISS=✓，BM25=✓）")
        return {
            "collection": collection,
            "source": path.name,
            "new_chunks": len(new_chunks),
            "replaced_chunks": replaced,
            "total_chunks": total,
            "embedder": embedder.name,
            "faiss": True,
            "bm25": True,
        }

    # --- TF-IDF 模式：回退旧的 numpy store ---
    store = _load_store(collection)
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

    if hasattr(embedder, "fit"):
        embedder.fit(store["docs"])
    store["matrix"] = embedder.encode(store["docs"])
    store["embedder"] = embedder.name
    store["faiss_ready"] = False
    store["bm25_ready"] = False
    _save_store(store, collection)

    log(f"知识库[{collection}]现有 {len(store['docs'])} 块（嵌入器 {embedder.name}，TF-IDF 回退模式）")
    return {
        "collection": collection,
        "source": path.name,
        "new_chunks": len(new_chunks),
        "replaced_chunks": replaced,
        "total_chunks": len(store["docs"]),
        "embedder": embedder.name,
        "faiss": False,
        "bm25": False,
    }


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 3, collection: str = DEFAULT_COLLECTION) -> list:
    """混合检索：HybridRRFRetriever（BGE 模式）或 numpy 余弦（TF-IDF 回退）。

    返回 [{text, source, score, vec_score, bm25_score, fusion}]
    """
    collection = _collection_name(collection)
    embedder = get_embedder()

    # --- BGE 模式：LangChain HybridRRFRetriever ---
    if embedder.name == "bge" and LANGCHAIN_AVAILABLE:
        db = _load_faiss_db(collection)
        bm25 = _load_bm25_okapi(collection)

        # 如果没有 LC 格式，尝试从旧 store 迁移
        if db is None:
            store = _load_store(collection)
            if store["docs"]:
                print(f"[rag] 自动迁移 collection {collection} 到 LangChain 格式")
                migrate_collection(collection)
                db = _load_faiss_db(collection)
                bm25 = _load_bm25_okapi(collection)

        if db is None:
            return []

        # 从 FAISS docstore 提取文档列表（用于对齐）
        # 注意：FAISS docstore 的顺序和 BM25Okapi 的文档顺序必须一致
        all_docs = [d.page_content for d in db.docstore._dict.values()]
        all_sources = [d.metadata.get("source", "?") for d in db.docstore._dict.values()]

        if not all_docs:
            return []

        # 如果 BM25 不存在或文档数不匹配，重新构建
        if bm25 is None:
            print(f"[rag] BM25 不存在，重新构建")
            bm25 = _build_bm25_okapi(all_docs)
            _save_bm25_okapi(bm25, collection)

        retriever = HybridRRFRetriever(
            faiss_db=db,
            bm25=bm25,
            docs=all_docs,
            sources=all_sources,
            top_k=top_k,
        )
        lc_docs = retriever.invoke(query)
        return [
            {
                "text": d.page_content,
                "source": d.metadata.get("source", ""),
                "score": d.metadata.get("score", 0.0),
                "vec_score": d.metadata.get("vec_score"),
                "bm25_score": d.metadata.get("bm25_score"),
                "fusion": d.metadata.get("fusion", "rrf"),
            }
            for d in lc_docs
        ]

    # --- TF-IDF 模式：回退旧的 numpy 余弦 ---
    store = _load_store(collection)
    if not store["docs"] or store["matrix"] is None:
        return []

    n = len(store["docs"])
    if hasattr(embedder, "fit") and getattr(embedder, "name", "") == "tfidf":
        embedder.fit(store["docs"])
    q_emb = embedder.encode([query])
    matrix = np.asarray(store["matrix"], dtype="float32")
    if q_emb.shape[1] != matrix.shape[1]:
        matrix = embedder.encode(store["docs"])

    vec_scores = _cosine(q_emb, matrix).tolist()
    idx = sorted(range(n), key=lambda i: -vec_scores[i])[:top_k]
    return [
        {
            "text": store["docs"][i],
            "source": store["sources"][i],
            "score": round(float(vec_scores[i]), 4),
            "vec_score": round(float(vec_scores[i]), 4),
            "bm25_score": None,
            "fusion": "vector_only",
        }
        for i in idx
    ]


# ---------------------------------------------------------------------------
# RAG 推理
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 缺口管理（不变）
# ---------------------------------------------------------------------------

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
    collection = _collection_name(collection)
    prompt = (prompt or "").strip()
    if not prompt:
        return
    rows = _read_gaps(collection)
    for r in rows:
        if r.get("prompt") == prompt and r.get("status") == "pending":
            r["hits"] = r.get("hits", 1) + 1
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
    log = log or (lambda _m: None)
    collection = _collection_name(collection)
    prompt = (prompt or "").strip()
    knowledge = (knowledge or "").strip()
    if not knowledge:
        raise ValueError("补充的知识内容不能为空")

    block = f"{prompt}\n{knowledge}\n" if prompt else f"{knowledge}\n"
    user_kb = _user_kb_path(collection)
    with open(user_kb, "a", encoding="utf-8") as fh:
        fh.write("\n" + block)
    log(f"知识已追加到 {user_kb.name}")

    rel = str(user_kb.relative_to(ROOT)).replace("\\", "/")
    ing = ingest_document(rel, log=log, collection=collection)

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


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def _get_collection_docs(collection: str) -> tuple[list, list, str]:
    """获取 collection 的文档列表、来源列表、嵌入器名（兼容新旧格式）。"""
    embedder = get_embedder()

    # 优先读 LangChain FAISS
    if embedder.name == "bge" and LANGCHAIN_AVAILABLE:
        db = _load_faiss_db(collection)
        if db is not None:
            docs = [d.page_content for d in db.docstore._dict.values()]
            sources = [d.metadata.get("source", "?") for d in db.docstore._dict.values()]
            return docs, sources, "bge"

    # 回退旧 store
    store = _load_store(collection)
    return store["docs"], store["sources"], store.get("embedder", embedder.name)


def stats(collection: str = DEFAULT_COLLECTION) -> dict:
    from collections import Counter
    collection = _collection_name(collection)
    docs, sources, embedder_name = _get_collection_docs(collection)
    pending = len([r for r in _read_gaps(collection) if r.get("status") == "pending"])
    return {
        "collection": collection,
        "total_chunks": len(docs),
        "sources": dict(Counter(sources)),
        "embedder": embedder_name,
        "pending_gaps": pending,
    }


def collections() -> list[dict]:
    """List available knowledge-base collections with lightweight stats."""
    names: set[str] = set()
    if _STORE_PATH.exists() or _GAPS_PATH.exists():
        names.add(DEFAULT_COLLECTION)
    # 检查 LC 格式
    if _lc_index_path(DEFAULT_COLLECTION).with_suffix(".faiss").exists():
        names.add(DEFAULT_COLLECTION)
    for p in VECTOR_DIR.iterdir():
        if p.is_dir():
            has_lc = (p / "lc_index.faiss").exists()
            has_old = (p / "store.pkl").exists()
            has_gaps = (p / "rag_gaps.jsonl").exists()
            if has_lc or has_old or has_gaps:
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
