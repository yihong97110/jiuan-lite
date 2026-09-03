"""RAG LangChain 测试脚本 v4 —— 修复 FAISS 内积 + BM25Okapi 直持

关键修复：
1. FAISS distance_strategy=MAX_INNER_PRODUCT（归一化后=余弦相似度，越大越好）
2. BM25 直接用 BM25Okapi 对象（不依赖 BM25Retriever 内部结构）
3. 文档顺序对齐：FAISS docstore 顺序 = BM25Okapi 文档顺序
"""
import sys, os, time, shutil
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/jiuan-lite")
os.chdir("/root/autodl-tmp/jiuan-lite")

# 清理旧索引（距离策略变了，旧索引不兼容）
vdb = Path("data/vector_db")
if vdb.exists():
    shutil.rmtree(vdb)
    print("✅ 已清空旧索引")
vdb.mkdir(parents=True, exist_ok=True)

print("\n=== 加载 RAG 模块 ===")
t0 = time.time()
from jiuan.pipeline import rag_backend
print(f"模块加载: {time.time()-t0:.1f}s | LangChain: {'✅' if rag_backend.LANGCHAIN_AVAILABLE else '❌'}")

# 测试文档
test_doc = """细胞是生物体结构和功能的基本单位，由细胞膜、细胞质和细胞核构成。

线粒体是细胞的动力工厂，通过有氧呼吸氧化有机物释放能量，产生ATP。

DNA是遗传信息的载体，由脱氧核糖、磷酸和碱基构成，呈双螺旋结构。

光合作用是植物利用光能将二氧化碳和水转化为有机物和氧气的过程。

酶是活细胞产生的生物催化剂，能降低化学反应的活化能，具有高效性和专一性。

生态系统的能量流动具有单向性和逐级递减的特点，营养级越高能量越少。
"""
Path("data/_test_bio.txt").write_text(test_doc, encoding="utf-8")

print("\n=== 入库（FAISS MAX_INNER_PRODUCT + BM25Okapi）===")
t0 = time.time()
result = rag_backend.ingest_document("data/_test_bio.txt", collection="test_bio", log=print)
print(f"入库结果: {result}")
print(f"耗时: {time.time()-t0:.1f}s")

print("\n=== 检索测试 ===")
queries = [
    ("细胞怎么产生能量？", "线粒体"),
    ("遗传物质是什么", "DNA"),
    ("植物怎么制造食物", "光合作用"),
]
all_pass = True
for q, expect in queries:
    print(f"\n查询: {q}")
    docs = rag_backend.retrieve(q, top_k=3, collection="test_bio")
    if not docs:
        print("  ❌ 无结果")
        all_pass = False
        continue
    for rank, d in enumerate(docs, 1):
        tag = "🎯" if rank == 1 else "  "
        vs = d.get("vec_score")
        bs = d.get("bm25_score")
        print(f"  {tag}#{rank} [融合={d['score']:.4f} vec={vs} bm25={bs}] {d['text'][:38]}...")
    hit = expect in docs[0]["text"]
    print(f"  {'✅ 命中' if hit else '❌ 未命中'} 期望含'{expect}'")
    if not hit:
        all_pass = False

print(f"\n{'='*40}")
print(f"{'🎉 全部通过！LangChain RAG 工作正常' if all_pass else '⚠️ 部分失败'}")
print(f"{'='*40}")
