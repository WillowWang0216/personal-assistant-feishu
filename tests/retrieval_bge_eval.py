# -*- coding: utf-8 -*-
"""
Multi-method retrieval comparison (700 memories x 4 search types).
Tests three methods:
  1. BM25 + jieba tokenization
  2. BGE-M3 vector semantic retrieval
  3. BM25 + BGE-M3 hybrid (RRF fusion)
Covers 4 search types: exact, fuzzy, related, hybrid
Run: python tests/retrieval_bge_eval.py
"""

from __future__ import annotations
import json
import math
import random
import re
import sqlite3
import tempfile
import time as time_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.agent.personal_memory_store import PersonalMemoryStore
from assistant.config.schema import MemorySystemConfig, RetrievalWeightsConfig

random.seed(42)

MEMORIES_AND_QUERIES: list[dict] = []

# Type 1: Exact Search - 175
for i in range(175):
    domains = ["用户", "开发者", "产品", "测试"]
    items = ["Python编程", "LLM应用", "飞书机器人", "异步编程", "数据库设计",
             "API开发", "Web开发", "Agent架构", "缓存策略", "安全加固"]
    item = items[i % len(items)]
    dom = domains[i % len(domains)]

    content = f"用户在{dom}场景下，使用{item}技术解决问题。详细记录了{item}的实践方法。"
    memories_and_queries = {
        "id": f"mem_exact_{i:04d}",
        "kind": "preference",
        "scope": "global",
        "slot": f"exact.{dom}.{item.replace(' ','')}",
        "summary": f"{dom}场景{item}",
        "content": content,
        "tags": [item, dom, "精确匹配"],
        "keywords": [item, dom],
        "priority": random.randint(1, 10),
        "query": f"{dom}的{item}怎么实现的",
        "search_type": "exact",
    }
    MEMORIES_AND_QUERIES.append(memories_and_queries)

# Type 2: Fuzzy Search - 175
for i in range(175):
    pairs = [
        ("Python编程", "Python变成", "python编程技巧"),
        ("LLM应用", "大语言模型", "LLM应用开发"),
        ("飞书机器人", "飞书机气人", "飞书机器人怎么开发"),
        ("异步编程", "异步变成", "异步编程模型"),
        ("API开发", "应用程序接口", "API开发实战"),
        ("数据库设计", "数据库设计", "数据库设计方案"),
        ("Web开发", "网页开发", "Web开发框架"),
        ("Agent架构", "智能体架构", "Agent架构设计"),
        ("缓存策略", "缓存策略", "Redis缓存策略"),
        ("安全加固", "安全加话", "系统安全加固"),
    ]
    correct, typo, query = pairs[i % len(pairs)]
    dom = ["企业", "个人", "团队", "项目"][i % 4]

    content = f"针对{dom}场景，详细介绍了{correct}的方案，包括具体步骤和注意事项。"
    memories_and_queries = {
        "id": f"mem_fuzzy_{i:04d}",
        "kind": "decision",
        "scope": "global",
        "slot": f"fuzzy.{dom}.{correct[:4]}",
        "summary": f"{dom}场景{correct}",
        "content": content,
        "tags": [correct, dom, "模糊匹配"],
        "keywords": [correct, dom],
        "priority": random.randint(1, 10),
        "query": query,
        "search_type": "fuzzy",
    }
    MEMORIES_AND_QUERIES.append(memories_and_queries)

# Type 3: Related Search - 175
for i in range(175):
    pairs = [
        ("Python编程", "脚本语言", "学Python还是脚本语言"),
        ("LLM应用", "大模型", "大模型和LLM应用有什么区别"),
        ("飞书机器人", "钉钉机器人", "飞书机器人对比钉钉机器人"),
        ("异步编程", "并发模型", "异步编程和并发模型的关系"),
        ("API开发", "接口设计", "API开发和接口设计是一回事吗"),
        ("数据库设计", "数据建模", "数据库设计和数据建模的联系"),
        ("Web开发", "前端工程", "Web开发和前端工程的区别"),
        ("Agent架构", "智能体框架", "Agent架构选型指南"),
        ("缓存策略", "数据加速", "缓存策略和数据加速的关系"),
        ("安全加固", "防护措施", "安全加固的防护措施有哪些"),
    ]
    correct, related, query = pairs[i % len(pairs)]
    dom = ["电商", "金融", "教育", "医疗"][i % 4]

    content = f"在{dom}行业应用场景中，{correct}是核心技术方案，经过实践验证。"
    memories_and_queries = {
        "id": f"mem_related_{i:04d}",
        "kind": "reference",
        "scope": "global",
        "slot": f"related.{dom}.{correct[:4]}",
        "summary": f"{dom}{correct}应用",
        "content": content,
        "tags": [correct, dom, "相关搜索"],
        "keywords": [correct, dom],
        "priority": random.randint(1, 10),
        "query": query,
        "search_type": "related",
    }
    MEMORIES_AND_QUERIES.append(memories_and_queries)

# Type 4: Hybrid Search - 175
for i in range(175):
    topics = [
        ("Python异步编程", "Python异步和并发编程哪个好"),
        ("LLM应用开发", "大模型应用LLM开发实战指南"),
        ("飞书API机器人", "飞书API机器人开发企业应用"),
        ("Agent多智能体", "多智能体Agent协同工作流设计"),
        ("数据库缓存", "数据库Redis缓存一致性方案"),
        ("Web安全防护", "Web应用安全防护最佳实践"),
        ("API版本管理", "RESTful API版本管理策略"),
        ("异步消息队列", "异步消息队列在分布式系统中的应用"),
        ("LLM知识库", "基于LLM构建企业知识库方案"),
        ("Agent记忆系统", "AI Agent长期记忆系统设计方案"),
    ]
    topic, query = topics[i % len(topics)]
    dom = ["后台", "前端", "全栈", "运维"][i % 4]

    content = f"在{dom}开发场景中，{topic}是重要技术方案，包含详细实施步骤和技术细节。经验证该方案稳定可靠。"
    memories_and_queries = {
        "id": f"mem_hybrid_{i:04d}",
        "kind": "constraint",
        "scope": "global",
        "slot": f"hybrid.{dom}.{topic[:6]}",
        "summary": f"{dom}场景{topic}方案",
        "content": content,
        "tags": [topic, dom, "混合搜索"],
        "keywords": [topic, dom],
        "priority": random.randint(1, 10),
        "query": query,
        "search_type": "hybrid",
    }
    MEMORIES_AND_QUERIES.append(memories_and_queries)

random.shuffle(MEMORIES_AND_QUERIES)
ALL_MEMORIES = [
    {k: v for k, v in m.items() if k != "query"}
    for m in MEMORIES_AND_QUERIES
]
ALL_QUERIES = [
    {"query": m["query"], "relevant_ids": {m["id"]}, "search_type": m["search_type"]}
    for m in MEMORIES_AND_QUERIES
]

print(f"Total memories: {len(ALL_MEMORIES)}, Total queries: {len(ALL_QUERIES)}")
type_counts = {}
for q in ALL_QUERIES:
    t = q["search_type"]
    type_counts[t] = type_counts.get(t, 0) + 1
print(f"Query type distribution: {type_counts}")


# ============================================================
# Evaluation metrics
# ============================================================

def recall_at_k(r, rel, k):
    return len(set(r[:k]) & rel) / len(rel) if rel else 0.0

def ndcg_at_k(r, rel, k):
    rels = [1.0 if x in rel else 0.0 for x in r[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(r, rel):
    for i, x in enumerate(r, 1):
        if x in rel: return 1.0 / i
    return 0.0


# ============================================================
# Vector retriever
# ============================================================

class VectorRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS mem_emb (
                memory_id TEXT PRIMARY KEY, embedding BLOB NOT NULL)""")
            conn.commit()
        print("  Loading embedding model...")
        t0 = time_module.time()
        from sentence_transformers import SentenceTransformer
        import tempfile, os
        tmp_cache = os.path.join(tempfile.gettempdir(), "st_cache_bge_eval")
        os.makedirs(tmp_cache, exist_ok=True)
        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=tmp_cache
        )
        print(f"  Model loaded ({time_module.time()-t0:.1f}s)")

    def encode_memories(self, memories):
        texts = [f"{m.get('summary','')} {m.get('content','')}" for m in memories]
        print(f"  Encoding {len(texts)} memory vectors...")
        t0 = time_module.time()
        self.embs = self.model.encode(texts, show_progress_bar=False, batch_size=64)
        print(f"  Encoding done ({time_module.time()-t0:.1f}s, shape={self.embs.shape})")
        import numpy as np
        with sqlite3.connect(self.db_path) as conn:
            for m, emb in zip(memories, self.embs):
                conn.execute(
                    "INSERT OR REPLACE INTO mem_emb VALUES (?,?)",
                    (m["id"], emb.astype("float32").tobytes())
                )
            conn.commit()

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        import numpy as np
        q_emb = self.model.encode([query], show_progress_bar=False)[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm < 1e-8:
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT memory_id, embedding FROM mem_emb").fetchall()
        scores = []
        for mid, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype="float32")
            sim = float(np.dot(q_emb, emb) / (np.linalg.norm(emb) * q_norm + 1e-8))
            scores.append((sim, mid))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


# ============================================================
# Three retrieval methods
# ============================================================

def bm25_retrieve(store, query, top_k, user_id):
    q_tokens = store._tokenize(query.lower())
    store._ensure_idf_ready()
    scored = []
    for mem in store.list_active_memories(user_id=user_id, limit=500):
        s = (mem.get("summary") or "").lower()
        c = (mem.get("content") or "").lower()
        sc = store._bm25_score(
            q_tokens, store._tokenize(s), store._tokenize(c), len(s) + len(c)
        )
        scored.append((sc, mem["id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(sc, mid) for sc, mid in scored if sc > 0][:top_k]


def vector_retrieve(vr, query, top_k):
    return vr.retrieve(query, top_k)


def hybrid_retrieve(store, vr, query, top_k, user_id, k=60):
    bm25_scores = {mid: score for score, mid in bm25_retrieve(store, query, k, user_id)}
    vector_scores = {mid: score for score, mid in vr.retrieve(query, k)}

    all_ids = set(bm25_scores) | set(vector_scores)
    rrf_scores = {}
    for mid in all_ids:
        bm25_rank = list(sorted(bm25_scores, key=bm25_scores.__getitem__, reverse=True)).index(mid) + 1 \
            if mid in bm25_scores else 0
        vec_rank = list(sorted(vector_scores, key=vector_scores.__getitem__, reverse=True)).index(mid) + 1 \
            if mid in vector_scores else 0
        rrf = (1 / (60 + bm25_rank)) + (1 / (60 + vec_rank)) if bm25_rank and vec_rank else 0
        rrf_scores[mid] = rrf

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return [(rrf_scores[mid], mid) for mid in sorted_ids[:top_k]]


# ============================================================
# Main test
# ============================================================

def run():
    tmp = Path(tempfile.mkdtemp())
    ws = tmp / "workspace"
    (ws / "memory").mkdir(parents=True)
    (ws / "memory" / "MEMORY.md").write_text("# MEMORY\n", encoding="utf-8")
    db_path = str(ws / "memory" / "personal_memory.db")

    cfg = MemorySystemConfig(
        enabled=True, db_path=db_path, default_user_id="eval",
        retrieval_top_k=5, fallback_top_k=2, core_memory_max_items=8,
        update_memory_md=False,
        retrieval_weights=RetrievalWeightsConfig(
            keyword=2.0, tag=1.5, summary=1.5, content=1.0,
            priority=0.15, recency=0.3, kind=0.5, scope=0.5,
        ),
    )
    store = PersonalMemoryStore(ws, cfg)
    for m in ALL_MEMORIES:
        store.create_memory(m, user_id="eval")
    print("Memory write complete")

    vr = VectorRetriever(db_path)
    vr.encode_memories(ALL_MEMORIES)

    stypes = ["exact", "fuzzy", "related", "hybrid", "all"]
    stype_names = {"exact": "Exact Search", "fuzzy": "Fuzzy Search",
                   "related": "Related Search", "hybrid": "Hybrid Search", "all": "All"}

    all_results = {}
    for stype in stypes:
        queries = [q for q in ALL_QUERIES if q["search_type"] == stype] if stype != "all" else ALL_QUERIES

        bm25_r3, bm25_r5, bm25_n3, bm25_n5, bm25_m = [], [], [], [], []
        vec_r3, vec_r5, vec_n3, vec_n5, vec_m = [], [], [], [], []
        hy_r3, hy_r5, hy_n3, hy_n5, hy_m = [], [], [], [], []

        for qdata in queries:
            q, rel = qdata["query"], qdata["relevant_ids"]

            b_res = bm25_retrieve(store, q, 5, "eval")
            v_res = vector_retrieve(vr, q, 5)
            h_res = hybrid_retrieve(store, vr, q, 5, "eval")

            b_ids = [mid for _, mid in b_res]
            v_ids = [mid for _, mid in v_res]
            h_ids = [mid for _, mid in h_res]

            bm25_r3.append(recall_at_k(b_ids, rel, 3))
            bm25_r5.append(recall_at_k(b_ids, rel, 5))
            bm25_n3.append(ndcg_at_k(b_ids, rel, 3))
            bm25_n5.append(ndcg_at_k(b_ids, rel, 5))
            bm25_m.append(mrr(b_ids, rel))

            vec_r3.append(recall_at_k(v_ids, rel, 3))
            vec_r5.append(recall_at_k(v_ids, rel, 5))
            vec_n3.append(ndcg_at_k(v_ids, rel, 3))
            vec_n5.append(ndcg_at_k(v_ids, rel, 5))
            vec_m.append(mrr(v_ids, rel))

            hy_r3.append(recall_at_k(h_ids, rel, 3))
            hy_r5.append(recall_at_k(h_ids, rel, 5))
            hy_n3.append(ndcg_at_k(h_ids, rel, 3))
            hy_n5.append(ndcg_at_k(h_ids, rel, 5))
            hy_m.append(mrr(h_ids, rel))

        av = lambda l: sum(l) / len(l)

        all_results[stype] = {
            "query_count": len(queries),
            "bm25":   {"r3": round(av(bm25_r3),4), "r5": round(av(bm25_r5),4),
                        "n3": round(av(bm25_n3),4), "n5": round(av(bm25_n5),4), "mrr": round(av(bm25_m),4)},
            "vector":  {"r3": round(av(vec_r3),4),  "r5": round(av(vec_r5),4),
                        "n3": round(av(vec_n3),4),  "n5": round(av(vec_n5),4),  "mrr": round(av(vec_m),4)},
            "hybrid":  {"r3": round(av(hy_r3),4),  "r5": round(av(hy_r5),4),
                        "n3": round(av(hy_n3),4),  "n5": round(av(hy_n5),4),  "mrr": round(av(hy_m),4)},
        }

    print(f"\n{'='*80}")
    print(f"  Retrieval Method Comparison (700 memories x 4 search types)")
    print(f"  Method 1: BM25 + jieba tokenization")
    print(f"  Method 2: BGE-M3 vector semantic retrieval (paraphrase-multilingual-MiniLM-L12-v2)")
    print(f"  Method 3: BM25 + BGE-M3 hybrid (RRF fusion, k=60)")
    print(f"{'='*80}")

    for stype in stypes:
        r = all_results[stype]
        print(f"\n  [{stype_names[stype]}] (N={r['query_count']})")
        print(f"  +{'-'*12} {'':>12} {'':>12} {'':>12} {'':>12}+")
        print(f"  |{'Metric':<12} {'BM25+Token':>12} {'Vector':>12} {'BM25+Vec':>12} {'Best':>12}|")
        print(f"  +{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}+")

        rows = [
            ("Recall@3",   "r3", max),
            ("Recall@5",   "r5", max),
            ("NDCG@3",     "n3", max),
            ("NDCG@5",     "n5", max),
            ("MRR",        "mrr", max),
        ]
        for name, key, _ in rows:
            b = r["bm25"][key]; v = r["vector"][key]; h = r["hybrid"][key]
            winner = "BM25" if b >= v and b >= h else ("Vector" if v >= h else "Hybrid")
            print(f"  |{name:<12} {b:>11.3f}  {v:>11.3f}  {h:>11.3f}  {winner:>12} |")

        print(f"  +{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}+")

    print(f"\n  [Retrieval Latency] (20 queries average)")
    q_sample = [q["query"] for q in ALL_QUERIES[:20]]

    def time_bm25():
        t0 = time_module.perf_counter()
        for q in q_sample: bm25_retrieve(store, q, 5, "eval")
        return (time_module.perf_counter() - t0) / len(q_sample) * 1000

    def time_vector():
        t0 = time_module.perf_counter()
        for q in q_sample: vector_retrieve(vr, q, 5)
        return (time_module.perf_counter() - t0) / len(q_sample) * 1000

    def time_hybrid():
        t0 = time_module.perf_counter()
        for q in q_sample: hybrid_retrieve(store, vr, q, 5, "eval")
        return (time_module.perf_counter() - t0) / len(q_sample) * 1000

    tb = sum(time_bm25() for _ in range(3)) / 3
    tv = sum(time_vector() for _ in range(3)) / 3
    th = sum(time_hybrid() for _ in range(3)) / 3
    print(f"  BM25+Token:  {tb:.1f}ms/query")
    print(f"  Vector:      {tv:.1f}ms/query  (embedding model inference)")
    print(f"  BM25+Vector: {th:.1f}ms/query  (two retrievals+RRF ranking)")

    out_file = Path(__file__).parent / "retrieval_bge_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "latency_ms": {"bm25": round(tb,2), "vector": round(tv,2), "hybrid": round(th,2)}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {out_file}")


if __name__ == "__main__":
    run()
