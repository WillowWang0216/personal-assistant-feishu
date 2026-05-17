# -*- coding: utf-8 -*-
"""
BM25-only vs BM25+7-factor hybrid retrieval comparison (700 memories x 50 queries).
Compare Recall@K, NDCG@K, MRR on the same dataset.
Run: python tests/retrieval_700_eval.py
"""

from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.agent.personal_memory_store import PersonalMemoryStore
from assistant.config.schema import MemorySystemConfig, RetrievalWeightsConfig

random.seed(42)

KIND_TEMPLATES = {
    "preference": {
        "slots": ["{domain}.reply.language", "{domain}.reply.style", "{domain}.detail.level",
                   "{domain}.interest.{topic}", "{domain}.tool.{name}"],
        "domains": ["user", "dev", "prod"],
        "topics": ["python", "llm", "agent", "web", "data"],
        "contents": [
            "用户偏好用{lang}回答",
            "回复风格要求{style}",
            "喜欢{level}级别的解释",
            "对{topic}领域特别感兴趣",
            "经常使用{tool}工具",
        ],
    },
    "decision": {
        "slots": ["{domain}.decision.{item}", "{domain}.stack.{name}"],
        "domains": ["proj", "arch", "infra"],
        "items": ["sqlite", "redis", "postgresql", "mongodb"],
        "contents": [
            "决定使用{item}方案",
            "技术栈选择{item}",
            "架构决策：采用{item}架构",
            "部署方案确定：{item}",
        ],
    },
    "reference": {
        "slots": ["{domain}.ref.{name}", "{domain}.doc.{name}"],
        "domains": ["lib", "api", "sdk"],
        "names": ["openai", "feishu", "sqlite", "redis", "jieba", "httpx"],
        "contents": [
            "参考文档：{name}官方文档",
            "API参考：{name}接口说明",
            "SDK使用：{name}集成方法",
            "第三方库：{name}使用指南",
        ],
    },
    "constraint": {
        "slots": ["{domain}.constraint.{name}", "{domain}.limit.{name}"],
        "domains": ["perf", "sec", "cost"],
        "names": ["latency", "privacy", "budget", "scale"],
        "contents": [
            "性能约束：{name}必须满足{val}",
            "安全约束：{name}相关要求",
            "成本约束：{name}控制在{val}",
        ],
    },
    "profile": {
        "slots": ["{domain}.profile.{name}", "{domain}.role.{name}"],
        "domains": ["self", "team", "org"],
        "names": ["role", "skill", "level", "company"],
        "contents": [
            "用户角色：{name}",
            "技能背景：{name}相关经验",
            "所属团队：{name}",
        ],
    },
}

LANGS = ["中文", "英文", "中英双语"]
STYLES = ["简洁", "详细", "结构化", "口语化"]
LEVELS = ["入门", "进阶", "高级"]
TOOLS = ["web_search", "memory_search", "read_file", "write_file"]
ITEMS = ["sqlite", "redis", "postgresql", "mongodb"]
VALS = ["<100ms", "<1s", "<5s", "<10s", "5%"]


def generate_memories(n: int = 700) -> list[dict]:
    memories = []
    kinds = ["preference", "decision", "reference", "constraint", "profile"]
    priorities = list(range(1, 11))
    slots_seen = set()
    while len(memories) < n:
        kind = random.choice(kinds)
        tpl = KIND_TEMPLATES[kind]
        slot_tpl = random.choice(tpl["slots"])
        domain = random.choice(tpl.get("domains", ["d1"]))
        slot = slot_tpl.format(
            domain=domain,
            item=random.choice(tpl.get("items", ["item"])),
            name=random.choice(tpl.get("names", ["name"])),
            topic=random.choice(tpl.get("topics", ["topic"])),
            tool=random.choice(tpl.get("tools", ["tool"])),
        )
        suffix = len([s for s in slots_seen if s.startswith(slot)])
        if suffix:
            slot = f"{slot}.{suffix}"
        slots_seen.add(slot)
        content_tpl = random.choice(tpl["contents"])
        content = content_tpl.format(
            lang=random.choice(LANGS), style=random.choice(STYLES),
            level=random.choice(LEVELS), topic=random.choice(tpl.get("topics", ["topic"])),
            tool=random.choice(TOOLS), item=random.choice(ITEMS),
            name=random.choice(tpl.get("names", ["name"])), val=random.choice(VALS),
        )
        memories.append({
            "id": f"mem_{len(memories):04d}",
            "kind": kind,
            "scope": "global",
            "slot": slot,
            "summary": content[:50],
            "content": content,
            "tags": random.sample(["python", "llm", "agent", "web", "db", "api", "dev", "test", "perf", "sec"], k=3),
            "keywords": random.sample(["python", "llm", "agent", "web", "db", "api", "dev", "test"], k=3),
            "priority": random.choice(priorities),
        })
    return memories


# ============================================================
# Evaluation metrics
# ============================================================

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def ndcg_at_k(retrieved, relevant, k):
    relevances = [1.0 if r in relevant else 0.0 for r in retrieved[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = [1.0] * min(len(relevant), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(retrieved, relevant):
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved, relevant, k):
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


class BM25OnlyStore(PersonalMemoryStore):
    def retrieve_bm25_only(self, query: str, top_k: int = 5, user_id: str = "default"):
        q_tokens = self._tokenize(query.lower())
        self._ensure_idf_ready()
        scored = []
        for memory in self.list_active_memories(user_id=user_id, limit=500):
            summary = (memory.get("summary") or "").lower()
            content = (memory.get("content") or "").lower()
            doc_len = len(summary) + len(content)
            score = self._bm25_score(
                q_tokens,
                self._tokenize(summary),
                self._tokenize(content),
                doc_len,
            )
            scored.append((score, memory["id"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for score, mid in scored if score > 0][:top_k]


# ============================================================
# Main test
# ============================================================

def run():
    tmp = Path(tempfile.mkdtemp())
    workspace = tmp / "workspace"
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "memory" / "MEMORY.md").write_text("# MEMORY\n", encoding="utf-8")

    memories = generate_memories(700)
    print(f"Generated memories: {len(memories)}")

    cfg = MemorySystemConfig(
        enabled=True,
        db_path=str(workspace / "memory" / "hybrid.db"),
        default_user_id="eval-user",
        retrieval_top_k=5, fallback_top_k=2, core_memory_max_items=8,
        update_memory_md=False,
        retrieval_weights=RetrievalWeightsConfig(
            keyword=2.0, tag=1.5, summary=1.5, content=1.0,
            priority=0.15, recency=0.3, kind=0.5, scope=0.5,
        ),
    )
    store = PersonalMemoryStore(workspace, cfg)

    cfg_bm25 = MemorySystemConfig(
        enabled=True,
        db_path=str(workspace / "memory" / "hybrid.db"),
        default_user_id="eval-user",
        retrieval_top_k=5, fallback_top_k=2, core_memory_max_items=8,
        update_memory_md=False,
        retrieval_weights=RetrievalWeightsConfig(
            keyword=0.0, tag=0.0, summary=0.0, content=1.0,
            priority=0.0, recency=0.0, kind=0.0, scope=0.0,
        ),
    )
    bm25_store = BM25OnlyStore(workspace, cfg_bm25)

    for m in memories:
        store.create_memory(m, user_id="eval-user")

    print("Database write complete")

    active = store.list_active_memories(user_id="eval-user", limit=700)
    sampled = active[:50]
    import re
    generated_queries = []
    for i, mem in enumerate(sampled):
        slot_parts = mem.get("slot", "").split(".")
        slot_kw = slot_parts[-1] if slot_parts else ""
        content = mem.get("content", "")
        cn_words = re.findall(r'[一-鿿]+', content)
        cn_kw = cn_words[1] if len(cn_words) > 1 else (cn_words[0] if cn_words else slot_kw)
        query = f"{slot_kw} {cn_kw}"
        generated_queries.append({
            "id": f"q_{i:02d}", "query": query, "relevant_ids": {mem["id"]},
        })

    print(f"Ground Truth queries: {len(generated_queries)}条\n")

    print(f"  {'Query':<28} {'Method':<14} {'R@3':>6} {'R@5':>6} {'N@3':>6} {'N@5':>6} {'MRR':>6} {'P@5':>6}")
    print(f"  {'-'*90}")

    mem_ids = {m["id"] for m in memories}
    hybrid_recall3, hybrid_recall5, hybrid_ndcg3, hybrid_ndcg5 = [], [], [], []
    hybrid_mrr, hybrid_precision5 = [], []
    bm25_recall3, bm25_recall5, bm25_ndcg3, bm25_ndcg5 = [], [], [], []
    bm25_mrr, bm25_precision5 = [], []

    for item in generated_queries:
        q = item["query"]
        rel = item["relevant_ids"]

        h_hits = store.retrieve(query=q, top_k=5, user_id="eval-user")
        h_ids = [h.get("id", "") for h in h_hits]
        h_ids = [r for r in h_ids if r in mem_ids]

        b_ids = bm25_store.retrieve_bm25_only(query=q, top_k=5, user_id="eval-user")
        b_ids = [r for r in b_ids if r in mem_ids]

        hr3 = recall_at_k(h_ids, rel, 3)
        hr5 = recall_at_k(h_ids, rel, 5)
        hn3 = ndcg_at_k(h_ids, rel, 3)
        hn5 = ndcg_at_k(h_ids, rel, 5)
        hm = mrr(h_ids, rel)
        hp5 = precision_at_k(h_ids, rel, 5)

        br3 = recall_at_k(b_ids, rel, 3)
        br5 = recall_at_k(b_ids, rel, 5)
        bn3 = ndcg_at_k(b_ids, rel, 3)
        bn5 = ndcg_at_k(b_ids, rel, 5)
        bm = mrr(b_ids, rel)
        bp5 = precision_at_k(b_ids, rel, 5)

        hybrid_recall3.append(hr3); hybrid_recall5.append(hr5)
        hybrid_ndcg3.append(hn3); hybrid_ndcg5.append(hn5)
        hybrid_mrr.append(hm); hybrid_precision5.append(hp5)
        bm25_recall3.append(br3); bm25_recall5.append(br5)
        bm25_ndcg3.append(bn3); bm25_ndcg5.append(bn5)
        bm25_mrr.append(bm); bm25_precision5.append(bp5)

        winner = "HYBRID" if hr3 >= br3 else "BM25"
        label = q[:26]
        print(f"  {label:<28} {'7-factor hybrid':<14} {hr3:>6.1%} {hr5:>6.1%} {hn3:>6.3f} {hn5:>6.3f} {hm:>6.3f} {hp5:>6.1%}")
        print(f"  {'':<28} {'BM25-only':<14} {br3:>6.1%} {br5:>6.1%} {bn3:>6.3f} {bn5:>6.3f} {bm:>6.3f} {bp5:>6.1%}")
        print(f"  {'':<28} {'Winner':<14} {'':<6} {'':<6} {'':<6} {'':<6} {'':<6} {winner:<6}")
        print()

    n = len(generated_queries)

    print(f"  {'='*90}")
    print(f"\n  Summary Comparison (700 memories x 50 queries)")
    print(f"\n  +{'-'*14} {'':>12} {'':>12} {'':>12}+")
    print(f"  |{'Metric':<14} {'BM25-only':>12} {'7-factor hybrid':>12} {'Improvement':>12}|")
    print(f"  +{'-'*14} {'-'*12} {'-'*12} {'-'*12}+")

    def avg(lst): return sum(lst) / len(lst)

    metrics = [
        ("Recall@3", avg(bm25_recall3), avg(hybrid_recall3)),
        ("Recall@5", avg(bm25_recall5), avg(hybrid_recall5)),
        ("NDCG@3",  avg(bm25_ndcg3),  avg(hybrid_ndcg3)),
        ("NDCG@5",  avg(bm25_ndcg5),  avg(hybrid_ndcg5)),
        ("MRR",      avg(bm25_mrr),    avg(hybrid_mrr)),
        ("Precision@5", avg(bm25_precision5), avg(hybrid_precision5)),
    ]
    for name, b, h in metrics:
        diff = h - b
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "=")
        print(f"  |{name:<14} {b:>11.3f}  {h:>11.3f}  {arrow}{abs(diff):>10.3f}  |")

    print(f"  +{'-'*14} {'-'*12} {'-'*12} {'-'*12}+")

    hybrid_wins = sum(1 for i in range(n) if hybrid_recall3[i] >= bm25_recall3[i])
    print(f"\n  Recall@3 winner stats: 7-factor hybrid {hybrid_wins}/50, BM25-only {n - hybrid_wins}/50")

    results = {
        "memory_count": len(memories),
        "query_count": n,
        "bm25_only": {
            "recall_at_3": round(avg(bm25_recall3), 4),
            "recall_at_5": round(avg(bm25_recall5), 4),
            "ndcg_at_3": round(avg(bm25_ndcg3), 4),
            "ndcg_at_5": round(avg(bm25_ndcg5), 4),
            "mrr": round(avg(bm25_mrr), 4),
            "precision_at_5": round(avg(bm25_precision5), 4),
        },
        "hybrid_7factor": {
            "recall_at_3": round(avg(hybrid_recall3), 4),
            "recall_at_5": round(avg(hybrid_recall5), 4),
            "ndcg_at_3": round(avg(hybrid_ndcg3), 4),
            "ndcg_at_5": round(avg(hybrid_ndcg5), 4),
            "mrr": round(avg(hybrid_mrr), 4),
            "precision_at_5": round(avg(hybrid_precision5), 4),
        },
    }
    out_file = Path(__file__).parent / "retrieval_700_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {out_file}")


if __name__ == "__main__":
    run()
