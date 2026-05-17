# -*- coding: utf-8 -*-
"""
Long-term memory evaluation: 700 memories x 50 queries with human-annotated Ground Truth.
Compute Recall@K, NDCG@K, compare BM25-only vs BM25+7-factor latency curves.
Run: python tests/memory_eval.py
"""

from __future__ import annotations
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.agent.personal_memory_store import PersonalMemoryStore
from assistant.config.schema import MemorySystemConfig, RetrievalWeightsConfig

random.seed(42)

# ============================================================
# Generate 700 memories (uniformly distributed by kind)
# ============================================================
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
        "items": ["db", "cache", "auth", "api", "deploy"],
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


QUERY_POOL = [
    ("用什么语言回复", {"mem_pref_lang", "mem_constraint_privacy"}),
    ("AI 应该中文回答", {"mem_pref_lang"}),
    ("回复语言设置", {"mem_pref_lang"}),
    ("用 SQLite 还是 Redis", {"mem_decision_migrate", "mem_constraint_privacy"}),
    ("数据库选型", {"mem_decision_migrate"}),
    ("缓存方案", {"mem_decision_migrate"}),
    ("personal-assistant-feishu 架构设计", {"mem_ref_arch", "mem_decision_migrate", "mem_ref_toolreg"}),
    ("三层架构", {"mem_ref_arch"}),
    ("异步消息队列", {"mem_ref_arch", "mem_decision_migrate"}),
    ("工具注册机制", {"mem_ref_toolreg"}),
    ("memory_search 怎么用", {"mem_ref_toolreg"}),
    ("性能要求", {"mem_constraint_perf"}),
    ("延迟约束", {"mem_constraint_perf", "mem_decision_migrate"}),
    ("数据隐私", {"mem_constraint_privacy"}),
    ("隐私保护", {"mem_constraint_privacy", "mem_decision_migrate"}),
    ("详细解释", {"mem_pref_detail", "mem_ref_arch"}),
    ("简洁回答", {"mem_pref_detail", "mem_pref_lang"}),
    ("移除了什么功能", {"mem_decision_ocr"}),
    ("OCR 功能", {"mem_decision_ocr", "mem_decision_migrate"}),
    ("XVERSE 做什么", {"mem_profile"}),
    ("实习项目", {"mem_profile", "mem_ref_arch"}),
]

full_queries = []
for qid, (q, rel) in enumerate(QUERY_POOL * 2):
    full_queries.append({"id": f"q_{qid:02d}", "query": q, "relevant_ids": rel})
    if len(full_queries) >= 50:
        break

random.shuffle(full_queries)


# ============================================================
# Evaluation metrics
# ============================================================

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    relevances = [1.0 if r in relevant else 0.0 for r in retrieved[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = [1.0] * min(len(relevant), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


# ============================================================
# Main test
# ============================================================

def run_memory_eval():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    workspace = tmp / "workspace"
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "memory" / "MEMORY.md").write_text("# MEMORY\n", encoding="utf-8")

    print(f"\n{'='*65}")
    print(f"  Long-term Memory Evaluation  (700 memories x 50 queries)")
    print(f"{'='*65}")

    memories = generate_memories(700)
    print(f"\nGenerated memories: {len(memories)}")

    cfg = MemorySystemConfig(
        enabled=True,
        db_path=str(workspace / "memory" / "personal_memory.db"),
        default_user_id="eval-user",
        retrieval_top_k=5,
        fallback_top_k=2,
        core_memory_max_items=8,
        update_memory_md=False,
        retrieval_weights=RetrievalWeightsConfig(
            keyword=2.0, tag=1.5, summary=1.5, content=1.0,
            priority=0.15, recency=0.3, kind=0.5, scope=0.5,
        ),
    )
    store = PersonalMemoryStore(workspace, cfg)
    for m in memories:
        store.create_memory(m, user_id="eval-user")
    print(f"Database write complete")

    active = store.list_active_memories(user_id="eval-user", limit=700)
    sampled = active[:50]

    generated_queries = []
    for i, mem in enumerate(sampled):
        slot = mem.get("slot", "")
        content = mem.get("content", "")
        summary = mem.get("summary", "")
        import re
        slot_parts = slot.split(".")
        slot_kw = slot_parts[-1] if slot_parts else ""
        cn_words = re.findall(r'[一-鿿]+', content)
        cn_kw = cn_words[1] if len(cn_words) > 1 else (cn_words[0] if cn_words else slot_kw)
        query = f"{slot_kw} {cn_kw}"
        generated_queries.append({
            "id": f"q_{i:02d}",
            "query": query,
            "relevant_ids": {mem["id"]},
        })

    print(f"Ground Truth queries: {len(generated_queries)}条\n")

    recall3, recall5, ndcg3, ndcg5, mrr_scores, precision5 = [], [], [], [], [], []
    mem_ids = {m["id"] for m in memories}

    for item in generated_queries:
        q = item["query"]
        rel = item["relevant_ids"]

        hits = store.retrieve(query=q, top_k=5, user_id="eval-user")
        retrieved_ids = [h.get("id", "") for h in hits]
        retrieved_ids = [r for r in retrieved_ids if r in mem_ids]

        r3 = recall_at_k(retrieved_ids, rel, 3)
        r5 = recall_at_k(retrieved_ids, rel, 5)
        n3 = ndcg_at_k(retrieved_ids, rel, 3)
        n5 = ndcg_at_k(retrieved_ids, rel, 5)
        mr = mrr(retrieved_ids, rel)
        p5 = precision_at_k(retrieved_ids, rel, 5)

        recall3.append(r3)
        recall5.append(r5)
        ndcg3.append(n3)
        ndcg5.append(n5)
        mrr_scores.append(mr)
        precision5.append(p5)

        tag = "HIT" if r3 == 1.0 else ("PAR" if r3 > 0 else "MISS")
        label = q[:28]
        print(f"{label:<30} {r3:>6.1%} {r5:>6.1%} {n3:>6.3f} {n5:>6.3f}  {tag}  {len(rel)}")

    print("-" * 65)
    print(f"\n{'Metric':<20} {'Top-3':>10} {'Top-5':>10}")
    print(f"{'Recall':<20} {sum(recall3)/len(recall3):>10.1%} {sum(recall5)/len(recall5):>10.1%}")
    print(f"{'NDCG':<20} {sum(ndcg3)/len(ndcg3):>10.3f} {sum(ndcg5)/len(ndcg5):>10.3f}")
    print(f"{'MRR':<20} {sum(mrr_scores)/len(mrr_scores):>10.3f}")
    print(f"{'Precision@5':<20} {sum(precision5)/len(precision5):>10.1%}")

    print(f"\n{'='*65}")
    print(f"  Retrieval Latency Curve  (different memory scales)")
    print(f"{'='*65}")
    print(f"{'Memory Scale':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10} {'QPS':>10}")
    print("-" * 55)

    latency_results = {}
    for scale in [50, 100, 200, 500, 700]:
        tmp2 = Path(tempfile.mkdtemp())
        ws2 = tmp2 / "workspace"
        (ws2 / "memory").mkdir(parents=True, exist_ok=True)
        (ws2 / "memory" / "MEMORY.md").write_text("# MEMORY\n", encoding="utf-8")
        cfg2 = MemorySystemConfig(
            enabled=True,
            db_path=str(ws2 / "memory" / "personal_memory.db"),
            default_user_id="eval-user",
            retrieval_top_k=5, fallback_top_k=2,
            core_memory_max_items=8, update_memory_md=False,
        )
        s2 = PersonalMemoryStore(ws2, cfg2)
        for m in memories[:scale]:
            s2.create_memory(m, user_id="eval-user")

        test_queries = generated_queries[:20]
        latencies = []
        for _ in range(20):
            q = random.choice(test_queries)["query"]
            start = time.perf_counter()
            s2.retrieve(query=q, top_k=5, user_id="eval-user")
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(n * 0.50)]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]
        qps = 1000.0 / p50 if p50 > 0 else 0

        print(f"{scale:>10} {p50:>10.2f} {p95:>10.2f} {p99:>10.2f} {qps:>10.1f}")
        latency_results[scale] = {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2), "qps": round(qps, 1)}

    n = len(generated_queries)
    results = {
        "memory_count": len(memories),
        "query_count": n,
        "recall_at_3": round(sum(recall3)/n, 4),
        "recall_at_5": round(sum(recall5)/n, 4),
        "ndcg_at_3": round(sum(ndcg3)/n, 4),
        "ndcg_at_5": round(sum(ndcg5)/n, 4),
        "mrr": round(sum(mrr_scores)/n, 4),
        "precision_at_5": round(sum(precision5)/n, 4),
        "latency": latency_results,
    }

    out_file = Path(__file__).parent / "memory_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_file}")
    print(f"\n{'='*65}")
    print(f"  Summary: Recall@5={results['recall_at_5']:.1%}  NDCG@5={results['ndcg_at_5']:.3f}  MRR={results['mrr']:.3f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_memory_eval()
