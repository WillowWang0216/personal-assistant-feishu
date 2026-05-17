# -*- coding: utf-8 -*-
"""
Vector search evaluation: BM25-only vs 7-factor hybrid vs vector semantic retrieval (700 memories x 50 queries).
Run: python tests/retrieval_vector_eval.py
"""

from __future__ import annotations
import json
import math
import random
import re
import sqlite3
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.agent.personal_memory_store import PersonalMemoryStore
from assistant.config.schema import MemorySystemConfig, RetrievalWeightsConfig

random.seed(42)

KIND_TEMPLATES = {
    "preference": {
        "slots": ["{domain}.reply.language","{domain}.reply.style","{domain}.detail.level",
                   "{domain}.interest.{topic}","{domain}.tool.{name}"],
        "domains": ["user","dev","prod"],
        "topics": ["python","llm","agent","web","data"],
        "contents": ["用户偏好用{lang}回答","回复风格要求{style}",
            "喜欢{level}级别的解释","对{topic}领域特别感兴趣","经常使用{tool}工具"],
    },
    "decision": {
        "slots": ["{domain}.decision.{item}","{domain}.stack.{name}"],
        "domains": ["proj","arch","infra"],
        "items": ["sqlite","redis","postgresql","mongodb"],
        "contents": ["决定使用{item}方案","技术栈选择{item}",
            "架构决策：采用{item}架构","部署方案确定：{item}"],
    },
    "reference": {
        "slots": ["{domain}.ref.{name}","{domain}.doc.{name}"],
        "domains": ["lib","api","sdk"],
        "names": ["openai","feishu","sqlite","redis","jieba","httpx"],
        "contents": ["参考文档：{name}官方文档","API参考：{name}接口说明",
            "SDK使用：{name}集成方法","第三方库：{name}使用指南"],
    },
    "constraint": {
        "slots": ["{domain}.constraint.{name}","{domain}.limit.{name}"],
        "domains": ["perf","sec","cost"],
        "names": ["latency","privacy","budget","scale"],
        "contents": ["性能约束：{name}必须满足{val}","安全约束：{name}相关要求","成本约束：{name}控制在{val}"],
    },
    "profile": {
        "slots": ["{domain}.profile.{name}","{domain}.role.{name}"],
        "domains": ["self","team","org"],
        "names": ["role","skill","level","company"],
        "contents": ["用户角色：{name}","技能背景：{name}相关经验","所属团队：{name}"],
    },
}
LANGS=["中文","英文","中英双语"]
STYLES=["简洁","详细","结构化","口语化"]
LEVELS=["入门","进阶","高级"]
TOOLS=["web_search","memory_search","read_file","write_file"]
ITEMS=["sqlite","redis","postgresql","mongodb"]
VALS=["<100ms","<1s","<5s","<10s","5%"]
POOL_3=["python","llm","agent","web","db","api","dev","test","perf","sec"]
POOL_2=["python","llm","agent","web","db","api","dev","test"]

def generate_memories(n=700):
    memories=[]
    kinds=["preference","decision","reference","constraint","profile"]
    priorities=list(range(1,11))
    slots_seen=set()
    while len(memories)<n:
        kind=random.choice(kinds)
        tpl=KIND_TEMPLATES[kind]
        slot_tpl=random.choice(tpl["slots"])
        domain=random.choice(tpl.get("domains",["d1"]))
        slot=slot_tpl.format(
            domain=domain, item=random.choice(tpl.get("items",["item"])),
            name=random.choice(tpl.get("names",["name"])),
            topic=random.choice(tpl.get("topics",["topic"])),
            tool=random.choice(tpl.get("tools",["tool"])),
        )
        suffix=len([s for s in slots_seen if s.startswith(slot)])
        if suffix: slot=f"{slot}.{suffix}"
        slots_seen.add(slot)
        ct=random.choice(tpl["contents"])
        content=ct.format(
            lang=random.choice(LANGS), style=random.choice(STYLES),
            level=random.choice(LEVELS), topic=random.choice(tpl.get("topics",["topic"])),
            tool=random.choice(TOOLS), item=random.choice(ITEMS),
            name=random.choice(tpl.get("names",["name"])), val=random.choice(VALS),
        )
        memories.append({
            "id":f"mem_{len(memories):04d}","kind":kind,"scope":"global",
            "slot":slot,"summary":content[:50],"content":content,
            "tags":random.sample(POOL_3,3),"keywords":random.sample(POOL_2,2),
            "priority":random.choice(priorities),
        })
    return memories

def recall_at_k(r, rel, k):
    return len(set(r[:k])&rel)/len(rel) if rel else 0.0

def ndcg_at_k(r, rel, k):
    import math
    rels=[1.0 if x in rel else 0.0 for x in r[:k]]
    dcg=sum(rel/math.log2(i+2) for i,rel in enumerate(rels))
    idcg=sum(1/math.log2(i+2) for i in range(min(len(rel),k)))
    return dcg/idcg if idcg>0 else 0.0

def mrr(r, rel):
    for i,x in enumerate(r,1):
        if x in rel: return 1.0/i
    return 0.0

def precision_at_k(r, rel, k):
    return len(set(r[:k])&rel)/k if k else 0.0

class VectorRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY, embedding BLOB NOT NULL)""")
            conn.commit()
        print("  Loading embedding model...")
        t0=time.time()
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print(f"  Model loaded ({time.time()-t0:.1f}s)")

    def encode_memories(self, memories):
        texts=[f"{m.get('summary','')} {m.get('content','')}" for m in memories]
        print(f"  Encoding {len(texts)} memory vectors...")
        t0=time.time()
        embeddings=self.model.encode(texts,show_progress_bar=False,batch_size=64)
        print(f"  Encoding done ({time.time()-t0:.1f}s, {embeddings.shape})")
        with sqlite3.connect(self.db_path) as conn:
            for m,emb in zip(memories,embeddings):
                conn.execute("INSERT OR REPLACE INTO memory_embeddings VALUES (?,?)",
                    (m["id"], emb.astype("float32").tobytes()))
            conn.commit()

    def retrieve(self, query: str, top_k: int=5) -> list[str]:
        import numpy as np
        q_emb=self.model.encode([query],show_progress_bar=False)[0]
        q_norm=np.linalg.norm(q_emb)
        if q_norm<1e-8: return []
        with sqlite3.connect(self.db_path) as conn:
            rows=conn.execute("SELECT memory_id,embedding FROM memory_embeddings").fetchall()
        scores=[]
        for mid,emb_bytes in rows:
            emb=np.frombuffer(emb_bytes,dtype="float32")
            sim=float(np.dot(q_emb,emb)/(np.linalg.norm(emb)*q_norm+1e-8))
            scores.append((sim,mid))
        scores.sort(key=lambda x:x[0],reverse=True)
        return [mid for _,mid in scores[:top_k]]

def run():
    tmp=Path(tempfile.mkdtemp())
    ws=tmp/"workspace"
    (ws/"memory").mkdir(parents=True)
    (ws/"memory"/"MEMORY.md").write_text("# MEMORY\n",encoding="utf-8")
    db_path=str(ws/"memory"/"personal_memory.db")

    memories=generate_memories(700)
    print(f"Generated memories: {len(memories)}")

    cfg=MemorySystemConfig(
        enabled=True, db_path=db_path, default_user_id="eval-user",
        retrieval_top_k=5, fallback_top_k=2, core_memory_max_items=8,
        update_memory_md=False,
        retrieval_weights=RetrievalWeightsConfig(
            keyword=2.0,tag=1.5,summary=1.5,content=1.0,
            priority=0.15,recency=0.3,kind=0.5,scope=0.5),
    )
    store=PersonalMemoryStore(ws,cfg)
    for m in memories: store.create_memory(m,user_id="eval-user")
    print("Database write complete")

    active=store.list_active_memories(user_id="eval-user",limit=700)
    print(f"Verified read: {len(active)} memories")

    vr=VectorRetriever(db_path)
    vr.encode_memories(memories)

    sampled=active[:50]
    queries=[]
    for i,mem in enumerate(sampled):
        slot_parts=mem.get("slot","").split(".")
        slot_kw=slot_parts[-1] if slot_parts else ""
        content=mem.get("content","")
        cn_words=re.findall(r'[一-鿿]+',content)
        cn_kw=cn_words[1] if len(cn_words)>1 else(cn_words[0] if cn_words else slot_kw)
        query=f"{slot_kw} {cn_kw}"
        queries.append({"id":f"q_{i:02d}","query":query,"relevant_ids":{mem["id"]}})

    print(f"Ground Truth queries: {len(queries)}条\n")

    mem_ids={m["id"] for m in memories}
    bm25_r3,b25_r5,b25_n3,b25_n5,b25_m,b25_p5=[],[],[],[],[],[]
    hy_r3,hy_r5,hy_n3,hy_n5,hy_m,hy_p5=[],[],[],[],[],[]
    v_r3,v_r5,v_n3,v_n5,v_m,v_p5=[],[],[],[],[],[]

    print(f"  {'Query':<28} {'BM25@3':>8} {'7-Factor@3':>12} {'Vector@3':>10}  {'Winner'}")
    print(f"  {'-'*65}")

    for item in queries:
        q,rel=item["query"],item["relevant_ids"]

        q_tokens=store._tokenize(q.lower())
        store._ensure_idf_ready()
        scored=[]
        for mem in store.list_active_memories(user_id="eval-user",limit=500):
            s=(mem.get("summary") or"").lower()
            c=(mem.get("content") or"").lower()
            sc=store._bm25_score(q_tokens,store._tokenize(s),store._tokenize(c),len(s)+len(c))
            scored.append((sc,mem["id"]))
        scored.sort(key=lambda x:x[0],reverse=True)
        b_ids=[mid for sc,mid in scored if sc>0][:5]
        b_ids=[r for r in b_ids if r in mem_ids]

        h_hits=store.retrieve(query=q, top_k=5, user_id="eval-user")
        h_ids=[h.get("id","") for h in h_hits]
        h_ids=[r for r in h_ids if r in mem_ids]

        v_ids=vr.retrieve(query=q, top_k=5)
        v_ids=[r for r in v_ids if r in mem_ids]

        def c(ids):
            return (recall_at_k(ids,rel,3),recall_at_k(ids,rel,5),
                    ndcg_at_k(ids,rel,3),ndcg_at_k(ids,rel,5),
                    mrr(ids,rel),precision_at_k(ids,rel,5))

        br3,br5,bn3,bn5,bm,bp=c(b_ids)
        hr3,hr5,hn3,hn5,hm,hp=c(h_ids)
        vr3,vr5,vn3,vn5,vm,vp=c(v_ids)

        bm25_r3.append(br3);b25_r5.append(br5);b25_n3.append(bn3);b25_n5.append(bn5);b25_m.append(bm);b25_p5.append(bp)
        hy_r3.append(hr3);hy_r5.append(hr5);hy_n3.append(hn3);hy_n5.append(hn5);hy_m.append(hm);hy_p5.append(hp)
        v_r3.append(vr3);v_r5.append(vr5);v_n3.append(vn3);v_n5.append(vn5);v_m.append(vm);v_p5.append(vp)

        winner="Vector" if vr3>=hr3 and vr3>=br3 else("7-Factor" if hr3>=br3 else "BM25")
        label=q[:26]
        print(f"  {label:<28} {br3:>7.0%} {hr3:>11.0%} {vr3:>9.0%}  {winner}")

    n=len(queries)
    av=lambda l:sum(l)/len(l)

    print(f"\n{'='*65}")
    print(f"  Summary Comparison (700 memories x 50 queries)")
    print(f"\n  +{'-'*14} {'BM25-only':>12} {'7-Factor':>12} {'Vector':>12}")
    print(f"  +{'-'*14} {'-'*12} {'-'*12} {'-'*12}")

    rows=[
        ("Recall@3",   av(bm25_r3),av(hy_r3),av(v_r3)),
        ("Recall@5",   av(b25_r5),av(hy_r5),av(v_r5)),
        ("NDCG@3",     av(b25_n3),av(hy_n3),av(v_n3)),
        ("NDCG@5",     av(b25_n5),av(hy_n5),av(v_n5)),
        ("MRR",        av(b25_m), av(hy_m), av(v_m)),
        ("Precision@5",av(b25_p5),av(hy_p5),av(v_p5)),
    ]
    for name,b,h,v in rows:
        best=max(b,h,v)
        def arrow(x): return "▲" if x==best else " "
        print(f"  |{name:<14} {b:>10.3f}{arrow(b)} {h:>10.3f}{arrow(h)} {v:>10.3f}{arrow(v)} |")

    print(f"  +{'-'*14} {'-'*12} {'-'*12} {'-'*12}")

    v_win=sum(1 for i in range(n) if v_r3[i]>=hy_r3[i] and v_r3[i]>=bm25_r3[i])
    hy_win=sum(1 for i in range(n) if hy_r3[i]>=v_r3[i] and hy_r3[i]>=bm25_r3[i])
    tie=n-v_win-hy_win
    print(f"\n  Recall@3 Winner Stats:")
    print(f"    Vector wins:   {v_win}/50 ({v_win/n:.0%})")
    print(f"    7-Factor wins: {hy_win}/50 ({hy_win/n:.0%})")
    print(f"    Tied:          {tie}/50 ({tie/n:.0%})")

    print(f"\n  Latency Comparison (20 queries average):")
    import time as time_module
    q_sample=[q["query"] for q in queries[:20]]

    def time_bm25():
        store._ensure_idf_ready()
        t0=time_module.perf_counter()
        for q in q_sample:
            q_tokens=store._tokenize(q.lower())
            scored=[]
            for mem in store.list_active_memories(user_id="eval-user",limit=500):
                s=(mem.get("summary") or"").lower();c=(mem.get("content") or"").lower()
                sc=store._bm25_score(q_tokens,store._tokenize(s),store._tokenize(c),len(s)+len(c))
                scored.append((sc,mem["id"]))
            scored.sort(key=lambda x:x[0],reverse=True)
        return (time_module.perf_counter()-t0)/len(q_sample)*1000

    def time_hybrid():
        t0=time_module.perf_counter()
        for q in q_sample: store.retrieve(query=q,top_k=5,user_id="eval-user")
        return (time_module.perf_counter()-t0)/len(q_sample)*1000

    def time_vector():
        t0=time_module.perf_counter()
        for q in q_sample: vr.retrieve(query=q,top_k=5)
        return (time_module.perf_counter()-t0)/len(q_sample)*1000

    tb=sum(time_bm25() for _ in range(3))/3
    th=sum(time_hybrid() for _ in range(3))/3
    tv=sum(time_vector() for _ in range(3))/3
    print(f"    BM25-only:   {tb:.1f}ms/query")
    print(f"    7-Factor:    {th:.1f}ms/query")
    print(f"    Vector:      {tv:.1f}ms/query  (embedding model: paraphrase-multilingual-MiniLM-L12-v2)")

    results={
        "memory_count":len(memories),"query_count":n,
        "bm25_only":    {k:round(v,4) for k,v in zip(["r3","r5","n3","n5","mrr","p5"],
                          [av(bm25_r3),av(b25_r5),av(b25_n3),av(b25_n5),av(b25_m),av(b25_p5)])},
        "hybrid_7factor":{k:round(v,4) for k,v in zip(["r3","r5","n3","n5","mrr","p5"],
                          [av(hy_r3),av(hy_r5),av(hy_n3),av(hy_n5),av(hy_m),av(hy_p5)])},
        "vector":        {k:round(v,4) for k,v in zip(["r3","r5","n3","n5","mrr","p5"],
                          [av(v_r3),av(v_r5),av(v_n3),av(v_n5),av(v_m),av(v_p5)])},
        "latency_ms":   {"bm25":round(tb,2),"hybrid":round(th,2),"vector":round(tv,2)},
    }
    out_file=Path(__file__).parent/"retrieval_vector_results.json"
    with open(out_file,"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"\n  Results saved: {out_file}")


if __name__=="__main__":
    run()
