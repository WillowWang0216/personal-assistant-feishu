"""personal-assistant Memory Management UI (Streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from assistant.agent.personal_memory_store import PersonalMemoryStore
from assistant.agent.memory_retriever import MemoryRetriever
from assistant.config.loader import load_config

st.set_page_config(
    page_title="personal-assistant Memory",
    page_icon="🐈",
    layout="wide",
)


def init_session_state():
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "store" not in st.session_state:
        ws = st.session_state.config.workspace_path
        mc = st.session_state.config.tools.memory_system
        st.session_state.store = PersonalMemoryStore(ws, mc)

    if "retriever" not in st.session_state:
        ws = st.session_state.config.workspace_path
        mc = st.session_state.config.tools.memory_system
        st.session_state.retriever = MemoryRetriever(ws, mc)


def render_memory_table(memories, store=None, key_suffix=""):
    if not memories:
        st.info("暂无记忆")
        return

    for mem in memories:
        kind = mem.get("kind", "unknown")
        slot = mem.get("slot", "")
        summary = mem.get("summary", mem.get("content", ""))
        scope = mem.get("scope", "")
        priority = mem.get("priority", 0)
        updated = mem.get("updated_at", "")
        mem_id = mem.get("id", "")

        color = {
            "constraint": "🔴",
            "preference": "🟢",
            "decision": "🟡",
            "reference": "🔵",
            "profile": "⚪",
        }.get(kind, "⚪")

        with st.expander(f"{color} [{kind}] {slot}"):
            st.markdown(f"**摘要:** {summary}")
            st.markdown(f"**内容:** {mem.get('content', '')}")
            cols = st.columns(3)
            cols[0].metric("优先级", priority)
            cols[1].metric("作用域", scope)
            cols[2].metric("更新时间", updated[:10] if updated else "N/A")

            tags = mem.get("tags", [])
            keywords = mem.get("keywords", [])
            if tags:
                st.markdown(f"**标签:** {', '.join(tags)}")
            if keywords:
                st.markdown(f"**关键词:** {', '.join(keywords)}")

            if store and mem_id:
                col1, col2 = st.columns(2)
                if col1.button("🗑️ 删除", key=f"del_{mem_id}_{key_suffix}"):
                    store.archive_memory(mem_id)
                    st.success("记忆已归档")
                    st.rerun()
                if col2.button("⭐ 提升优先级", key=f"promote_{mem_id}_{key_suffix}"):
                    new_priority = min(10, priority + 1)
                    store.update_memory(mem_id, {"priority": new_priority})
                    st.success(f"优先级提升为 {new_priority}")
                    st.rerun()


st.title("🐈 personal-assistant 记忆管理")

init_session_state()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 全部记忆", "➕ 新增记忆", "🔍 检索测试", "📊 统计信息", "⚙️ 配置"])

with tab1:
    st.subheader("全部活跃记忆")
    if st.button("刷新", type="secondary"):
        st.rerun()

    memories = st.session_state.store.list_active_memories(limit=500)
    st.info(f"共 {len(memories)} 条记忆")
    render_memory_table(memories, store=st.session_state.store, key_suffix="all")

with tab2:
    st.subheader("手动新增记忆")
    st.caption("直接写入 PersonalMemoryStore，不经过 LLM 提取")

    col1, col2 = st.columns(2)
    kind = col1.selectbox("记忆类型", ["preference", "decision", "constraint", "reference", "profile"])
    scope = col2.selectbox("作用域", ["global", "topic", "project"])

    content = st.text_area("正文内容", placeholder="输入记忆的具体内容...", height=100)
    summary = st.text_input("摘要（可不填，自动取正文前50字）", placeholder="一句话概括这条记忆")
    priority = st.slider("优先级", 0, 10, 5)

    tags_raw = st.text_input("标签（逗号分隔，可不填）", placeholder="例如: 沟通风格, 开发偏好")
    keywords_raw = st.text_input("关键词（逗号分隔，可不填）", placeholder="例如: 简洁, 短")

    if st.button("✅ 写入记忆", type="primary"):
        if not content.strip():
            st.error("正文内容不能为空")
        else:
            import uuid
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
            auto_summary = summary.strip() or content[:50]

            kind_lower = kind.lower()
            namespace = "manual"
            key = (auto_summary or content[:40]).lower().replace(" ", "_")[:40]
            slot = f"{kind_lower}.{namespace}.{key}"

            memory = {
                "id": f"mem_{uuid.uuid4().hex[:12]}",
                "kind": kind_lower,
                "scope": scope,
                "scope_key": namespace,
                "slot": slot,
                "content": content.strip(),
                "summary": auto_summary,
                "tags": tags,
                "keywords": keywords,
                "priority": priority,
                "source_refs": ["ui:manual"],
            }

            st.session_state.store.create_memory(memory)
            st.success(f"记忆已写入: {slot}")
            st.rerun()

with tab3:
    st.subheader("记忆检索测试")
    query = st.text_area("查询语句", placeholder="输入查询词，回车确认...")
    col1, col2 = st.columns([1, 1])
    top_k = col1.slider("返回数量", 1, 20, 5)
    user_id = col2.text_input("用户ID", value="shared")

    if query:
        results = st.session_state.retriever.retrieve_for_prompt(
            user_text=query,
            session_state=None,
            recent_messages=[],
            user_id=user_id,
        )
        results = results[:top_k]
        st.info(f"检索到 {len(results)} 条记忆")
        render_memory_table(results, store=None, key_suffix="search")

with tab4:
    st.subheader("记忆库统计")
    stats = st.session_state.store.get_stats(user_id="shared")

    cols = st.columns(4)
    cols[0].metric("活跃记忆", stats.get("active", 0))
    cols[1].metric("已替代", stats.get("superseded", 0))
    cols[2].metric("待合并候选", stats.get("candidates_unmerged", 0))
    cols[3].metric("历史事件", stats.get("events_total", 0))

    st.divider()
    st.subheader("最近事件")
    events = st.session_state.store.list_recent_events(limit=10)
    for evt in events:
        action = evt.get("action", "")
        reason = evt.get("reason", "")
        created = evt.get("created_at", "")[:19]
        st.markdown(f"- **{action}** | {reason} | {created}")

with tab5:
    st.subheader("当前配置")
    mc = st.session_state.config.tools.memory_system
    st.json({
        "enabled": mc.enabled,
        "db_path": str(mc.db_path),
        "retrieval_top_k": mc.retrieval_top_k,
        "core_memory_max_items": mc.core_memory_max_items,
        "max_candidates_per_run": mc.max_candidates_per_run,
    })

    st.divider()
    st.subheader("检索权重")
    rw = mc.retrieval_weights
    st.json({
        "keyword (BM25)": rw.keyword,
        "tag": rw.tag,
        "priority": rw.priority,
        "recency": rw.recency,
        "kind": rw.kind,
        "scope": rw.scope,
    })
