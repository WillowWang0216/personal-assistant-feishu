"""Personal memory search tool — lets the agent actively query the long-term memory store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from assistant.agent.tools.base import Tool
from assistant.agent.personal_memory_store import PersonalMemoryStore
from assistant.config.schema import MemorySystemConfig


class MemorySearchTool(Tool):
    """
    Search the personal long-term memory database.

    Used when the agent needs to actively recall facts, preferences, decisions,
    and project states that may not appear in auto-injected memory context.
    """

    def __init__(self, workspace: Path, config: MemorySystemConfig) -> None:
        self._store = PersonalMemoryStore(workspace, config)
        self._config = config

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search the personal long-term memory database for stored facts, "
            "preferences, decisions, and project notes. "
            "Use this when the auto-injected memory context is insufficient or "
            "when you need to actively look up specific topics. "
            "Returns ranked results with slot, summary, content, and metadata."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query or keywords describing what you want to recall. "
                        "Example: 'PLM_Sol training hyperparameters', 'CoA schema layers', "
                        "'用户偏好语言'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 20).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "Optional scope filter: 'global' | 'topic' | 'project'. "
                        "Leave empty to search across all scopes."
                    ),
                },
                "scope_key": {
                    "type": "string",
                    "description": (
                        "Optional scope key for finer filtering when scope='topic' or 'project'. "
                        "Example: 'plm-sol', 'personal-assistant-memory', 'coa'."
                    ),
                },
                "kind": {
                    "type": "string",
                    "description": (
                        "Optional kind filter: 'preference' | 'decision' | 'reference' | "
                        "'constraint' | 'profile'. Leave empty for all kinds."
                    ),
                },
                "slot_prefix": {
                    "type": "string",
                    "description": (
                        "Optional slot prefix filter. Returns only memories whose slot starts "
                        "with this string. Example: 'project.plm_sol', 'decision.coa'."
                    ),
                },
            },
            "required": ["query"],
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        scope: str | None = None,
        scope_key: str | None = None,
        kind: str | None = None,
        slot_prefix: str | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        top_k = max(1, min(20, int(top_k)))

        scope_hints: dict[str, Any] = {}
        if scope:
            scope_hints["scope"] = scope
        if scope_key:
            scope_hints["scope_key"] = scope_key

        # Isolate by user_id (group chat=chat_id, private chat=sender_id)
        results = self._store.retrieve(
            query=query,
            scope_hints=scope_hints,
            top_k=top_k,
            user_id=user_id,
        )

        # Detect fallback retrieval (normal results identical to empty-query results -> core memory hit)
        is_fallback = False
        if results:
            normal_ids = {r.get("id") for r in results}
            fallback_check = self._store.retrieve(query="", scope_hints={}, top_k=top_k, user_id=user_id)
            fallback_ids = {r.get("id") for r in fallback_check}
            if normal_ids and normal_ids == fallback_ids:
                is_fallback = True

        # Post-filter: kind / slot_prefix (not natively supported by store.retrieve)
        if kind:
            results = [r for r in results if r.get("kind") == kind]
        if slot_prefix:
            results = [r for r in results if (r.get("slot") or "").startswith(slot_prefix)]

        if not results:
            return (
                f"No memories found for query: '{query}'"
                + (f" (scope={scope}" + (f", scope_key={scope_key}" if scope_key else "") + ")" if scope else "")
                + (f" (kind={kind})" if kind else "")
                + (f" (slot_prefix={slot_prefix})" if slot_prefix else "")
                + "."
            )

        # Mark as used, affects recency scoring
        self._store.mark_used([r["id"] for r in results if r.get("id")])

        lines: list[str] = [
            f"{'⚠️ No direct match found. Showing fallback (core) memories instead:' if is_fallback else ''}"
            f"Found {len(results)} memory result(s) for query: '{query}'\n"
        ]
        for i, item in enumerate(results, 1):
            slot = item.get("slot") or item.get("kind") or "—"
            summary = item.get("summary") or ""
            content = item.get("content") or ""
            kind_val = item.get("kind") or ""
            scope_val = item.get("scope") or ""
            scope_key_val = item.get("scope_key") or ""
            updated_at = (item.get("updated_at") or "")[:10]
            evidence = item.get("evidence_count") or 1

            lines.append(f"### [{i}] {slot}")
            lines.append(f"- **kind**: {kind_val}  **scope**: {scope_val}" + (f"/{scope_key_val}" if scope_key_val else ""))
            lines.append(f"- **updated**: {updated_at}  **evidence**: {evidence}")
            lines.append(f"- **summary**: {summary}")
            if content and content.strip() != summary.strip():
                content_preview = content[:400] + ("…" if len(content) > 400 else "")
                lines.append(f"- **content**: {content_preview}")
            lines.append("")

        return "\n".join(lines)
