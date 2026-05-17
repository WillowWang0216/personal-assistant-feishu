"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant.bus.events import InboundMessage, OutboundMessage
from assistant.bus.queue import MessageBus
from assistant.providers.base import LLMProvider
from assistant.agent.context import ContextBuilder
from assistant.agent.memory_compiler import MemoryCompiler
from assistant.agent.tools.registry import ToolRegistry
from assistant.agent.tools.filesystem import (
    AppendFileTool,
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from assistant.agent.tools.shell import ExecTool
from assistant.agent.tools.web import WebSearchTool, WebFetchTool
from assistant.agent.tools.message import MessageTool
from assistant.agent.tools.image_generate import ImageGenerateTool
from assistant.agent.tools.notion import NotionTool
from assistant.agent.tools.spawn import SpawnTool
from assistant.agent.tools.session_manage import SessionManageTool
from assistant.agent.tools.cron import CronTool
from assistant.agent.tools.memory_search import MemorySearchTool
from assistant.agent.tools.meeting_summary import MeetingSummaryTool
from assistant.agent.subagent import SubagentManager
from assistant.session.compressor import SessionContextCompressor
from assistant.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from assistant.config.schema import (
        ExecToolConfig,
        FeishuConfig,
        IflytekConfig,
        ImageGenConfig,
        ContextCompressionConfig,
        MemorySystemConfig,
        MineruConfig,
        NotionToolConfig,
        ToolHistoryConfig,
        WebSearchConfig,
    )
    from assistant.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_tokens: int = 4096,
        context_window_tokens: int | None = None,
        token_budget_mode: str = "output",
        merge_subagent_usage: bool = True,
        max_iterations: int = 30,
        reasoning_effort: str | None = None,
        web_search_config: WebSearchConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        mineru_config: MineruConfig | None = None,
        iflytek_config: IflytekConfig | None = None,
        image_gen_config: ImageGenConfig | None = None,
        notion_config: NotionToolConfig | None = None,
        tool_history_config: ToolHistoryConfig | None = None,
        context_compression_config: ContextCompressionConfig | None = None,
        memory_system_config: MemorySystemConfig | None = None,
        feishu_config: FeishuConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_tokens = max(1, int(max_tokens))
        self.context_window_tokens = self._safe_int(context_window_tokens)
        self.token_budget_mode = token_budget_mode if token_budget_mode in {"output", "context"} else "output"
        self.merge_subagent_usage = merge_subagent_usage
        self.max_iterations = max_iterations
        self.reasoning_effort = reasoning_effort
        self.web_search_config = web_search_config or WebSearchConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.mineru_config = mineru_config or MineruConfig()
        self.iflytek_config = iflytek_config or IflytekConfig()
        self.image_gen_config = image_gen_config or ImageGenConfig()
        self.notion_config = notion_config or NotionToolConfig()
        self.tool_history_config = tool_history_config or ToolHistoryConfig()
        self.context_compression_config = context_compression_config or ContextCompressionConfig()
        self.memory_system_config = memory_system_config or MemorySystemConfig()
        self.feishu_config = feishu_config or FeishuConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        legacy_keep_recent_tool = getattr(self.context_compression_config, "keep_recent_tool_messages", 0)
        self._keep_recent_tool_messages = max(
            0,
            int(getattr(self.tool_history_config, "keep_recent_messages", legacy_keep_recent_tool)),
        )

        # Initialize core components
        self.context = ContextBuilder(workspace, memory_system_config=self.memory_system_config)
        self.sessions = SessionManager(workspace)
        self.compressor = SessionContextCompressor(
            provider=provider,
            sessions_dir=self.sessions.sessions_dir,
            config=self.context_compression_config,
            default_model=self.model,
            keep_recent_tool_messages=self._keep_recent_tool_messages,
        )
        self.memory_compiler = MemoryCompiler(
            workspace=workspace,
            provider=provider,
            config=self.memory_system_config,
            default_model=self.model,
        )
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            web_search_config=self.web_search_config,
            exec_config=self.exec_config,
            mineru_config=self.mineru_config,
            notion_config=self.notion_config,
            image_gen_config=self.image_gen_config,
            feishu_config=self.feishu_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._tool_digest_max_events = max(1, self.tool_history_config.max_events)
        self._tool_digest_max_chars = max(200, self.tool_history_config.max_chars)
        self._tool_preview_chars = max(80, self.tool_history_config.preview_chars)
        self._keep_recent_dialog_messages = max(1, self.context_compression_config.keep_recent_messages)
        self._history_dialog_limit: int | None = None
        self._history_tool_limit: int | None = None
        self._history_max_messages = 50
        self._history_precompress_max_messages = self._history_max_messages
        self._history_postcompress_max_messages = self._history_max_messages
        self._history_no_gap_cap = self._history_max_messages
        if self.context_compression_config.enabled:
            self._history_dialog_limit = self._keep_recent_dialog_messages
            self._history_tool_limit = self._keep_recent_tool_messages
            self._history_precompress_max_messages = max(
                self._history_max_messages,
                self.context_compression_config.trigger_by_message_count,
            )
            self._history_postcompress_max_messages = max(
                10,
                self._keep_recent_dialog_messages + self._keep_recent_tool_messages + 5,
            )
            self._history_no_gap_cap = max(
                self._history_precompress_max_messages,
                self.context_compression_config.trigger_by_message_count,
            )
        # Register all built-in tools
        self._register_default_tools()

    def _resolve_history_limit(self, session: Session, session_summary: str) -> int:
        """Resolve history size while preventing any summary-to-window gaps."""
        if not self.context_compression_config.enabled:
            return self._history_precompress_max_messages

        active_count = sum(1 for m in session.messages if m.get("include_in_context", True))
        active_floor = max(1, min(active_count, self._history_no_gap_cap))

        # With summary enabled, always include all active (not-yet-compressed) messages
        # up to next compression cap to avoid memory gaps between summary and recent window.
        if session_summary:
            return max(self._history_postcompress_max_messages, active_floor)
        return max(self._history_precompress_max_messages, active_floor)

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Safely convert to integer: None/bool returns 0, negative becomes 0, normal conversion."""
        try:
            if value is None:
                return 0
            if isinstance(value, bool):
                return 0
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _accumulate_usage(cls, target: dict[str, int], usage: dict[str, Any] | None) -> None:
        """Accumulate token usage across LLM calls for the agent lifecycle."""
        if not usage:
            return
        target["prompt_tokens"] += cls._safe_int(usage.get("prompt_tokens"))
        target["completion_tokens"] += cls._safe_int(usage.get("completion_tokens"))
        target["total_tokens"] += cls._safe_int(usage.get("total_tokens"))
        target["cache_tokens"] += cls._safe_int(usage.get("cache_tokens"))

    @classmethod
    def _build_token_monitor(
        cls,
        usage: dict[str, int],
        output_budget_tokens: int,
        context_window_tokens: int = 0,
        token_budget_mode: str = "output",
        tool_calls_completed: int = 0,
    ) -> dict[str, Any]:
        """
        Build token usage monitoring data (for Feishu card charts).
        Supports multiple cache_tokens field names for provider compatibility:
        prompt_tokens_details.cached_tokens / prompt_cache_hit_tokens /
        cache_read_input_tokens / cached_prompt_tokens / input_cached_tokens
        """
        prompt_tokens = cls._safe_int(usage.get("prompt_tokens"))
        output_tokens = cls._safe_int(usage.get("completion_tokens"))
        cache_tokens = cls._safe_int(usage.get("cache_tokens"))
        total_tokens = cls._safe_int(usage.get("total_tokens"))

        # Provider usage semantics are not always consistent:
        # some return prompt_tokens as total input, some as uncached input only.
        # Reconcile via total_tokens - completion_tokens and take the safer upper bound.
        derived_input_tokens = max(0, total_tokens - output_tokens)
        input_tokens = max(prompt_tokens, derived_input_tokens)
        cache_tokens = min(cache_tokens, input_tokens)
        input_uncached_tokens = max(0, input_tokens - cache_tokens)
        normalized_total_tokens = max(total_tokens, input_tokens + output_tokens)

        # Output budget calculation
        output_budget = max(1, cls._safe_int(output_budget_tokens))
        output_used = output_tokens
        output_raw_residue = output_budget - output_used
        output_residue = max(0, output_raw_residue)
        output_ratio = min(1.0, output_used / output_budget)

        # Context budget calculation (input + output vs context_window_tokens)
        has_context_budget = cls._safe_int(context_window_tokens) > 0
        context_budget = cls._safe_int(context_window_tokens) if has_context_budget else output_budget
        context_used = input_tokens + output_tokens
        context_raw_residue = context_budget - context_used
        context_residue = max(0, context_raw_residue)
        context_ratio = min(1.0, context_used / context_budget)

        # Select which budget mode to use based on config
        effective_mode = "context" if token_budget_mode == "context" and has_context_budget else "output"
        selected_residue = context_residue if effective_mode == "context" else output_residue
        selected_budget = context_budget if effective_mode == "context" else output_budget
        selected_used = context_used if effective_mode == "context" else output_used
        selected_ratio = context_ratio if effective_mode == "context" else output_ratio

        # Build ECharts chart data
        chart_values = [
            {"category": "token用量", "item": "input", "value": input_tokens},
            {"category": "token用量", "item": "output", "value": output_tokens},
        ]
        completed_tools = max(0, cls._safe_int(tool_calls_completed))
        if completed_tools > 0:
            chart_values.append({"category": "token用量", "item": "tool_calls", "value": completed_tools})

        return {
            "input_tokens": input_tokens,
            "prompt_tokens_raw": prompt_tokens,
            "input_tokens_derived_from_total": derived_input_tokens,
            "input_uncached_tokens": input_uncached_tokens,
            "output_tokens": output_tokens,
            "cache_tokens": cache_tokens,
            "task_total_tokens": normalized_total_tokens,
            "output_budget_total_tokens": output_budget,
            "output_budget_used_tokens": output_used,
            "output_budget_residue_tokens": output_residue,
            "output_budget_usage_ratio": output_ratio,
            "output_budget_usage_percent": round(output_ratio * 100, 2),
            "output_budget_exceeded": output_raw_residue < 0,
            "context_window_total_tokens": cls._safe_int(context_window_tokens),
            "context_budget_total_tokens": context_budget,
            "context_budget_used_tokens": context_used,
            "context_budget_residue_tokens": context_residue,
            "context_budget_usage_ratio": context_ratio,
            "context_budget_usage_percent": round(context_ratio * 100, 2),
            "context_budget_exceeded": context_raw_residue < 0,
            "selected_budget_mode": effective_mode,
            "selected_budget_total_tokens": selected_budget,
            "selected_budget_used_tokens": selected_used,
            "selected_budget_residue_tokens": selected_residue,
            "selected_budget_usage_ratio": selected_ratio,
            "selected_budget_usage_percent": round(selected_ratio * 100, 2),
            "chart": {
                "type": "bar",
                "direction": "horizontal",
                "title": {"text": "token用量占比图"},
                "data": {"values": chart_values},
                "xField": "value",
                "yField": "category",
                "seriesField": "item",
                "stack": True,
                "legends": {"visible": True, "orient": "bottom"},
                "label": {"visible": True, "formatter": "value"},
            },
        }

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (workspace-restricted if enabled)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(AppendFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell execution tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))

        # Web tools: search and fetch
        self.tools.register(WebSearchTool(
            api_key=self.web_search_config.api_key or None,
            max_results=self.web_search_config.max_results,
            endpoint=self.web_search_config.endpoint,
            country=self.web_search_config.country,
            language=self.web_search_config.language,
            tbs=self.web_search_config.tbs,
            page=self.web_search_config.page,
            autocorrect=self.web_search_config.autocorrect,
            search_type=self.web_search_config.search_type,
        ))
        self.tools.register(WebFetchTool())

        # PDF parsing tool (MinerU)
        if self.mineru_config and self.mineru_config.enabled:
            from assistant.agent.tools.pdf_mineru import MineruPdfParseTool
            self.tools.register(MineruPdfParseTool(
                config=self.mineru_config,
                allowed_dir=allowed_dir,
            ))

        # Message sending tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Image generation tool
        image_tool = ImageGenerateTool(
            config=self.image_gen_config,
            feishu_config=self.feishu_config,
            workspace=self.workspace,
            allowed_dir=allowed_dir,
        )
        self.tools.register(image_tool)

        # Notion database management tool
        self.tools.register(NotionTool(
            config=self.notion_config,
            allowed_dir=allowed_dir,
        ))

        # Subagent spawning tool
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Session management tool
        self.tools.register(SessionManageTool(manager=self.sessions))

        # Cron scheduling tool
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Personal memory search tool
        if self.memory_system_config and self.memory_system_config.enabled:
            self.tools.register(MemorySearchTool(
                workspace=self.workspace,
                config=self.memory_system_config,
            ))

        # Meeting summary tool (voice-to-text)
        if self.iflytek_config and self.iflytek_config.enabled:
            if self.iflytek_config.app_id and self.iflytek_config.secret_key:
                from assistant.agent.tools.transcription import IflytekTranscriptionProvider
                transcriber = IflytekTranscriptionProvider(
                    app_id=self.iflytek_config.app_id,
                    secret_key=self.iflytek_config.secret_key,
                    language=self.iflytek_config.language or "cn",
                )
                self.tools.register(MeetingSummaryTool(
                    llm_provider=self.provider,
                    transcription_provider=transcriber,
                    workspace=self.workspace,
                    allowed_dir=allowed_dir,
                ))
            else:
                logger.warning("IflytekConfig enabled but app_id or secret_key not set. MeetingSummaryTool not registered.")

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )

                # Process message
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        """
        # Handle system messages (subagent announces)
        if msg.channel == "system":
            return await self._process_system_message(msg)

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")

        # Step 2: Get or create session
        active_key = self.sessions.get_active_session_key(msg.channel, msg.chat_id)
        session_key_override = msg.metadata.get("session_key_override") if isinstance(msg.metadata, dict) else None
        session_key = active_key or session_key_override or msg.session_key
        session = self.sessions.get_or_create(session_key)

        # Step 3: Context compression check
        await self.compressor.compress_if_needed(session)
        session_summary = self.compressor.get_summary(session.key)
        history_limit = self._resolve_history_limit(session, session_summary)

        # Step 4: Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)

        image_tool = self.tools.get("image_generate")
        if isinstance(image_tool, ImageGenerateTool):
            image_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        session_tool = self.tools.get("session_manage")
        if isinstance(session_tool, SessionManageTool):
            session_tool.set_context(msg.channel, msg.chat_id)

        # Step 5: Build LLM message list
        user_id = msg.chat_id
        messages = self.context.build_messages(
            history=session.get_history(
                max_messages=history_limit,
                max_dialog_messages=self._history_dialog_limit,
                max_tool_messages=self._history_tool_limit,
                tool_max_events=self._tool_digest_max_events,
                tool_preview_chars=self._tool_preview_chars,
                tool_max_chars=self._tool_digest_max_chars,
            ),
            current_message=msg.content,
            session_summary=session_summary,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            user_id=user_id,
        )

        # Step 6: ReAct main loop initialization
        iteration = 0
        final_content = None
        task_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_tokens": 0,
        }
        # Register token monitor factory for message tool
        if isinstance(message_tool, MessageTool):
            message_tool.set_token_monitor_factory(
                lambda: self._build_token_monitor(
                    task_usage,
                    self.max_tokens,
                    self.context_window_tokens,
                    self.token_budget_mode,
                )
            )

        # Feishu streaming state
        feishu_stream_enabled = msg.channel == "feishu" and self.feishu_config.streaming_enabled
        stream_id = self._build_stream_id(msg) if feishu_stream_enabled else ""
        stream_initialized = False
        stream_started_at: float | None = None
        tool_log_entries: list[str] = []
        completed_tool_calls = 0

        # Append user message to session history
        session.add_message("user", msg.content)
        try:
            while iteration < self.max_iterations:
                iteration += 1

                # Call LLM
                response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
                self._accumulate_usage(task_usage, response.usage)

                # Process tool calls
                if response.has_tool_calls:
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                            }
                        }
                        for tc in response.tool_calls
                    ]
                    messages = self.context.add_assistant_message(
                        messages, response.content, tool_call_dicts
                    )

                    session.add_message("assistant", response.content, tool_calls=tool_call_dicts)

                    # Execute tool calls one by one
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)
                        panel_args_str = self._format_tool_arguments_for_panel(tool_call.arguments, max_value_chars=1000)
                        logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")

                        # Feishu channel: real-time streaming tool status
                        if feishu_stream_enabled:
                            if not stream_initialized:
                                await self._publish_feishu_stream_init(
                                    msg=msg,
                                    stream_id=stream_id,
                                    token_monitor=self._build_token_monitor(
                                        task_usage,
                                        self.max_tokens,
                                        self.context_window_tokens,
                                        self.token_budget_mode,
                                        tool_calls_completed=completed_tool_calls,
                                    ),
                                    initial_text="正在调用工具，请稍候...",
                                )
                                stream_initialized = True
                                stream_started_at = time.time()

                            call_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            entry_prefix = f"###### {len(tool_log_entries) + 1}. `{tool_call.name}`"
                            tool_log_entries.append(
                                (
                                    f"{entry_prefix}\n"
                                    f"- 调用时间: {call_started_at}\n"
                                    f"- 状态: 执行中\n"
                                    f"- 参数:\n"
                                    f"```json\n{panel_args_str}\n```"
                                )
                            )
                            await self._publish_feishu_tool_update(
                                msg=msg,
                                stream_id=stream_id,
                                tool_logs_markdown=self._build_tool_panel_markdown(tool_log_entries),
                                token_monitor=self._build_token_monitor(
                                    task_usage,
                                    self.max_tokens,
                                    self.context_window_tokens,
                                    self.token_budget_mode,
                                    tool_calls_completed=completed_tool_calls,
                                ),
                            )
                        else:
                            # Non-Feishu channels: push normal tool call notification
                            push_message = OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content=(
                                    f"正在调用工具： `{tool_call.name}`\n"
                                    f"参数列表：\n"
                                    f"```json\n{args_str}\n```"
                                ),
                                metadata={
                                    "token_monitor": self._build_token_monitor(
                                        task_usage,
                                        self.max_tokens,
                                        self.context_window_tokens,
                                        self.token_budget_mode,
                                    )
                                },
                            )
                            await self.bus.publish_outbound(push_message)

                        # Execute tool
                        started = time.perf_counter()
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        result_text = result if isinstance(result, str) else str(result)
                        completed_tool_calls += 1

                        # Update tool log panel status on completion
                        if feishu_stream_enabled and tool_log_entries:
                            elapsed = time.perf_counter() - started
                            call_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            entry_prefix = f"###### {len(tool_log_entries)}. `{tool_call.name}`"
                            tool_log_entries[-1] = (
                                f"{entry_prefix}\n"
                                f"- 调用时间: {call_started_at}\n"
                                f"- 完成时间: {call_finished_at}\n"
                                f"- 状态: 已完成 ({elapsed:.2f}s)\n"
                                f"- 参数:\n"
                                f"```json\n{panel_args_str}\n```"
                            )
                            await self._publish_feishu_tool_update(
                                msg=msg,
                                stream_id=stream_id,
                                tool_logs_markdown=self._build_tool_panel_markdown(tool_log_entries),
                                token_monitor=self._build_token_monitor(
                                    task_usage,
                                    self.max_tokens,
                                    self.context_window_tokens,
                                    self.token_budget_mode,
                                    tool_calls_completed=completed_tool_calls,
                                ),
                            )

                        # Append tool result to session history and message list
                        session.add_message("tool", result_text, tool_call_id=tool_call.id, name=tool_call.name)
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                else:
                    # LLM no longer calling tools - ReAct loop ends
                    final_content = response.content
                    break
        finally:
            # Cleanup: remove token monitor factory
            if isinstance(message_tool, MessageTool):
                message_tool.set_token_monitor_factory(None)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Step 7: Session persistence
        session.add_message("assistant", final_content)
        await self.compressor.compress_if_needed(session)
        self.sessions.save(session)

        # Step 8: Build final token usage monitoring data
        token_monitor = self._build_token_monitor(
            task_usage,
            self.max_tokens,
            self.context_window_tokens,
            self.token_budget_mode,
            tool_calls_completed=completed_tool_calls if feishu_stream_enabled else 0,
        )

        # Step 8a: Feishu channel: use streaming card push
        if self._should_publish_feishu_streaming(msg, final_content):
            await self._publish_feishu_streaming_response(
                msg,
                final_content,
                token_monitor,
                stream_id=stream_id,
                send_init=not stream_initialized,
                stream_started_at=stream_started_at,
            )
            return None

        # Step 8b: Other channels: normal message push
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata={"token_monitor": token_monitor},
        )

    def _should_publish_feishu_streaming(self, msg: InboundMessage, final_content: str) -> bool:
        """Use pseudo streaming for Feishu outbound delivery."""
        return (
            msg.channel == "feishu"
            and self.feishu_config.streaming_enabled
            and bool(final_content)
        )

    @staticmethod
    def _build_stream_id(msg: InboundMessage) -> str:
        """Generate stream identifier shared across tool logs and final answer."""
        return f"{msg.channel}:{msg.chat_id}:{int(time.time() * 1000)}"

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        """Truncate overly long text with an appended note."""
        if len(value) <= limit:
            return value
        return f"{value[:limit]}\n...\n(内容已截断，原始长度: {len(value)} 字符)"

    @classmethod
    def _truncate_tool_argument_value(cls, value: Any, max_value_chars: int) -> Any:
        """
        Recursively truncate tool argument values, only truncating overly long strings
        while preserving structural integrity of the arguments.
        """
        if isinstance(value, str):
            if len(value) <= max_value_chars:
                return value
            return (
                value[:max_value_chars]
                + f"... (内容已截断，原始长度: {len(value)} 字符)"
            )

        if isinstance(value, list):
            return [cls._truncate_tool_argument_value(item, max_value_chars) for item in value]

        if isinstance(value, dict):
            return {
                key: cls._truncate_tool_argument_value(item, max_value_chars)
                for key, item in value.items()
            }

        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except TypeError:
            serialized = str(value)

        if len(serialized) <= max_value_chars:
            return value

        return cls._truncate_text(serialized, max_value_chars)

    @classmethod
    def _format_tool_arguments_for_panel(cls, arguments: Any, max_value_chars: int = 1000) -> str:
        """Format tool arguments as JSON string, truncating only overly long values."""
        normalized = cls._truncate_tool_argument_value(arguments, max_value_chars)
        return json.dumps(normalized, indent=2, ensure_ascii=False)

    def _build_tool_panel_markdown(self, entries: list[str]) -> str:
        """Build a Markdown panel for tool call records."""
        header = "###### 工具调用记录"
        if not entries:
            return f"{header}\n\n暂无工具调用。"
        return f"{header}\n\n" + "\n\n".join(entries)

    async def _publish_feishu_stream_init(
        self,
        msg: InboundMessage,
        stream_id: str,
        token_monitor: dict[str, Any],
        initial_text: str = "",
    ) -> None:
        """Publish Feishu streaming initialization event (init)."""
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=initial_text,
                metadata={
                    "token_monitor": token_monitor,
                    "feishu_stream": {
                        "action": "init",
                        "stream_id": stream_id,
                        "full_text": initial_text,
                    },
                },
            )
        )

    async def _publish_feishu_tool_update(
        self,
        msg: InboundMessage,
        stream_id: str,
        tool_logs_markdown: str,
        token_monitor: dict[str, Any],
    ) -> None:
        """Publish Feishu tool call update event (tool_update)."""
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="",
                metadata={
                    "token_monitor": token_monitor,
                    "feishu_stream": {
                        "action": "tool_update",
                        "stream_id": stream_id,
                        "tool_logs_markdown": tool_logs_markdown,
                        "force": True,
                    },
                },
            )
        )

    async def _publish_feishu_streaming_response(
        self,
        msg: InboundMessage,
        final_content: str,
        token_monitor: dict[str, Any],
        stream_id: str | None = None,
        send_init: bool = True,
        stream_started_at: float | None = None,
    ) -> None:
        """
        Publish Feishu streaming response (init -> append x N -> finalize).
        """
        stream_id = stream_id or self._build_stream_id(msg)
        chunk_chars = max(20, int(self.feishu_config.streaming_print_step_default) * 16)
        interval = max(0.08, float(self.feishu_config.streaming_print_frequency_ms_default) / 1000.0 * 4.0)
        timeout_sec = max(1, int(self.feishu_config.streaming_preemptive_timeout_sec))
        stream_start_time = stream_started_at

        # Step 1: init
        if send_init:
            await self._publish_feishu_stream_init(
                msg=msg,
                stream_id=stream_id,
                token_monitor=token_monitor,
                initial_text="",
            )
            if stream_start_time is None:
                stream_start_time = time.time()

        if stream_start_time is None:
            stream_start_time = time.time()

        # Step 2: Chunked push (append)
        total = len(final_content)
        cursor = 0
        while cursor < total:
            next_cursor = min(total, cursor + chunk_chars)
            if (time.time() - stream_start_time) > timeout_sec:
                streamed_text = final_content[:next_cursor]
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=streamed_text,
                        metadata={
                            "token_monitor": token_monitor,
                            "feishu_stream": {
                                "action": "finalize",
                                "stream_id": stream_id,
                                "full_text": streamed_text,
                            },
                        },
                    )
                )
                await self._publish_feishu_timeout_fallback(
                    msg=msg,
                    remaining_content=final_content[next_cursor:],
                    token_monitor=token_monitor,
                )
                return

            cursor = next_cursor
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=final_content[:cursor],
                    metadata={
                        "token_monitor": token_monitor,
                        "feishu_stream": {
                            "action": "append",
                            "stream_id": stream_id,
                            "full_text": final_content[:cursor],
                        },
                    },
                )
            )
            if cursor < total:
                await asyncio.sleep(interval)

        # Step 3: finalize
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                metadata={
                    "token_monitor": token_monitor,
                    "feishu_stream": {
                        "action": "finalize",
                        "stream_id": stream_id,
                        "full_text": final_content,
                    },
                },
            )
        )

    async def _publish_feishu_timeout_fallback(
        self,
        msg: InboundMessage,
        remaining_content: str,
        token_monitor: dict[str, Any],
    ) -> None:
        """Feishu preemptive timeout fallback handling."""
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="当前回答已超出流式窗口限制，后续内容将切换为普通消息继续发送。",
                metadata={"token_monitor": token_monitor},
            )
        )

        if not remaining_content:
            return

        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=remaining_content,
                metadata={"token_monitor": token_monitor},
            )
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        # Use the origin session for context
        active_key = self.sessions.get_active_session_key(origin_channel, origin_chat_id)
        session_key = active_key or f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        await self.compressor.compress_if_needed(session)
        session_summary = self.compressor.get_summary(session.key)
        history_limit = self._resolve_history_limit(session, session_summary)

        # Update tool contexts to point to original channel
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)

        image_tool = self.tools.get("image_generate")
        if isinstance(image_tool, ImageGenerateTool):
            image_tool.set_context(origin_channel, origin_chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)

        session_tool = self.tools.get("session_manage")
        if isinstance(session_tool, SessionManageTool):
            session_tool.set_context(origin_channel, origin_chat_id)

        # Build message list
        messages = self.context.build_messages(
            history=session.get_history(
                max_messages=history_limit,
                max_dialog_messages=self._history_dialog_limit,
                max_tool_messages=self._history_tool_limit,
                tool_max_events=self._tool_digest_max_events,
                tool_preview_chars=self._tool_preview_chars,
                tool_max_chars=self._tool_digest_max_chars,
            ),
            current_message=msg.content,
            session_summary=session_summary,
            channel=origin_channel,
            chat_id=origin_chat_id,
            user_id=origin_chat_id,
        )

        # Agent loop (same as _process_message but no Feishu streaming logs)
        iteration = 0
        final_content = None
        task_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_tokens": 0,
        }
        # Merge subagent token usage if from subagent
        if (
            self.merge_subagent_usage
            and msg.sender_id == "subagent"
            and isinstance(msg.metadata, dict)
            and isinstance(msg.metadata.get("subagent_usage"), dict)
        ):
            self._accumulate_usage(task_usage, msg.metadata.get("subagent_usage"))
        if isinstance(message_tool, MessageTool):
            message_tool.set_token_monitor_factory(
                lambda: self._build_token_monitor(
                    task_usage,
                    self.max_tokens,
                    self.context_window_tokens,
                    self.token_budget_mode,
                )
            )

        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        try:
            while iteration < self.max_iterations:
                iteration += 1

                response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
                self._accumulate_usage(task_usage, response.usage)

                if response.has_tool_calls:
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                            }
                        }
                        for tc in response.tool_calls
                    ]
                    messages = self.context.add_assistant_message(
                        messages, response.content, tool_call_dicts
                    )

                    session.add_message("assistant", response.content, tool_calls=tool_call_dicts)

                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                        started = time.perf_counter()
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        result_text = result if isinstance(result, str) else str(result)

                        session.add_message("tool", result_text, tool_call_id=tool_call.id, name=tool_call.name)

                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                else:
                    final_content = response.content
                    break
        finally:
            if isinstance(message_tool, MessageTool):
                message_tool.set_token_monitor_factory(None)

        if final_content is None:
            final_content = "Background task completed."

        session.add_message("assistant", final_content)
        await self.compressor.compress_if_needed(session)
        self.sessions.save(session)

        token_monitor = self._build_token_monitor(
            task_usage,
            self.max_tokens,
            self.context_window_tokens,
            self.token_budget_mode,
        )

        # Route to original channel
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content,
            metadata={"token_monitor": token_monitor},
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            metadata={"session_key_override": session_key},
        )

        response = await self._process_message(msg)
        return response.content if response else ""

    async def process_memory_daily_update(self, diary_path: Path, extracted_from: str | None = None) -> dict[str, int]:
        """Run daily memory extraction+merge and refresh MEMORY.md."""
        source = extracted_from or diary_path.name
        return await self.memory_compiler.daily_update_from_file(diary_path, extracted_from=source)
