"""Meeting audio summarization tool: transcribe audio to text, then summarize with LLM."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assistant.agent.tools.base import Tool

if TYPE_CHECKING:
    from assistant.providers.base import LLMProvider
    from assistant.agent.tools.transcription import IflytekTranscriptionProvider


class MeetingSummaryTool(Tool):
    """
    Tool to summarize meeting audio recordings.

    Pipeline: audio file -> iFlytek transcription -> LLM structured summary -> save .md to workspace.
    Supported formats: mp3, wav, m4a, aac, pcm, opus, flac, ogg, amr, speex.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        transcription_provider: IflytekTranscriptionProvider,
        workspace: Path,
        allowed_dir: Path | None = None,
    ):
        self._llm = llm_provider
        self._transcriber = transcription_provider
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "meeting_summary"

    @property
    def description(self) -> str:
        return (
            "Transcribe a meeting audio file and generate a structured summary in Chinese. "
            "Input: path to an audio file (mp3, wav, m4a, aac, etc.) saved locally. "
            "Output: a structured meeting summary (theme, discussion points, decisions, action items) "
            "saved as 会议摘要_YYYYMMDD_HHMMSS.md in the workspace, "
            "and the full summary text returned as the tool result. "
            "Supports Chinese-language meetings. "
            "The audio file is typically located in the media directory or workspace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {
                    "type": "string",
                    "description": (
                        "Path to the meeting audio file. "
                        "The file should be a recording in mp3, wav, m4a, aac, opus, flac, ogg, amr or speex format. "
                        "Audio files received via Feishu are typically stored in the media directory."
                    ),
                },
                "meeting_title": {
                    "type": "string",
                    "description": (
                        "Optional meeting title or topic. "
                        "If not provided, the LLM will infer it from the transcript content."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": "Language of the meeting audio: 'cn' for Chinese (default), 'en' for English.",
                    "default": "cn",
                },
            },
            "required": ["audio_path"],
        }

    async def execute(
        self,
        audio_path: str,
        meeting_title: str | None = None,
        language: str = "cn",
        **kwargs: Any,
    ) -> str:
        # Step 1: Validate audio file path
        resolved = self._resolve_path(audio_path)
        if not resolved.exists():
            return f"Error: Audio file not found: {audio_path}"
        if not resolved.is_file():
            return f"Error: Not a file: {audio_path}"

        file_size_mb = resolved.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:
            return f"Error: Audio file too large ({file_size_mb:.1f}MB). Maximum allowed is 500MB."

        # Step 2: Transcribe audio file
        audio_path_str = str(resolved)
        try:
            transcript = await asyncio.wait_for(
                self._transcriber.transcribe(resolved),
                timeout=180.0,
            )
        except asyncio.TimeoutError:
            return "Error: Transcription timed out after 180 seconds. The audio file may be too long."

        if not transcript or not transcript.strip():
            return "Error: Transcription returned empty result. Please check the audio file quality."

        # Step 3: Summarize transcript using LLM
        summary_prompt = _build_summary_prompt(transcript, meeting_title, language)
        try:
            summary_text = await self._llm.chat(
                messages=[
                    {"role": "user", "content": summary_prompt},
                ],
                system=None,
            )
            if not summary_text:
                return "Error: LLM returned empty summary."
        except Exception as e:
            return f"Error: LLM summarization failed: {str(e)}"

        # Step 4: Save summary as .md file in workspace
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"会议摘要_{timestamp}.md"
        output_path = self._workspace / output_filename

        md_content = _build_markdown(
            summary=summary_text,
            audio_file=resolved.name,
            timestamp=timestamp,
            language=language,
        )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md_content, encoding="utf-8")
            saved_path = str(output_path)
        except Exception as e:
            saved_path = f"<存档失败: {e}>"

        # Step 5: Build final result
        result_lines = [
            f"✅ 会议摘要已生成",
            f"📁 存档: {saved_path}",
            "",
            "=" * 60,
            summary_text,
            "=" * 60,
            f"\n*原始音频: {resolved.name}*",
        ]
        return "\n".join(result_lines)

    def _resolve_path(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if self._allowed_dir and not str(resolved).startswith(str(self._allowed_dir.resolve())):
            raise PermissionError(f"Path {path} is outside allowed directory {self._allowed_dir}")
        return resolved


def _build_summary_prompt(
    transcript: str,
    meeting_title: str | None,
    language: str,
) -> str:
    title_hint = f"\n会议主题（已知）：{meeting_title}" if meeting_title else ""
    return f"""你是一位专业的会议记录助手。请根据以下会议录音转写文本，整理成一份结构化的中文会议摘要。{title_hint}

## 要求
- 语言：中文（尽量通顺流畅，修正口语中的不通顺表达）
- 格式：Markdown，包含以下部分：
  1. **会议主题**：一句话概括会议主题
  2. **会议时间**：如能推断出时间则填写，否则写"时间未标注"
  3. **参会人员**：如能推断出人员则填写，否则写"未标注"
  4. **讨论要点**：列出主要讨论内容（3-8条，用简洁的要点形式）
  5. **决议事项**：列出会议达成的决议或结论（如有）
  6. **待跟进事项**：列出需要后续跟进的行动项和负责人（如能推断）
- 摘要应简洁明了，便于快速阅读和传播
- 如果转写文本不完整或质量较低，请基于现有内容尽力整理，并在对应部分注明"信息不足"

## 会议录音转写文本
---
{transcript}
---
"""


def _build_markdown(
    summary: str,
    audio_file: str,
    timestamp: str,
    language: str,
) -> str:
    header = (
        f"# 会议摘要\n\n"
        f"- **原始音频**: {audio_file}\n"
        f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"- **语言**: {'中文' if language == 'cn' else '英文'}\n"
        f"\n---\n"
    )
    return header + summary + "\n"
