"""Voice transcription providers: Groq Whisper and iFlytek file transcription."""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error(f"Groq transcription error: {e}")
            return ""


class IflytekTranscriptionProvider:
    """
    iFlytek file transcription provider using WebAPI.

    Documentation: https://www.xfyun.cn/doc/asr/ifasr_new/API.html
    Two-step flow: 1) Upload file to get orderId; 2) Poll for transcription result.
    """

    def __init__(self, app_id: str, secret_key: str, language: str = "cn"):
        self.app_id = app_id
        self.secret_key = secret_key
        self.language = language

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe audio file using iFlytek file transcription API.

        Args:
            file_path: Audio file path.

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self.app_id or not self.secret_key:
            logger.warning("iFlytek transcription credentials not configured")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return ""

        try:
            # Generate signature
            ts = str(int(time.time()))
            base_string = self.app_id + ts
            md5_str = hashlib.md5(base_string.encode()).hexdigest()
            signa = base64.b64encode(
                hmac.new(
                    self.secret_key.encode(),
                    md5_str.encode(),
                    hashlib.sha1,
                ).digest()
            ).decode()

            # Upload file
            order_id = await self._upload_file(path, signa, ts)
            if not order_id:
                return ""

            # Poll for transcription result (max 120 seconds)
            text = await self._poll_result(order_id, signa, ts)
            return text

        except Exception as e:
            logger.error(f"iFlytek transcription error: {e}")
            return ""

    async def _upload_file(self, path: Path, signa: str, ts: str) -> str | None:
        """Upload audio file and get orderId."""
        upload_url = "https://raasr.xfyun.cn/v2/api/upload"

        duration_ms = self._get_audio_duration_ms(path)

        params: dict[str, Any] = {
            "appId": self.app_id,
            "signa": signa,
            "ts": ts,
            "fileName": path.name,
            "fileSize": path.stat().st_size,
            "duration": str(duration_ms),
        }

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {"file": (path.name, f)}
                    response = await client.post(
                        upload_url,
                        params=params,
                        files=files,
                        timeout=60.0,
                    )
                response.raise_for_status()
                data = response.json()

                if data.get("code") != "000000":
                    logger.error(f"iFlytek upload failed: {data.get('descInfo')} ({data.get('code')})")
                    return None

                order_id = data.get("content", {}).get("orderId")
                if not order_id:
                    logger.error("iFlytek upload succeeded but no orderId returned")
                    return None

                logger.info(f"iFlytek upload succeeded, orderId={order_id}")
                return order_id

        except Exception as e:
            logger.error(f"iFlytek upload request error: {e}")
            return None

    async def _poll_result(self, order_id: str, signa: str, ts: str, max_wait: int = 120) -> str:
        """Poll for transcription result."""
        result_url = "https://raasr.xfyun.cn/v2/api/getResult"
        params: dict[str, Any] = {
            "appId": self.app_id,
            "signa": signa,
            "ts": ts,
            "orderId": order_id,
        }

        start = time.monotonic()
        poll_interval = 3

        async with httpx.AsyncClient() as client:
            while time.monotonic() - start < max_wait:
                await asyncio.sleep(poll_interval)

                try:
                    response = await client.post(
                        result_url,
                        data=params,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    if data.get("code") != "000000":
                        if data.get("code") in ("10003", "10004", "10005"):
                            continue
                        logger.error(f"iFlytek getResult failed: {data.get('descInfo')} ({data.get('code')})")
                        return ""

                    content = data.get("content", {})
                    status = content.get("orderInfo", {}).get("status", 0)

                    if status == 9:
                        order_result = content.get("orderResult", {})
                        if isinstance(order_result, str):
                            try:
                                order_result = json.loads(order_result)
                            except Exception:
                                pass
                        return self._parse_transcription_result(order_result)

                    elif status == -1:
                        logger.error("iFlytek transcription failed: audio file error or timeout")
                        return ""

                    continue

                except Exception as e:
                    logger.warning(f"iFlytek poll request error: {e}, retrying...")
                    continue

            logger.error(f"iFlytek polling timed out after {max_wait}s")
            return ""

    def _parse_transcription_result(self, result: Any) -> str:
        """Parse iFlytek transcription result into concatenated text."""
        if not result:
            return ""

        lines: list[str] = []
        try:
            data_list = result.get("data", [])
            if isinstance(data_list, list):
                for item in data_list:
                    ws = item.get("ws", [])
                    if isinstance(ws, list):
                        sentence_parts = []
                        for w_item in ws:
                            cw_list = w_item.get("cw", [])
                            if isinstance(cw_list, list):
                                for cw in cw_list:
                                    word = cw.get("w", "")
                                    if word:
                                        sentence_parts.append(word)
                            else:
                                word = w_item.get("w", "")
                                if word:
                                    sentence_parts.append(word)
                        if sentence_parts:
                            lines.append("".join(sentence_parts))
                    else:
                        text = item.get("text", "") or item.get("onebest", "")
                        if text:
                            lines.append(text)
            else:
                text = result.get("text", "") or result.get("onebest", "")
                if text:
                    lines.append(text)
        except Exception as e:
            logger.warning(f"Failed to parse iFlytek transcription result: {e}")
            if isinstance(result, str):
                return result
            return str(result)

        return "\n".join(lines) if lines else ""

    def _get_audio_duration_ms(self, path: Path) -> int:
        """Estimate audio duration in milliseconds from file size."""
        size_bytes = path.stat().st_size
        estimated_seconds = size_bytes / (16 * 1024)
        estimated_seconds = max(1, min(estimated_seconds, 5 * 3600))
        return int(estimated_seconds * 1000)
