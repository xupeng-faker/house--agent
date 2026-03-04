"""LLM chat completions client with retry and Qwen thinking-tag removal."""
import asyncio
import re
import time
from typing import Any

import httpx

from config import MODEL_PORT
from logger import model_log, _safe_json

_client: httpx.AsyncClient | None = None
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120.0, trust_env=False)
    return _client


async def chat_completions(
    model_ip: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Call model chat completions endpoint with retries."""
    if session_id:
        url = f"http://{model_ip}:{MODEL_PORT}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Session-ID": session_id}
    else:
        url = f"http://{model_ip}:{MODEL_PORT}/v2/chat/completions"
        headers = {"Content-Type": "application/json"}

    body: dict[str, Any] = {
        "model": "",
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools

    model_log.debug("[IN] session=%s url=%s msgs=%d tools=%s",
                    session_id, url, len(messages), len(tools) if tools else 0)

    client = _get_client()
    max_retries = 3
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            result = r.json()
            dur = int((time.time() - t0) * 1000)
            model_log.debug("[OUT] session=%s dur=%dms", session_id, dur)
            return result
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            dur = int((time.time() - t0) * 1000)
            model_log.warning("[ERR] session=%s attempt=%d dur=%dms err=%s",
                              session_id, attempt + 1, dur, e)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    return {}


def parse_response(response: dict) -> tuple[str | None, list | None]:
    """Extract content and tool_calls from chat completion response."""
    choices = response.get("choices") or []
    if not choices:
        return None, None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content:
        content = _THINK_RE.sub("", content).strip()
    return content, msg.get("tool_calls")
