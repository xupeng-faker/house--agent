"""调用评测/本地模型：model_ip:8888，OpenAI 兼容的 chat completions。"""
import asyncio
import re
import time
from typing import Any

import httpx

from config import MODEL_PORT
from logger import log_model_input, log_model_output, service_log

# 全局复用 httpx 客户端，减少连接开销
_model_client: httpx.AsyncClient | None = None


def _get_model_client() -> httpx.AsyncClient:
    global _model_client
    if _model_client is None or _model_client.is_closed:
        _model_client = httpx.AsyncClient(timeout=120.0, trust_env=False)
    return _model_client


async def chat_completions(
    model_ip: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    url = f"http://{model_ip}:{MODEL_PORT}/v2/chat/completions"
    headers = {"Content-Type": "application/json"}
    if session_id:
        url = f"http://{model_ip}:{MODEL_PORT}/v1/chat/completions"
        headers["Session-ID"] = session_id

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

    log_model_input(session_id or "", url, messages, tools)

    client = _get_model_client()
    max_retries = 3
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            result = r.json()
            dur = int((time.time() - t0) * 1000)
            log_model_output(session_id or "", result, dur)
            return result
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            dur = int((time.time() - t0) * 1000)
            service_log.warning(
                "[MODEL_ERR] session=%s attempt=%d/%d duration=%dms error=%s",
                session_id, attempt + 1, max_retries, dur, e,
            )
            if attempt == max_retries - 1:
                raise e
            delay = 2 ** attempt
            await asyncio.sleep(delay)

    return {}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_assistant_message(response: dict) -> tuple[str | None, list | None]:
    """从 chat completion 响应中取出 assistant 的 content 与 tool_calls。"""
    choices = response.get("choices") or []
    if not choices:
        return None, None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    # 安全剥离 Qwen3 thinking 块
    if content:
        content = _THINK_RE.sub("", content).strip()
    return content, msg.get("tool_calls")
