"""调用评测/本地模型：model_ip:8888，OpenAI 兼容的 chat completions。"""
import json
from typing import Any

import httpx

from config import MODEL_PORT


async def chat_completions(
    model_ip: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    请求 POST http://{model_ip}:{MODEL_PORT}/v1/chat/completions 或 v2。
    若 session_id 存在则请求头带 Session-ID（评测用）；否则用 v2 无需 Session-ID。
    """
    url = f"http://{model_ip}:{MODEL_PORT}/v2/chat/completions"
    headers = {"Content-Type": "application/json"}
    if session_id:
        url = f"http://{model_ip}:{MODEL_PORT}/v1/chat/completions"
        headers["Session-ID"] = session_id

    body: dict[str, Any] = {
        "model": "",
        "messages": messages,
        "stream": False,
    }
    if tools:
        body["tools"] = tools

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=body, headers=headers)
        r.raise_for_status()
        return r.json()


def parse_assistant_message(response: dict) -> tuple[str | None, list | None]:
    """从 chat completion 响应中取出 assistant 的 content 与 tool_calls。"""
    choices = response.get("choices") or []
    if not choices:
        return None, None
    msg = choices[0].get("message") or {}
    return msg.get("content"), msg.get("tool_calls")
