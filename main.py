"""
租房 Agent 主入口：提供 POST /api/v1/chat，对接模型与租房仿真 API，遵循 agent 输入输出约定。
"""
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_client import chat_completions, parse_assistant_message
from prompts import SYSTEM_PROMPT
from rental_tools import get_tools_schema, init_houses, run_tool
from session_store import (
    append_messages,
    get_messages,
    is_initialized,
    set_initialized,
)

app = FastAPI(title="租房 Agent", description="需求理解、房源筛选与推荐")


@app.get("/")
def root():
    return {"service": "租房 Agent", "chat": "POST /api/v1/chat"}

# 单轮内最大工具调用轮数，防止死循环
MAX_TOOL_ROUNDS = 20


class ChatRequest(BaseModel):
    model_ip: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    status: str
    tool_results: list[dict]
    timestamp: int
    duration_ms: int


def _is_house_result_json(text: str) -> bool:
    """判断是否为约定的房源推荐 JSON：含 message 和 houses 且为合法 JSON。"""
    if not text or not text.strip():
        return False
    try:
        d = json.loads(text.strip())
        return isinstance(d, dict) and "message" in d and "houses" in d
    except json.JSONDecodeError:
        return False


def _try_extract_json(text: str) -> str | None:
    """尝试从文本中提取合法的房源推荐 JSON 子串。"""
    if not text:
        return None
    # 简单查找最外层 {}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        if _is_house_result_json(candidate):
            return candidate
    return None


def _extract_house_ids_from_content(content: str) -> list[str] | None:
    """若 content 为约定 JSON 则解析出 houses 列表并返回；否则返回 None。"""
    if not _is_house_result_json(content):
        return None
    try:
        d = json.loads(content.strip())
        houses = d.get("houses")
        if isinstance(houses, list) and len(houses) <= 5:
            return [str(h) for h in houses]
        return None
    except (json.JSONDecodeError, TypeError):
        return None


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    # 使用 session_id 作为租房仿真 API 的 X-User-ID（评测若下发工号可改为从请求体取）
    user_id = session_id

    # 新会话：调用房源数据重置
    if not is_initialized(session_id):
        try:
            await init_houses(user_id)
        except Exception as e:
            pass  # 若评测环境未提供 init 接口则忽略
        set_initialized(session_id)

    # 追加本轮用户消息
    append_messages(session_id, {"role": "user", "content": user_message})

    tools_schema = get_tools_schema()
    messages = get_messages(session_id)
    # 发给模型的 messages：系统提示 + 已有对话（不含 system，在下面加）
    model_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 只把 user/assistant/tool 消息发给模型，保留 tool_calls 结构
    for m in messages:
        role = m.get("role")
        if role == "user":
            model_messages.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            model_messages.append(msg)
        elif role == "tool":
            model_messages.append(
                {"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": m.get("content", "")}
            )

    tool_results: list[dict] = []
    start_ts = time.time()
    response_text = ""
    round_count = 0

    while round_count < MAX_TOOL_ROUNDS:
        round_count += 1
        response = await chat_completions(req.model_ip, model_messages, tools=tools_schema, session_id=session_id)
        content, tool_calls = parse_assistant_message(response)

        if not tool_calls:
            response_text = content or ""
            
            # 尝试清洗 JSON：若包含合法的房源推荐 JSON，则提取出来作为最终回复
            cleaned_json = _try_extract_json(response_text)
            if cleaned_json:
                response_text = cleaned_json

            # 将 assistant 回复写入会话
            append_messages(session_id, {"role": "assistant", "content": response_text})
            break

        # 有 tool_calls：执行并追加 tool 结果
        assistant_msg = {"role": "assistant", "content": content or "", "tool_calls": tool_calls}
        model_messages.append(assistant_msg)
        append_messages(session_id, assistant_msg)

        tool_outputs: list[dict] = []
        for tc in tool_calls:
            tid = tc.get("id", "")
            fn = (tc.get("function") or {}).get("name", "")
            try:
                args_str = (tc.get("function") or {}).get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except json.JSONDecodeError:
                args = {}
            try:
                out = await run_tool(fn, args, user_id)
            except Exception as e:
                out = json.dumps({"error": str(e)})
            tool_results.append({"name": fn, "success": "error" not in out.lower()[:200], "output": out[:2000]})
            model_messages.append({"role": "tool", "tool_call_id": tid, "content": out})
            tool_outputs.append({"role": "tool", "tool_call_id": tid, "content": out})
        append_messages(session_id, *tool_outputs)

    duration_ms = int((time.time() - start_ts) * 1000)
    return ChatResponse(
        session_id=session_id,
        response=response_text,
        status="success",
        tool_results=tool_results,
        timestamp=int(start_ts),
        duration_ms=duration_ms,
    )
