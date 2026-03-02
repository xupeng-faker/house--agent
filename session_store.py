"""会话存储：按 session_id 维护多轮对话消息列表。"""
from typing import Any

# session_id -> { "messages": [{"role","content"} | tool_calls], "initialized": bool }
_sessions: dict[str, dict[str, Any]] = {}


def get_messages(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "initialized": False}
    return _sessions[session_id]["messages"]


def append_messages(session_id: str, *msgs: dict) -> None:
    get_messages(session_id)
    _sessions[session_id]["messages"].extend(msgs)


def set_messages(session_id: str, messages: list[dict]) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "initialized": False}
    _sessions[session_id]["messages"] = list(messages)


def is_initialized(session_id: str) -> bool:
    if session_id not in _sessions:
        return False
    return _sessions[session_id].get("initialized", False)


def set_initialized(session_id: str, value: bool = True) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "initialized": False}
    _sessions[session_id]["initialized"] = value
