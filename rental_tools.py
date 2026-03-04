"""
租房仿真 API 工具：OpenAI 格式的 tools 定义 + 执行器。
请求头 X-User-ID 使用 session_id（或评测下发的用户工号）。
"""
import json
import time
from typing import Any

import httpx

from config import FAKE_APP_BASE_URL
from logger import log_tool_call, log_tool_result, tool_log

# 全局复用 httpx 客户端
_api_client: httpx.AsyncClient | None = None


def _get_api_client() -> httpx.AsyncClient:
    global _api_client
    if _api_client is None or _api_client.is_closed:
        _api_client = httpx.AsyncClient(timeout=30.0, trust_env=False)
    return _api_client


def _headers(user_id: str, need_user_id: bool = True) -> dict:
    h = {"Content-Type": "application/json"}
    if need_user_id:
        h["X-User-ID"] = user_id
    return h


async def _get(url: str, params: dict | None, user_id: str, need_user_id: bool = True) -> str:
    client = _get_api_client()
    tool_log.debug("[HTTP_GET] url=%s params=%s user=%s", url, params, user_id)
    r = await client.get(url, params=params or {}, headers=_headers(user_id, need_user_id))
    r.raise_for_status()
    tool_log.debug("[HTTP_RESP] url=%s status=%d body=%s", url, r.status_code, r.text[:2000])
    return r.text


async def _post(url: str, user_id: str, need_user_id: bool = True) -> str:
    client = _get_api_client()
    tool_log.debug("[HTTP_POST] url=%s user=%s", url, user_id)
    r = await client.post(url, headers=_headers(user_id, need_user_id))
    r.raise_for_status()
    tool_log.debug("[HTTP_RESP] url=%s status=%d body=%s", url, r.status_code, r.text[:2000])
    return r.text


async def init_houses(user_id: str) -> str:
    """新会话时调用：重置当前用户视角下的房源数据。"""
    tool_log.info("[INIT] user=%s 初始化房源数据", user_id)
    url = f"{FAKE_APP_BASE_URL}/api/houses/init"
    result = await _post(url, user_id=user_id)
    tool_log.info("[INIT] user=%s 完成, result=%s", user_id, result[:500])
    return result


async def run_tool(tool_name: str, arguments: dict[str, Any], user_id: str) -> str:
    """根据 tool_name 和 arguments 调用租房仿真 API，返回结果 JSON 字符串。"""
    log_tool_call(user_id, tool_name, arguments)
    t0 = time.time()

    try:
        result = await _dispatch_tool(tool_name, arguments, user_id)
        dur = int((time.time() - t0) * 1000)
        success = "error" not in result.lower()[:200]
        log_tool_result(user_id, tool_name, dur, result, success)
        return result
    except Exception as exc:
        dur = int((time.time() - t0) * 1000)
        err_msg = json.dumps({"error": str(exc)})
        log_tool_result(user_id, tool_name, dur, err_msg, False)
        raise


async def _dispatch_tool(tool_name: str, arguments: dict[str, Any], user_id: str) -> str:
    base = FAKE_APP_BASE_URL.rstrip("/")

    if tool_name == "get_landmarks":
        url = f"{base}/api/landmarks"
        return await _get(url, {k: v for k, v in arguments.items() if v is not None}, user_id, need_user_id=False)

    if tool_name == "get_landmark_by_name":
        name = arguments.get("name", "")
        url = f"{base}/api/landmarks/name/{name}"
        return await _get(url, None, user_id, need_user_id=False)

    if tool_name == "search_landmarks":
        url = f"{base}/api/landmarks/search"
        return await _get(url, {k: v for k, v in arguments.items() if v is not None}, user_id, need_user_id=False)

    if tool_name == "get_landmark_by_id":
        id_ = arguments.get("id", "")
        url = f"{base}/api/landmarks/{id_}"
        return await _get(url, None, user_id, need_user_id=False)

    if tool_name == "get_house_by_id":
        house_id = arguments.get("house_id", "")
        url = f"{base}/api/houses/{house_id}"
        return await _get(url, None, user_id, need_user_id=True)

    if tool_name == "get_house_listings":
        house_id = arguments.get("house_id", "")
        url = f"{base}/api/houses/listings/{house_id}"
        return await _get(url, None, user_id, need_user_id=True)

    if tool_name == "get_houses_by_community":
        url = f"{base}/api/houses/by_community"
        params = {k: v for k, v in arguments.items() if v is not None}
        return await _get(url, params, user_id, need_user_id=True)

    if tool_name == "get_houses_by_platform":
        url = f"{base}/api/houses/by_platform"
        params = {k: v for k, v in arguments.items() if v is not None}
        return await _get(url, params, user_id, need_user_id=True)

    if tool_name == "get_houses_nearby":
        url = f"{base}/api/houses/nearby"
        params = {k: v for k, v in arguments.items() if v is not None}
        return await _get(url, params, user_id, need_user_id=True)

    if tool_name == "get_nearby_landmarks":
        url = f"{base}/api/houses/nearby_landmarks"
        params = {k: v for k, v in arguments.items() if v is not None}
        return await _get(url, params, user_id, need_user_id=True)

    if tool_name == "rent_house":
        house_id = arguments.get("house_id", "")
        listing_platform = arguments.get("listing_platform", "安居客")
        url = f"{base}/api/houses/{house_id}/rent?listing_platform={listing_platform}"
        return await _post(url, user_id=user_id, need_user_id=True)

    if tool_name == "terminate_rental":
        house_id = arguments.get("house_id", "")
        listing_platform = arguments.get("listing_platform", "安居客")
        url = f"{base}/api/houses/{house_id}/terminate?listing_platform={listing_platform}"
        return await _post(url, user_id=user_id, need_user_id=True)

    if tool_name == "take_offline":
        house_id = arguments.get("house_id", "")
        listing_platform = arguments.get("listing_platform", "安居客")
        url = f"{base}/api/houses/{house_id}/offline?listing_platform={listing_platform}"
        return await _post(url, user_id=user_id, need_user_id=True)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def get_tools_schema(round_num: int = 1) -> list[dict]:
    """返回 OpenAI 格式的 tools 列表。round_num=1 仅搜索，round_num>=2 含租房等。"""
    search_tools = [
        {"type": "function", "function": {"name": "search_landmarks", "description": "搜地标得ID", "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "category": {"type": "string"}, "district": {"type": "string"}}, "required": ["q"]}}},
        {"type": "function", "function": {"name": "get_houses_nearby", "description": "地标附近查房", "parameters": {"type": "object", "properties": {"landmark_id": {"type": "string"}, "max_distance": {"type": "number"}}, "required": ["landmark_id"]}}},
        {"type": "function", "function": {"name": "get_houses_by_platform", "description": "条件查房。district/price/bedrooms等。地标用search+get_houses_nearby", "parameters": {"type": "object", "properties": {"district": {"type": "string"}, "area": {"type": "string"}, "min_price": {"type": "integer"}, "max_price": {"type": "integer"}, "bedrooms": {"type": "string"}, "rental_type": {"type": "string"}, "decoration": {"type": "string"}, "elevator": {"type": "string"}, "min_area": {"type": "integer"}, "max_area": {"type": "integer"}, "max_subway_dist": {"type": "integer"}, "subway_station": {"type": "string"}, "commute_to_xierqi_max": {"type": "integer"}, "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]}, "sort_by": {"type": "string", "enum": ["price", "area", "subway"]}, "sort_order": {"type": "string", "enum": ["asc", "desc"]}}}}},
        {"type": "function", "function": {"name": "get_house_by_id", "description": "查房源详情含tags", "parameters": {"type": "object", "properties": {"house_id": {"type": "string"}}, "required": ["house_id"]}}},
        {"type": "function", "function": {"name": "get_houses_by_community", "description": "按小区名查房", "parameters": {"type": "object", "properties": {"community": {"type": "string"}}, "required": ["community"]}}},
        {"type": "function", "function": {"name": "get_nearby_landmarks", "description": "查小区周边公园商超", "parameters": {"type": "object", "properties": {"community": {"type": "string"}, "type": {"type": "string"}}, "required": ["community"]}}},
    ]
    action_tools = [
        {"type": "function", "function": {"name": "get_house_listings", "description": "查房源各平台挂牌价", "parameters": {"type": "object", "properties": {"house_id": {"type": "string"}}, "required": ["house_id"]}}},
        {"type": "function", "function": {"name": "rent_house", "description": "办理租房", "parameters": {"type": "object", "properties": {"house_id": {"type": "string"}, "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]}}, "required": ["house_id", "listing_platform"]}}},
        {"type": "function", "function": {"name": "terminate_rental", "description": "退租", "parameters": {"type": "object", "properties": {"house_id": {"type": "string"}, "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]}}, "required": ["house_id", "listing_platform"]}}},
        {"type": "function", "function": {"name": "take_offline", "description": "下架", "parameters": {"type": "object", "properties": {"house_id": {"type": "string"}, "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]}}, "required": ["house_id", "listing_platform"]}}},
    ]
    if round_num >= 2:
        return search_tools + action_tools
    return search_tools
