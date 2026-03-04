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
        _api_client = httpx.AsyncClient(timeout=30.0)
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


def get_tools_schema() -> list[dict]:
    """返回 OpenAI 格式的 tools 列表，供模型调用。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_landmarks",
                "description": "查地标。category: subway/company/landmark, district: 行政区",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "district": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_landmark_by_name",
                "description": "按名查地标ID。name: 地标名",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_landmarks",
                "description": "搜地标。q: 关键词",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "category": {"type": "string"},
                        "district": {"type": "string"},
                    },
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_landmark_by_id",
                "description": "按ID查地标详情",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_house_by_id",
                "description": "查房源详情，含价格、面积、朝向、装修、tags（如仅限小型犬、押一付三、包物业费、房东直租、可短租、仅线下看房、仅工作日看房、线上VR看房等）",
                "parameters": {
                    "type": "object",
                    "properties": {"house_id": {"type": "string"}},
                    "required": ["house_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_house_listings",
                "description": "查房源在各平台(链家/安居客/58同城)的挂牌记录与价格。办理租房前若未指定平台，可先调用此接口获取挂牌平台列表，再选其一调用 rent_house。",
                "parameters": {
                    "type": "object",
                    "properties": {"house_id": {"type": "string"}},
                    "required": ["house_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_houses_by_community",
                "description": "查小区房源",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "community": {"type": "string"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                        "page": {"type": "integer"},
                        "page_size": {"type": "integer"},
                    },
                    "required": ["community"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_houses_by_platform",
                "description": "条件查房。params: district, price, bedrooms, subway_dist, commute_to_xierqi_max... listing_platform 为可选，不传时搜索所有平台；若指定链家/58 无结果，必须不传或改安居客重试。注意：望京南、立水桥站、金融街、公司附近等商圈/地标需求，应先用 search_landmarks 获 landmark_id，再用 get_houses_nearby，勿用本接口按 district 搜。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "listing_platform": {"type": "string", "description": "可选，仅在有明确平台需求时传入；链家/58 可能无数据，无结果时可不传重试", "enum": ["链家", "安居客", "58同城"]},
                        "district": {"type": "string"},
                        "area": {"type": "string"},
                        "min_price": {"type": "integer"},
                        "max_price": {"type": "integer"},
                        "bedrooms": {"type": "string"},
                        "rental_type": {"type": "string"},
                        "decoration": {"type": "string"},
                        "orientation": {"type": "string"},
                        "elevator": {"type": "string"},
                        "min_area": {"type": "integer"},
                        "max_area": {"type": "integer"},
                        "subway_line": {"type": "string"},
                        "max_subway_dist": {"type": "integer"},
                        "subway_station": {"type": "string"},
                        "commute_to_xierqi_max": {"type": "integer"},
                        "sort_by": {"type": "string", "enum": ["price", "area", "subway"]},
                        "sort_order": {"type": "string", "enum": ["asc", "desc"]},
                        "page": {"type": "integer"},
                        "page_size": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_houses_nearby",
                "description": "地标附近查房。landmark_id必填",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "landmark_id": {"type": "string"},
                        "max_distance": {"type": "number"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                        "page": {"type": "integer"},
                        "page_size": {"type": "integer"},
                    },
                    "required": ["landmark_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_nearby_landmarks",
                "description": "查小区周边配套。type=park 查公园(遛狗等)，type=shopping 查商超。需先有房源得小区名，再查该小区周边。用于「附近有公园」「能遛狗」等需求。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "community": {"type": "string"},
                        "type": {"type": "string", "description": "park=公园, shopping=商超"},
                        "max_distance_m": {"type": "number"},
                    },
                    "required": ["community"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rent_house",
                "description": "执行租房操作。用户说「帮我办理租房」「就租这套」「帮我预约」「我就租了」时必须调用。平台未指定时先 get_house_listings 查挂牌平台再选其一，或默认安居客。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                    },
                    "required": ["house_id", "listing_platform"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminate_rental",
                "description": "执行退租操作。用户说「帮我退掉」「退租」「不租了」时必须调用。平台未指定时先 get_house_listings 或默认安居客。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                    },
                    "required": ["house_id", "listing_platform"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "take_offline",
                "description": "执行下架操作。用户说「下架」时必须调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                    },
                    "required": ["house_id", "listing_platform"],
                },
            },
        },
    ]
