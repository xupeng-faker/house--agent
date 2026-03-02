"""
租房仿真 API 工具：OpenAI 格式的 tools 定义 + 执行器。
请求头 X-User-ID 使用 session_id（或评测下发的用户工号）。
"""
import json
from typing import Any

import httpx

from config import FAKE_APP_BASE_URL


def _headers(user_id: str, need_user_id: bool = True) -> dict:
    h = {"Content-Type": "application/json"}
    if need_user_id:
        h["X-User-ID"] = user_id
    return h


async def _get(url: str, params: dict | None, user_id: str, need_user_id: bool = True) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params or {}, headers=_headers(user_id, need_user_id))
        r.raise_for_status()
        return r.text


async def _post(url: str, user_id: str, need_user_id: bool = True) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, headers=_headers(user_id, need_user_id))
        r.raise_for_status()
        return r.text


async def init_houses(user_id: str) -> str:
    """新会话时调用：重置当前用户视角下的房源数据。"""
    url = f"{FAKE_APP_BASE_URL}/api/houses/init"
    return await _post(url, user_id=user_id)


async def run_tool(tool_name: str, arguments: dict[str, Any], user_id: str) -> str:
    """根据 tool_name 和 arguments 调用租房仿真 API，返回结果 JSON 字符串。"""
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

    if tool_name == "get_landmark_stats":
        url = f"{base}/api/landmarks/stats"
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

    if tool_name == "get_house_stats":
        url = f"{base}/api/houses/stats"
        return await _get(url, None, user_id, need_user_id=True)

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
                "description": "获取地标列表。支持按 category(subway/company/landmark)、district(如海淀、朝阳)筛选。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "地标类别：subway/company/landmark"},
                        "district": {"type": "string", "description": "行政区，如海淀、朝阳"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_landmark_by_name",
                "description": "按名称精确查询地标，如西二旗站、百度。返回地标 id、经纬度等，用于后续 nearby 查房。",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "地标名称"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_landmarks",
                "description": "关键词模糊搜索地标。q 必填；可选 category、district。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "搜索关键词，必填"},
                        "category": {"type": "string", "description": "subway/company/landmark"},
                        "district": {"type": "string", "description": "行政区"},
                    },
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_landmark_by_id",
                "description": "按地标 id 查询地标详情。",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "description": "地标 ID"}},
                    "required": ["id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_landmark_stats",
                "description": "获取地标统计信息（总数、按类别分布等）。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_house_by_id",
                "description": "根据房源 ID 获取单套房源详情。",
                "parameters": {
                    "type": "object",
                    "properties": {"house_id": {"type": "string", "description": "房源 ID，如 HF_2001"}},
                    "required": ["house_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_house_listings",
                "description": "根据房源 ID 获取该房源在各平台(链家/安居客/58同城)的全部挂牌记录。",
                "parameters": {
                    "type": "object",
                    "properties": {"house_id": {"type": "string", "description": "房源 ID"}},
                    "required": ["house_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_houses_by_community",
                "description": "按小区名查询该小区下可租房源。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "community": {"type": "string", "description": "小区名"},
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
                "description": "按挂牌平台与多条件筛选可租房源。建议尽可能多地填充参数以精确匹配用户需求。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                        "district": {"type": "string", "description": "行政区，逗号分隔，如 '海淀,朝阳'"},
                        "area": {"type": "string", "description": "商圈，逗号分隔"},
                        "min_price": {"type": "integer", "description": "最低月租金(元)"},
                        "max_price": {"type": "integer", "description": "最高月租金(元)"},
                        "bedrooms": {"type": "string", "description": "卧室数，逗号分隔如 '1,2'"},
                        "rental_type": {"type": "string", "description": "整租或合租"},
                        "decoration": {"type": "string", "description": "精装/简装等"},
                        "orientation": {"type": "string", "description": "朝向，如 '朝南'"},
                        "elevator": {"type": "string", "description": "是否有电梯：true/false"},
                        "min_area": {"type": "integer"},
                        "max_area": {"type": "integer"},
                        "subway_line": {"type": "string"},
                        "max_subway_dist": {"type": "integer", "description": "最大地铁距离(米)。'近地铁'建议设为 800，'地铁可达'建议设为 1000"},
                        "subway_station": {"type": "string"},
                        "commute_to_xierqi_max": {"type": "integer", "description": "到西二旗通勤时间上限(分钟)，如 30, 45, 60"},
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
                "description": "以地标为圆心查附近可租房源。注意：必须先调用 get_landmark_by_name 或 search_landmarks 获取 landmark_id 后才能使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "landmark_id": {"type": "string", "description": "地标 ID (必填，非名称)"},
                        "max_distance": {"type": "number", "description": "最大直线距离(米)，默认2000"},
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
                "description": "查询某小区周边地标，用于判断生活便利程度（查商超）或环境（查公园）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "community": {"type": "string", "description": "小区名"},
                        "type": {"type": "string", "description": "shopping(商超) 或 park(公园)"},
                        "max_distance_m": {"type": "number", "description": "最大距离(米)，默认 3000"},
                    },
                    "required": ["community"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_house_stats",
                "description": "获取房源统计信息（总套数、按状态/行政区/户型分布、价格区间等）。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rent_house",
                "description": "租房：将指定房源设为已租。必须调用此接口才算完成租房操作。",
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
                "description": "退租：将房源恢复为可租。必须调用此接口才算完成退租。",
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
                "description": "下架：将房源设为下架。必须调用此接口才算完成下架。",
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
