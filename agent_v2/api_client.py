"""HTTP client for the rental simulation API."""
import json
import time
from typing import Any

import httpx

from config import FAKE_APP_BASE_URL
from logger import tool_log

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0, trust_env=False)
    return _client


def _headers(user_id: str, need_uid: bool = True) -> dict:
    h = {"Content-Type": "application/json"}
    if need_uid:
        h["X-User-ID"] = user_id
    return h


async def _get(url: str, params: dict | None, user_id: str, need_uid: bool = True) -> str:
    c = _get_client()
    r = await c.get(url, params=params or {}, headers=_headers(user_id, need_uid))
    r.raise_for_status()
    return r.text


async def _post(url: str, user_id: str, need_uid: bool = True) -> str:
    c = _get_client()
    r = await c.post(url, headers=_headers(user_id, need_uid))
    r.raise_for_status()
    return r.text


async def init_houses(user_id: str) -> str:
    tool_log.info("[INIT] user=%s", user_id)
    return await _post(f"{FAKE_APP_BASE_URL}/api/houses/init", user_id)


async def run_tool(name: str, args: dict[str, Any], user_id: str) -> str:
    """Dispatch a tool call to the rental API and return raw JSON string."""
    tool_log.info("[CALL] tool=%s args=%s", name, json.dumps(args, ensure_ascii=False)[:500])
    t0 = time.time()
    try:
        result = await _dispatch(name, args, user_id)
        dur = int((time.time() - t0) * 1000)
        tool_log.info("[RESULT] tool=%s dur=%dms len=%d", name, dur, len(result))
        return result
    except Exception as exc:
        dur = int((time.time() - t0) * 1000)
        tool_log.error("[ERROR] tool=%s dur=%dms err=%s", name, dur, exc)
        return json.dumps({"error": str(exc)})


async def _dispatch(name: str, args: dict[str, Any], uid: str) -> str:
    base = FAKE_APP_BASE_URL.rstrip("/")
    clean = {k: v for k, v in args.items() if v is not None}

    if name == "get_landmarks":
        return await _get(f"{base}/api/landmarks", clean, uid, need_uid=False)

    if name == "get_landmark_by_name":
        return await _get(f"{base}/api/landmarks/name/{clean.get('name', '')}", None, uid, need_uid=False)

    if name == "search_landmarks":
        return await _get(f"{base}/api/landmarks/search", clean, uid, need_uid=False)

    if name == "get_landmark_by_id":
        return await _get(f"{base}/api/landmarks/{clean.get('id', '')}", None, uid, need_uid=False)

    if name == "get_house_by_id":
        return await _get(f"{base}/api/houses/{clean.get('house_id', '')}", None, uid)

    if name == "get_house_listings":
        return await _get(f"{base}/api/houses/listings/{clean.get('house_id', '')}", None, uid)

    if name == "get_houses_by_community":
        return await _get(f"{base}/api/houses/by_community", clean, uid)

    if name == "get_houses_by_platform":
        return await _get(f"{base}/api/houses/by_platform", clean, uid)

    if name == "get_houses_nearby":
        return await _get(f"{base}/api/houses/nearby", clean, uid)

    if name == "get_nearby_landmarks":
        return await _get(f"{base}/api/houses/nearby_landmarks", clean, uid)

    if name == "rent_house":
        hid = clean.get("house_id", "")
        plat = clean.get("listing_platform", "安居客")
        return await _post(f"{base}/api/houses/{hid}/rent?listing_platform={plat}", uid)

    if name == "terminate_rental":
        hid = clean.get("house_id", "")
        plat = clean.get("listing_platform", "安居客")
        return await _post(f"{base}/api/houses/{hid}/terminate?listing_platform={plat}", uid)

    if name == "take_offline":
        hid = clean.get("house_id", "")
        plat = clean.get("listing_platform", "安居客")
        return await _post(f"{base}/api/houses/{hid}/offline?listing_platform={plat}", uid)

    return json.dumps({"error": f"Unknown tool: {name}"})


def extract_items(data: Any) -> list:
    """Extract the items list from various API response shapes."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if "items" in data and isinstance(data["items"], list):
        return data["items"]
    if "houses" in data and isinstance(data["houses"], list):
        return data["houses"]
    inner = data.get("data")
    if isinstance(inner, list):
        return inner
    if isinstance(inner, dict) and isinstance(inner.get("items"), list):
        return inner["items"]
    if isinstance(inner, dict) and isinstance(inner.get("houses"), list):
        return inner["houses"]
    return []


def extract_house_ids(raw: str) -> list[str]:
    """Parse API JSON response and return list of house_id strings."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    ids = []
    for item in extract_items(data):
        if isinstance(item, dict):
            hid = item.get("house_id") or item.get("id")
            if hid and str(hid).startswith("HF_"):
                ids.append(str(hid))
    return ids
