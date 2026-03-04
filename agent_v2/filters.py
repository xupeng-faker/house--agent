"""Multi-turn tag/field filtering with fallback re-search."""
from __future__ import annotations

import json
from typing import Any

from api_client import run_tool, extract_house_ids, extract_items
from logger import log, tool_log
from session import Session

# Equivalent tags: requiring one also accepts its equivalents
TAG_EQUIVALENTS: dict[str, list[str]] = {
    "包宽带": ["包宽带", "免宽带费"],
    "免宽带费": ["免宽带费", "包宽带"],
    "包水电费": ["包水电费", "免水电费"],
    "免水电费": ["免水电费", "包水电费"],
    "包物业费": ["包物业费", "免物业费"],
    "免物业费": ["免物业费", "包物业费"],
    "免车位费": ["免车位费", "包车位"],
    "包车位": ["包车位", "免车位费"],
    "可养猫": ["可养猫", "可养宠物"],
    "可养狗": ["可养狗", "可养宠物"],
    "可养宠物": ["可养宠物", "可养猫", "可养狗"],
    "近公园": ["近公园"],
    "全天可看房": ["全天可看房", "线下+线上"],
}

# For "周末看房", accept any weekend tag
WEEKEND_VIEWING_TAGS = {"周末14-18点", "周末9-12点", "周末9-18点", "仅周末看房", "全天可看房", "线下+线上"}
WORKDAY_VIEWING_TAGS = {"工作日14-18点", "工作日9-12点", "工作日9-18点", "仅工作日看房", "全天可看房", "线下+线上"}


def house_matches(h: dict, tags_require: list[str], tags_exclude: list[str],
                  field_filters: list[tuple[str, str, Any]]) -> bool:
    """Check if a house dict matches all filter conditions."""
    tags = h.get("tags") or []

    # Check required tags
    for req in tags_require:
        accepted = TAG_EQUIVALENTS.get(req, [req])
        if not any(t in tags for t in accepted):
            return False

    # Check excluded tags
    for exc in tags_exclude:
        if exc in tags:
            return False

    # Check field filters
    for field, op, value in field_filters:
        actual = h.get(field)
        if op == "eq":
            if isinstance(value, bool):
                actual_bool = actual is True or str(actual).lower() in ("true", "1", "有")
                if actual_bool != value:
                    return False
            elif str(actual) != str(value):
                return False
        elif op == "contains":
            if value not in str(actual or ""):
                return False
        elif op == "not_in":
            if str(actual or "") in value:
                return False

    return True


async def filter_candidates(
    house_ids: list[str],
    tags_require: list[str],
    tags_exclude: list[str],
    field_filters: list[tuple[str, str, Any]],
    user_id: str,
) -> tuple[list[str], list[dict]]:
    """Filter house IDs by fetching details and checking conditions.
    Returns (matched_ids, tool_results)."""
    matched: list[str] = []
    tool_results: list[dict] = []

    for hid in house_ids:
        try:
            raw = await run_tool("get_house_by_id", {"house_id": hid}, user_id)
            tool_results.append({"name": "get_house_by_id", "success": True, "output": raw[:500]})
        except Exception:
            continue

        try:
            data = json.loads(raw)
            h = data.get("data") or data.get("house") or data
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(h, dict):
            continue

        if house_matches(h, tags_require, tags_exclude, field_filters):
            matched.append(hid)

    return matched, tool_results


async def filter_with_fallback(
    sess: Session,
    tags_require: list[str],
    tags_exclude: list[str],
    field_filters: list[tuple[str, str, Any]],
    user_id: str,
) -> tuple[list[str], list[dict]]:
    """Multi-turn filter with fallback:
    1. Filter current candidates (top 5)
    2. If empty, filter all_results (up to 50)
    3. If still empty, re-search with accumulated params and filter
    Returns (matched_ids, tool_results)."""

    # Step 1: filter current candidates
    if sess.candidates:
        matched, tr = await filter_candidates(
            sess.candidates, tags_require, tags_exclude, field_filters, user_id
        )
        if matched:
            return matched[:5], tr

    # Step 2: filter broader result set
    broader = [hid for hid in sess.all_results if hid not in sess.candidates]
    if broader:
        matched2, tr2 = await filter_candidates(
            broader[:20], tags_require, tags_exclude, field_filters, user_id
        )
        if matched2:
            return matched2[:5], tr2

    # Step 3: re-search with accumulated params
    params = sess.build_search_params()
    if params.get("district") or params.get("area"):
        log.info("[FILTER_FALLBACK] Re-searching with params: %s", params)
        try:
            raw = await run_tool("get_houses_by_platform", params, user_id)
            all_ids = extract_house_ids(raw)
            if all_ids:
                sess.all_results = all_ids
                matched3, tr3 = await filter_candidates(
                    all_ids[:30], tags_require, tags_exclude, field_filters, user_id
                )
                if matched3:
                    return matched3[:5], tr3
        except Exception as e:
            log.warning("[FILTER_FALLBACK] Re-search failed: %s", e)

    return [], []


async def do_search(params: dict[str, Any], user_id: str) -> tuple[str, list[str], list[str]]:
    """Execute get_houses_by_platform search.
    Returns (raw_output, all_ids, top5_ids)."""
    # Ensure page_size
    if "page_size" not in params:
        params["page_size"] = 50

    raw = await run_tool("get_houses_by_platform", params, user_id)
    all_ids = extract_house_ids(raw)

    # Platform fallback: if 链家/58同城 returns empty, try 安居客
    if not all_ids and params.get("listing_platform") in ("链家", "58同城"):
        retry = {k: v for k, v in params.items() if k != "listing_platform"}
        retry["listing_platform"] = "安居客"
        log.info("[SEARCH] Platform fallback to 安居客")
        raw = await run_tool("get_houses_by_platform", retry, user_id)
        all_ids = extract_house_ids(raw)

    # Subway station fallback
    if not all_ids and params.get("subway_station"):
        retry = {k: v for k, v in params.items() if k != "subway_station"}
        if not retry.get("max_subway_dist"):
            retry["max_subway_dist"] = 800
        log.info("[SEARCH] Subway station fallback")
        raw = await run_tool("get_houses_by_platform", retry, user_id)
        all_ids = extract_house_ids(raw)

    return raw, all_ids, all_ids[:5]


async def do_landmark_search(landmark_q: str, max_dist: int, user_id: str) -> tuple[list[str], list[str]]:
    """Search houses near a landmark.
    Returns (all_ids, top5_ids)."""
    try:
        raw_search = await run_tool("search_landmarks", {"q": landmark_q}, user_id)
        data = json.loads(raw_search)
    except Exception as e:
        log.warning("[LANDMARK] search failed: %s", e)
        return [], []

    landmarks = data.get("landmarks") or data.get("items") or extract_items(data) or []
    if not landmarks or not isinstance(landmarks[0], dict):
        return [], []

    lid = landmarks[0].get("id") or landmarks[0].get("landmark_id") or ""
    if not lid:
        return [], []

    try:
        raw = await run_tool("get_houses_nearby", {
            "landmark_id": lid, "max_distance": max_dist, "page_size": 50
        }, user_id)
        all_ids = extract_house_ids(raw)
        return all_ids, all_ids[:5]
    except Exception as e:
        log.warning("[LANDMARK] nearby search failed: %s", e)
        return [], []


async def do_community_search(community: str, user_id: str) -> tuple[list[str], list[str]]:
    """Search houses by community name.
    Returns (all_ids, top5_ids)."""
    try:
        raw = await run_tool("get_houses_by_community", {"community": community}, user_id)
        all_ids = extract_house_ids(raw)
        return all_ids, all_ids[:5]
    except Exception as e:
        log.warning("[COMMUNITY] search failed: %s", e)
        return [], []
