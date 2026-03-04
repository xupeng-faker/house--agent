"""Zero-LLM shortcuts for rent, terminate, compare-price, nearby-landmarks."""
from __future__ import annotations

import json
import re
from typing import Any

from api_client import run_tool, extract_items
from logger import log
from session import Session


def _make_response(message: str, houses: list[str]) -> str:
    return json.dumps({"message": message, "houses": houses}, ensure_ascii=False)


async def do_compare_price(sess: Session, user_id: str) -> tuple[str, list[dict]] | None:
    """Compare prices for the first candidate across platforms."""
    if not sess.candidates:
        return None
    house_id = sess.candidates[0]
    tool_results: list[dict] = []

    try:
        raw = await run_tool("get_house_listings", {"house_id": house_id}, user_id)
        tool_results.append({"name": "get_house_listings", "success": True, "output": raw[:2000]})
    except Exception:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    listings = extract_items(data) or data.get("listings") or data.get("items") or []
    cheapest_plat = "安居客"
    cheapest_price = float("inf")
    price_lines: list[str] = []

    for item in listings:
        if not isinstance(item, dict):
            continue
        plat = item.get("listing_platform") or item.get("platform") or ""
        price = item.get("price") or item.get("rent") or 0
        try:
            p = int(price)
        except (TypeError, ValueError):
            continue
        price_lines.append(f"{plat} {p}元/月")
        if p < cheapest_price:
            cheapest_price = p
            cheapest_plat = plat or "安居客"

    msg = "、".join(price_lines) if price_lines else "暂无挂牌信息"
    if cheapest_plat and cheapest_price != float("inf"):
        msg = f"{house_id}各平台挂牌价：{msg}。最便宜为{cheapest_plat}，{int(cheapest_price)}元/月"

    return _make_response(msg, [house_id]), tool_results


async def do_rent(sess: Session, msg: str, user_id: str) -> tuple[str, list[dict]] | None:
    """Rent the first candidate, optionally at cheapest platform."""
    if not sess.candidates:
        return None
    house_id = sess.candidates[0]
    tool_results: list[dict] = []

    # Determine platform: if user says "最便宜的平台", get listings first
    platform = "安居客"
    if "最便宜" in msg:
        try:
            raw = await run_tool("get_house_listings", {"house_id": house_id}, user_id)
            tool_results.append({"name": "get_house_listings", "success": True, "output": raw[:2000]})
            data = json.loads(raw)
            listings = extract_items(data) or data.get("listings") or []
            cheapest_price = float("inf")
            for item in listings:
                if not isinstance(item, dict):
                    continue
                p = int(item.get("price") or item.get("rent") or 999999)
                plat = item.get("listing_platform") or item.get("platform") or ""
                if p < cheapest_price:
                    cheapest_price = p
                    platform = plat or "安居客"
        except Exception:
            pass

    try:
        raw_rent = await run_tool("rent_house", {"house_id": house_id, "listing_platform": platform}, user_id)
        tool_results.append({"name": "rent_house", "success": True, "output": raw_rent[:2000]})
        sess.rented_house = (house_id, platform)
        return _make_response(f"已为您在{platform}办理租房，房源{house_id}。", [house_id]), tool_results
    except Exception as e:
        log.warning("[RENT] failed: %s", e)
        return None


async def do_terminate(sess: Session, user_id: str) -> tuple[str, list[dict]] | None:
    """Terminate rental for the most recently rented house."""
    if not sess.rented_house:
        return None
    house_id, platform = sess.rented_house
    tool_results: list[dict] = []

    try:
        raw = await run_tool("terminate_rental", {"house_id": house_id, "listing_platform": platform}, user_id)
        tool_results.append({"name": "terminate_rental", "success": True, "output": raw[:2000]})
        sess.rented_house = None
        return _make_response(f"已为您办理退租，房源{house_id}。", [house_id]), tool_results
    except Exception as e:
        log.warning("[TERMINATE] failed: %s", e)
        return None


_LM_TYPE_MAP = {
    "公园": "公园", "菜市场": "菜市场", "医院": "医院", "学校": "学校",
    "商场": "商超", "商超": "商超", "餐饮": "餐饮", "餐馆": "餐饮",
    "健身房": "健身房", "警察局": "警察局", "派出所": "警察局",
    "银行": "银行", "加油站": "加油站",
}


async def do_nearby_landmark(sess: Session, msg: str, user_id: str) -> tuple[str, list[dict]] | None:
    """Query nearby landmarks for the first candidate's community."""
    if not sess.candidates:
        return None
    house_id = sess.candidates[0]

    lm_type = None
    for kw, t in _LM_TYPE_MAP.items():
        if kw in msg:
            lm_type = t
            break
    if not lm_type:
        return None

    try:
        raw_house = await run_tool("get_house_by_id", {"house_id": house_id}, user_id)
        h = json.loads(raw_house)
        h = h.get("data") or h.get("house") or h
        community = h.get("community") or ""
        if not community:
            return None

        raw_lm = await run_tool("get_nearby_landmarks", {"community": community, "type": lm_type}, user_id)
        data = json.loads(raw_lm)
        landmarks = data.get("landmarks") or data.get("items") or extract_items(data) or []

        if not landmarks:
            result = _make_response(f"{community}附近暂无{lm_type}信息", sess.candidates[:5])
        else:
            names = []
            for lm in landmarks[:5]:
                if isinstance(lm, dict):
                    n = lm.get("name") or lm.get("landmark_name") or ""
                    d = lm.get("distance") or ""
                    if n:
                        names.append(f"{n}({d}m)" if d else n)
            result = _make_response(
                f"附近有{lm_type}：{'、'.join(names)}" if names else f"{community}附近暂无{lm_type}信息",
                sess.candidates[:5],
            )
        return result, [{"name": "get_nearby_landmarks", "success": True, "output": raw_lm[:500]}]
    except Exception as e:
        log.warning("[NEARBY_LM] failed: %s", e)
        return None
