"""Core agent: intent routing, conversation loop, response generation."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from api_client import run_tool, extract_house_ids, extract_items
from filters import (
    do_community_search,
    do_landmark_search,
    do_search,
    filter_with_fallback,
)
from intent import (
    Intent,
    classify_intent,
    extract_community_name,
    extract_filter_specs,
    extract_landmark_query,
    extract_search_params,
    get_canned_response,
    LANDMARK_KEYWORDS,
)
from llm_client import chat_completions, parse_response
from logger import log, log_req_resp
from prompts import SYSTEM_PROMPT
from session import Session
from shortcuts import do_compare_price, do_nearby_landmark, do_rent, do_terminate
from tools import get_tools_schema


def _resp_json(message: str, houses: list[str]) -> str:
    return json.dumps({"message": message, "houses": houses[:5]}, ensure_ascii=False)


def _is_valid_house_json(text: str) -> bool:
    try:
        d = json.loads(text.strip())
        return isinstance(d, dict) and "message" in d and "houses" in d
    except (json.JSONDecodeError, TypeError):
        return False


def _try_extract_json(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        if _is_valid_house_json(candidate):
            return candidate
    return None


def _strip_markdown(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _normalize_response(text: str, valid_ids: set[str] | None = None) -> str:
    """Ensure response is valid JSON with message+houses or plain text."""
    if not text or not text.strip():
        return _resp_json("查询失败，请重试", [])

    # Try to extract JSON
    extracted = _try_extract_json(text)
    if extracted:
        try:
            d = json.loads(extracted)
            msg = _strip_markdown(d.get("message", ""))
            houses = d.get("houses", [])
            if not isinstance(houses, list):
                houses = []
            normalized = []
            for h in houses[:5]:
                hid = str(h) if not isinstance(h, dict) else str(h.get("house_id") or h.get("id") or "")
                if hid.startswith("HF_"):
                    if valid_ids is None or hid in valid_ids:
                        normalized.append(hid)
            return _resp_json(msg, normalized)
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to find HF_ IDs in text
    ids = re.findall(r"(HF_\d+)", text)
    if ids:
        unique = list(dict.fromkeys(ids))[:5]
        return _resp_json(_strip_markdown(text), unique)

    return text


def _compress_tool_output(tool_name: str, output: str) -> str:
    """Compress tool output to reduce token usage."""
    if not output:
        return ""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return output[:1000]

    if tool_name == "get_nearby_landmarks":
        landmarks = data.get("landmarks") or data.get("items") or extract_items(data) or []
        if landmarks and isinstance(landmarks[0], dict):
            by_type: dict[str, list[str]] = {}
            for lm in landmarks[:15]:
                name = lm.get("name") or lm.get("landmark_name") or ""
                dist = lm.get("distance") or ""
                t = lm.get("type") or lm.get("category") or "地标"
                if name:
                    by_type.setdefault(t, []).append(f"{name}({dist}m)" if dist else name)
            lines = [f"{k}：{', '.join(v[:5])}" for k, v in by_type.items()]
            return "\n".join(lines) if lines else "暂无周边地标"
        return output[:500]

    items = extract_items(data)
    if items and isinstance(items[0], dict):
        header = "id|price|area|dist|subway|bed|type|decor|elev|tags"
        rows = [header]
        for item in items[:10]:
            hid = str(item.get("house_id") or item.get("id") or "")
            price = str(item.get("price") or "")
            area = str(item.get("area_sqm") or item.get("area") or "")
            dist = str(item.get("district") or "")
            sub = str(item.get("subway_distance") or "")
            bed = str(item.get("bedrooms") or "")
            typ = str(item.get("rental_type") or "")
            deco = str(item.get("decoration") or "")
            elev = "Y" if item.get("elevator") else "N"
            tags = ",".join((item.get("tags") or [])[:4])
            rows.append(f"{hid}|{price}|{area}|{dist}|{sub}|{bed}|{typ}|{deco}|{elev}|{tags}")
        return "\n".join(rows)

    return output[:1000]


async def handle_message(
    sess: Session,
    user_message: str,
    model_ip: str,
    user_id: str,
) -> tuple[str, list[dict]]:
    """Main entry: process user message and return (response_text, tool_results)."""
    sess.turn += 1
    has_candidates = bool(sess.candidates)
    intent = classify_intent(user_message, has_candidates, sess.turn)

    log.info("[INTENT] session=%s turn=%d intent=%s msg=%s",
             sess.session_id, sess.turn, intent.value, user_message[:100])

    # ── Canned responses ──────────────────────────────────────────────
    canned = get_canned_response(intent)
    if canned:
        sess.append_msg({"role": "user", "content": user_message})
        sess.append_msg({"role": "assistant", "content": canned})
        return canned, []

    # ── Terminate ─────────────────────────────────────────────────────
    if intent == Intent.TERMINATE:
        sess.append_msg({"role": "user", "content": user_message})
        result = await do_terminate(sess, user_id)
        if result:
            resp, tr = result
            sess.append_msg({"role": "assistant", "content": resp})
            return resp, tr
        # Fall through to LLM if no rented house found
        return await _llm_fallback(sess, user_message, model_ip, user_id)

    # ── Compare price ─────────────────────────────────────────────────
    if intent == Intent.COMPARE_PRICE:
        sess.append_msg({"role": "user", "content": user_message})
        result = await do_compare_price(sess, user_id)
        if result:
            resp, tr = result
            sess.append_msg({"role": "assistant", "content": resp})
            return resp, tr
        return await _llm_fallback(sess, user_message, model_ip, user_id)

    # ── Rent ──────────────────────────────────────────────────────────
    if intent == Intent.RENT:
        sess.append_msg({"role": "user", "content": user_message})
        result = await do_rent(sess, user_message, user_id)
        if result:
            resp, tr = result
            sess.append_msg({"role": "assistant", "content": resp})
            return resp, tr
        return await _llm_fallback(sess, user_message, model_ip, user_id)

    # ── Nearby landmark ───────────────────────────────────────────────
    if intent == Intent.NEARBY_LANDMARK:
        sess.append_msg({"role": "user", "content": user_message})
        result = await do_nearby_landmark(sess, user_message, user_id)
        if result:
            resp, tr = result
            sess.append_msg({"role": "assistant", "content": resp})
            return resp, tr
        return await _llm_fallback(sess, user_message, model_ip, user_id)

    # ── Community query ───────────────────────────────────────────────
    if intent == Intent.COMMUNITY_QUERY:
        sess.append_msg({"role": "user", "content": user_message})
        community = extract_community_name(user_message)
        if community:
            all_ids, top5 = await do_community_search(community, user_id)
            sess.all_results = all_ids
            sess.candidates = top5
            msg = f"为您找到{len(all_ids)}套{community}在租房源" if top5 else f"暂未找到{community}在租房源"
            resp = _resp_json(msg, top5)
            sess.append_msg({"role": "assistant", "content": resp})
            return resp, [{"name": "get_houses_by_community", "success": bool(top5), "output": resp}]
        return await _llm_fallback(sess, user_message, model_ip, user_id)

    # ── Landmark search ───────────────────────────────────────────────
    if intent == Intent.LANDMARK_SEARCH:
        sess.append_msg({"role": "user", "content": user_message})
        lm_query = extract_landmark_query(user_message)
        if lm_query:
            lm_name, max_dist = lm_query
            # Also extract other params
            params = extract_search_params(user_message)
            sess.merge_requirements(params)

            all_ids, top5 = await do_landmark_search(lm_name, max_dist, user_id)

            # Post-filter by tags/fields if user specified extra conditions
            specs = extract_filter_specs(user_message)
            tags_req = specs.get("tags_require", [])
            tags_exc = specs.get("tags_exclude", [])
            ff = specs.get("field_filters", [])

            if all_ids and (tags_req or tags_exc or ff):
                from filters import filter_candidates
                matched, _ = await filter_candidates(all_ids[:20], tags_req, tags_exc, ff, user_id)
                if matched:
                    top5 = matched[:5]

            # Also filter by params (bedrooms, price, rental_type, decoration)
            if all_ids and params:
                filtered = await _post_filter_by_params(all_ids[:30], params, user_id)
                if filtered:
                    top5 = filtered[:5]
                    all_ids = filtered

            sess.all_results = all_ids
            sess.candidates = top5
            msg = f"为您找到{len(top5)}套{lm_name}附近房源" if top5 else f"暂未找到{lm_name}附近在租房源"
            resp = _resp_json(msg, top5)
            sess.append_msg({"role": "assistant", "content": resp})
            return resp, [{"name": "get_houses_nearby", "success": bool(top5), "output": resp}]
        return await _llm_fallback(sess, user_message, model_ip, user_id)

    # ── Filter (multi-turn) ───────────────────────────────────────────
    if intent == Intent.FILTER:
        sess.append_msg({"role": "user", "content": user_message})
        specs = extract_filter_specs(user_message)
        tags_req = specs.get("tags_require", [])
        tags_exc = specs.get("tags_exclude", [])
        ff = specs.get("field_filters", [])
        updated_params = specs.get("updated_params", {})

        # Merge accumulated requirements
        if tags_req:
            sess.merge_requirements({"tags_require": tags_req})
        if tags_exc:
            sess.merge_requirements({"tags_exclude": tags_exc})
        if ff:
            sess.merge_requirements({"field_filters": ff})
        if updated_params:
            sess.merge_requirements(updated_params)

        # Combine all accumulated tag/field filters
        all_tags_req = sess.requirements.get("tags_require", [])
        all_tags_exc = sess.requirements.get("tags_exclude", [])
        all_ff = sess.requirements.get("field_filters", [])

        matched, tr = await filter_with_fallback(sess, all_tags_req, all_tags_exc, all_ff, user_id)
        if matched:
            sess.candidates = matched[:5]
            msg = f"已为您筛选出{len(matched)}套符合条件的房源"
            resp = _resp_json(msg, matched[:5])
        else:
            # Keep old candidates but inform user
            msg = "暂无完全符合所有条件的房源，建议调整部分筛选条件"
            resp = _resp_json(msg, sess.candidates[:5])

        sess.append_msg({"role": "assistant", "content": resp})
        return resp, tr

    # ── Search (first round or new search) ────────────────────────────
    if intent == Intent.SEARCH:
        sess.append_msg({"role": "user", "content": user_message})
        params = extract_search_params(user_message)
        sess.merge_requirements(params)

        # Also extract tag requirements from the first message
        specs = extract_filter_specs(user_message)
        tags_req = specs.get("tags_require", [])
        tags_exc = specs.get("tags_exclude", [])
        ff = specs.get("field_filters", [])
        if tags_req:
            sess.merge_requirements({"tags_require": tags_req})
        if tags_exc:
            sess.merge_requirements({"tags_exclude": tags_exc})
        if ff:
            sess.merge_requirements({"field_filters": ff})

        # Check if this is a landmark query embedded in a search
        lm_query = extract_landmark_query(user_message)
        if lm_query:
            lm_name, max_dist = lm_query
            all_ids, top5 = await do_landmark_search(lm_name, max_dist, user_id)
        else:
            search_params = sess.build_search_params()
            if not search_params.get("district") and not search_params.get("area"):
                # No district -- fall through to LLM
                return await _llm_fallback(sess, user_message, model_ip, user_id)
            _, all_ids, top5 = await do_search(search_params, user_id)

        # Post-filter by accumulated tags if any
        all_tags_req = sess.requirements.get("tags_require", [])
        all_tags_exc = sess.requirements.get("tags_exclude", [])
        all_ff = sess.requirements.get("field_filters", [])
        if all_ids and (all_tags_req or all_tags_exc or all_ff):
            from filters import filter_candidates
            matched, _ = await filter_candidates(all_ids[:30], all_tags_req, all_tags_exc, all_ff, user_id)
            if matched:
                top5 = matched[:5]
                all_ids = matched + [i for i in all_ids if i not in matched]

        sess.all_results = all_ids
        sess.candidates = top5
        msg = f"为您找到{len(top5)}套符合条件的房源" if top5 else "暂无符合条件的房源，建议调整筛选条件"
        resp = _resp_json(msg, top5)
        sess.append_msg({"role": "assistant", "content": resp})
        return resp, [{"name": "search", "success": bool(top5), "output": resp}]

    # ── Complex / fallback to LLM ─────────────────────────────────────
    sess.append_msg({"role": "user", "content": user_message})
    return await _llm_fallback(sess, user_message, model_ip, user_id)


async def _post_filter_by_params(
    house_ids: list[str], params: dict[str, Any], user_id: str
) -> list[str]:
    """Post-filter house IDs by checking details against params like bedrooms, price, rental_type."""
    matched = []
    for hid in house_ids:
        try:
            raw = await run_tool("get_house_by_id", {"house_id": hid}, user_id)
            h = json.loads(raw)
            h = h.get("data") or h.get("house") or h
        except Exception:
            continue
        if not isinstance(h, dict):
            continue

        ok = True
        if params.get("bedrooms"):
            if str(h.get("bedrooms")) != str(params["bedrooms"]):
                ok = False
        if params.get("max_price"):
            if (h.get("price") or 999999) > int(params["max_price"]):
                ok = False
        if params.get("min_price"):
            if (h.get("price") or 0) < int(params["min_price"]):
                ok = False
        if params.get("rental_type"):
            if h.get("rental_type") != params["rental_type"]:
                ok = False
        if params.get("decoration"):
            if params["decoration"] == "毛坯":
                if h.get("decoration") not in ("毛坯", "空房"):
                    ok = False
            elif h.get("decoration") != params["decoration"]:
                ok = False
        if params.get("elevator") == "true":
            if not (h.get("elevator") is True or str(h.get("elevator")).lower() in ("true", "1")):
                ok = False
        if ok:
            matched.append(hid)
    return matched


async def _llm_fallback(
    sess: Session,
    user_message: str,
    model_ip: str,
    user_id: str,
) -> tuple[str, list[dict]]:
    """Fall back to LLM tool-calling loop."""
    log.info("[LLM_FALLBACK] session=%s", sess.session_id)

    # Build messages for LLM
    system_content = SYSTEM_PROMPT
    # Add accumulated requirements context
    if sess.requirements:
        parts = []
        for k, v in sess.requirements.items():
            if k in ("tags_require", "tags_exclude", "field_filters"):
                continue
            if v is not None:
                parts.append(f"{k}={v}")
        if parts:
            system_content += f"\n\n[已确认需求：{', '.join(parts)}]"
    if sess.candidates:
        system_content += f"\n\n[上一轮候选房源：{', '.join(sess.candidates[:5])}]"

    model_messages: list[dict] = [{"role": "system", "content": system_content}]

    # Include recent conversation history (last 10 messages)
    recent = sess.messages[-10:]
    for m in recent:
        msg = {"role": m.get("role"), "content": m.get("content", "") or ""}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        model_messages.append(msg)

    tool_results: list[dict] = []
    valid_ids: set[str] = set()
    response_text = ""
    max_rounds = 5

    for round_num in range(1, max_rounds + 1):
        tools_schema = get_tools_schema(round_num=round_num)
        response = await chat_completions(model_ip, model_messages, tools=tools_schema, session_id=sess.session_id)
        content, tool_calls = parse_response(response)

        if not tool_calls:
            response_text = content or ""
            response_text = _normalize_response(response_text, valid_ids or None)
            sess.append_msg({"role": "assistant", "content": response_text})

            # Update candidates from response
            try:
                d = json.loads(response_text)
                new_houses = d.get("houses", [])
                if new_houses:
                    sess.candidates = [str(h) for h in new_houses if str(h).startswith("HF_")][:5]
            except (json.JSONDecodeError, TypeError):
                pass
            break

        # Execute tool calls
        assistant_msg = {"role": "assistant", "content": content or "", "tool_calls": tool_calls}
        model_messages.append(assistant_msg)
        sess.append_msg(assistant_msg)

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
                raw_out = await run_tool(fn, args, user_id)
            except Exception as e:
                raw_out = json.dumps({"error": str(e)})

            # Collect valid house IDs
            for hid in extract_house_ids(raw_out):
                valid_ids.add(hid)

            compressed = _compress_tool_output(fn, raw_out)
            tool_results.append({"name": fn, "success": "error" not in raw_out.lower()[:200], "output": raw_out[:2000]})

            tool_msg = {"role": "tool", "tool_call_id": tid, "content": compressed}
            model_messages.append(tool_msg)
            tool_outputs.append(tool_msg)

        sess.append_msg(*tool_outputs)

    # Fallback if loop exhausted
    if not response_text:
        if valid_ids:
            ids_list = list(valid_ids)[:5]
            response_text = _resp_json(f"为您找到{len(ids_list)}套符合条件的房源", ids_list)
        else:
            response_text = _resp_json("查询失败，请重试", [])
        sess.append_msg({"role": "assistant", "content": response_text})
        try:
            d = json.loads(response_text)
            new_houses = d.get("houses", [])
            if new_houses:
                sess.candidates = new_houses[:5]
        except (json.JSONDecodeError, TypeError):
            pass

    return response_text, tool_results
