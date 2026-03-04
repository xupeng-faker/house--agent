"""
租房 Agent 主入口：提供 POST /api/v1/chat，对接模型与租房仿真 API，遵循 agent 输入输出约定。
"""
import json
import re
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_client import chat_completions, parse_assistant_message
from logger import log_request, log_request_response, log_response, service_log, tool_log
from prompts import SYSTEM_PROMPT
from rental_tools import get_tools_schema, init_houses, run_tool
from session_store import (
    append_messages,
    get_messages,
    is_initialized,
    set_initialized,
)

app = FastAPI(title="租房 Agent", description="需求理解、房源筛选与推荐")

service_log.info("租房 Agent 服务启动")


@app.get("/")
def root():
    return {"service": "租房 Agent", "chat": "POST /api/v1/chat"}


MAX_TOOL_ROUNDS = 5


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


# --------------- 辅助函数 ---------------

def _is_house_result_json(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        d = json.loads(text.strip())
        return isinstance(d, dict) and "message" in d and "houses" in d
    except json.JSONDecodeError:
        return False


def _try_extract_json(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        if _is_house_result_json(candidate):
            return candidate
    return None


def _normalize_house_id(x: Any) -> str | None:
    """将 houses 元素转为 HF_xxx 字符串。"""
    if isinstance(x, str) and x.startswith("HF_"):
        return x
    if isinstance(x, dict):
        return str(x.get("house_id") or x.get("id") or "")
    return str(x) if x else None


def _strip_markdown(text: str) -> str:
    """去除 message 中的 Markdown 格式，保留纯文本。"""
    if not text:
        return text
    # 去除 **bold** 格式
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # 去除 *italic* 格式
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # 将多余换行合并为单个换行
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# 已知虚假/占位房源 ID 模式（模型臆造）
_FAKE_HOUSE_ID_PATTERNS = frozenset({"HF_12345", "HF_67890", "HF_11111", "HF_22222", "HF_99999", "HF_00000"})


def _is_likely_fake_house_id(hid: str) -> bool:
    """检测是否为明显占位符 ID。"""
    if not hid or not hid.startswith("HF_"):
        return True
    if hid in _FAKE_HOUSE_ID_PATTERNS:
        return True
    # HF_12345、HF_67890 等 5 位连续数字常见占位
    suffix = hid[3:]
    if suffix.isdigit() and len(suffix) >= 5:
        num = int(suffix)
        if num in (12345, 67890, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999):
            return True
    return False


def _unwrap_nested_json(msg: str) -> tuple[str, list[str]] | None:
    """若 message 为内嵌 JSON 字符串，解析并返回 (message, houses)；否则返回 None。"""
    if not msg or not isinstance(msg, str) or not msg.strip().startswith("{"):
        return None
    try:
        inner = json.loads(msg.strip())
        if isinstance(inner, dict) and "message" in inner and "houses" in inner:
            inner_msg = inner.get("message", "")
            inner_houses = inner.get("houses", [])
            if not isinstance(inner_houses, list):
                inner_houses = []
            return str(inner_msg), [str(h) for h in inner_houses if isinstance(h, str) and h.startswith("HF_")]
    except json.JSONDecodeError:
        pass
    return None


def _clean_and_enforce_limit(json_str: str, valid_house_ids: set[str] | None = None) -> str:
    """严格输出仅含 message 和 houses 的 JSON，符合评测要求。"""
    try:
        raw = json_str.strip()
        if raw.startswith("\n"):
            raw = raw.lstrip("\n")
        d = json.loads(raw)
        houses = d.get("houses", [])
        if not isinstance(houses, list):
            houses = []
        message = _strip_markdown(d.get("message", ""))

        # 修复双重 JSON 编码：若 message 为内嵌 JSON，解析取内层
        unwrapped = _unwrap_nested_json(message)
        if unwrapped:
            message, houses = unwrapped

        normalized = []
        for h in houses[:5]:
            hid = _normalize_house_id(h)
            if not hid or not hid.startswith("HF_"):
                continue
            # 虚假 ID 防护
            if _is_likely_fake_house_id(hid):
                continue
            if valid_house_ids is not None and hid not in valid_house_ids:
                continue
            normalized.append(hid)

        # 若过滤后 houses 为空但原有过可疑 ID，修正 message
        if not normalized and houses and any(_is_likely_fake_house_id(_normalize_house_id(h) or "") for h in houses[:5]):
            message = "暂未找到符合条件的房源，建议调整筛选条件"

        return json.dumps({"message": message, "houses": normalized}, ensure_ascii=False)
    except Exception:
        return json_str


# get_houses_by_platform 支持的参数（API 无 tags 参数，需过滤）
_PLATFORM_VALID_KEYS = frozenset({
    "listing_platform", "district", "area", "min_price", "max_price", "bedrooms",
    "rental_type", "decoration", "orientation", "elevator", "min_area", "max_area",
    "subway_line", "max_subway_dist", "subway_station", "commute_to_xierqi_max",
    "sort_by", "sort_order", "page", "page_size",
})

_SEARCH_TOOLS = frozenset({
    "get_houses_by_platform", "get_houses_nearby", "get_houses_by_community",
    "search_landmarks", "get_landmark_by_name", "get_landmarks", "get_landmark_by_id",
})


def _looks_like_tool_call(text: str) -> bool:
    """检测文本是否形如 tool call JSON，避免将原始格式返回用户。"""
    if not text or not text.strip():
        return False
    try:
        d = json.loads(text.strip())
        return isinstance(d, dict) and "name" in d and "arguments" in d
    except json.JSONDecodeError:
        return False


def _parse_tool_call_content(text: str) -> tuple[str, dict] | None:
    """若 content 形如 {"name":"xxx","arguments":{...}} 则解析，否则返回 None。"""
    if not text or not text.strip():
        return None
    try:
        d = json.loads(text.strip())
        if not isinstance(d, dict):
            return None
        name = d.get("name")
        args = d.get("arguments")
        if name and isinstance(args, dict):
            return str(name), args
    except json.JSONDecodeError:
        pass
    return None


def _has_minimal_search_params(name: str, args: dict) -> bool:
    """检查是否有最小可执行参数。get_houses_by_platform 仅 district 或 max_price 即可。"""
    if name == "get_houses_by_platform":
        return bool(args.get("district") or args.get("area") or args.get("max_price") or args.get("min_price") or args.get("bedrooms"))
    if name == "get_houses_nearby":
        return bool(args.get("landmark_id"))
    if name == "get_houses_by_community":
        return bool(args.get("community"))
    return bool(args)


async def _try_execute_tool_call_from_content(content: str, user_id: str) -> str | None:
    """尝试将 content 中的 tool call 形态解析并执行，返回格式化结果或 None。"""
    parsed = _parse_tool_call_content(content)
    if not parsed:
        return None
    name, args = parsed
    if name not in _SEARCH_TOOLS:
        return None
    # 过滤非法参数（如 tags），放宽：仅 district 或 max_price 也允许执行
    if name == "get_houses_by_platform":
        args = {k: v for k, v in args.items() if k in _PLATFORM_VALID_KEYS and v is not None}
    else:
        args = {k: v for k, v in args.items() if v is not None}
    if not _has_minimal_search_params(name, args):
        return None
    try:
        raw = await run_tool(name, args, user_id)
        if not raw or "error" in raw.lower()[:200]:
            return None
        data = json.loads(raw)
        items = _extract_items(data)
        house_ids = []
        for item in (items or []):
            if isinstance(item, dict):
                hid = item.get("house_id") or item.get("id")
                if hid:
                    house_ids.append(str(hid))
        house_ids = house_ids[:5]
        msg = f"为您找到{len(house_ids)}套符合条件的房源" if house_ids else "暂无符合条件的房源，建议调整筛选条件"
        return json.dumps({"message": msg, "houses": house_ids}, ensure_ascii=False)
    except (json.JSONDecodeError, Exception) as e:
        service_log.warning("[TOOL_CALL_FALLBACK] 兜底执行失败: %s", e)
        return None


def _extract_house_ids_from_tool_output(output: str) -> list[str]:
    """从工具返回的 JSON 中提取 house_id 列表。"""
    ids = []
    try:
        data = json.loads(output)
        for item in _extract_items(data) or []:
            if isinstance(item, dict):
                hid = item.get("house_id") or item.get("id")
                if hid and str(hid).startswith("HF_"):
                    ids.append(str(hid))
    except (json.JSONDecodeError, TypeError):
        pass
    return ids


def _ensure_strict_json_response(text: str, valid_house_ids: set[str] | None = None) -> str:
    """最终兜底：规范为仅 message+houses 的 JSON，符合评测要求。"""
    if not text or not text.strip():
        return text
    extracted = _try_extract_json(text)
    if extracted:
        return _clean_and_enforce_limit(extracted, valid_house_ids)
    fallback = _fallback_extract_houses(text)
    if fallback:
        return _clean_and_enforce_limit(fallback, valid_house_ids)
    return text


def _fallback_extract_houses(text: str) -> str | None:
    ids = re.findall(r"(HF_\d+)", text)
    if ids:
        unique_ids = list(dict.fromkeys(ids))[:5]
        return json.dumps({"message": text, "houses": unique_ids}, ensure_ascii=False)
    return None


def _extract_items(data: Any) -> list | None:
    """从 API 返回的各种嵌套结构中提取 items 列表。
    支持: {"items":[...]}, {"data":{"items":[...]}}, {"data":[...]}, [...]
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return None
    # 直接有 items
    if "items" in data and isinstance(data["items"], list):
        return data["items"]
    # 嵌套在 data 下
    inner = data.get("data")
    if isinstance(inner, list):
        return inner
    if isinstance(inner, dict) and "items" in inner and isinstance(inner["items"], list):
        return inner["items"]
    return None


def _compress_tool_output(tool_name: str, output: str) -> str:
    if not output:
        return ""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return output[:1000]

    items = _extract_items(data)

    if items is not None and len(items) > 0 and isinstance(items[0], dict):
        slim_items = []
        for item in items:
            slim_item = {
                "id": item.get("house_id") or item.get("id"),
                "price": item.get("price"),
                "area": item.get("area"),
                "district": item.get("district"),
                "subway": item.get("subway_distance"),
                "bed": item.get("bedrooms"),
                "floor": item.get("floor"),
                "type": item.get("rental_type"),
                "orient": item.get("orientation"),
                "deco": item.get("decoration"),
                "elev": item.get("elevator"),
                "comm": item.get("community"),
                "noise": item.get("noise_level"),
                "avail": item.get("available_date"),
                "tags": item.get("tags"),
                "platform": item.get("listing_platform") or item.get("platform"),
                "status": item.get("status"),
            }
            slim_items.append({k: v for k, v in slim_item.items() if v is not None})
        return json.dumps(slim_items, ensure_ascii=False)

    return output[:1500]


# --------------- 短路与意图预处理 ---------------

_CN_NUM = {"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"}
_DISTRICTS = ["海淀", "朝阳", "通州", "昌平", "大兴", "房山", "西城", "丰台", "顺义", "东城"]
_BEDROOM_RE = re.compile(r"([一二两三四五1-5])\s*(?:居|室|房)")
_PRICE_MAX_RE = re.compile(r"(?:租金|预算|月租|价格).*?(\d+(?:k|千)?|\d{3,5})\s*(?:元|块)?(?:以[下内里]|以内|之内|之下)?", re.IGNORECASE)
_PRICE_MAX_RE2 = re.compile(r"(\d+(?:k|千)?|\d{3,5})\s*(?:元|块)?(?:以[下内里]|以内|之内|之下)", re.IGNORECASE)
_PRICE_RANGE_RE = re.compile(r"(\d+(?:k|千)?|\d{3,5})\s*(?:到|至|-|~)\s*(\d+(?:k|千)?|\d{3,5})", re.IGNORECASE)
_AREA_MIN_RE = re.compile(r"(\d{2,3})\s*(?:平|㎡)(?:以上|米以上)?")
_COMMUTE_RE = re.compile(r"通勤\s*(\d{1,3})\s*分钟")
_SUBWAY_LINE_RE = re.compile(r"(\d{1,2})号线")
_SUBWAY_DIST_RE = re.compile(r"(\d{3,4})\s*(?:米|m)(?:以?内)?")
_PLATFORMS = {"链家": "链家", "安居客": "安居客", "58同城": "58同城", "58": "58同城"}
# 地标/商圈 -> 行政区，用于无明确区域时的推断
_LANDMARK_DISTRICT = {"望京南": "朝阳", "望京": "朝阳", "立水桥": "朝阳", "双合站": "朝阳", "金融街": "西城", "西二旗": "海淀", "车公庄": "西城"}


def _extract_requirements_summary(messages: list[dict]) -> str | None:
    """从对话历史提取已确认需求，用于多轮上下文继承。"""
    parts = []
    all_text = " ".join(m.get("content", "") or "" for m in messages if m.get("role") == "user")
    for d in _DISTRICTS:
        if d in all_text:
            parts.append(f"区域={d}")
            break
    range_m = _PRICE_RANGE_RE.search(all_text)
    if range_m:
        parts.append(f"预算={range_m.group(1)}-{range_m.group(2)}元")
    else:
        price_m = _PRICE_MAX_RE.search(all_text) or _PRICE_MAX_RE2.search(all_text)
        if price_m:
            parts.append(f"预算≤{price_m.group(1)}元")
    bed_match = _BEDROOM_RE.search(all_text)
    if bed_match:
        raw = bed_match.group(1)
        parts.append(f"户型={_CN_NUM.get(raw, raw)}居室")
    if "整租" in all_text:
        parts.append("整租")
    elif "合租" in all_text:
        parts.append("合租")
    extras = []
    if any(kw in all_text for kw in ["养狗", "养猫", "宠物", "金毛"]):
        extras.append("养宠物")
    if any(kw in all_text for kw in ["不养宠物", "不养狗", "不养猫", "室友不养"]):
        extras.append("禁止宠物")
    if any(kw in all_text for kw in ["公园", "遛狗"]):
        extras.append("近公园")
    if any(kw in all_text for kw in ["VR", "线上看房", "不用跑现场"]):
        extras.append("线上看房")
    if any(kw in all_text for kw in ["月付", "押一付一"]):
        extras.append("月付")
    if any(kw in all_text for kw in ["房东直租", "无中介"]):
        extras.append("房东直租")
    if any(kw in all_text for kw in ["电梯", "有电梯"]):
        extras.append("需电梯")
    if any(kw in all_text for kw in ["民水民电", "民电"]):
        extras.append("民水民电")
    if any(kw in all_text for kw in ["租2个月", "租3个月", "短租", "可月租"]):
        extras.append("短租期")
    if any(kw in all_text for kw in ["链家", "安居客", "58"]):
        extras.append("平台偏好")
    if extras:
        parts.append("其他=" + "、".join(extras))
    if not parts:
        return None
    return "当前已确认需求：" + "，".join(parts)


def _extract_last_house_ids(messages: list[dict]) -> list[str]:
    """从对话历史中提取最近一次 assistant 回复的 houses 列表。"""
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "") or ""
        if not content.strip().startswith("{"):
            continue
        try:
            d = json.loads(content.strip())
            houses = d.get("houses", [])
            if isinstance(houses, list) and houses:
                return [str(h) for h in houses if isinstance(h, str) and h.startswith("HF_")][:5]
        except json.JSONDecodeError:
            pass
    return []


def _try_canned_response(msg: str) -> str | None:
    """尝试匹配基础对话，返回固定回复或 None。"""
    msg_lower = msg.lower().strip()
    
    # 纯打招呼
    if msg_lower in ["你好", "hello", "hi", "嗨", "您好", "你好呀", "hi~", "hello~"]:
        return "您好！我是智能租房助手，请问有什么可以帮您？"
    
    # 能力/自我介绍询问（可能和问候组合，如 "你好，你可以做什么"）
    capability_kws = ["你可以做什么", "你是谁", "你能做什么", "你的功能", "你能帮我做什么", "你会什么"]
    if any(x in msg_lower for x in capability_kws):
        return "您好！我是智能租房助手，帮您找房、查房、租房"
    
    # 道谢/结束
    if msg_lower in ["谢谢", "感谢", "多谢", "谢了", "thanks", "thank you", "好的谢谢"]:
        return "不客气，如有其他需求随时联系我！"
    if msg_lower in ["再见", "拜拜", "bye", "结束"]:
        return "再见，祝您找到满意的房子！"
    
    return None


def _try_direct_search(msg: str) -> dict | None:
    """从明确的租房需求中提取参数，直接构建 API 查询参数。"""
    # 排除：操作类、纯地标无区域（附近/上班/公司 等需 get_houses_nearby）
    # 放宽：含「站」或「望京」时，若同时有行政区(朝阳/海淀等)，仍可走短路
    if any(kw in msg for kw in ["办理", "预约", "退租", "退掉", "下架"]):
        return None
    has_district = any(d in msg for d in _DISTRICTS)
    if not has_district and any(kw in msg for kw in ["附近", "上班", "公司", "SOHO", "商圈", "商城", "国贸", "百度", "小米"]):
        return None
    # 2026-03-04 重构：若有明确小区名、大学、地标等，不走短路（交给 nearby/community）
    # 避免： "我想在望京西园找房" -> 提取 "望京"(朝阳) -> 搜全朝阳 -> 错误
    if any(kw in msg for kw in ["小区", "园", "里", "寓", "苑", "舍", "家", "村", "大厦", "中心", "广场", "大学", "学院", "医院", "公园"]):
        # 简单 heuristic: 常见小区后缀
        return None
    if not any(kw in msg for kw in ["找", "租", "房", "居室", "居", "套", "单间", "推荐", "看看"]):
        return None

    params: dict[str, Any] = {}

    for d in _DISTRICTS:
        if d in msg:
            params["district"] = d
            break
    if not params.get("district"):
        for lm, dist in _LANDMARK_DISTRICT.items():
            if lm in msg:
                params["district"] = dist
                break

    bed_match = _BEDROOM_RE.search(msg)
    if bed_match:
        raw = bed_match.group(1)
        params["bedrooms"] = _CN_NUM.get(raw, raw)

    range_match = _PRICE_RANGE_RE.search(msg)
    if range_match:
        def _parse_price(s: str) -> int:
            s = s.lower()
            if "k" in s:
                return int(float(s.replace("k", "")) * 1000)
            if "千" in s:
                return int(float(s.replace("千", "")) * 1000)
            return int(s)
        
        p1 = _parse_price(range_match.group(1))
        p2 = _parse_price(range_match.group(2))
        params["min_price"] = min(p1, p2)
        params["max_price"] = max(p1, p2)
    else:
        price_match = _PRICE_MAX_RE.search(msg) or _PRICE_MAX_RE2.search(msg)
        if price_match:
            def _parse_price(s: str) -> int:
                s = s.lower()
                if "k" in s:
                    return int(float(s.replace("k", "")) * 1000)
                if "千" in s:
                    return int(float(s.replace("千", "")) * 1000)
                return int(s)
            params["max_price"] = _parse_price(price_match.group(1))

    area_match = _AREA_MIN_RE.search(msg)
    if area_match:
        params["min_area"] = int(area_match.group(1))

    if "精装" in msg:
        params["decoration"] = "精装"
    elif "豪装" in msg or "豪华" in msg:
        params["decoration"] = "豪华"

    if "电梯" in msg:
        params["elevator"] = "true"

    subway_dist_match = _SUBWAY_DIST_RE.search(msg)
    if "近地铁" in msg:
        params["max_subway_dist"] = 800
    elif "地铁可达" in msg:
        params["max_subway_dist"] = 1000
    elif subway_dist_match and ("离地铁" in msg or "米" in msg or "地铁" in msg):
        params["max_subway_dist"] = int(subway_dist_match.group(1))

    commute_match = _COMMUTE_RE.search(msg)
    if commute_match:
        params["commute_to_xierqi_max"] = int(commute_match.group(1))

    if "整租" in msg:
        params["rental_type"] = "整租"
    elif "合租" in msg or "单间" in msg:
        params["rental_type"] = "合租"

    # 平台兼容：链家/58 数据可能为空，不传 listing_platform 避免误判无房
    for label, platform in _PLATFORMS.items():
        if label in msg:
            if platform in ("链家", "58同城"):
                break  # 不加入 listing_platform，让 API 返回所有平台数据
            params["listing_platform"] = platform
            break

    if "从低到高" in msg or "从便宜到贵" in msg:
        params["sort_by"] = "price"
        params["sort_order"] = "asc"
    elif "从高到低" in msg or "从贵到便宜" in msg:
        params["sort_by"] = "price"
        params["sort_order"] = "desc"
    elif "从大到小" in msg:
        params["sort_by"] = "area"
        params["sort_order"] = "desc"
    elif "从小到大" in msg:
        params["sort_by"] = "area"
        params["sort_order"] = "asc"
    elif "离地铁" in msg and ("从近到远" in msg or "排" in msg):
        params["sort_by"] = "subway"
        params["sort_order"] = "asc"
    elif "按租金" in msg:
        params["sort_by"] = "price"
        params["sort_order"] = "asc" if "低" in msg else "desc"
    elif "按面积" in msg:
        params["sort_by"] = "area"
        params["sort_order"] = "desc" if "大" in msg else "asc"

    subway_line_match = _SUBWAY_LINE_RE.search(msg)
    if subway_line_match:
        params["subway_line"] = subway_line_match.group(1)

    if not params.get("district"):
        return None
    has_filter = any(params.get(k) for k in (
        "bedrooms", "max_price", "min_price", "rental_type", "decoration",
        "elevator", "max_subway_dist", "commute_to_xierqi_max", "listing_platform",
        "sort_by", "subway_line",
    ))
    if not has_filter:
        return None

    return params


async def _do_direct_search(params: dict, user_id: str) -> str:
    """直接调用 get_houses_by_platform 并格式化为标准 JSON 输出。链家/58 无数据时自动回退安居客。"""
    raw_out = await run_tool("get_houses_by_platform", params, user_id)

    try:
        data = json.loads(raw_out)
    except json.JSONDecodeError:
        service_log.error("[DIRECT] raw_out 非 JSON: %s", raw_out[:500])
        return json.dumps({"message": "查询出错，请稍后重试", "houses": []}, ensure_ascii=False)

    items = _extract_items(data) or []
    did_platform_fallback = False
    # 平台兼容：链家/58 返回空时，用安居客重试
    if not items and params.get("listing_platform") in ("链家", "58同城"):
        retry_params = {k: v for k, v in params.items() if k != "listing_platform"}
        retry_params["listing_platform"] = "安居客"
        service_log.info("[DIRECT] 链家/58 无数据，回退安居客重试")
        raw_out = await run_tool("get_houses_by_platform", retry_params, user_id)
        try:
            data = json.loads(raw_out)
            items = _extract_items(data) or []
        except json.JSONDecodeError:
            pass
        if items:
            params = retry_params
            did_platform_fallback = True

    service_log.info("[DIRECT] API返回结构 keys=%s, 提取到 %d 条 items",
                     list(data.keys()) if isinstance(data, dict) else type(data).__name__, len(items))

    house_ids = []
    for item in items:
        if isinstance(item, dict):
            hid = item.get("house_id") or item.get("id")
            if hid:
                house_ids.append(str(hid))

    house_ids = house_ids[:5]

    if house_ids:
        msg = f"为您找到{len(house_ids)}套符合条件的房源"
        if did_platform_fallback:
            msg = "当前为您展示安居客房源。" + msg
    else:
        msg = "暂无符合条件的房源，建议调整筛选条件"

    return json.dumps({"message": msg, "houses": house_ids}, ensure_ascii=False)


# --------------- 核心 chat 接口 ---------------

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    log_request(session_id, user_message, req.model_ip)
    user_id = session_id

    # 新会话：调用房源数据重置
    if not is_initialized(session_id):
        service_log.info("[INIT] session=%s 新会话，初始化房源数据", session_id)
        try:
            await init_houses(user_id)
        except Exception as e:
            service_log.error("[INIT] session=%s 初始化失败: %s", session_id, e)
        set_initialized(session_id)

    # --- 短路 1：基础对话 ---
    canned = _try_canned_response(user_message)
    if canned:
        service_log.info("[SHORT] session=%s 基础对话短路, response=%s", session_id, canned)
        append_messages(session_id, {"role": "user", "content": user_message})
        append_messages(session_id, {"role": "assistant", "content": canned})
        log_request_response(session_id, user_message, canned)
        return ChatResponse(
            session_id=session_id, response=canned, status="success",
            tool_results=[], timestamp=int(time.time()), duration_ms=0,
        )

    # --- 短路 2：明确结构化查询，直接调 API（仅新会话首轮） ---
    messages_so_far = get_messages(session_id)
    if len(messages_so_far) == 0:
        search_params = _try_direct_search(user_message)
        if search_params is not None:
            service_log.info("[SHORT] session=%s 直接搜索短路, params=%s", session_id, search_params)
            start_ts = time.time()
            append_messages(session_id, {"role": "user", "content": user_message})
            try:
                direct_result = await _do_direct_search(search_params, user_id)
                direct_result = _ensure_strict_json_response(direct_result)
                dur = int((time.time() - start_ts) * 1000)
                log_response(session_id, "success", dur, direct_result)
                log_request_response(session_id, user_message, direct_result)
                append_messages(session_id, {"role": "assistant", "content": direct_result})
                return ChatResponse(
                    session_id=session_id, response=direct_result, status="success",
                    tool_results=[{"name": "get_houses_by_platform", "success": True, "output": direct_result}],
                    timestamp=int(start_ts), duration_ms=dur,
                )
            except Exception as e:
                service_log.warning("[SHORT] session=%s 直接搜索失败，回落到模型: %s", session_id, e)

    # 追加本轮用户消息
    append_messages(session_id, {"role": "user", "content": user_message})

    tools_schema = get_tools_schema()
    messages = get_messages(session_id)
    
    # 构建模型消息：系统提示 + 需求摘要（多轮时）+ 上一轮候选房源 + 压缩历史
    system_content = SYSTEM_PROMPT
    summary = _extract_requirements_summary(messages)
    if summary and len(messages) >= 2:
        system_content = system_content + f"\n\n[{summary}]"
    last_house_ids = _extract_last_house_ids(messages)
    if last_house_ids and len(messages) >= 4:
        system_content = system_content + f"\n\n[上一轮候选房源：{', '.join(last_house_ids)}。本轮新增条件请在上述房源中筛选，用 get_house_by_id 查详情后过滤，不得重新全量搜索。]"
    model_messages: list[dict] = [{"role": "system", "content": system_content}]

    KEEP_RECENT = 18  # 略增历史窗口，减少长对话截断
    total_msgs = len(messages)
    
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content", "") or ""
        is_old = i < (total_msgs - KEEP_RECENT)
        
        if role == "user":
            model_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            msg = {"role": "assistant", "content": content}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            model_messages.append(msg)
        elif role == "tool":
            if is_old:
                model_messages.append(
                    {"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": "History tool output."}
                )
            else:
                model_messages.append(
                    {"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": content}
                )

    tool_results: list[dict] = []
    start_ts = time.time()
    response_text = ""
    round_count = 0

    while round_count < MAX_TOOL_ROUNDS:
        round_count += 1
        service_log.info("[LOOP] session=%s round=%d/%d 开始模型调用", session_id, round_count, MAX_TOOL_ROUNDS)
        
        current_tools = tools_schema if round_count <= 5 else None
        
        response = await chat_completions(req.model_ip, model_messages, tools=current_tools, session_id=session_id)
        content, tool_calls = parse_assistant_message(response)

        if not tool_calls:
            response_text = content or ""
            service_log.info("[LOOP] session=%s round=%d 模型返回文本(无tool_calls), len=%d", session_id, round_count, len(response_text))
            # 收集本轮及之前轮次工具返回的有效房源 ID
            valid_house_ids = set()
            for tr in tool_results:
                for hid in _extract_house_ids_from_tool_output(tr.get("output", "") or ""):
                    valid_house_ids.add(hid)
            # 兜底：若 content 为 tool call 形态（模型误输出），尝试执行；严禁将原始 tool call JSON 返回用户
            parsed = _parse_tool_call_content(response_text)
            if parsed:
                fallback_result = await _try_execute_tool_call_from_content(response_text, user_id)
                if fallback_result:
                    try:
                        fb = json.loads(fallback_result)
                        for h in fb.get("houses", []):
                            if isinstance(h, str) and h.startswith("HF_"):
                                valid_house_ids.add(h)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    response_text = _ensure_strict_json_response(fallback_result, valid_house_ids or None)
                else:
                    response_text = json.dumps({"message": "查询暂时失败，请稍后重试或调整条件", "houses": []}, ensure_ascii=False)
            elif _looks_like_tool_call(response_text):
                response_text = json.dumps({"message": "查询暂时失败，请稍后重试或调整条件", "houses": []}, ensure_ascii=False)
            else:
                response_text = _ensure_strict_json_response(response_text, valid_house_ids or None)
            append_messages(session_id, {"role": "assistant", "content": response_text})
            break

        # 有 tool_calls：执行并追加 tool 结果
        tool_names = [((tc.get("function") or {}).get("name", "")) for tc in tool_calls]
        service_log.info("[LOOP] session=%s round=%d 模型请求工具调用: %s", session_id, round_count, tool_names)

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

            tool_log.info("[EXEC] session=%s round=%d tool=%s tid=%s args=%s", session_id, round_count, fn, tid, json.dumps(args, ensure_ascii=False)[:500])
            t_tool = time.time()
            try:
                raw_out = await run_tool(fn, args, user_id)
            except Exception as e:
                raw_out = json.dumps({"error": str(e)})
                tool_log.error("[EXEC] session=%s tool=%s 异常: %s", session_id, fn, e)
            tool_dur = int((time.time() - t_tool) * 1000)
            
            compressed_out = _compress_tool_output(fn, raw_out)
            success = "error" not in raw_out.lower()[:200]
            tool_results.append({"name": fn, "success": success, "output": raw_out[:2000]})
            tool_log.info("[EXEC] session=%s tool=%s success=%s duration=%dms compressed_len=%d", session_id, fn, success, tool_dur, len(compressed_out))

            model_messages.append({"role": "tool", "tool_call_id": tid, "content": compressed_out})
            tool_outputs.append({"role": "tool", "tool_call_id": tid, "content": compressed_out})
            
        append_messages(session_id, *tool_outputs)

    duration_ms = int((time.time() - start_ts) * 1000)
    log_response(session_id, "success", duration_ms, response_text)
    log_request_response(session_id, user_message, response_text)
    return ChatResponse(
        session_id=session_id, response=response_text, status="success",
        tool_results=tool_results, timestamp=int(start_ts), duration_ms=duration_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8191, reload=True)
