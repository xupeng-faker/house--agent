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
from logger import log_request, log_response, service_log, tool_log
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
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        if _is_house_result_json(candidate):
            return candidate
    return None


def _clean_and_enforce_limit(json_str: str) -> str:
    try:
        d = json.loads(json_str)
        houses = d.get("houses", [])
        if not isinstance(houses, list):
            houses = []
        if len(houses) > 5:
            houses = houses[:5]
        return json.dumps({"message": d.get("message", ""), "houses": houses}, ensure_ascii=False)
    except Exception:
        return json_str


def _fallback_extract_houses(text: str) -> str | None:
    ids = re.findall(r"(HF_\d+)", text)
    if ids:
        unique_ids = list(dict.fromkeys(ids))[:5]
        return json.dumps({"message": text, "houses": unique_ids}, ensure_ascii=False)
    return None


def _compress_tool_output(tool_name: str, output: str) -> str:
    if not output:
        return ""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return output[:1000]

    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        items = None

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
            }
            slim_items.append({k: v for k, v in slim_item.items() if v is not None})
        return json.dumps(slim_items, ensure_ascii=False)

    return output[:1500]


# --------------- 短路与意图预处理 ---------------

# 中文数字映射
_CN_NUM = {"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"}

# 行政区列表
_DISTRICTS = ["海淀", "朝阳", "通州", "昌平", "大兴", "房山", "西城", "丰台", "顺义", "东城"]

# 户型映射：中文 -> bedrooms 参数
_BEDROOM_RE = re.compile(r"([一二两三四五1-5])\s*居")

# 租金上限
_PRICE_MAX_RE = re.compile(r"(?:租金|预算|月租|价格).*?(\d{3,5})\s*(?:元|块)?(?:以[下内里]|以内|之内|之下)?")
_PRICE_MAX_RE2 = re.compile(r"(\d{3,5})\s*(?:元|块)?(?:以[下内里]|以内|之内|之下)")

# 面积
_AREA_MIN_RE = re.compile(r"(\d{2,3})\s*(?:平|㎡)(?:以上|米以上)?")


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
    """尝试从明确的租房需求中提取参数，直接构建 API 查询参数。
    返回 get_houses_by_platform 的参数字典，或 None。
    """
    # 必须包含租房意图关键词
    if not any(kw in msg for kw in ["找", "租", "房", "居室", "居", "套"]):
        return None
    
    params: dict[str, Any] = {}
    
    # 1. 提取行政区
    for d in _DISTRICTS:
        if d in msg:
            params["district"] = d
            break
    
    # 2. 提取户型
    bed_match = _BEDROOM_RE.search(msg)
    if bed_match:
        raw = bed_match.group(1)
        params["bedrooms"] = _CN_NUM.get(raw, raw)
    
    # 3. 提取租金上限
    price_match = _PRICE_MAX_RE.search(msg) or _PRICE_MAX_RE2.search(msg)
    if price_match:
        params["max_price"] = int(price_match.group(1))
    
    # 4. 面积下限
    area_match = _AREA_MIN_RE.search(msg)
    if area_match:
        params["min_area"] = int(area_match.group(1))
    
    # 5. 装修
    if "精装" in msg:
        params["decoration"] = "精装"
    elif "豪装" in msg or "豪华" in msg:
        params["decoration"] = "豪华"
    
    # 6. 电梯
    if "电梯" in msg or "有电梯" in msg:
        params["elevator"] = "true"
    
    # 7. 近地铁
    if "近地铁" in msg:
        params["max_subway_dist"] = 800
    elif "地铁可达" in msg:
        params["max_subway_dist"] = 1000
    
    # 8. 排序：按价格/地铁
    if "便宜" in msg or "价格" in msg:
        params["sort_by"] = "price"
        params["sort_order"] = "asc"
    elif "近地铁" in msg or "离地铁" in msg:
        params["sort_by"] = "subway"
        params["sort_order"] = "asc"
    
    # 必须至少有一个有效筛选条件（区域 或 户型 或 价格），否则不够精准
    if not (params.get("district") or params.get("bedrooms") or params.get("max_price")):
        return None
    
    return params


async def _do_direct_search(params: dict, user_id: str) -> str:
    """直接调用 get_houses_by_platform 并格式化为标准 JSON 输出。"""
    raw_out = await run_tool("get_houses_by_platform", params, user_id)
    
    try:
        data = json.loads(raw_out)
    except json.JSONDecodeError:
        return json.dumps({"message": "查询出错，请稍后重试", "houses": []}, ensure_ascii=False)
    
    # 从返回数据中提取房源 ID
    items = []
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    
    house_ids = []
    for item in items:
        hid = item.get("house_id") or item.get("id")
        if hid:
            house_ids.append(str(hid))
    
    house_ids = house_ids[:5]
    
    if house_ids:
        msg = f"为您找到{len(house_ids)}套符合条件的房源"
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
                dur = int((time.time() - start_ts) * 1000)
                log_response(session_id, "success", dur, direct_result)
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
    
    # 构建模型消息：系统提示 + 压缩历史
    model_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    KEEP_RECENT = 10
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
        
        current_tools = tools_schema if round_count <= 3 else None
        
        response = await chat_completions(req.model_ip, model_messages, tools=current_tools, session_id=session_id)
        content, tool_calls = parse_assistant_message(response)

        if not tool_calls:
            response_text = content or ""
            service_log.info("[LOOP] session=%s round=%d 模型返回文本(无tool_calls), len=%d", session_id, round_count, len(response_text))
            
            cleaned_json = _try_extract_json(response_text)
            if cleaned_json:
                response_text = _clean_and_enforce_limit(cleaned_json)
            else:
                fallback_json = _fallback_extract_houses(response_text)
                if fallback_json:
                    response_text = fallback_json

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
    return ChatResponse(
        session_id=session_id, response=response_text, status="success",
        tool_results=tool_results, timestamp=int(start_ts), duration_ms=duration_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8191, reload=True)
