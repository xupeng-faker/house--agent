"""
统一日志管理模块。

日志分四类（各自独立 logger），同时输出到控制台和按日期轮转的文件：
  - service   : 服务生命周期（启动、请求进出、异常）
  - tool      : 工具调用（名称、参数、返回值、耗时）
  - model_in  : 发送给模型的完整 messages + tools schema
  - model_out : 模型返回的完整 response JSON
"""
import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

os.makedirs(LOG_DIR, exist_ok=True)

_LOGGER_NAMES = ("service", "tool", "model_in", "model_out")

_FMT = "%(asctime)s | %(name)-9s | %(levelname)-5s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_REQRESP_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"


def _make_file_handler(name: str) -> TimedRotatingFileHandler:
    path = os.path.join(LOG_DIR, f"{name}.log")
    handler = TimedRotatingFileHandler(
        path, when="midnight", interval=1, backupCount=7, encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    return handler


def _make_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    return handler


_console = _make_console_handler()

for _name in _LOGGER_NAMES:
    _logger = logging.getLogger(_name)
    _logger.setLevel(LOG_LEVEL)
    _logger.propagate = False
    _logger.addHandler(_console)
    _logger.addHandler(_make_file_handler(_name))


_reqresp_logger = logging.getLogger("reqresp")
_reqresp_logger.setLevel(LOG_LEVEL)
_reqresp_logger.propagate = False
_reqresp_console = logging.StreamHandler()
_reqresp_console.setLevel(logging.DEBUG)
_reqresp_console.setFormatter(logging.Formatter(_REQRESP_FMT, datefmt=_DATE_FMT))
_reqresp_logger.addHandler(_reqresp_console)
_reqresp_file = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "reqresp.log"), when="midnight", interval=1, backupCount=7, encoding="utf-8",
)
_reqresp_file.setLevel(logging.DEBUG)
_reqresp_file.setFormatter(logging.Formatter(_REQRESP_FMT, datefmt=_DATE_FMT))
_reqresp_logger.addHandler(_reqresp_file)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _safe_json(obj, max_len: int = 0) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if max_len and len(s) > max_len:
        return s[:max_len] + "...(truncated)"
    return s


# ---- 便捷方法 ----

service_log = get_logger("service")
tool_log = get_logger("tool")
model_in_log = get_logger("model_in")
model_out_log = get_logger("model_out")


def log_request(session_id: str, message: str, model_ip: str) -> None:
    service_log.info(
        "[REQ] session=%s model_ip=%s message=%s",
        session_id, model_ip, message[:200],
    )


def log_response(session_id: str, status: str, duration_ms: int, response: str) -> None:
    service_log.info(
        "[RESP] session=%s status=%s duration=%dms response=%s",
        session_id, status, duration_ms, response[:500],
    )


def log_tool_call(session_id: str, tool_name: str, arguments: dict) -> None:
    tool_log.info(
        "[CALL] session=%s tool=%s args=%s",
        session_id, tool_name, _safe_json(arguments),
    )


def log_tool_result(session_id: str, tool_name: str, duration_ms: int, output: str, success: bool) -> None:
    tool_log.info(
        "[RESULT] session=%s tool=%s success=%s duration=%dms output=%s",
        session_id, tool_name, success, duration_ms, output[:3000],
    )


def log_model_input(session_id: str, url: str, messages: list, tools: list | None) -> None:
    model_in_log.debug(
        "[MODEL_IN] session=%s url=%s\nmessages=%s\ntools=%s",
        session_id, url,
        _safe_json(messages),
        _safe_json(tools) if tools else "None",
    )


def log_model_output(session_id: str, response: dict, duration_ms: int) -> None:
    model_out_log.debug(
        "[MODEL_OUT] session=%s duration=%dms\nresponse=%s",
        session_id, duration_ms,
        _safe_json(response),
    )


def log_request_response(session_id: str, request_message: str, response_content: str) -> None:
    from datetime import datetime
    payload = json.dumps({
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "request_message": request_message,
        "response_content": response_content,
    }, ensure_ascii=False, indent=2)
    _reqresp_logger.info("请求响应 | %s", payload)


# ---- 筛选日志（JSON 行格式：每次筛选的消息名称、全集、给用户的房源）----

_filter_logger = logging.getLogger("filter")
_filter_logger.setLevel(logging.DEBUG)
_filter_logger.propagate = False
_filter_logger.addHandler(_console)
_filter_file_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "filter.log"),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
_filter_file_handler.setFormatter(logging.Formatter("%(message)s"))
_filter_logger.addHandler(_filter_file_handler)


def log_filter(
    session_id: str,
    message_name: str,
    full_house_ids: list[str],
    user_house_ids: list[str],
    qc_input_house_ids: list[str] | None = None,
) -> None:
    """记录每次筛选：消息名称、筛选出来房源的全集、给用户的房源；若经过二次质检则记录质检前后列表。JSON 格式，一行一条。"""
    from datetime import datetime
    payload = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "message_name": message_name,
        "full_house_ids": full_house_ids,
        "user_house_ids": user_house_ids,
    }
    if qc_input_house_ids is not None:
        payload["secondary_qc_applied"] = True
        payload["qc_input_house_ids"] = qc_input_house_ids
        payload["qc_output_house_ids"] = user_house_ids
    else:
        payload["secondary_qc_applied"] = False
    line = json.dumps(payload, ensure_ascii=False)
    _filter_logger.info("%s", line)
