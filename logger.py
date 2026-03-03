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
