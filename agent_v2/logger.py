import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_FMT = "%(asctime)s | %(name)-9s | %(levelname)-5s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _make_handler(name: str) -> TimedRotatingFileHandler:
    h = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}.log"),
        when="midnight", backupCount=7, encoding="utf-8",
    )
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    return h


_console = logging.StreamHandler()
_console.setLevel(logging.DEBUG)
_console.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

for _n in ("service", "tool", "model", "reqresp"):
    _l = logging.getLogger(_n)
    _l.setLevel(logging.DEBUG)
    _l.propagate = False
    _l.addHandler(_console)
    _l.addHandler(_make_handler(_n))

log = logging.getLogger("service")
tool_log = logging.getLogger("tool")
model_log = logging.getLogger("model")
reqresp_log = logging.getLogger("reqresp")


def _safe_json(obj, max_len: int = 0) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if max_len and len(s) > max_len:
        return s[:max_len] + "…"
    return s


def log_req_resp(session_id: str, req_msg: str, resp: str) -> None:
    from datetime import datetime
    reqresp_log.info("请求响应 | %s", json.dumps({
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "request_message": req_msg,
        "response_content": resp,
    }, ensure_ascii=False, indent=2))
