"""Rental Agent v2 -- FastAPI entry point."""
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import handle_message
from api_client import init_houses
from config import USER_ID
from logger import log, log_req_resp
from session import get_session

app = FastAPI(title="租房 Agent v2")
log.info("租房 Agent v2 服务启动")


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


@app.get("/")
def root():
    return {"service": "租房 Agent v2", "chat": "POST /api/v1/chat"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    user_id = USER_ID
    sess = get_session(session_id)

    # Initialize houses on first message of each session
    if not sess.initialized:
        try:
            await init_houses(user_id)
        except Exception as e:
            log.warning("[INIT] init_houses failed: %s", e)
        sess.initialized = True

    log.info("[REQ] session=%s model_ip=%s msg=%s", session_id, req.model_ip, user_message[:200])
    start_ts = time.time()

    response_text, tool_results = await handle_message(sess, user_message, req.model_ip, user_id)

    duration_ms = int((time.time() - start_ts) * 1000)
    log.info("[RESP] session=%s dur=%dms resp=%s", session_id, duration_ms, response_text[:500])
    log_req_resp(session_id, user_message, response_text)

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        status="success",
        tool_results=tool_results,
        timestamp=int(start_ts),
        duration_ms=duration_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8191, reload=True)
