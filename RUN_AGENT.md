# 租房 Agent 运行说明

本目录为租房 Agent 实现，提供 `POST /api/v1/chat` 接口，对接模型与租房仿真 API，满足赛题目标与 agent 输入输出约定。

## 功能概览

- **需求交互与解析**：自然语言输入，提取预算、地段、户型、通勤、配套等；模糊需求时主动追问。
- **房源信息处理**：通过工具调用租房仿真 API，支持地标查询、按条件筛选、按地标附近查房、小区周边配套等。
- **多维度分析**：对候选房源从通勤、租金、配套、设施等维度分析，形成优缺点与推荐理由。
- **结果输出**：最终推荐最多 5 套高匹配度房源时，按约定返回 JSON：`{"message":"...", "houses":["HF_xxx", ...]}`。

## 环境要求

- Python 3.10+
- 租房仿真服务已启动（见 README 中接口列表，默认 `http://IP:8080`）
- 模型服务可访问（判题器通过 `model_ip` 下发，端口 8888）

## 配置

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `FAKE_APP_BASE_URL` | 租房仿真 API 基地址 | `http://localhost:8080` |

评测时若内网有指定 IP，请设置该变量。

## 安装与启动

```bash
cd "/Users/fakertony/Documents/cursor/house- agent"
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8191
```

服务将在 `http://localhost:8191` 提供：

- `GET /`：简单状态
- `POST /api/v1/chat`：对话接口

## 请求与响应格式

与 `agent.md` 约定一致。

**请求** `POST /api/v1/chat`：

```json
{
  "model_ip": "xxx.xxx.xx.x",
  "session_id": "abc123",
  "message": "我想在海淀近地铁找一套两居，预算 8000 以内"
}
```

**响应**：

- 普通对话：`response` 为自然语言文本。
- 房源查询完成后：`response` 为合法 JSON 字符串，包含 `message` 与 `houses`（房源 ID 列表，最多 5 个）。

```json
{
  "session_id": "abc123",
  "response": "{\"message\": \"为您找到以下符合条件的房源\", \"houses\": [\"HF_4\", \"HF_6\"]}",
  "status": "success",
  "tool_results": [...],
  "timestamp": 1704067200,
  "duration_ms": 1500
}
```

## 会话与 X-User-ID

- 同一 `session_id` 的多轮请求会共用会话上下文。
- 新会话会先调用租房仿真接口 `POST /api/houses/init` 做房源数据重置（若环境提供）。
- 调用所有房源相关接口时，使用 `session_id` 作为 `X-User-ID` 请求头。若评测平台在请求体中下发用户工号，可在 `main.py` 中改为优先使用该工号。

## 项目结构

```
main.py              # FastAPI 入口，/api/v1/chat
config.py            # FAKE_APP_BASE_URL 等配置
llm_client.py        # 调用 model_ip:8888 的 chat completions
prompts.py           # Agent 系统提示词
rental_tools.py      # 租房仿真 API 的 tools 定义与执行
session_store.py     # 会话消息存储
fake_app_agent_tools.json  # 租房仿真 OpenAPI 描述
目标.md / agent.md   # 赛题目标与接口约定
```
