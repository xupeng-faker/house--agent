"""配置：租房仿真 API 基地址等"""
import os

# 租房仿真服务 base URL，评测时由环境或内网说明指定
FAKE_APP_BASE_URL = os.getenv("FAKE_APP_BASE_URL", "http://localhost:8080")

# 模型服务端口（固定）
MODEL_PORT = 8888
