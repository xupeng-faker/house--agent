"""配置：租房仿真 API 基地址等"""
import os

# 租房仿真服务 base URL，评测时由环境或内网说明指定
FAKE_APP_BASE_URL = os.getenv("FAKE_APP_BASE_URL", "http://localhost:8080")

# 模型服务端口（固定）
MODEL_PORT = 8888

# 返回房源前是否启用模型二次质检筛选（根据用户需求与房源详情再筛一轮）
ENABLE_SECONDARY_QUALITY_CHECK = os.getenv("ENABLE_SECONDARY_QUALITY_CHECK", "true").lower() in ("1", "true", "yes")
