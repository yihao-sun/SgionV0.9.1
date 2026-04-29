# 响应生成模块配置
import os

RESPONSE_CONFIG = {
    "mode": "hybrid",           # "api", "local", "hybrid"
    "api_key": os.environ.get("DEEPSEEK_API_KEY", "sk-f14a069d26c64bed93dcd691c4862a08"),
    "api_model": "deepseek-chat",
    "local_model_path": "models/response_generator",  # 微调后的模型目录
    "state_dim": 768,
    "fallback_threshold": 0.3,
    "max_fallback_per_session": 3,
}
