from .base import ResponseGenerator
from .deepseek_gen import DeepSeekGenerator
from .local_gen import LocalGenerator
from .hybrid_gen import HybridGenerator
from .config import RESPONSE_CONFIG

__all__ = [
    "ResponseGenerator",
    "DeepSeekGenerator",
    "LocalGenerator",
    "HybridGenerator",
    "RESPONSE_CONFIG",
    "create_response_generator"
]

def create_response_generator(config: dict = None):
    if config is None:
        config = RESPONSE_CONFIG
    mode = config["mode"]
    if mode == "api":
        return DeepSeekGenerator(api_key=config["api_key"], model=config["api_model"])
    elif mode == "local":
        return LocalGenerator(model_path=config["local_model_path"], state_dim=config["state_dim"])
    else:  # hybrid
        local = LocalGenerator(model_path=config["local_model_path"], state_dim=config["state_dim"])
        api = DeepSeekGenerator(api_key=config["api_key"], model=config["api_model"])
        return HybridGenerator(local, api, config["fallback_threshold"], config["max_fallback_per_session"])
