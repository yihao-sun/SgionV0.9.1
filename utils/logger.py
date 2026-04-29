import logging
import logging.handlers
import os
from .config_loader import Config

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

_loggers = {}

def get_logger(name):
    if name in _loggers:
        return _loggers[name]
    cfg = Config()
    log_level = cfg.get('logging.level', 'INFO')
    module_level = cfg.get(f'logging.modules.{name}', log_level)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, module_level))
    
    # 避免重复添加 handler
    if not logger.handlers:
        # 使用绝对路径确保日志文件在项目根目录的 logs 文件夹中
        log_file = os.path.join(PROJECT_ROOT, 'logs', 'ee.log')
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True, mode=0o755)
        
        # 使用普通的FileHandler避免文件滚动导致的权限问题
        file_handler = logging.FileHandler(
            log_file
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    _loggers[name] = logger
    return logger
