import yaml
from pathlib import Path
import copy

class Config:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def copy(self):
        """返回配置对象的深拷贝，用于安全传递给可能修改配置的模块"""
        new_config = Config.__new__(Config)
        new_config._data = copy.deepcopy(self._data)
        return new_config
    
    def reload(self):
        self._load()