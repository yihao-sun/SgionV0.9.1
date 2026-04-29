import pickle
import json
from pathlib import Path
from datetime import datetime

class StateManager:
    VERSION = 1
    
    def __init__(self, save_dir="./saved_states"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, state_dict, name="latest"):
        """state_dict 包含所有需要持久化的模块状态"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_v{self.VERSION}_{timestamp}.pkl"
        filepath = self.save_dir / filename
        state_dict['_metadata'] = {'version': self.VERSION, 'timestamp': timestamp}
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
        # 同时保存一份 JSON 元数据便于查看
        meta_path = filepath.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump({'version': self.VERSION, 'timestamp': timestamp, 'keys': list(state_dict.keys())}, f)
        return str(filepath)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        # 版本兼容处理（预留）
        version = state.get('_metadata', {}).get('version', 0)
        if version > self.VERSION:
            raise ValueError(f"State version {version} > current {self.VERSION}")
        return state
    
    def list_saves(self):
        return list(self.save_dir.glob("*.pkl"))