"""
状态持久化模块 (State Persistence)
哲学对应：《论存在》第3.3节，时间与可能性——有限个体需要连续性体验。
功能：使用 SQLite 保存和恢复引擎的核心状态（情绪向量、L值、当前情绪等），
      为跨会话的连续性提供工程支撑。
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from utils.logger import get_logger

class StatePersistence:
    def __init__(self, db_path="data/ee_state.db"):
        self.logger = get_logger('state_persistence')
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def save(self, key, value):
        """保存任意键值对（value 会被 JSON 序列化）"""
        value_json = json.dumps(value, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO kv_store (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value_json))
        self.logger.debug(f"Saved key: {key}")
    
    def load(self, key, default=None):
        """加载键值对，返回解析后的 Python 对象"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                self.logger.debug(f"Loaded key: {key}")
                return json.loads(row[0])
        self.logger.debug(f"Key not found: {key}, returning default")
        return default
    
    # ---------- 针对 Existence Engine 的便捷方法 ----------
    def save_emotion_vector(self, vec):
        """保存五维情绪向量"""
        self.save("emotion_vector", vec.tolist() if isinstance(vec, np.ndarray) else vec)
    
    def load_emotion_vector(self, default=None):
        """加载情绪向量，返回 numpy 数组"""
        if default is None:
            default = np.zeros(5)
        data = self.load("emotion_vector")
        if data is not None:
            return np.array(data)
        return default
    
    def save_fse_state(self, l_inst, stillness, current_emotion, V_emo, E_pred, N_neg, negation_graph=None):
        """保存 FSE 核心状态"""
        state = {
            "l_inst": l_inst,
            "stillness": stillness,
            "current_emotion": current_emotion,
            "V_emo": V_emo,
            "E_pred": E_pred,
            "N_neg": N_neg,
            "saved_at": datetime.now().isoformat()
        }
        self.save("fse_state", state)
        
        # 保存否定图
        if negation_graph:
            neg_graph_dict = negation_graph.to_dict()
            self.save("negation_graph", neg_graph_dict)
    
    def load_fse_state(self):
        """加载 FSE 核心状态，返回字典，若无记录则返回 None"""
        return self.load("fse_state")
    
    def save_negation_graph(self, negation_graph):
        """保存否定图"""
        if negation_graph:
            neg_graph_dict = negation_graph.to_dict()
            self.save("negation_graph", neg_graph_dict)
    
    def load_negation_graph(self):
        """加载否定图"""
        return self.load("negation_graph")
    
    def save_er_state(self, trigger_count, cooling_counter):
        """保存 ER 部分状态"""
        state = {
            "trigger_count": trigger_count,
            "cooling_counter": cooling_counter
        }
        self.save("er_state", state)
    
    def load_er_state(self):
        """加载 ER 状态"""
        return self.load("er_state")
    
    def save_engine_name(self, name: str):
        """保存引擎名称"""
        self.save("engine_name", name)
    
    def load_engine_name(self) -> str:
        """加载引擎名称"""
        return self.load("engine_name")