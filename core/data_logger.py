"""
数据日志记录器 (Data Logger)
功能：自动持久化交互事件、过程元信息快照，为长期模式识别提供数据。
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any


class DataLogger:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 事件日志文件（JSONL 格式，每行一个事件）
        self.event_log_path = self.data_dir / "event_log.jsonl"
        self.max_event_file_size_mb = 10  # 超过此大小则轮转
        
        # 过程元信息快照目录
        self.process_meta_dir = self.data_dir / "process_meta_snapshots"
        self.process_meta_dir.mkdir(parents=True, exist_ok=True)
        
        # 计数器，用于定期保存
        self.interaction_counter = 0
        self.save_interval = 100  # 每 100 轮保存一次过程元信息快照
    
    def log_event(self, event: Dict[str, Any]):
        """
        追加一条交互事件到 JSONL 文件。
        自动添加记录时间戳。
        """
        # 避免修改原事件字典
        event_copy = event.copy()
        event_copy['_logged_at'] = time.time()
        try:
            with open(self.event_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_copy, ensure_ascii=False) + '\n')
            self._rotate_if_needed()
        except Exception as e:
            # 降级：打印错误但不中断主流程
            print(f"[DataLogger] Failed to log event: {e}")
    
    def _rotate_if_needed(self):
        """若事件日志文件过大，重命名为带时间戳的归档文件"""
        if self.event_log_path.exists():
            size_mb = self.event_log_path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_event_file_size_mb:
                archive_name = f"event_log_{int(time.time())}.jsonl"
                self.event_log_path.rename(self.data_dir / archive_name)
    
    def save_process_meta_snapshot(self, process_meta, force: bool = False):
        """
        保存过程元信息的完整快照（投射、反哺、螺旋历史、僵化度等）。
        若 force=False，则按 save_interval 频率保存；若 force=True 则立即保存。
        """
        if not force:
            self.interaction_counter += 1
            if self.interaction_counter % self.save_interval != 0:
                return
        
        try:
            # 将 deque 转换为 list 以便 JSON 序列化
            snapshot = {
                'timestamp': time.time(),
                'projections': list(process_meta.projections),
                'nourishments': list(process_meta.nourishments),
                'spiral_history': getattr(process_meta, 'spiral_history', []),
                'stiffness_history': list(process_meta.stiffness_history),
                'coupling_mode': process_meta.coupling_mode,
                'reset_count': process_meta.reset_count
            }
            filename = f"pm_snapshot_{int(time.time())}.json"
            filepath = self.process_meta_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DataLogger] Failed to save process meta snapshot: {e}")
    
    def export_all_on_shutdown(self, engine):
        """
        引擎关闭时调用，强制保存所有待持久化数据。
        """
        if engine is None:
            return
        try:
            if hasattr(engine, 'process_meta'):
                self.save_process_meta_snapshot(engine.process_meta, force=True)
            # 保存全息种子
            if hasattr(engine, 'save_seed'):
                seed_path = self.data_dir / f"shutdown_seed_{int(time.time())}.json"
                engine.save_seed(str(seed_path), termination_reason="user_shutdown")
        except Exception as e:
            print(f"[DataLogger] Failed to export on shutdown: {e}")