"""
双通路记忆 (Dual Path Memory)
哲学对应：意象层概念文档第5节，区分实时通路（向量检索）和沉思通路（结构共鸣检索）。
功能：存储交互快照（ImageEntry），支持基于结构坐标和呼吸印记的共鸣检索。
"""

import time
import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from core.structural_coordinator import StructuralCoordinate
from core.image_base import ImageEntry
from utils.logger import get_logger

class MemorySnapshot:
    """记忆快照：一次有意义的交互记录"""
    def __init__(self, entry_id: str, user_coord: StructuralCoordinate, engine_coord: StructuralCoordinate,
                 breath: Dict[str, float], summary: str, timestamp: float = None, emotion_vector: list = None):
        self.id = entry_id  # 可用 UUID 或时间戳生成
        self.user_coord = user_coord
        self.engine_coord = engine_coord
        self.breath = breath  # {'proj_intensity', 'nour_success', 'stiffness'}
        self.summary = summary
        self.timestamp = timestamp or time.time()
        self.emotion_vector = emotion_vector or []  # 存储时的情绪向量
    
    def to_dict(self) -> Dict:
        """将快照转换为字典"""
        return {
            'id': self.id,
            'user_coord': {
                'major': self.user_coord.major,
                'middle': self.user_coord.middle,
                'fine': self.user_coord.fine
            },
            'engine_coord': {
                'major': self.engine_coord.major,
                'middle': self.engine_coord.middle,
                'fine': self.engine_coord.fine
            },
            'breath': self.breath,
            'summary': self.summary,
            'timestamp': self.timestamp,
            'emotion_vector': self.emotion_vector
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemorySnapshot':
        """从字典创建快照"""
        user_coord = StructuralCoordinate(
            data['user_coord']['major'],
            data['user_coord']['middle'],
            data['user_coord']['fine']
        )
        engine_coord = StructuralCoordinate(
            data['engine_coord']['major'],
            data['engine_coord']['middle'],
            data['engine_coord']['fine']
        )
        return cls(
            entry_id=data['id'],
            user_coord=user_coord,
            engine_coord=engine_coord,
            breath=data['breath'],
            summary=data['summary'],
            timestamp=data['timestamp'],
            emotion_vector=data['emotion_vector']
        )


class DualPathMemory:
    def __init__(self, lps=None, image_base=None, max_snapshots: int = 1000, engine=None):
        self.lps = lps  # 实时通路：LPS 向量检索
        self.image_base = image_base
        self.max_snapshots = max_snapshots
        self.snapshots: List[MemorySnapshot] = []
        self.logger = get_logger('dual_path_memory')
        self.engine = engine
        self._snapshots_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'dual_memory_snapshots.json')
        # 加载持久化的快照
        self._load_snapshots()
    
    def _load_snapshots(self):
        """加载持久化的快照"""
        try:
            if os.path.exists(self._snapshots_file):
                with open(self._snapshots_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.snapshots = [MemorySnapshot.from_dict(item) for item in data]
                    self.logger.info(f"加载了 {len(self.snapshots)} 个快照")
        except Exception as e:
            self.logger.warning(f"加载快照失败: {e}")
    
    def _save_snapshots(self):
        """保存快照到文件"""
        try:
            # 确保数据目录存在
            os.makedirs(os.path.dirname(self._snapshots_file), exist_ok=True)
            # 只保存最近的 max_snapshots 个快照
            snapshots_to_save = self.snapshots[-self.max_snapshots:]
            data = [snapshot.to_dict() for snapshot in snapshots_to_save]
            with open(self._snapshots_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"保存了 {len(snapshots_to_save)} 个快照")
        except Exception as e:
            self.logger.warning(f"保存快照失败: {e}")

    def store_snapshot(self, user_coord: StructuralCoordinate, engine_coord: StructuralCoordinate,
                       breath: Dict[str, float], summary: str, emotion_vector: list = None) -> str:
        """存储一次交互快照，返回快照 ID"""
        entry_id = f"mem_{int(time.time() * 1000)}"
        snapshot = MemorySnapshot(entry_id, user_coord, engine_coord, breath, summary, emotion_vector=emotion_vector)
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        # 保存快照到文件
        self._save_snapshots()
        self.logger.debug(f"Stored snapshot {entry_id} with engine coord {engine_coord}")
        return entry_id

    def _coordinate_similarity(self, coord1: StructuralCoordinate, coord2: StructuralCoordinate) -> float:
        """计算结构坐标的相似度（0-1）"""
        # 大层距离权重 0.5，中层 0.3，细微层 0.2
        major_match = 1.0 if coord1.major == coord2.major else 0.0
        middle_match = 1.0 if coord1.middle == coord2.middle else 0.0
        fine_match = 1.0 if coord1.fine == coord2.fine else 0.0
        return 0.5 * major_match + 0.3 * middle_match + 0.2 * fine_match

    def _breath_similarity(self, breath1: Dict[str, float], breath2: Dict[str, float]) -> float:
        """计算呼吸印记的余弦相似度"""
        vec1 = np.array([breath1.get('proj_intensity', 0.5),
                         breath1.get('nour_success', 0.5),
                         breath1.get('stiffness', 0.0)])
        vec2 = np.array([breath2.get('proj_intensity', 0.5),
                         breath2.get('nour_success', 0.5),
                         breath2.get('stiffness', 0.0)])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))



    def _coord_by_xiantian(self, xiantian_code: int) -> StructuralCoordinate:
        """根据先天序编码查找对应的结构坐标"""
        if self.image_base:
            card = self.image_base.get_card_by_xiantian(xiantian_code)
            if card:
                return StructuralCoordinate(card.major, card.middle, card.fine)
        # 回退：直接计算一个坐标
        major = (xiantian_code >> 2) % 4
        middle = (xiantian_code >> 1) % 2
        fine = xiantian_code % 2
        return StructuralCoordinate(major, middle, fine)

    def _coord_by_tarot(self, tarot_code: int) -> StructuralCoordinate:
        """根据塔罗序编码查找对应的结构坐标"""
        major = tarot_code // 16
        middle = (tarot_code % 16) // 4
        fine = tarot_code % 4
        return StructuralCoordinate(major, middle, fine)

    def _random_walk(self, current_coord: StructuralCoordinate, stiffness: float) -> StructuralCoordinate:
        """有倾向的随机漫步，模拟大衍筮法概率分布"""
        import random
        r = random.randint(0, 15)
        
        # 根据僵化度提高跳跃概率
        jump_threshold = 14 if stiffness > 0.5 else 15
        
        if r >= jump_threshold:
            # 低概率跨相位跳跃（极性翻转）
            if current_coord.xiantian_code == 7:
                return self._coord_by_xiantian(0)
            elif current_coord.xiantian_code == 0:
                return self._coord_by_xiantian(7)
            else:
                # 跳至相邻大层的对偶
                opposite = current_coord.get_opposite_xiantian()
                return self._coord_by_xiantian(opposite)
        elif r >= 12:
            # 中概率：对偶检索
            opposite = current_coord.get_opposite_xiantian()
            return self._coord_by_xiantian(opposite)
        else:
            # 高概率：平滑移动（塔罗序 ±1）
            new_tarot = (current_coord.as_tarot_code() + random.choice([-1, 1])) % 64
            return self._coord_by_tarot(new_tarot)

    def _cosine_distance(self, vec1, vec2):
        import numpy as np
        v1, v2 = np.array(vec1), np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 1.0
        return 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def contemplative_retrieval(self, current_coord: StructuralCoordinate, current_breath: Dict[str, float],
                                top_k: int = 5, resonance_threshold: float = 0.6, walk_probability: float = 0.3) -> List[Tuple[MemorySnapshot, float]]:
        """
        沉思通路检索：基于结构共鸣的意象检索。
        
        维度隔离原则：初筛必须基于低维结构坐标，禁止在高维语义空间进行最近邻检索。
        仅匹配大层、中层、细微层后，再使用呼吸和情绪向量进行二次排序。
        
        共鸣度 = 0.4 * 坐标相似度 + 0.3 * 呼吸相似度 + 0.3 * 情绪相似度
        """
        import random
        # 以 walk_probability 概率使用宫殿漫步，否则使用原有随机漫步
        if random.random() < walk_probability and hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'palace_retriever'):
            mode = 'subjective'
            walk_steps = 3
            # 生成查询向量
            summary = current_breath.get('summary', '')
            if hasattr(self.engine, 'lps') and self.engine.lps and hasattr(self.engine.lps, 'encoder'):
                query_vec = self.engine.lps.encoder.encode([summary])[0]
                # 执行宫殿漫步检索
                palace_results = self.engine.palace_retriever.retrieve_by_walk(
                    query_vec, start_room=None, walk_steps=walk_steps, mode=mode
                )
                # 转换为 MemorySnapshot 格式返回
                results = []
                for result in palace_results[:top_k]:
                    # 创建临时 MemorySnapshot 对象
                    # 注意：这里使用简化的方式，实际应用中可能需要更复杂的转换
                    snapshot = MemorySnapshot(
                        entry_id=f"palace_{result.get('id')}",
                        user_coord=current_coord,
                        engine_coord=current_coord,  # 使用当前坐标作为引擎坐标
                        breath=current_breath,  # 使用当前呼吸
                        summary=result.get('text', ''),
                        timestamp=result.get('last_accessed', time.time())
                    )
                    # 使用漫步检索的相似度作为共鸣度
                    resonance = result.get('_walk_sim', 0.0)
                    if resonance >= resonance_threshold:
                        results.append((snapshot, resonance))
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
        
        # 原有逻辑：使用随机漫步
        if random.random() < walk_probability:
            stiffness = current_breath.get('stiffness', 0)
            walked_coord = self._random_walk(current_coord, stiffness)
            self.logger.debug(f"Using walked coordinate: {walked_coord} (original: {current_coord})")
            retrieval_coord = walked_coord
        else:
            retrieval_coord = current_coord

        results = []
        # 从 current_breath 中获取当前僵化度
        stiffness = current_breath.get('stiffness', 0.0)
        # 获取当前情绪向量
        current_emotion = None
        if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'fse') and hasattr(self.engine.fse, 'E_vec'):
            current_emotion = self.engine.fse.E_vec
        
        for snapshot in self.snapshots:
            # 维度隔离原则：先进行结构坐标初筛
            coord_sim = self._coordinate_similarity(retrieval_coord, snapshot.engine_coord)
            
            # 只有结构坐标相似度达到一定阈值才进行后续计算
            if coord_sim < 0.3:  # 结构相似度阈值，确保初筛基于低维结构坐标
                continue
                
            breath_sim = self._breath_similarity(current_breath, snapshot.breath)
            
            # 情绪相似度
            emotion_sim = 0.5
            if current_emotion is not None and len(current_emotion) > 0 and hasattr(snapshot, 'emotion_vector') and snapshot.emotion_vector:
                emotion_sim = 1.0 - self._cosine_distance(current_emotion, snapshot.emotion_vector)
            
            # 综合共鸣度：结构 0.4 + 呼吸 0.3 + 情绪 0.3
            base_resonance = 0.4 * coord_sim + 0.3 * breath_sim + 0.3 * emotion_sim
            
            # 触觉柔软度对偶共鸣增益：高僵化时提升高柔软度快照的权重
            tactile_gain = 1.0
            if stiffness > 0.5:
                snapshot_softness = snapshot.breath.get('tactile_softness', 0.5) if hasattr(snapshot, 'breath') else 0.5
                if snapshot_softness > 0.7:
                    tactile_gain = 1.5  # 对偶共鸣增益
                elif snapshot_softness < 0.3:
                    tactile_gain = 0.8  # 粗粝快照轻微降权
            
            resonance = base_resonance * tactile_gain
            
            if resonance >= resonance_threshold:
                results.append((snapshot, resonance))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_inspiration(self, current_coord: StructuralCoordinate, current_breath: Dict[str, float]) -> Optional[str]:
        """获取灵感火花：最高共鸣快照的摘要 + 转化提示"""
        results = self.contemplative_retrieval(current_coord, current_breath, top_k=1)
        if results:
            snapshot, resonance = results[0]
            # 获取当前坐标对应的意象条目，提取转化提示
            card = self.image_base.get_card_by_coordinate(current_coord) if self.image_base else None
            hints = card.transition_hints if card else []
            hints_str = "、".join(hints[:2]) if hints else "继续呼吸"
            return f"共鸣度 {resonance:.2f}：{snapshot.summary}（转化提示：{hints_str}）"
        return None
    
    def export_core_memories(self, k: int = 10) -> List[dict]:
        """
        导出前 k 条高共鸣权重的记忆摘要，供 DigitalSeed 使用。
        返回格式与 DigitalSeed.core_memories 兼容。
        """
        core = []
        
        # 为每个快照计算共鸣度并排序
        resonance_snapshots = []
        
        # 使用默认呼吸状态计算共鸣度
        default_breath = {'proj_intensity': 0.5, 'nour_success': 0.5, 'stiffness': 0.0}
        
        # 使用一个默认坐标（可以使用第一个快照的坐标或默认坐标）
        if self.snapshots:
            default_coord = self.snapshots[0].engine_coord
        else:
            # 创建一个默认坐标
            default_coord = StructuralCoordinate(0, 0, 0)
        
        # 计算每个快照的共鸣度
        for snapshot in self.snapshots:
            # 计算当前快照与默认状态的共鸣度
            coord_sim = self._coordinate_similarity(default_coord, snapshot.engine_coord)
            breath_sim = self._breath_similarity(default_breath, snapshot.breath)
            resonance = 0.7 * coord_sim + 0.3 * breath_sim
            resonance_snapshots.append((snapshot, resonance))
        
        # 按共鸣度排序
        sorted_snapshots = sorted(resonance_snapshots, key=lambda x: x[1], reverse=True)[:k]
        
        for snap, resonance in sorted_snapshots:
            # 提取坐标和摘要
            coord = (snap.engine_coord.major, snap.engine_coord.middle, snap.engine_coord.fine)
            summary = snap.summary[:100]  # 截断
            # 情感价（可从呼吸印记推断：反哺成功率 - 0.5）
            valence = snap.breath.get('nour_success', 0.5) - 0.5
            
            core.append({
                'coordinate': coord,
                'summary': summary,
                'affective_valence': valence,
                'resonance_count': int(resonance * 10)  # 近似
            })
        return core