import random
import numpy as np
from typing import List, Dict, Optional

class PalaceRetriever:
    def __init__(self, engine):
        self.engine = engine
        self.lps = engine.lps
    
    def _get_current_room(self, mode: str) -> int:
        """获取当前边界所在的房间 ID"""
        if mode == 'subjective':
            coord = self.engine.structural_coordinator.get_current_coordinate()
            return coord.as_tarot_code()
        elif mode == 'objective':
            # 客观模式：从最近一次交互中提取客观分类，若无则默认 0
            return getattr(self.engine, '_last_objective_room', 0)
        return 0
    
    def _walk_next(self, current_room: int, mode: str) -> int:
        """根据模式和内在状态，漫步到下一个房间"""
        if mode == 'subjective':
            return self._walk_subjective(current_room)
        else:
            return self._walk_objective(current_room)
    
    def _walk_subjective(self, room: int) -> int:
        """主观模式：塔罗序 ±1、对偶跳跃、大层跃迁，受僵化度和情绪调制"""
        stiffness = self.engine.process_meta.get_coupling_stiffness()
        # 测试模式：若存在 _test_stiffness 属性，则使用它
        if hasattr(self.engine.process_meta, '_test_stiffness'):
            stiffness = self.engine.process_meta._test_stiffness
        emotion = self.engine.fse.current_emotion
        
        # 基础概率：平滑转移 75%，对偶跳跃 20%，极性跃迁 5%
        # 僵化时提高跃迁概率
        leap_bonus = min(0.15, stiffness * 0.2)
        r = random.random()
        
        if r < 0.75 - leap_bonus:
            # 平滑转移：塔罗序 ±1
            delta = random.choice([-1, 1])
            return (room + delta) % 64
        elif r < 0.95:
            # 对偶跳跃：先天序对偶（高三位与低三位分别取反）
            inner = (room >> 3) & 0x07
            outer = room & 0x07
            opp_inner = 7 - inner
            opp_outer = 7 - outer
            return (opp_inner << 3) | opp_outer
        else:
            # 极性跃迁：大层循环跳跃（major 改变）
            major = room // 16
            new_major = (major + 2) % 4
            return new_major * 16 + (room % 16)
    
    def _walk_objective(self, room: int) -> int:
        """客观模式：沿聚合（向坤）或扩散（向乾）方向漫步，受欲望调制"""
        desire = self.engine.desire_spectrum.get_dominant_desire()
        inner = (room >> 3) & 0x07
        outer = room & 0x07
        
        if desire in ('existence', 'release', 'converge'):
            # 向内聚合：向坤(0)方向移动
            new_inner = max(0, inner - 1) if inner > 0 else 0
            new_outer = max(0, outer - 1) if outer > 0 else 0
        else:
            # 向外扩散：向乾(7)方向移动
            new_inner = min(7, inner + 1) if inner < 7 else 7
            new_outer = min(7, outer + 1) if outer < 7 else 7
        
        return (new_inner << 3) | new_outer
    
    def retrieve_by_walk(self, query_vec, start_room=None, walk_steps=3, mode='subjective') -> List[Dict]:
        """漫步检索主方法"""
        if start_room is None:
            start_room = self._get_current_room(mode)
        
        memories = []
        visited = set()
        current_room = start_room
        
        for _ in range(walk_steps):
            if current_room in visited:
                break
            visited.add(current_room)
            
            # 获取该房间的所有记忆
            room_memories = self.lps.query_by_tag(min_potency=0.2, **{f'{mode}_room': current_room})
            for mem in room_memories:
                mem['found_in_room'] = current_room
            memories.extend(room_memories)
            
            # 漫步到下一个房间
            current_room = self._walk_next(current_room, mode)
        
        # 按向量相似度重排
        return self._rerank_by_similarity(memories, query_vec)
    
    def _rerank_by_similarity(self, memories: List[Dict], query_vec) -> List[Dict]:
        """基于查询向量对记忆进行重排"""
        if not memories or query_vec is None:
            return memories
        for mem in memories:
            emb = mem.get('embedding')
            if emb is not None:
                mem['_walk_sim'] = np.dot(query_vec, emb)
            else:
                mem['_walk_sim'] = 0.0
        memories.sort(key=lambda x: x.get('_walk_sim', 0), reverse=True)
        return memories