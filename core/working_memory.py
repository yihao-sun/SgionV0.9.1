import time
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Optional, Tuple
import re

class WorkingMemory:
    def __init__(self, max_age_seconds: int = 3600, max_entries: int = 50):
        self.max_age = max_age_seconds
        self.max_entries = max_entries
        self.entries = deque()  # 缓存最近条目
        self._keyword_index: Dict[str, List[int]] = {}
    
    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        return list(set(words))[:5]
    
    def add(self, user_input: str, response: str, emotion: str, major: int, l_inst: float, engine):
        """每轮对话后调用：存入工作记忆缓存，并写入LPS沉积层"""
        # 1. 工作记忆缓存
        keywords = self._extract_keywords(user_input)
        entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'response_summary': response[:80] + "…" if len(response) > 80 else response,
            'emotion': emotion,
            'major': major,
            'l_inst': round(l_inst, 2),
            'keywords': keywords
        }
        self.entries.append(entry)
        self._prune_cache()
        
        # 2. LPS沉积层写入（势能0.3）
        if hasattr(engine, 'lps') and engine.lps:
            print(f"[DEBUG] 执行LPS沉积写入: user_input={user_input[:30]}...")
            summary = f"用户: {user_input} | 息观({emotion}): {response[:100]}"
            # 获取完整主观坐标
            coord = engine.structural_coordinator.get_current_coordinate()
            subjective_room = coord.as_tarot_code()
            # 获取客观分类
            objective_room = 0
            if hasattr(engine, 'objective_classifier'):
                objective_room = engine.objective_classifier.classify(user_input, response)
            
            # 推断输入的大致大层相位（基于简单规则）
            input_major = None
            # 三重标签体系：新增简单推理标签和随机标签
            reasoned_card = None
            random_card = None
            if hasattr(engine, 'structural_coordinator'):
                # 简单推理：22张大牌中选一张
                if hasattr(engine.structural_coordinator, '_infer_major_arcana'):
                    reasoned_card = engine.structural_coordinator._infer_major_arcana(user_input)
                # 随机面相：完整78张中抽一张
                if hasattr(engine.structural_coordinator, 'draw_random_card'):
                    random_card = engine.structural_coordinator.draw_random_card()
                # 保持向后兼容，获取输入大层
                if hasattr(engine.structural_coordinator, 'infer_input_major'):
                    input_major = engine.structural_coordinator.infer_input_major(user_input)

            tags = {
                'type': 'sediment',
                'emotion': emotion,
                'major_phase': major,
                'subjective_room': subjective_room,
                'objective_room': objective_room,
                'input_major': input_major,
                'l_inst': round(l_inst, 2),
                'timestamp': time.time(),
                'date_str': time.strftime('%Y-%m-%d %H:%M'),
                'keywords': keywords,
                # 新增：三重标签
                'reasoned_card': reasoned_card,     # 简单推理的大牌ID（22张之一或None）
                'random_card': random_card,         # 随机抽取的完整塔罗牌ID（78张之一）
            }
            embedding = engine.lps.encoder.encode([summary])[0] if engine.lps.encoder else None
            node_id = engine.lps.add_if_new(summary, embedding, potency=0.3, tags=tags)
            print(f"[DEBUG] LPS沉积写入结果: node_id={node_id}")
    
    def _prune_cache(self):
        now = time.time()
        while self.entries and now - self.entries[0]['timestamp'] > self.max_age:
            self.entries.popleft()
        while len(self.entries) > self.max_entries:
            self.entries.popleft()
    
    def get_context_for_llm(self) -> str:
        if not self.entries:
            return ""
        lines = []
        for e in list(self.entries)[-5:]:
            lines.append(f"用户: {e['user_input']} | 息观({e['emotion']}): {e['response_summary']}")
        return "最近对话：\n" + "\n".join(lines)
    
    def retrieve_by_time_range(self, start_time: float, end_time: float, engine) -> List[Dict]:
        """检索指定时间范围内的沉积条目，按时间顺序返回"""
        if not hasattr(engine, 'lps'):
            return []
        
        candidates = engine.lps.query_by_tag(type='sediment')
        results = []
        for item in candidates:
            # 确保tags存在且timestamp是数字
            tags = item.get('tags', {})
            ts = tags.get('timestamp', 0)
            if isinstance(ts, (int, float)) and start_time <= ts <= end_time:
                results.append(item)
        
        # 按时间升序排序
        results.sort(key=lambda x: (x.get('tags', {}) or {}).get('timestamp', 0))
        return results
    
    def retrieve_by_date_str(self, date_str: str, engine) -> List[Dict]:
        """按日期字符串检索（如 '2026-04-22'）"""
        start = datetime.strptime(date_str, '%Y-%m-%d').timestamp()
        end = start + 86400  # 24小时
        return self.retrieve_by_time_range(start, end, engine)
    
    def retrieve_by_keyword(self, keyword: str, time_range: Optional[Tuple[float, float]] = None, engine=None) -> List[Dict]:
        """按关键词检索沉积条目，可选时间范围"""
        if not engine or not hasattr(engine, 'lps'):
            return []
        
        candidates = engine.lps.query_by_tag(type='sediment')
        results = []
        for item in candidates:
            tags = item['tags']
            # 时间过滤
            if time_range:
                ts = tags.get('timestamp', 0)
                if not (time_range[0] <= ts <= time_range[1]):
                    continue
            # 关键词匹配（检索文本或标签中的关键词）
            text = item.get('text', '') or ''
            keywords = tags.get('keywords', [])
            # 处理NumPy数组的情况
            import numpy as np
            if isinstance(keywords, np.ndarray):
                # 检查数组是否为空
                if keywords.size == 0:
                    keywords = []
                else:
                    # 转换为Python列表
                    keywords = keywords.tolist()
            else:
                # 处理其他情况
                keywords = keywords or []
            if keyword in text or keyword in keywords:
                results.append(item)
                # 检索命中，计数+1
                self._increment_retrieval_count(item, engine)
        
        # 按时间降序（最近的优先）
        results.sort(key=lambda x: (x.get('tags', {}) or {}).get('timestamp', 0), reverse=True)
        return results
    
    def _increment_retrieval_count(self, item: Dict, engine):
        """递增检索计数，达到3次自动提升势能至固化层"""
        tags = item.get('tags', {})
        count = tags.get('retrieval_count', 0) + 1
        tags['retrieval_count'] = count
        
        # 更新LPS条目标签
        item['tags'] = tags
        
        if count >= 3:
            current_potency = item.get('potency', 0.3)
            if current_potency < 0.7:
                delta = 0.7 - current_potency
                engine.lps.update_potency(item['id'], delta)