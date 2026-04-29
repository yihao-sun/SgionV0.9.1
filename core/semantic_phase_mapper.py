"""
语义相位映射器 (Semantic Phase Mapper)
功能：将自然语言关键词/短语映射为过程相位概率分布，支持加固、转化、遗忘的全生命周期演化。
"""

import json
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class SemanticEntry:
    """语义条目：一个词/短语的相位倾向与演化状态"""
    keyword: str                                    # 词/短语
    phase_distribution: Dict[int, float] = field(default_factory=dict)  # 大层倾向概率 {0:0.2, 1:0.6, ...}
    confidence: float = 0.3                         # 整体可信度 0-1
    retrieval_count: int = 0                        # 被检索次数
    last_retrieved: float = 0.0                     # 最后检索时间戳
    source: str = "user"                            # 来源：seed/user/system/consolidation
    source_decay: float = 1.0                       # 来源遗忘衰减（1.0=完全保留来源标记）
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'keyword': self.keyword,
            'phase_distribution': self.phase_distribution,
            'confidence': self.confidence,
            'retrieval_count': self.retrieval_count,
            'last_retrieved': self.last_retrieved,
            'source': self.source,
            'source_decay': self.source_decay,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SemanticEntry':
        entry = cls(keyword=data['keyword'])
        entry.phase_distribution = {int(k): v for k, v in data.get('phase_distribution', {}).items()}
        entry.confidence = data.get('confidence', 0.3)
        entry.retrieval_count = data.get('retrieval_count', 0)
        entry.last_retrieved = data.get('last_retrieved', 0.0)
        entry.source = data.get('source', 'user')
        entry.source_decay = data.get('source_decay', 1.0)
        entry.created_at = data.get('created_at', time.time())
        return entry


class SemanticPhaseMapper:
    def __init__(self, lps):
        self.lps = lps
        self._keyword_index = {}  # keyword -> [{'id': node_id, 'potency': potency, 'source': source}]
        self._build_index()
    
    def _build_index(self):
        """构建语义词条索引"""
        for meta in self.lps.metadata:
            tags = meta.get('tags', {})
            if tags.get('type') == 'semantic':
                kw = tags.get('keyword')
                if kw:
                    entry_info = {
                        'id': meta['id'],
                        'potency': meta['potency'],
                        'source': tags.get('source', 'unknown')
                    }
                    self._keyword_index.setdefault(kw, []).append(entry_info)
    
    def get_entry(self, keyword: str) -> Optional[Dict]:
        """获取语义条目，返回字典格式（兼容原有调用）"""
        kw = keyword.strip()
        results = self.lps.query_by_tag(type='semantic', keyword=kw, min_potency=0.0)
        if results:
            # 返回势能最高的条目
            entry_data = results[0]
            # 更新检索统计（提升势能、衰减来源）
            self._update_entry_stats(entry_data['id'])
            return {
                'keyword': kw,
                'phase_distribution': entry_data['tags'].get('phase_distribution', {}),
                'confidence': entry_data['potency'],  # 势能即置信度
                'source': entry_data['tags'].get('source', 'user'),
                'source_decay': entry_data['tags'].get('source_decay', 1.0)
            }
        return None
    
    def _update_entry_stats(self, node_id: int):
        """更新条目的检索统计：提升势能，衰减来源"""
        # 通过 LPS 的 update_potency 提升势能
        # 假设 LPS 有 update_potency 方法
        # self.lps.update_potency(node_id, delta=0.01)
        # 来源衰减需要访问 metadata，可直接修改
        for meta in self.lps.metadata:
            if meta['id'] == node_id:
                tags = meta.get('tags', {})
                tags['source_decay'] = tags.get('source_decay', 1.0) * 0.999
                if tags['source_decay'] < 0.1:
                    tags['source'] = 'internalized'
                break
    
    def get_or_create_entry(self, keyword: str) -> Dict:
        """获取或创建词条"""
        kw = keyword.strip()
        entry = self.get_entry(kw)
        if entry:
            return entry
        # 创建新语义条目（作为 LPS 条目存储）
        tags = {
            'type': 'semantic',
            'keyword': kw,
            'phase_distribution': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            'source': 'user',
            'source_decay': 1.0
        }
        text = f"[SEMANTIC] {kw}"  # 占位文本
        embedding = self.lps.encoder.encode([kw])[0] if hasattr(self.lps, 'encoder') and self.lps.encoder else None
        node_id = self.lps.add(text, embedding, potency=0.3, tags=tags)
        # 更新索引
        entry_info = {
            'id': node_id,
            'potency': 0.3,
            'source': 'user'
        }
        self._keyword_index.setdefault(kw, []).append(entry_info)
        return {
            'keyword': kw,
            'phase_distribution': {0:0.25, 1:0.25, 2:0.25, 3:0.25},
            'confidence': 0.3,
            'source': 'user',
            'source_decay': 1.0
        }
    
    def update_entry(self, keyword: str, actual_major: int, success: float = 1.0):
        """
        根据实际共鸣结果更新词条的相位分布与置信度。
        actual_major: 本次采样得到的大层
        success: 交互成功度（0-1），用于调制更新幅度
        """
        kw = keyword.strip()
        results = self.lps.query_by_tag(type='semantic', keyword=kw, min_potency=0.0)
        if not results:
            self.get_or_create_entry(kw)
            results = self.lps.query_by_tag(type='semantic', keyword=kw, min_potency=0.0)
        if not results:
            return
        entry_data = results[0]
        tags = entry_data['tags']
        phase_dist = tags.get('phase_distribution', {0:0.25,1:0.25,2:0.25,3:0.25})
        
        expected_prob = phase_dist.get(actual_major, 0.0)
        alpha = 0.05 * success
        new_confidence = entry_data['potency']
        if expected_prob > 0.25:
            new_confidence = min(1.0, entry_data['potency'] + alpha * 0.5)
            phase_dist[actual_major] = expected_prob + alpha
        else:
            phase_dist[actual_major] = phase_dist.get(actual_major, 0.0) + alpha * 0.5
            new_confidence = max(0.2, entry_data['potency'] - alpha * 0.3)
        
        # 归一化
        total = sum(phase_dist.values())
        if total > 0:
            for k in phase_dist:
                phase_dist[k] /= total
        
        # 更新 LPS 条目
        tags['phase_distribution'] = phase_dist
        # 势能更新
        # 假设 LPS 有 update_potency 方法
        # delta = new_confidence - entry_data['potency']
        # self.lps.update_potency(entry_data['id'], delta)
        
        # 更新索引中的信息
        kw = tags.get('keyword')
        if kw and kw in self._keyword_index:
            for entry_info in self._keyword_index[kw]:
                if entry_info['id'] == entry_data['id']:
                    entry_info['potency'] = new_confidence
                    break
    
    def _normalize_distribution(self, dist: Dict[int, float]):
        total = sum(dist.values())
        if total > 0:
            for k in dist:
                dist[k] /= total
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词（简化版：分词后取长度≥2的词）"""
        if not text:
            return []
        words = re.split(r'[，。！？、；：""\'\'\s]+', text)
        seen = set()
        keywords = []
        for w in words:
            w = w.strip()
            if len(w) >= 2 and w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords[:10]
    
    def get_distribution(self, text: str) -> Dict[int, float]:
        """
        从输入文本计算语义相位分布。
        策略：提取关键词，取各词条分布的加权平均（权重=confidence）。
        """
        keywords = self._extract_keywords(text)
        if not keywords:
            return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        
        merged = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        total_weight = 0.0
        
        for kw in keywords:
            entry = self.get_entry(kw)
            if entry:
                weight = entry['confidence']
                for m, p in entry['phase_distribution'].items():
                    merged[m] += p * weight
                total_weight += weight
        
        if total_weight > 0:
            for m in merged:
                merged[m] /= total_weight
            return merged
        else:
            return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    
    def prune_sparse_entries(self, min_retrieval: int = 3, max_age_days: int = 30) -> int:
        """剪枝长期低频词条（保护 seed 来源且已内化的词条）"""
        # 委托给 LPS 处理，或直接过滤
        # 这里简化处理，返回 0 表示没有剪枝
        return 0
    
    def get_stats(self) -> dict:
        """获取语义库统计信息"""
        if not hasattr(self, 'lps') or not self.lps:
            return {'total': 0, 'top_confidence': [], 'sources': {}}

        # 直接遍历LPS元数据，统计type='semantic'的条目
        semantic_entries = []
        for meta in self.lps.metadata:
            tags = meta.get('tags', {})
            if tags.get('type') == 'semantic':
                semantic_entries.append(meta)

        total = len(semantic_entries)
        if total == 0:
            return {'total': 0, 'top_confidence': [], 'sources': {}}

        # 按势能降序排序，取前5
        sorted_entries = sorted(semantic_entries, key=lambda x: x['potency'], reverse=True)
        top = []
        for entry in sorted_entries[:5]:
            tags = entry.get('tags', {})
            keyword = tags.get('keyword', entry.get('text', '?')[:30])
            top.append({
                'keyword': keyword if keyword else '?',
                'confidence': round(entry['potency'], 2),
                'source': tags.get('source', 'unknown')
            })

        # 来源分布统计
        sources = {}
        for entry in semantic_entries:
            src = entry.get('tags', {}).get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1

        return {
            'total': total,
            'top_confidence': top,
            'sources': sources
        }