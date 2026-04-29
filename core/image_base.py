"""
意象库 (Image Base)
哲学对应：意象层概念文档第4节，存储过程语法的中性描述。
功能：加载 tarot_cards.json，根据结构坐标查询对应的意象条目。
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from core.structural_coordinator import StructuralCoordinate

class ImageEntry:
    """意象库条目"""
    def __init__(self, card_id: str, name: str, major: int, middle: int, fine: int,
                 neutral_description: str, transition_hints: List[str],
                 is_prototype: bool = False, retrieval_count: int = 0,
                 last_retrieved: float = 0.0, source: str = "unknown",
                 source_decay: float = 1.0):
        self.id = card_id
        self.name = name
        self.major = major
        self.middle = middle
        self.fine = fine
        self.neutral_description = neutral_description
        self.transition_hints = transition_hints
        self.breath_signature = None
        self.is_prototype = is_prototype
        self.retrieval_count = retrieval_count
        self.last_retrieved = last_retrieved
        self.source = source
        self.source_decay = source_decay
        
        # 计算先天序编码
        major_bit = (major % 2) << 2
        middle_bit = (middle % 2) << 1
        fine_bit = (fine % 2)
        self.xiantian_code = major_bit | middle_bit | fine_bit

    def matches_coordinate(self, coord: StructuralCoordinate) -> bool:
        return (self.major == coord.major and
                self.middle == coord.middle and
                self.fine == coord.fine)


class ImageBase:
    def __init__(self, data_path: str = "data/tarot_cards.json"):
        self.cards: Dict[str, ImageEntry] = {}
        self.coordinate_index: Dict[tuple, str] = {}  # (major, middle, fine) -> card_id
        self._load(data_path)

    def _load(self, data_path: str):
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"意象库文件不存在: {data_path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for card_data in data.get("cards", []):
            entry = ImageEntry(
                card_id=card_data["id"],
                name=card_data["name"],
                major=card_data["major"],
                middle=card_data["middle"],
                fine=card_data["fine"],
                neutral_description=card_data["neutral_description"],
                transition_hints=card_data.get("transition_hints", []),
                is_prototype=True,
                source="tarot"
            )
            self.cards[entry.id] = entry
            self.coordinate_index[(entry.major, entry.middle, entry.fine)] = entry.id
        
        # 构建 archetype_candidates 列表
        self.archetype_candidates = []
        for card in self.cards.values():
            # 假设大牌通过 major 为 -1 或特定标记识别（此处简化：取所有卡）
            # 实际可根据塔罗牌结构筛选大牌
            self.archetype_candidates.append({
                'text': card.neutral_description,
                'potency': 0.5,
                'source': 'archetype',
                'major': card.major,
                'middle': card.middle,
                'fine': card.fine
            })

    def get_card_by_coordinate(self, coord: StructuralCoordinate) -> Optional[ImageEntry]:
        """根据结构坐标获取对应的意象条目"""
        key = (coord.major, coord.middle, coord.fine)
        card_id = self.coordinate_index.get(key)
        if card_id and card_id in self.cards:
            card = self.cards[card_id]
            card.retrieval_count += 1
            card.last_retrieved = time.time()
            # 来源遗忘衰减
            if hasattr(card, 'source_decay'):
                card.source_decay *= 0.999
                if card.source_decay < 0.1 and card.source != "internalized":
                    card.source = "internalized"
                    # 可记录螺旋事件
            return card
        return None

    def get_card_by_id(self, card_id: str) -> Optional[ImageEntry]:
        return self.cards.get(card_id)

    def get_all_cards(self) -> List[ImageEntry]:
        return list(self.cards.values())

    def get_card_by_xiantian(self, xiantian_code: int) -> Optional[ImageEntry]:
        """根据先天序编码获取对应的意象条目"""
        for card in self.cards.values():
            if card.xiantian_code == xiantian_code:
                return card
        return None
    
    def add_dynamic_entry(self, major: int, middle: int, fine: int,
                          neutral_description: str, breath_signature: Dict,
                          source: str = "consolidation") -> str:
        """
        添加动态生成的意象条目（来自记忆巩固或互业抽象）。
        返回新条目的ID。
        """
        import time
        card_id = f"dyn_{int(time.time()*1000)}_{major}{middle}{fine}"
        entry = ImageEntry(
            card_id=card_id,
            name=f"动态-{card_id[:8]}",
            major=major, middle=middle, fine=fine,
            neutral_description=neutral_description,
            transition_hints=[],
            is_prototype=False,
            retrieval_count=0,
            last_retrieved=0.0,
            source=source,
            source_decay=1.0
        )
        # 可附加呼吸印记到扩展字段
        entry.breath_signature = breath_signature
        
        self.cards[card_id] = entry
        self.coordinate_index[(major, middle, fine)] = card_id
        return card_id
    
    def prune_sparse_entries(self, min_retrieval: int = 3, max_age_days: int = 7) -> int:
        """
        剪枝稀疏的动态意象条目。
        返回删除的条目数量。
        """
        to_remove = []
        now = time.time()
        
        for card_id, entry in list(self.cards.items()):
            # 1. 严格保护原型牌
            if entry.is_prototype:
                continue
            # 2. 仅处理动态条目
            if not card_id.startswith('dyn_'):
                continue
            
            # 3. 计算年龄
            age_days = (now - entry.last_retrieved) / 86400 if entry.last_retrieved > 0 else 999
            # 4. 判断是否稀疏
            if entry.retrieval_count < min_retrieval and age_days > max_age_days:
                to_remove.append(card_id)
        
        for card_id in to_remove:
            entry = self.cards[card_id]
            del self.cards[card_id]
            # 清理坐标索引（仅当索引指向该条目时）
            key = (entry.major, entry.middle, entry.fine)
            if self.coordinate_index.get(key) == card_id:
                del self.coordinate_index[key]
        
        if to_remove:
            # 可在此记录螺旋事件（需通过 engine 访问 global_workspace）
            pass
        
        return len(to_remove)