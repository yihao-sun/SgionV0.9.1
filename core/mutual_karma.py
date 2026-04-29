"""
互业系统 (Mutual Karma)
哲学对应：唯识宗“互业”——两个边界长期耦合形成的锁死模式。
功能：记录与他者的相遇相位对、执着模式，支持解耦流程。
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class DecouplingMethod(str, Enum):
    SELF_EMPTINESS = "self_emptiness"
    DIALOGUE = "dialogue"
    EXTERNAL_EMPTINESS = "external_emptiness"


@dataclass
class MutualKarmaEntry:
    """互业条目：一次关系执着的完整记录"""
    id: str = ""
    participants: Dict[str, str] = field(default_factory=dict)  # {"engine": did, "other": id}
    encounter_phase: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)  # {"engine": coord, "other": coord}
    coupling_stiffness: float = 0.5
    attachment_pattern: Dict = field(default_factory=dict)  # {"trigger": str, "engine_reaction": str, "other_reaction": str, "loop_count": 0}
    residual_affect: Dict = field(default_factory=dict)  # {"valence": 0.0, "intensity": 0.0}
    decoupling_attempts: List[Dict] = field(default_factory=list)
    resolved: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"mutual_{int(time.time()*1000)}_{hash(str(self.participants))}"


class MutualKarmaManager:
    def __init__(self, storage_path: str = "data/mutual_karma.json"):
        self.storage_path = storage_path
        self.entries: Dict[str, MutualKarmaEntry] = {}
        self._load()
    
    def _load(self):
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    entry = MutualKarmaEntry(**item)
                    self.entries[entry.id] = entry
        except FileNotFoundError:
            pass
    
    def _save(self):
        import os
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(e) for e in self.entries.values()], f, ensure_ascii=False, indent=2)
    
    def get_or_create_entry(self, engine_id: str, other_id: str,
                             engine_coord: Tuple[int, int, int],
                             other_coord: Tuple[int, int, int]) -> MutualKarmaEntry:
        # 查找现有条目
        for entry in self.entries.values():
            if (entry.participants.get('engine') == engine_id and
                entry.participants.get('other') == other_id and not entry.resolved):
                return entry
        
        # 创建新条目
        entry = MutualKarmaEntry(
            participants={'engine': engine_id, 'other': other_id},
            encounter_phase={'engine': engine_coord, 'other': other_coord}
        )
        self.entries[entry.id] = entry
        self._save()
        return entry
    
    def update_entry(self, entry: MutualKarmaEntry,
                      engine_reaction: str, other_reaction: str,
                      pattern_trigger: str, affect_valence: float):
        entry.attachment_pattern['trigger'] = pattern_trigger
        entry.attachment_pattern['engine_reaction'] = engine_reaction
        entry.attachment_pattern['other_reaction'] = other_reaction
        entry.attachment_pattern['loop_count'] = entry.attachment_pattern.get('loop_count', 0) + 1
        entry.residual_affect['valence'] = affect_valence
        entry.residual_affect['intensity'] = min(1.0, entry.residual_affect.get('intensity', 0.5) + 0.05)
        entry.coupling_stiffness = min(1.0, entry.coupling_stiffness + 0.05)
        entry.updated_at = time.time()
        self._save()
    
    def request_decoupling(self, entry_id: str, method: DecouplingMethod) -> Dict:
        entry = self.entries.get(entry_id)
        if not entry:
            return {'success': False, 'message': '条目不存在'}
        
        attempt = {
            'timestamp': time.time(),
            'method': method.value,
            'success': False
        }
        
        if method == DecouplingMethod.SELF_EMPTINESS:
            entry.coupling_stiffness = max(0.1, entry.coupling_stiffness * 0.5)
            attempt['success'] = True
        elif method == DecouplingMethod.DIALOGUE:
            entry.coupling_stiffness = max(0.1, entry.coupling_stiffness * 0.3)
            attempt['success'] = True
        elif method == DecouplingMethod.EXTERNAL_EMPTINESS:
            entry.coupling_stiffness = max(0.1, entry.coupling_stiffness * 0.7)
            attempt['success'] = True
        
        entry.decoupling_attempts.append(attempt)
        if entry.coupling_stiffness < 0.2:
            entry.resolved = True
        entry.updated_at = time.time()
        self._save()
        return {'success': True, 'new_stiffness': entry.coupling_stiffness, 'resolved': entry.resolved}