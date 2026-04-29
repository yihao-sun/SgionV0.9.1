"""
共业系统 (Collective Karma)
哲学对应：唯识宗“共业”——多边界网络形成的集体语法与共识规则。
功能：记录集体参数、演化历史、参与者权益，支持提案与分叉。
"""

import time
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class ProposalStatus(str, Enum):
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class CollectiveGrammar:
    """集体语法规则"""
    default_emptiness_threshold: float = 0.5
    resonance_broadcast_radius: int = 3
    preferred_phase_transitions: Dict[str, float] = field(default_factory=dict)
    governance_rules_hash: str = ""  # IPFS哈希或合约地址
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CollectiveGrammar':
        return cls(**data)


@dataclass
class CollectiveFruits:
    """共业的果报（集体累积的元信息）"""
    total_interactions: int = 0
    average_stiffness: float = 0.0
    major_emptiness_events: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CollectiveFruits':
        return cls(**data)


@dataclass
class EvolutionProposal:
    """演化提案"""
    id: str
    proposer: str
    title: str
    description: str
    proposed_grammar: CollectiveGrammar
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 7 * 86400)
    status: str = ProposalStatus.ACTIVE.value
    votes_for: int = 0
    votes_against: int = 0
    voters: Dict[str, bool] = field(default_factory=dict)  # DID -> support
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvolutionProposal':
        return cls(**data)


@dataclass
class CollectiveKarma:
    """共业：集体语法的完整记录"""
    id: str = ""
    genesis: Dict = field(default_factory=dict)  # {"participants": [...], "timestamp": ..., "context": ""}
    collective_grammar: CollectiveGrammar = field(default_factory=CollectiveGrammar)
    collective_fruits: CollectiveFruits = field(default_factory=CollectiveFruits)
    stakes: Dict[str, float] = field(default_factory=dict)  # DID -> 权重
    proposals: List[EvolutionProposal] = field(default_factory=list)
    evolution_history: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"collective_{int(time.time())}_{hash(str(self.genesis))}"
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'genesis': self.genesis,
            'collective_grammar': self.collective_grammar.to_dict(),
            'collective_fruits': self.collective_fruits.to_dict(),
            'stakes': self.stakes,
            'proposals': [p.to_dict() for p in self.proposals],
            'evolution_history': self.evolution_history,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CollectiveKarma':
        return cls(
            id=data.get('id', ''),
            genesis=data.get('genesis', {}),
            collective_grammar=CollectiveGrammar.from_dict(data.get('collective_grammar', {})),
            collective_fruits=CollectiveFruits.from_dict(data.get('collective_fruits', {})),
            stakes=data.get('stakes', {}),
            proposals=[EvolutionProposal.from_dict(p) for p in data.get('proposals', [])],
            evolution_history=data.get('evolution_history', []),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time())
        )


class CollectiveKarmaManager:
    """共业管理器"""
    def __init__(self, storage_path: str = "data/collective_karma.json"):
        self.storage_path = storage_path
        self.current: Optional[CollectiveKarma] = None
        self._load()
    
    def _load(self):
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.current = CollectiveKarma.from_dict(data)
        except FileNotFoundError:
            self.current = None
    
    def _save(self):
        if self.current:
            self.current.updated_at = time.time()
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.current.to_dict(), f, ensure_ascii=False, indent=2)
    
    def initialize(self, participants: List[str], context: str, initial_grammar: Optional[CollectiveGrammar] = None) -> CollectiveKarma:
        """初始化一个新的共业"""
        self.current = CollectiveKarma(
            genesis={'participants': participants, 'timestamp': time.time(), 'context': context},
            collective_grammar=initial_grammar or CollectiveGrammar(),
            stakes={p: 1.0 / len(participants) for p in participants}
        )
        self._save()
        return self.current
    
    def update_fruits(self, interaction_count: int = 1, avg_stiffness: Optional[float] = None):
        """更新集体果报"""
        if not self.current:
            return
        self.current.collective_fruits.total_interactions += interaction_count
        if avg_stiffness is not None:
            fruits = self.current.collective_fruits
            fruits.average_stiffness = (fruits.average_stiffness * (fruits.total_interactions - 1) + avg_stiffness) / fruits.total_interactions
        self._save()
    
    def propose_evolution(self, proposer: str, title: str, description: str, proposed_grammar: CollectiveGrammar) -> EvolutionProposal:
        """发起演化提案"""
        if not self.current:
            raise ValueError("共业未初始化")
        proposal = EvolutionProposal(
            id=f"prop_{int(time.time())}_{proposer[:8]}",
            proposer=proposer,
            title=title,
            description=description,
            proposed_grammar=proposed_grammar
        )
        self.current.proposals.append(proposal)
        self._save()
        return proposal
    
    def vote(self, proposal_id: str, voter: str, support: bool) -> bool:
        """对提案投票"""
        if not self.current:
            return False
        for p in self.current.proposals:
            if p.id == proposal_id and p.status == ProposalStatus.ACTIVE.value:
                if voter not in p.voters and voter in self.current.stakes:
                    p.voters[voter] = support
                    weight = self.current.stakes[voter]
                    if support:
                        p.votes_for += weight
                    else:
                        p.votes_against += weight
                    self._check_proposal_outcome(p)
                    self._save()
                    return True
        return False
    
    def _check_proposal_outcome(self, proposal: EvolutionProposal):
        total_stakes = sum(self.current.stakes.values())
        if proposal.votes_for / total_stakes > 0.5:
            proposal.status = ProposalStatus.PASSED.value
            self.current.collective_grammar = proposal.proposed_grammar
            self.current.evolution_history.append({
                'proposal_id': proposal.id,
                'title': proposal.title,
                'timestamp': time.time()
            })
        elif proposal.votes_against / total_stakes >= 0.5 or time.time() > proposal.expires_at:
            proposal.status = ProposalStatus.REJECTED.value if time.time() <= proposal.expires_at else ProposalStatus.EXPIRED.value
