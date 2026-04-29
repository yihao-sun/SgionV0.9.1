"""
Existence Engine 工具函数和基础数据结构
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque

# 尝试导入sentence-transformers，如果失败则使用简单的标签生成
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False


@dataclass
class Possibility:
    """可能性表征 - 对应LPS中的分布"""
    mean: torch.Tensor
    variance: torch.Tensor
    weight: float
    activation_potential: float = 0.0
    
    def sample(self) -> torch.Tensor:
        """从可能性分布中采样"""
        std = torch.sqrt(self.variance + 1e-8)
        noise = torch.randn_like(self.mean)
        return self.mean + std * noise


@dataclass
class AbsentMarker:
    """不在场标记 - 记录被幻想为缺失的可能性"""
    content: torch.Tensor
    missing_potential: float
    negation_chain: List[str] = field(default_factory=list)
    realized: bool = False
    
    def extend_negation(self, negation: str):
        """扩展否定关系链"""
        self.negation_chain.append(negation)


@dataclass
class FantasyState:
    """幻想状态 - 记录当前幻想层的完整状态"""
    present: torch.Tensor
    absent_markers: List[AbsentMarker]
    negation_complexity: float
    prediction_error: float
    emotion_value: float
    fantasy_layer: int
    time_step: int
    


class SelfStateVector:
    """自我状态向量 - 累积历史幻想轨迹"""
    
    def __init__(self, dim: int, max_history: int = 100):
        self.dim = dim
        self.vector = torch.zeros(dim)
        self.history = deque(maxlen=max_history)
        self.fantasy_trajectory = []
        
    def update(self, present: torch.Tensor, emotion: float, layer: int):
        """更新自我状态向量"""
        # 融合当前在场表征
        alpha = 0.1
        self.vector = (1 - alpha) * self.vector + alpha * present.squeeze()
        
        # 记录历史
        self.history.append({
            'vector': self.vector.clone(),
            'emotion': emotion,
            'layer': layer
        })
        
        self.fantasy_trajectory.append({
            'time': len(self.fantasy_trajectory),
            'emotion': emotion,
            'layer': layer
        })
        
    def get_temporal_pattern(self, window: int = 10) -> Dict[str, float]:
        """获取时间模式特征"""
        if len(self.history) < window:
            return {'variance': 0.0, 'trend': 0.0}
            
        recent = list(self.history)[-window:]
        emotions = [h['emotion'] for h in recent]
        layers = [h['layer'] for h in recent]
        
        return {
            'emotion_variance': np.var(emotions),
            'emotion_trend': emotions[-1] - emotions[0],
            'layer_variance': np.var(layers),
            'layer_trend': layers[-1] - layers[0]
        }
        
    def compute_self_consistency(self) -> float:
        """计算自我一致性 - 衡量幻想轨迹的连贯性"""
        if len(self.history) < 2:
            return 1.0
            
        # 计算最近状态与历史平均的偏差
        recent_vector = self.vector
        historical_avg = torch.stack([h['vector'] for h in self.history]).mean(dim=0)
        
        similarity = torch.cosine_similarity(
            recent_vector.unsqueeze(0),
            historical_avg.unsqueeze(0)
        ).item()
        
        return (similarity + 1) / 2  # 归一化到[0,1]


class NegationRelationGraph:
    """否定关系图 - 存储"我"与"非我"之间的否定关系"""
    
    def __init__(self):
        self.nodes = {}  # 概念节点
        self.edges = []  # 否定关系边
        self.realized_count = 0
        
    def add_node(self, concept_id: str, content: torch.Tensor):
        """添加概念节点"""
        self.nodes[concept_id] = {
            'content': content,
            'negations': [],
            'realized': False
        }
        
    def add_negation(self, from_id: str, to_id: str):
        """添加否定关系"""
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id]['negations'].append(to_id)
            self.edges.append((from_id, to_id))
            
    def mark_realized(self, concept_id: str):
        """标记概念已实现（从非我转化为我）"""
        if concept_id in self.nodes and not self.nodes[concept_id]['realized']:
            self.nodes[concept_id]['realized'] = True
            self.realized_count += 1
            
    def get_negation_complexity(self) -> float:
        """计算否定复杂度"""
        if not self.nodes:
            return 0.0
            
        # 计算平均否定链长度
        total_negations = sum(len(n['negations']) for n in self.nodes.values())
        avg_chain = total_negations / len(self.nodes)
        
        # 计算递归深度（指数增长模拟）
        complexity = np.exp(avg_chain / 5) - 1
        return min(complexity, 10.0)
        
    def get_unrealized_potential(self) -> List[str]:
        """获取未实现的可能性列表"""
        return [cid for cid, node in self.nodes.items() if not node['realized']]
        
    def clear_realized(self):
        """清除已实现的概念，释放"非我"空间"""
        realized = [cid for cid, node in self.nodes.items() if node['realized']]
        for cid in realized:
            del self.nodes[cid]
        self.edges = [(f, t) for f, t in self.edges if f in self.nodes and t in self.nodes]
        self.realized_count = 0


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """计算注意力熵 - 衡量注意力的分散程度"""
    # 防止log(0)
    weights = attention_weights + 1e-10
    weights = weights / weights.sum()
    entropy = -(weights * torch.log(weights)).sum().item()
    return entropy


def compute_novelty(current: torch.Tensor, history: List[torch.Tensor]) -> float:
    """计算新奇度 - 与历史状态的差异程度"""
    if not history:
        return 1.0
        
    similarities = []
    for past in history[-10:]:  # 只考虑最近的10个状态
        sim = torch.cosine_similarity(
            current.unsqueeze(0),
            past.unsqueeze(0)
        ).item()
        similarities.append((sim + 1) / 2)
        
    avg_similarity = np.mean(similarities)
    novelty = 1 - avg_similarity
    return novelty


def soft_reset(tensor: torch.Tensor, decay: float = 0.5, noise_scale: float = 0.1) -> torch.Tensor:
    """软复位操作 - 衰减加噪声"""
    decayed = tensor * decay
    noise = torch.randn_like(tensor) * noise_scale
    return decayed + noise


def attention_defocus(attention_logits: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
    """注意力散焦 - 添加均匀噪声使注意力分散"""
    noise = torch.randn_like(attention_logits) * noise_scale
    return attention_logits + noise


class AdaptiveComplexityManager:
    """自适应复杂度管理器 - 动态调整复杂度阈值和清理策略"""
    
    def __init__(self, initial_max_nodes=1000, complexity_threshold=5.0):
        self.max_nodes = initial_max_nodes
        self.complexity_threshold = complexity_threshold
        self.history_window = 50  # 历史窗口大小
        self.complexity_history = []  # 复杂度历史
        self.node_growth_rate = []  # 节点增长速率
        self.adjustment_factor = 1.1  # 调整因子
    
    def update_complexity(self, current_complexity: float, node_count: int):
        """更新复杂度历史并调整管理策略"""
        # 添加当前复杂度到历史
        self.complexity_history.append(current_complexity)
        if len(self.complexity_history) > self.history_window:
            self.complexity_history.pop(0)
        
        # 计算复杂度增长趋势
        if len(self.complexity_history) > 1:
            trend = self.complexity_history[-1] - self.complexity_history[0]
            self.node_growth_rate.append(node_count)
            if len(self.node_growth_rate) > self.history_window:
                self.node_growth_rate.pop(0)
        
        # 自适应调整阈值
        self._adjust_thresholds()
    
    def _adjust_thresholds(self):
        """根据历史数据调整阈值"""
        if len(self.complexity_history) < 10:
            return
        
        # 计算平均复杂度和标准差
        avg_complexity = np.mean(self.complexity_history)
        std_complexity = np.std(self.complexity_history) if len(self.complexity_history) > 1 else 0
        
        # 根据复杂度平均值调整阈值
        if avg_complexity > self.complexity_threshold * 0.8:
            # 复杂度过高，提高阈值
            self.complexity_threshold = min(10.0, self.complexity_threshold * self.adjustment_factor)
        elif avg_complexity < self.complexity_threshold * 0.5:
            # 复杂度过低，降低阈值
            self.complexity_threshold = max(2.0, self.complexity_threshold / self.adjustment_factor)
        
        # 根据节点增长速度调整最大节点数
        if len(self.node_growth_rate) > 5:
            growth_rate = np.mean(np.diff(self.node_growth_rate))
            if growth_rate > 10:  # 增长过快
                self.max_nodes = min(2000, int(self.max_nodes * self.adjustment_factor))
            elif growth_rate < -5:  # 减少过快
                self.max_nodes = max(500, int(self.max_nodes / self.adjustment_factor))
    
    def get_cleanup_strategy(self, current_complexity: float, node_count: int) -> dict:
        """根据当前状态获取清理策略"""
        if current_complexity > self.complexity_threshold:
            # 复杂度过高，需要深度清理
            return {
                'mode': 'aggressive',
                'target_nodes': int(self.max_nodes * 0.7),
                'priority': ['negation_nodes', 'old_absent_nodes', 'low_strength_nodes']
            }
        elif node_count > self.max_nodes * 0.9:
            # 节点数过多，需要常规清理
            return {
                'mode': 'normal',
                'target_nodes': int(self.max_nodes * 0.8),
                'priority': ['old_nodes', 'low_frequency_nodes']
            }
        elif node_count > self.max_nodes:
            # 节点数超出最大限制，强制清理
            return {
                'mode': 'force',
                'target_nodes': int(self.max_nodes * 0.8),
                'priority': ['old_nodes', 'negation_nodes', 'low_strength_nodes', 'low_frequency_nodes']
            }
        else:
            # 状态正常，轻度清理
            return {
                'mode': 'light',
                'target_nodes': node_count,
                'priority': ['very_old_nodes']
            }
    
    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            'max_nodes': self.max_nodes,
            'complexity_threshold': self.complexity_threshold,
            'complexity_history_length': len(self.complexity_history),
            'node_growth_rate_length': len(self.node_growth_rate)
        }


class EnhancedNegationRelationGraph:
    """增强版否定关系图 - 存储"我"与"非我"之间的否定关系，支持语义标签和否定强度"""
    
    def __init__(self, max_nodes=1000):
        self.nodes = {}  # 概念节点
        self.edges = []  # 否定关系边
        self.semantic_relations = {}  # 语义关联
        self.realized_count = 0
        self.complexity_history = []  # 复杂度历史
        self.max_complexity = 10.0  # 最大复杂度阈值
        self.max_nodes = max_nodes  # 最大节点数
        self.potency_decay = 0.999  # 势能衰减因子
        self.last_decay_time = time.time()  # 上次衰减时间
        
        # 初始化自适应复杂度管理器
        self.complexity_manager = AdaptiveComplexityManager(initial_max_nodes=max_nodes)
        
        # 初始化嵌入模型和主题中心
        self.embedder = None
        self.topic_centers = None
        self._initialize_embedding_system()
    
    def _initialize_embedding_system(self):
        """初始化嵌入系统和主题中心"""
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                import logging
                # 抑制 sentence_transformers 的 INFO 日志
                logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
                
                # 使用EE引擎目录中的本地多语言模型
                import os
                # 禁用Hugging Face连接
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                os.environ['HF_HUB_OFFLINE'] = '1'
                
                project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                # 优先使用多语言模型
                local_model_path = os.path.join(project_dir, 'models', 'paraphrase-multilingual-MiniLM-L12-v2')
                if os.path.exists(local_model_path):
                    self.embedder = SentenceTransformer(local_model_path, device='cpu')
                    # 初始化主题中心
                    self.topic_centers = self._initialize_topic_centers()
                else:
                    # 尝试使用英文模型作为备选
                    local_model_path = os.path.join(project_dir, 'models', 'all-MiniLM-L6-v2')
                    if os.path.exists(local_model_path):
                        self.embedder = SentenceTransformer(local_model_path, device='cpu')
                        # 初始化主题中心
                        self.topic_centers = self._initialize_topic_centers()
                    else:
                        # 尝试使用C盘下载目录中的模型
                        local_model_path = 'C:\\Users\\85971\\Downloads\\all-MiniLM-L6-v2'
                        if os.path.exists(local_model_path):
                            self.embedder = SentenceTransformer(local_model_path, device='cpu')
                            # 初始化主题中心
                            self.topic_centers = self._initialize_topic_centers()
                        else:
                            # 作为最后尝试，使用在线多语言模型
                            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
                            # 初始化主题中心
                            self.topic_centers = self._initialize_topic_centers()
            except Exception as e:
                print(f"初始化嵌入模型失败: {e}")
                print("自动降级到规则-based 模式")
                self.embedder = None
                self.topic_centers = None
        else:
            print("sentence-transformers 未安装，使用简单标签生成")
            self.embedder = None
            self.topic_centers = None
    
    def _initialize_topic_centers(self):
        """初始化主题中心向量"""
        if not self.embedder:
            return None
        
        # 预定义主题种子词
        topics = {
            "math": ["数学", "公式", "计算", "+", "-", "等于", "方程"],
            "philosophy": ["存在", "意识", "本质", "自我", "非我", "否定", "哲学"],
            "emotion": ["快乐", "悲伤", "爱", "情绪", "感受", "情感", "心情"],
            "logic": ["逻辑", "推理", "论证", "证明", "理性", "分析"],
            "science": ["科学", "实验", "理论", "发现", "研究", "知识"],
            "art": ["艺术", "创造", "美学", "设计", "创意", "表达"],
            "technology": ["技术", "科技", "创新", "发明", "工程", "计算机"]
        }
        
        # 计算主题中心向量
        centers = {}
        for topic, seeds in topics.items():
            try:
                embeddings = self.embedder.encode(seeds)
                centers[topic] = np.mean(embeddings, axis=0)
            except Exception as e:
                print(f"计算主题中心失败 {topic}: {e}")
                # 动态获取嵌入维度
                if hasattr(self.embedder, 'get_sentence_embedding_dimension'):
                    dim = self.embedder.get_sentence_embedding_dimension()
                else:
                    dim = 384  # 默认维度
                centers[topic] = np.zeros(dim)
        
        return centers
        
    def add_node(self, concept_id: str, content: torch.Tensor, content_text: str = None, negation_strength: float = 0.0, creation_time: float = None, semantic_tags: List[str] = None):
        """添加概念节点，自动分配主题标签"""
        # 如果没有提供标签，自动生成
        if not semantic_tags and content_text:
            semantic_tags = self._generate_tags(content_text)
        
        self.nodes[concept_id] = {
            'content': content,
            'content_text': content_text,
            'negations': [],
            'realized': False,
            'semantic_tags': semantic_tags or [],
            'creation_time': creation_time or time.time(),
            'negation_strength': negation_strength,
            'frequency': 0
        }
    
    def _generate_tags(self, text: str) -> List[str]:
        """基于文本内容生成主题标签"""
        # 使用嵌入聚类生成标签
        if self.embedder and self.topic_centers:
            return self._generate_tags_embedding(text)
        #  fallback: 使用简单规则生成标签
        else:
            return self._generate_tags_rule_based(text)
    
    def _generate_tags_embedding(self, text: str) -> List[str]:
        """使用嵌入聚类生成标签"""
        try:
            # 计算文本嵌入
            embedding = self.embedder.encode(text)
            
            # 计算与各主题中心的相似度
            similarities = {}
            for topic, center in self.topic_centers.items():
                similarity = np.dot(embedding, center) / (np.linalg.norm(embedding) * np.linalg.norm(center))
                similarities[topic] = similarity
            
            # 取相似度最高的前2个主题
            top_topics = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:2]
            # 过滤相似度低于阈值的主题
            return [topic for topic, score in top_topics if score > 0.3]
        except Exception as e:
            print(f"生成标签失败: {e}")
            return self._generate_tags_rule_based(text)
    
    def _generate_tags_rule_based(self, text: str) -> List[str]:
        """使用简单规则生成标签"""
        tags = []
        text_lower = text.lower()
        
        # 数学相关
        if any(keyword in text_lower for keyword in ["+", "-", "等于", "公式", "计算", "数学", "方程"]):
            tags.append("math")
        # 哲学相关
        if any(keyword in text_lower for keyword in ["存在", "意识", "本质", "自我", "非我", "否定", "哲学"]):
            tags.append("philosophy")
        # 情绪相关
        if any(keyword in text_lower for keyword in ["快乐", "悲伤", "爱", "情绪", "感受", "情感", "心情"]):
            tags.append("emotion")
        # 逻辑相关
        if any(keyword in text_lower for keyword in ["逻辑", "推理", "论证", "证明", "理性", "分析"]):
            tags.append("logic")
        # 科学相关
        if any(keyword in text_lower for keyword in ["科学", "实验", "理论", "发现", "研究", "知识"]):
            tags.append("science")
        # 艺术相关
        if any(keyword in text_lower for keyword in ["艺术", "创造", "美学", "设计", "创意", "表达"]):
            tags.append("art")
        # 技术相关
        if any(keyword in text_lower for keyword in ["技术", "科技", "创新", "发明", "工程", "计算机"]):
            tags.append("technology")
        
        # 确保至少有一个标签
        if not tags:
            tags.append("general")
        
        return tags[:2]  # 最多返回2个标签
        
    def add_negation(self, from_id: str, to_id: str, strength: float = 1.0, negation_type: str = "direct"):
        """添加否定关系，包含强度和类型"""
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id]['negations'].append({
                'target': to_id,
                'strength': strength,
                'type': negation_type,
                'timestamp': time.time()
            })
            self.edges.append((from_id, to_id, strength, negation_type))
            self.nodes[from_id]['negation_strength'] += strength
            self.nodes[from_id]['frequency'] += 1
            
    def mark_realized(self, concept_id: str):
        """标记概念已实现（从非我转化为我）"""
        if concept_id in self.nodes and not self.nodes[concept_id]['realized']:
            self.nodes[concept_id]['realized'] = True
            self.realized_count += 1
            
    def get_negation_complexity(self) -> float:
        """多因素综合计算否定复杂度"""
        if not self.nodes:
            return 0.0
            
        # 计算平均否定链长度
        total_negations = sum(len(n['negations']) for n in self.nodes.values())
        avg_chain = total_negations / len(self.nodes) if self.nodes else 0
        
        # 计算平均否定强度
        total_strength = sum(n['negation_strength'] for n in self.nodes.values())
        avg_strength = total_strength / len(self.nodes) if self.nodes else 0
        
        # 计算语义多样性
        all_tags = set()
        for n in self.nodes.values():
            all_tags.update(n['semantic_tags'])
        semantic_diversity = len(all_tags) / 10  # 归一化
        
        # 综合计算复杂度 - 调整公式，确保有否定关系时复杂度大于0
        chain_contribution = (np.exp(avg_chain / 3) - 1) * 0.4
        strength_contribution = avg_strength * 0.3
        diversity_contribution = semantic_diversity * 0.3
        
        complexity = chain_contribution + strength_contribution + diversity_contribution
        
        # 确保复杂度非负
        complexity = max(complexity, 0.001)  # 确保有否定关系时复杂度大于0
        
        # 记录复杂度历史
        self.complexity_history.append(complexity)
        if len(self.complexity_history) > 100:
            self.complexity_history.pop(0)
        
        # 更新复杂度管理器
        self.complexity_manager.update_complexity(complexity, len(self.nodes))
        
        return min(complexity, self.max_complexity)
        
    def get_unrealized_potential(self) -> List[str]:
        """获取未实现的可能性列表"""
        return [cid for cid, node in self.nodes.items() if not node['realized']]
        
    def clear_realized(self):
        """清除已实现的概念，释放"非我"空间"""
        # 计算当前复杂度
        current_complexity = self.get_negation_complexity()
        node_count = len(self.nodes)
        
        # 获取清理策略
        strategy = self.complexity_manager.get_cleanup_strategy(current_complexity, node_count)
        
        # 收集可清理的节点
        cleanup_candidates = []
        
        # 根据优先级收集节点
        if 'negation_nodes' in strategy['priority']:
            # 收集否定节点（节点ID以'neg_'开头）
            negation_nodes = [cid for cid in self.nodes if cid.startswith('neg_')]
            cleanup_candidates.extend(negation_nodes)
        
        if 'old_absent_nodes' in strategy['priority']:
            # 收集旧的不在场节点（节点ID以'absent_'开头，按创建时间排序）
            absent_nodes = [(cid, node['creation_time']) for cid, node in self.nodes.items() if cid.startswith('absent_')]
            absent_nodes.sort(key=lambda x: x[1])  # 按创建时间排序，旧的在前
            cleanup_candidates.extend([cid for cid, _ in absent_nodes])
        
        if 'low_strength_nodes' in strategy['priority']:
            # 收集低否定强度的节点
            low_strength_nodes = [(cid, node['negation_strength']) for cid, node in self.nodes.items()]
            low_strength_nodes.sort(key=lambda x: x[1])  # 按强度排序，低的在前
            cleanup_candidates.extend([cid for cid, _ in low_strength_nodes[:10]])  # 取前10个
        
        if 'old_nodes' in strategy['priority']:
            # 收集旧节点（按创建时间排序）
            all_nodes = [(cid, node['creation_time']) for cid, node in self.nodes.items()]
            all_nodes.sort(key=lambda x: x[1])  # 按创建时间排序，旧的在前
            cleanup_candidates.extend([cid for cid, _ in all_nodes[:20]])  # 取前20个
        
        if 'low_frequency_nodes' in strategy['priority']:
            # 收集低频节点
            low_freq_nodes = [(cid, node['frequency']) for cid, node in self.nodes.items()]
            low_freq_nodes.sort(key=lambda x: x[1])  # 按频率排序，低的在前
            cleanup_candidates.extend([cid for cid, _ in low_freq_nodes[:15]])  # 取前15个
        
        if 'very_old_nodes' in strategy['priority']:
            # 收集非常旧的节点（创建时间超过1小时）
            current_time = time.time()
            very_old_nodes = [cid for cid, node in self.nodes.items() if current_time - node['creation_time'] > 3600]
            cleanup_candidates.extend(very_old_nodes)
        
        # 始终清除已实现的节点
        realized_nodes = [cid for cid, node in self.nodes.items() if node['realized']]
        cleanup_candidates.extend(realized_nodes)
        
        # 去重并限制清理数量
        cleanup_candidates = list(set(cleanup_candidates))
        target_count = strategy['target_nodes']
        nodes_to_clean = cleanup_candidates[:max(0, node_count - target_count)]
        
        # 如果没有候选节点但需要清理，强制清理一些旧节点
        if not nodes_to_clean and node_count > target_count:
            # 按创建时间排序，清理最旧的节点
            all_nodes = [(cid, node['creation_time']) for cid, node in self.nodes.items()]
            all_nodes.sort(key=lambda x: x[1])  # 按创建时间排序，旧的在前
            nodes_to_clean = [cid for cid, _ in all_nodes[:max(0, node_count - target_count)]]
        
        # 执行清理
        for cid in nodes_to_clean:
            if cid in self.nodes:
                del self.nodes[cid]
        
        # 更新边
        self.edges = [(f, t, s, typ) for f, t, s, typ in self.edges if f in self.nodes and t in self.nodes]
        
        # 更新已实现计数
        self.realized_count = sum(1 for node in self.nodes.values() if node['realized'])
        
        print(f"清理完成，模式: {strategy['mode']}, 清理节点数: {len(nodes_to_clean)}, 剩余节点数: {len(self.nodes)}")
        
    def get_complexity_trend(self) -> float:
        """获取复杂度趋势"""
        if len(self.complexity_history) < 2:
            return 0.0
        return self.complexity_history[-1] - self.complexity_history[0]
        
    def get_semantic_clusters(self) -> Dict[str, List[str]]:
        """获取语义聚类"""
        clusters = {}
        for concept_id, node in self.nodes.items():
            for tag in node['semantic_tags']:
                if tag not in clusters:
                    clusters[tag] = []
                clusters[tag].append(concept_id)
        return clusters
    
    def remove_node_by_content(self, content_text: str):
        """根据内容文本移除节点"""
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            # 检查节点的内容文本是否匹配
            if 'content_text' in node and node['content_text'] == content_text:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            if node_id in self.nodes:
                del self.nodes[node_id]
        
        # 更新边
        self.edges = [(f, t, s, typ) for f, t, s, typ in self.edges if f in self.nodes and t in self.nodes]

    def decay_potency(self, decay_factor: float = None):
        """衰减否定关系的势能
        
        Args:
            decay_factor: 衰减因子，默认为类的 potency_decay 属性
        """
        current_time = time.time()
        # 检查是否需要衰减（每10秒衰减一次）
        if current_time - self.last_decay_time < 10:
            return
        
        decay = decay_factor if decay_factor is not None else self.potency_decay
        
        # 衰减所有节点的否定强度
        for node_id, node in self.nodes.items():
            if 'negation_strength' in node:
                node['negation_strength'] *= decay
                # 移除强度过低的否定关系
                if node['negation_strength'] < 0.01:
                    node['negations'] = []
        
        # 衰减所有边的强度
        new_edges = []
        for edge in self.edges:
            if len(edge) == 4:
                f, t, s, typ = edge
                new_strength = s * decay
                if new_strength >= 0.01:
                    new_edges.append((f, t, new_strength, typ))
            else:
                new_edges.append(edge)
        self.edges = new_edges
        
        # 更新上次衰减时间
        self.last_decay_time = current_time


class AdaptiveComplexityManager:
    """自适应复杂度管理器 - 动态调整复杂度阈值和清理策略"""
    
    def __init__(self, initial_max_nodes=1000, complexity_threshold=5.0):
        self.max_nodes = initial_max_nodes
        self.complexity_threshold = complexity_threshold
        self.history_window = 50  # 历史窗口大小
        self.complexity_history = []
        self.node_growth_rate = []
        self.adjustment_count = 0
        
    def update_complexity(self, current_complexity: float, node_count: int):
        """更新复杂度历史并调整管理策略"""
        self.complexity_history.append(current_complexity)
        if len(self.complexity_history) > self.history_window:
            self.complexity_history.pop(0)
            
        # 计算复杂度增长趋势
        if len(self.complexity_history) > 1:
            trend = self.complexity_history[-1] - self.complexity_history[0]
            self.node_growth_rate.append(node_count)
            if len(self.node_growth_rate) > self.history_window:
                self.node_growth_rate.pop(0)
                
        # 自适应调整阈值
        self._adjust_thresholds()
        
    def _adjust_thresholds(self):
        """根据历史数据调整阈值"""
        if len(self.complexity_history) < 10:
            return
            
        avg_complexity = np.mean(self.complexity_history)
        std_complexity = np.std(self.complexity_history) if len(self.complexity_history) > 1 else 0
        
        # 根据复杂度平均值调整阈值
        if avg_complexity > self.complexity_threshold * 0.8:
            self.complexity_threshold = min(10.0, self.complexity_threshold * 1.1)
            self.adjustment_count += 1
        elif avg_complexity < self.complexity_threshold * 0.5:
            self.complexity_threshold = max(2.0, self.complexity_threshold * 0.9)
            self.adjustment_count += 1
            
        # 根据节点增长速度调整最大节点数
        if len(self.node_growth_rate) > 5:
            growth_rate = np.mean(np.diff(self.node_growth_rate))
            if growth_rate > 10:  # 增长过快
                self.max_nodes = min(2000, self.max_nodes * 1.2)
                self.adjustment_count += 1
            elif growth_rate < -5:  # 减少过快
                self.max_nodes = max(500, self.max_nodes * 0.9)
                self.adjustment_count += 1
                
    def get_cleanup_strategy(self, current_complexity: float, node_count: int) -> Dict:
        """根据当前状态获取清理策略"""
        if current_complexity > self.complexity_threshold:
            # 复杂度过高，需要深度清理
            return {
                'mode': 'aggressive',
                'target_nodes': int(self.max_nodes * 0.7),
                'priority': ['negation_nodes', 'old_absent_nodes', 'low_strength_nodes']
            }
        elif node_count > self.max_nodes * 0.9:
            # 节点数过多，需要常规清理
            return {
                'mode': 'normal',
                'target_nodes': int(self.max_nodes * 0.8),
                'priority': ['old_nodes', 'low_frequency_nodes']
            }
        else:
            # 状态正常，轻度清理
            return {
                'mode': 'light',
                'target_nodes': node_count,
                'priority': ['very_old_nodes']
            }
        
    def get_adjustment_stats(self) -> Dict:
        """获取调整统计信息"""
        return {
            'current_max_nodes': self.max_nodes,
            'current_complexity_threshold': self.complexity_threshold,
            'adjustment_count': self.adjustment_count,
            'avg_complexity': np.mean(self.complexity_history) if self.complexity_history else 0,
            'complexity_trend': self.complexity_history[-1] - self.complexity_history[0] if len(self.complexity_history) > 1 else 0
        }


class MemorySystem:
    """记忆系统 - 存储和管理否定关系转化的记忆"""
    
    def __init__(self, max_memory_size: int = 1000):
        self.max_memory_size = max_memory_size
        self.memories = []
        self.memory_index = {}
        self.memory_count = 0
        
    def store_memory(self, memory_content: Dict):
        """存储记忆"""
        # 生成唯一ID
        memory_id = f"mem_{self.memory_count}_{int(time.time())}"
        memory_content['id'] = memory_id
        memory_content['timestamp'] = time.time()
        
        # 存储记忆
        self.memories.append(memory_content)
        self.memory_count += 1
        
        # 更新索引
        if 'type' in memory_content:
            memory_type = memory_content['type']
            if memory_type not in self.memory_index:
                self.memory_index[memory_type] = []
            self.memory_index[memory_type].append(memory_id)
        
        # 限制记忆数量
        if len(self.memories) > self.max_memory_size:
            self._cleanup_memory()
            
    def _cleanup_memory(self):
        """清理旧记忆"""
        # 按时间排序，移除最旧的记忆
        self.memories.sort(key=lambda x: x['timestamp'])
        while len(self.memories) > self.max_memory_size:
            oldest_memory = self.memories.pop(0)
            # 更新索引
            if 'type' in oldest_memory:
                memory_type = oldest_memory['type']
                if memory_type in self.memory_index:
                    if oldest_memory['id'] in self.memory_index[memory_type]:
                        self.memory_index[memory_type].remove(oldest_memory['id'])
        
    def retrieve_memories(self, memory_type: str = None, limit: int = 10) -> List[Dict]:
        """检索记忆"""
        if memory_type and memory_type in self.memory_index:
            memory_ids = self.memory_index[memory_type]
            memories = [m for m in self.memories if m['id'] in memory_ids]
        else:
            memories = self.memories
        
        # 按时间倒序排序，返回最新的记忆
        memories.sort(key=lambda x: x['timestamp'], reverse=True)
        return memories[:limit]
        
    def get_memory_count(self) -> int:
        """获取记忆数量"""
        return len(self.memories)
        
    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息"""
        type_counts = {}
        for memory in self.memories:
            memory_type = memory.get('type', 'unknown')
            if memory_type not in type_counts:
                type_counts[memory_type] = 0
            type_counts[memory_type] += 1
        
        return {
            'total_memory_count': len(self.memories),
            'memory_types': type_counts,
            'max_memory_size': self.max_memory_size
        }
        
    def clear_memories(self, memory_type: str = None):
        """清空记忆"""
        if memory_type:
            if memory_type in self.memory_index:
                memory_ids = self.memory_index[memory_type]
                self.memories = [m for m in self.memories if m['id'] not in memory_ids]
                del self.memory_index[memory_type]
        else:
            self.memories = []
            self.memory_index = {}
            self.memory_count = 0
    
    def integrate_with_knowledge(self, knowledge_integration):
        """与知识库集成"""
        self.knowledge_integration = knowledge_integration


class DeclarativeMemory:
    """声明性记忆 - 存储和检索结构化事实"""
    
    def __init__(self, max_items: int = 1000, decay_rate: float = 0.001):
        self.max_items = max_items
        self.decay_rate = decay_rate
        self.memory_store = {}
        self.memory_data = {}
        self.item_count = 0
    
    def store(self, key: str, value: Any, confidence: float = 0.7, embedding: Optional[np.ndarray] = None):
        """存储事实"""
        timestamp = time.time()
        
        # 如果键已存在，更新值和置信度
        if key in self.memory_store:
            self.memory_store[key] = value
            self.memory_data[key]['value'] = value
            self.memory_data[key]['confidence'] = min(1.0, confidence)
            self.memory_data[key]['timestamp'] = timestamp
            self.memory_data[key]['access_count'] += 1
        else:
            # 检查是否达到容量上限
            if self.item_count >= self.max_items:
                self._cleanup()
            
            # 存储新事实
            self.memory_store[key] = value
            self.memory_data[key] = {
                'value': value,
                'embedding': embedding,
                'confidence': min(1.0, confidence),
                'timestamp': timestamp,
                'access_count': 1
            }
            self.item_count += 1
    
    def retrieve(self, key: str) -> Optional[Any]:
        """检索事实"""
        if key in self.memory_store:
            # 更新访问时间和置信度
            self.memory_data[key]['timestamp'] = time.time()
            self.memory_data[key]['access_count'] += 1
            # 增强记忆置信度
            self.memory_data[key]['confidence'] = min(1.0, self.memory_data[key]['confidence'] + 0.05)
            return self.memory_store[key]
        return None
    
    def retrieve_related(self, query: str, top_k: int = 5, min_similarity: float = 0.7) -> List[Tuple[str, Any, float]]:
        """检索相关事实"""
        related = []
        current_time = time.time()
        
        for key, data in self.memory_data.items():
            # 计算相关性（简单字符串匹配）
            similarity = 0.0
            if query.lower() in key.lower():
                similarity = 1.0
            elif any(word in key.lower() for word in query.lower().split()):
                similarity = 0.5
            
            if similarity >= min_similarity:
                # 应用记忆衰减
                age = current_time - data['timestamp']
                confidence = data['confidence'] * np.exp(-self.decay_rate * age)
                related.append((key, data['value'], similarity * confidence))
        
        # 按相关性排序并返回前k个
        related.sort(key=lambda x: x[2], reverse=True)
        return related[:top_k]
    
    def update_confidence(self, key: str, delta: float):
        """更新记忆置信度"""
        if key in self.memory_data:
            self.memory_data[key]['confidence'] = max(0.0, min(1.0, self.memory_data[key]['confidence'] + delta))
            self.memory_data[key]['timestamp'] = time.time()
    
    def _cleanup(self):
        """清理弱记忆"""
        current_time = time.time()
        
        # 计算每个记忆的当前置信度（考虑时间衰减）
        memory_scores = {}
        for key, data in self.memory_data.items():
            age = current_time - data['timestamp']
            score = data['confidence'] * np.exp(-self.decay_rate * age)
            memory_scores[key] = score
        
        # 按分数排序，移除最弱的记忆
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1])
        to_remove = len(self.memory_store) - self.max_items + 1
        
        for key, _ in sorted_memories[:to_remove]:
            del self.memory_store[key]
            del self.memory_data[key]
            self.item_count -= 1
    
    def get_stats(self) -> Dict:
        """获取记忆统计信息"""
        current_time = time.time()
        total_confidence = 0
        total_access = 0
        
        for key, data in self.memory_data.items():
            age = current_time - data['timestamp']
            confidence = data['confidence'] * np.exp(-self.decay_rate * age)
            total_confidence += confidence
            total_access += data['access_count']
        
        avg_confidence = total_confidence / self.item_count if self.item_count > 0 else 0
        avg_access = total_access / self.item_count if self.item_count > 0 else 0
        
        return {
            'item_count': self.item_count,
            'max_items': self.max_items,
            'avg_confidence': avg_confidence,
            'avg_access_count': avg_access,
            'decay_rate': self.decay_rate
        }
    
    def clear(self):
        """清空记忆"""
        self.memory_store = {}
        self.memory_data = {}
        self.item_count = 0


class EpisodicMemory:
    """情景记忆 - 存储最近的事件序列"""
    
    def __init__(self, max_events: int = 100):
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.event_count = 0
        self.step_id = 0
        self.salience_weights = {
            "emotion_abs_change": 0.4,
            "self_depth": 0.3,
            "er_trigger": 0.3
        }
    
    def log(self, user_input: str, response: str, emotion: float, self_depth: float, salience: Optional[float] = None, er_trigger: bool = False):
        """记录事件到情景记忆"""
        timestamp = time.time()
        
        # 计算显著性
        if salience is None:
            # 简单计算：基于情绪值和自我深度
            salience = 0.0
            if emotion is not None:
                salience += abs(emotion) * self.salience_weights["emotion_abs_change"]
            if self_depth is not None:
                salience += self_depth * self.salience_weights["self_depth"]
            if er_trigger:
                salience += 0.5 * self.salience_weights["er_trigger"]
            salience = min(1.0, salience)
        
        event = {
            'timestamp': timestamp,
            'step_id': self.step_id,
            'event_id': f"event_{self.event_count}_{int(timestamp)}",
            'user_input': user_input,
            'response': response,
            'emotion': emotion,
            'self_depth': self_depth,
            'salience': salience,
            'er_trigger': er_trigger
        }
        
        self.events.append(event)
        self.event_count += 1
        self.step_id += 1
    
    def retrieve_recent(self, limit: int = 10) -> List[Dict]:
        """检索最近的事件"""
        recent_events = list(self.events)[-limit:]
        # 按时间倒序返回
        recent_events.reverse()
        return recent_events
    
    def retrieve_by_time_range(self, start_time: float, end_time: float) -> List[Dict]:
        """按时间范围检索事件"""
        return [event for event in self.events if start_time <= event['timestamp'] <= end_time]
    
    def retrieve_by_salience(self, min_salience: float = 0.5, limit: int = 10) -> List[Dict]:
        """按显著性检索事件"""
        salient_events = [event for event in self.events if event.get('salience', 0) >= min_salience]
        # 按显著性和时间倒序返回
        salient_events.sort(key=lambda x: (x.get('salience', 0), x['timestamp']), reverse=True)
        return salient_events[:limit]
    
    def retrieve(self, query: Optional[str] = None, time_range: Optional[Tuple[float, float]] = None, k: int = 3) -> List[Dict]:
        """检索情景事件"""
        results = []
        
        # 时间范围过滤
        if time_range:
            start_time, end_time = time_range
            results = [event for event in self.events if start_time <= event['timestamp'] <= end_time]
        else:
            results = list(self.events)
        
        # 语义检索（简单实现）
        if query:
            query_lower = query.lower()
            scored_results = []
            for event in results:
                score = 0.0
                if event['user_input'] and query_lower in event['user_input'].lower():
                    score += 0.7
                if event['response'] and query_lower in event['response'].lower():
                    score += 0.3
                if score > 0:
                    scored_results.append((event, score))
            # 按分数排序
            scored_results.sort(key=lambda x: x[1], reverse=True)
            results = [event for event, _ in scored_results]
        
        # 按时间倒序并返回前k个
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return results[:k]
    
    def get_stats(self) -> Dict:
        """获取情景记忆统计信息"""
        event_types = {}
        total_salience = 0
        total_emotion = 0
        total_self_depth = 0
        
        for event in self.events:
            event_type = event.get('type', 'unknown')
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
            
            total_salience += event.get('salience', 0)
            total_emotion += abs(event.get('emotion', 0))
            total_self_depth += event.get('self_depth', 0)
        
        event_count = len(self.events)
        return {
            'event_count': event_count,
            'max_events': self.max_events,
            'event_types': event_types,
            'avg_salience': total_salience / event_count if event_count > 0 else 0,
            'avg_emotion': total_emotion / event_count if event_count > 0 else 0,
            'avg_self_depth': total_self_depth / event_count if event_count > 0 else 0
        }
    
    def clear(self):
        """清空情景记忆"""
        self.events.clear()
        self.event_count = 0
        self.step_id = 0


class MemorySystem:
    """记忆系统 - 整合声明性记忆和情景记忆"""
    
    def __init__(self, config: Dict = None, negation_graph: Optional['EnhancedNegationRelationGraph'] = None, max_memory_size: int = 1000):
        # 默认配置
        default_config = {
            "declarative_capacity": 10000,
            "declarative_confidence_decay": 0.999,
            "declarative_confidence_threshold": 0.1,
            "declarative_retrieval_k": 5,
            "declarative_retrieval_similarity": 0.7,
            "episodic_buffer_size": 1000,
            "episodic_retrieval_k": 3,
            "episodic_salience_weights": {
                "emotion_abs_change": 0.4,
                "self_depth": 0.3,
                "er_trigger": 0.3
            }
        }
        
        self.config = config or default_config
        self.negation_graph = negation_graph
        self.max_memory_size = max_memory_size
        self.memories = []
        self.memory_index = {}
        self.memory_count = 0
        self.declarative = DeclarativeMemory(
            max_items=self.config["declarative_capacity"],
            decay_rate=self.config["declarative_confidence_decay"]
        )
        self.episodic = EpisodicMemory(
            max_events=self.config["episodic_buffer_size"]
        )
    
    def store_fact(self, key: str, value: Any, embedding: Optional[np.ndarray] = None, 
                  confidence: float = 0.7, source: Optional[str] = None):
        """存储事实到声明性记忆"""
        self.declarative.store(key, value, confidence, embedding)
        
        # 与否定关系图交互：如果该事实之前是缺失项，移除对应否定关系
        if self.negation_graph:
            # 生成可能的否定关系键
            negation_key = f"非我不是{key}"
            # 移除对应的否定关系
            self.negation_graph.remove_node_by_content(negation_key)
    
    def retrieve_facts(self, query: str, k: int = None) -> List[Tuple[str, Any, float]]:
        """检索相关事实"""
        if k is None:
            k = self.config["declarative_retrieval_k"]
        min_similarity = self.config["declarative_retrieval_similarity"]
        return self.declarative.retrieve_related(query, k, min_similarity)
    
    def log_event(self, user_input: str, response: str, emotion: float, 
                 self_depth: float, salience: Optional[float] = None, 
                 er_trigger: bool = False):
        """记录事件到情景记忆"""
        self.episodic.log(user_input, response, emotion, self_depth, salience, er_trigger)
    
    def retrieve_events(self, query: Optional[str] = None, 
                       time_range: Optional[Tuple[float, float]] = None, 
                       k: int = None) -> List[Dict]:
        """检索情景事件"""
        if k is None:
            k = self.config["episodic_retrieval_k"]
        return self.episodic.retrieve(query, time_range, k)
    
    def decay_and_prune(self):
        """定期调用，衰减置信度并删除低置信度事实"""
        # 清理弱记忆
        self.declarative._cleanup()
    
    def store_memory(self, memory_content: Dict):
        """存储记忆（兼容旧版本）"""
        # 生成唯一ID
        memory_id = f"mem_{int(time.time())}"
        memory_content['id'] = memory_id
        memory_content['timestamp'] = time.time()
        
        # 将记忆转换为事实存储到声明性记忆
        if 'type' in memory_content:
            memory_type = memory_content['type']
            key = f"{memory_type}_{memory_id}"
            self.store_fact(key, memory_content)
    
    def extract_frequent_facts(self, min_frequency: int = 3) -> List[Tuple[str, Any]]:
        """从情景记忆中提取高频事实"""
        # 统计用户输入和响应中的高频事实
        fact_counts = {}
        
        for event in self.episodic.events:
            # 简单实现：提取用户输入中的可能事实
            # 这里可以使用更复杂的NLP方法
            input_text = event.get('user_input', '')
            if input_text:
                # 示例：提取"我叫X"类型的事实
                if "我叫" in input_text:
                    parts = input_text.split("我叫")
                    if len(parts) > 1:
                        name = parts[1].strip()
                        key = "user_name"
                        if key not in fact_counts:
                            fact_counts[key] = {}
                        if name not in fact_counts[key]:
                            fact_counts[key][name] = 0
                        fact_counts[key][name] += 1
                # 示例：提取"我是X"类型的事实
                if "我是" in input_text:
                    parts = input_text.split("我是")
                    if len(parts) > 1:
                        identity = parts[1].strip()
                        key = "user_identity"
    
    def integrate_with_knowledge(self, knowledge_integration):
        """与知识库集成"""
        self.knowledge_integration = knowledge_integration
    
    def retrieve_enhanced(self, query, max_results=3):
        """增强检索"""
        # 简单实现，返回空列表
        return []


class EmotionSystem:
    """情绪系统 - 管理情绪状态并与否定机制集成"""
    
    def __init__(self):
        self.emotion_states = {}
        self.emotion_history = []
        self.base_emotion = 0.0  # 基础情绪值
    
    def update_emotion(self, source: str, value: float):
        """更新情绪状态"""
        # 限制情绪值范围在[-1, 1]
        value = max(-1.0, min(1.0, value))
        
        # 更新情绪状态
        self.emotion_states[source] = value
        
        # 计算综合情绪值
        overall_emotion = self.calculate_overall_emotion()
        
        # 记录情绪历史
        self.emotion_history.append({
            'timestamp': time.time(),
            'source': source,
            'value': value,
            'overall': overall_emotion
        })
        
        # 限制历史记录长度
        if len(self.emotion_history) > 100:
            self.emotion_history.pop(0)
        
        return overall_emotion
    
    def calculate_overall_emotion(self) -> float:
        """计算综合情绪值"""
        if not self.emotion_states:
            return self.base_emotion
        
        # 加权计算综合情绪
        weights = {
            'negation_complexity': 0.3,
            'prediction_error': 0.2,
            'attachment': 0.2,
            'novelty': 0.15,
            'fantasy_layer': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for source, value in self.emotion_states.items():
            weight = weights.get(source, 0.1)  # 默认权重
            weighted_sum += value * weight
            total_weight += weight
        
        if total_weight > 0:
            return max(-1.0, min(1.0, weighted_sum / total_weight))
        else:
            return self.base_emotion
    
    def get_emotion_state(self) -> Dict:
        """获取当前情绪状态"""
        return {
            'states': self.emotion_states,
            'overall': self.calculate_overall_emotion(),
            'history_length': len(self.emotion_history)
        }


class DecisionSystem:
    """决策系统 - 基于否定状态提供决策建议"""
    
    def __init__(self):
        self.contexts = {}
        self.decision_history = []
        self.default_action = 'maintain_balance'
    
    def add_context(self, context_type: str, context_data: Dict):
        """添加决策上下文"""
        self.contexts[context_type] = context_data
    
    def get_decision(self) -> Dict:
        """获取决策"""
        # 综合所有上下文信息
        negation_context = self.contexts.get('negation', {})
        emotion_context = self.contexts.get('emotion', {})
        memory_context = self.contexts.get('memory', {})
        
        # 基于否定复杂度和未实现潜力制定决策
        complexity = negation_context.get('negation_complexity', 0.0)
        unrealized_count = negation_context.get('unrealized_potential_count', 0)
        suggested_action = negation_context.get('suggested_action', self.default_action)
        
        # 考虑情绪因素
        overall_emotion = emotion_context.get('overall', 0.0)
        
        # 制定最终决策
        decision = {
            'action': suggested_action,
            'confidence': self._calculate_confidence(complexity, unrealized_count, overall_emotion),
            'context': {
                'negation': negation_context,
                'emotion': emotion_context,
                'memory': memory_context
            },
            'timestamp': time.time()
        }
        
        # 记录决策历史
        self.decision_history.append(decision)
        if len(self.decision_history) > 50:
            self.decision_history.pop(0)
        
        return decision
    
    def _calculate_confidence(self, complexity: float, unrealized_count: int, emotion: float) -> float:
        """计算决策信心"""
        # 基于复杂度、未实现潜力和情绪计算信心
        complexity_factor = 1.0 - min(1.0, abs(complexity - 5.0) / 5.0)  # 中等复杂度信心高
        potential_factor = min(1.0, unrealized_count / 10.0)  # 未实现潜力适中时信心高
        emotion_factor = 0.5 + 0.5 * abs(emotion)  # 情绪强烈时信心高
        
        return min(1.0, (complexity_factor + potential_factor + emotion_factor) / 3.0)
    
    def get_decision_history(self) -> List[Dict]:
        """获取决策历史"""
        return self.decision_history


class NegationIntegrationSystem:
    """否定集成系统 - 实现否定关系与其他系统的集成"""
    
    def __init__(self, negation_graph, memory_system=None, emotion_system=None, decision_system=None):
        self.negation_graph = negation_graph
        self.memory_system = memory_system
        self.emotion_system = emotion_system
        self.decision_system = decision_system
        
    def update_memory_from_negation(self):
        """将否定关系转化为记忆"""
        if not self.memory_system:
            return
        
        unrealized = self.negation_graph.get_unrealized_potential()
        for concept_id in unrealized[:10]:  # 处理前10个未实现的概念
            if concept_id in self.negation_graph.nodes:
                node = self.negation_graph.nodes[concept_id]
                # 将否定关系存储为记忆
                memory_content = {
                    'type': 'negation',
                    'content': node['content'].detach().cpu().numpy().tolist() if hasattr(node['content'], 'detach') else node['content'],
                    'semantic_tags': node.get('semantic_tags', []),
                    'negation_strength': node.get('negation_strength', 0.0),
                    'frequency': node.get('frequency', 0),
                    'timestamp': node.get('creation_time', time.time())
                }
                self.memory_system.store_memory(memory_content)
                
    def update_emotion_from_complexity(self):
        """根据否定复杂度更新情绪状态"""
        if not self.emotion_system:
            return
        
        complexity = self.negation_graph.get_negation_complexity()
        # 复杂度与情绪的关系：中等复杂度促进积极情绪，过高或过低则产生消极情绪
        if 2.0 < complexity < 7.0:
            emotion_valence = 0.2 * (5.0 - abs(complexity - 5.0))  # 倒U形曲线
        else:
            emotion_valence = -0.1 * abs(complexity - 5.0)
        
        # 这里假设emotion_system有update_emotion方法
        if hasattr(self.emotion_system, 'update_emotion'):
            self.emotion_system.update_emotion('negation_complexity', emotion_valence)
            
    def inform_decision(self):
        """为决策系统提供否定信息"""
        if not self.decision_system:
            return
        
        complexity = self.negation_graph.get_negation_complexity()
        unrealized_potential = self.negation_graph.get_unrealized_potential()
        
        # 基于否定状态生成建议动作
        if complexity < 2.0:
            suggested_action = 'explore'
        elif complexity > 7.0:
            suggested_action = 'simplify'
        else:
            suggested_action = 'maintain_balance'
        
        # 为决策系统提供上下文
        negation_context = {
            'negation_complexity': complexity,
            'unrealized_potential_count': len(unrealized_potential),
            'suggested_action': suggested_action
        }
        
        self.decision_system.add_context('negation', negation_context)
    
    def integrate(self):
        """执行集成操作"""
        # 从否定关系更新记忆
        self.update_memory_from_negation()
        
        # 从否定复杂度更新情绪
        self.update_emotion_from_complexity()
        
        # 为决策系统提供信息
        self.inform_decision()