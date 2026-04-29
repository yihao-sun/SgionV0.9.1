#!/usr/bin/env python3
"""
否定关系图 (Negation Graph)
哲学对应：《论存在》第3.1.1节，非我的否定意义。
功能：分层存储（核心/动态/短期）否定关系，势能衰减与TTL遗忘，压抑内容检测。
主要类：LayeredNegGraph, NegationLayer, NegationNode
"""

import time
from typing import Dict, List, Optional
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger('negation_graph')


class NegationNode:
    """
    否定关系图节点
    """
    
    def __init__(self, node_id: str, description: str, potency: float = 1.0, parent_id: Optional[str] = None, created_step: int = 0):
        """
        初始化否定关系图节点
        
        Args:
            node_id: 节点唯一标识符
            description: 节点描述（文本描述或嵌入向量）
            potency: 势能值
            parent_id: 父节点ID（用于否定链）
            created_step: 创建时的步数（用于TTL）
        """
        self.id = node_id
        self.description = description   # 文本描述或嵌入向量
        self.potency = potency           # 势能 float
        self.parent_id = parent_id       # 父节点ID（用于否定链）
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.protected = False
        self.created_step = created_step  # 创建步数，用于TTL计算
    
    def __repr__(self):
        return f"NegationNode(id={self.id}, description={self.description}, potency={self.potency:.3f})"


class NegationLayer:
    """
    否定关系图层
    """
    
    def __init__(self, name: str, decay_rate: float, prune_threshold: float, max_nodes: int, protected: bool = False, ttl_steps: Optional[int] = None):
        """
        初始化否定关系图层
        
        Args:
            name: 层名称
            decay_rate: 势能衰减率
            prune_threshold: 剪枝阈值
            max_nodes: 最大节点数
            protected: 是否默认保护节点
            ttl_steps: 短期层节点生存时间（步数）
        """
        self.name = name
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self.max_nodes = max_nodes
        self.protected = protected
        self.ttl_steps = ttl_steps
        self.nodes: Dict[str, NegationNode] = {}
        self.next_id_counter = 0
        self.current_step = 0
        
        # 新增：势能分桶索引（0-19，每0.05一桶）
        self.potency_buckets = {i: set() for i in range(20)}
        # 新增：该层总势能缓存
        self._total_potency = 0.0
    
    def _next_id(self) -> str:
        """
        生成唯一节点ID
        
        Returns:
            唯一节点ID
        """
        node_id = f"{self.name}_{self.next_id_counter}"
        self.next_id_counter += 1
        return node_id
    
    def _get_bucket(self, potency: float) -> int:
        """根据势能返回桶索引（0-19）"""
        return min(19, int(potency * 20))
    
    def add(self, description: str, initial_potency: float = 1.0, parent_id: Optional[str] = None) -> str:
        """
        添加否定关系节点
        
        Args:
            description: 节点描述
            initial_potency: 初始势能
            parent_id: 父节点ID
        
        Returns:
            新节点的ID
        """
        node_id = self._next_id()
        node = NegationNode(node_id, description, initial_potency, parent_id, created_step=self.current_step)
        if self.protected:
            node.protected = True
        self.nodes[node_id] = node
        
        # 更新分桶索引
        bucket = self._get_bucket(initial_potency)
        self.potency_buckets[bucket].add(node_id)
        self._total_potency += initial_potency
        
        logger.debug(f"Adding negation to {self.name} layer: {description} with potency {initial_potency}")
        
        if len(self.nodes) > self.max_nodes:
            self.prune()
        
        return node_id
    
    def update_potency(self, node_id: str, delta: float):
        """
        更新节点势能
        
        Args:
            node_id: 节点ID
            delta: 势能变化量
        """
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        old_bucket = self._get_bucket(node.potency)
        old_potency = node.potency
        
        node.potency = max(0.0, node.potency + delta)
        node.last_accessed = time.time()
        
        # 更新分桶索引
        new_bucket = self._get_bucket(node.potency)
        if old_bucket != new_bucket:
            self.potency_buckets[old_bucket].discard(node_id)
            self.potency_buckets[new_bucket].add(node_id)
        
        # 更新总势能缓存
        self._total_potency += (node.potency - old_potency)
        
        logger.debug(f"Updated potency for node {node_id} in {self.name} layer: {node.potency:.3f}")
    
    def access(self, node_id: str):
        """
        访问节点，更新最后访问时间
        
        Args:
            node_id: 节点ID
        """
        if node_id in self.nodes:
            self.nodes[node_id].last_accessed = time.time()
    
    def _remove_node(self, node_id: str):
        """内部方法：删除节点并更新缓存与分桶"""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        bucket = self._get_bucket(node.potency)
        self.potency_buckets[bucket].discard(node_id)
        self._total_potency -= node.potency
        del self.nodes[node_id]
    
    def decay(self):
        """图层势能衰减，同步更新缓存和分桶"""
        if self.ttl_steps is not None:
            self.current_step += 1
            nodes_to_remove = []
            for node_id, node in self.nodes.items():
                age_in_steps = self.current_step - node.created_step
                if age_in_steps > self.ttl_steps:
                    nodes_to_remove.append(node_id)
            for node_id in nodes_to_remove:
                self._remove_node(node_id)
        
        potency_delta = 0.0
        for node_id, node in list(self.nodes.items()):
            if not node.protected:
                old_bucket = self._get_bucket(node.potency)
                old_potency = node.potency
                node.potency *= self.decay_rate
                potency_delta += (node.potency - old_potency)
                new_bucket = self._get_bucket(node.potency)
                if old_bucket != new_bucket:
                    self.potency_buckets[old_bucket].discard(node_id)
                    self.potency_buckets[new_bucket].add(node_id)
        
        self._total_potency += potency_delta
        logger.debug(f"Decayed {self.name} layer, total potency now {self._total_potency:.3f}")
    
    def prune(self, threshold: Optional[float] = None, current_step: Optional[int] = None):
        if threshold is None:
            threshold = self.prune_threshold
        
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            if not node.protected and node.potency < threshold:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            self._remove_node(node_id)
        
        if nodes_to_remove:
            logger.info(f"Pruned {len(nodes_to_remove)} nodes in {self.name} layer")
    
    def get_topk(self, k: int) -> List[NegationNode]:
        """从高势能桶向低势能桶收集，直到凑够 k 个节点"""
        results = []
        for bucket_idx in range(19, -1, -1):
            for node_id in self.potency_buckets[bucket_idx]:
                if node_id in self.nodes:
                    results.append(self.nodes[node_id])
                    if len(results) >= k:
                        return results[:k]
        return results
    
    def total_potency(self) -> float:
        """返回该层总势能（O(1)缓存）"""
        return self._total_potency
    
    def clear(self, keep_protected: bool = True):
        """
        清空非保护节点
        
        Args:
            keep_protected: 是否保留受保护节点
        """
        if keep_protected:
            # 只保留受保护节点
            protected_nodes = {node_id: node for node_id, node in self.nodes.items() if node.protected}
            self.nodes = protected_nodes
        else:
            # 清空所有节点
            self.nodes = {}
        
        # 重新初始化分桶索引和总势能缓存
        self.potency_buckets = {i: set() for i in range(20)}
        self._total_potency = 0.0
        
        # 重新计算总势能和分桶索引
        for node_id, node in self.nodes.items():
            bucket = self._get_bucket(node.potency)
            self.potency_buckets[bucket].add(node_id)
            self._total_potency += node.potency
        
        logger.info(f"Cleared {self.name} layer, remaining nodes: {len(self.nodes)}")
    
    def __len__(self):
        """
        返回节点数量
        
        Returns:
            节点数量
        """
        return len(self.nodes)


class LayeredNegGraph:
    """
    分层否定关系图
    """
    
    def __init__(self, config):
        """
        初始化分层否定关系图
        
        Args:
            config: 配置字典
        """
        self.config = config
        # 核心层：慢衰减，不参与常规剪枝
        self.core = NegationLayer(
            name='core',
            decay_rate=config.get('memory.core_decay_rate', 0.999),
            prune_threshold=config.get('memory.core_prune_threshold', 0.2),
            max_nodes=config.get('memory.core_max_nodes', 2000),
            protected=True   # 核心层节点默认保护
        )
        # 动态层：正常衰减，定期剪枝
        self.dynamic = NegationLayer(
            name='dynamic',
            decay_rate=config.get('memory.dynamic_decay_rate', 0.99),
            prune_threshold=config.get('memory.dynamic_prune_threshold', 0.1),
            max_nodes=config.get('memory.dynamic_max_nodes', 10000)
        )
        # 短期层：快速衰减，仅保留最近N步
        self.short_term = NegationLayer(
            name='short_term',
            decay_rate=config.get('memory.short_term_decay_rate', 0.95),
            prune_threshold=config.get('memory.short_term_prune_threshold', 0.05),
            max_nodes=config.get('memory.short_term_max_nodes', 1000),
            ttl_steps=config.get('memory.short_term_ttl', 1000)
        )
        self.logger = get_logger('neg_graph')
        # 节点ID到层的映射，用于快速查找
        self.node_to_layer = {}
        self._total_potency = 0.0  # 缓存三层总势能
    
    def add_negation(self, description, layer='dynamic', initial_potency=1.0, parent_id=None):
        if layer == 'core':
            node_id = self.core.add(description, initial_potency, parent_id)
        elif layer == 'short_term':
            node_id = self.short_term.add(description, initial_potency, parent_id)
        else:
            node_id = self.dynamic.add(description, initial_potency, parent_id)
        
        self.node_to_layer[node_id] = layer
        self._total_potency = self.core.total_potency() + self.dynamic.total_potency() + self.short_term.total_potency()
        return node_id
    
    def update_potency(self, node_id, delta, layer='dynamic'):
        actual_layer = self.node_to_layer.get(node_id, layer)
        if actual_layer == 'core':
            self.core.update_potency(node_id, delta)
        elif actual_layer == 'short_term':
            self.short_term.update_potency(node_id, delta)
        else:
            self.dynamic.update_potency(node_id, delta)
        self._total_potency = self.core.total_potency() + self.dynamic.total_potency() + self.short_term.total_potency()
    
    def decay_all(self):
        """
        所有层势能衰减
        """
        # 记录衰减前的节点数
        core_before = len(self.core)
        dynamic_before = len(self.dynamic)
        short_term_before = len(self.short_term)
        
        self.core.decay()
        self.dynamic.decay()
        self.short_term.decay()
        
        # 更新总势能缓存
        self._total_potency = self.core.total_potency() + self.dynamic.total_potency() + self.short_term.total_potency()
        
        # 记录衰减后的节点数
        core_after = len(self.core)
        dynamic_after = len(self.dynamic)
        short_term_after = len(self.short_term)
        
        # 输出各层节点数变化
        logger.info(f"LayeredNegGraph decay_all: core={core_before}->{core_after}, "
                   f"dynamic={dynamic_before}->{dynamic_after}, "
                   f"short_term={short_term_before}->{short_term_after}, "
                   f"total_potency={self._total_potency:.3f}")
    
    def prune(self):
        """
        剪枝操作
        """
        self.dynamic.prune()
        self.short_term.prune()
        # core层不自动剪枝，但可设置容量上限后移除最低势能节点
        if len(self.core) > self.core.max_nodes:
            # 按势能排序，保留势能高的节点
            sorted_nodes = sorted(self.core.nodes.values(), key=lambda x: x.potency, reverse=True)
            nodes_to_keep = sorted_nodes[:self.core.max_nodes]
            keep_ids = {node.id for node in nodes_to_keep}
            
            nodes_to_remove = [node_id for node_id in self.core.nodes if node_id not in keep_ids]
            for node_id in nodes_to_remove:
                self.core._remove_node(node_id)
                if node_id in self.node_to_layer:
                    del self.node_to_layer[node_id]
            
            logger.info(f"Pruning core layer to max_nodes, removed {len(nodes_to_remove)} nodes")
        
        # 更新总势能缓存
        self._total_potency = self.core.total_potency() + self.dynamic.total_potency() + self.short_term.total_potency()
    
    def get_total_potency(self):
        """返回三层总势能（O(1)）"""
        return self._total_potency
    
    def get_topk(self, k, exclude_protected=True, layers=['core','dynamic','short_term']):
        """
        合并各层结果，按势能排序返回topk
        
        Args:
            k: 返回节点数量
            exclude_protected: 是否排除受保护节点
            layers: 参与排序的层
        
        Returns:
            势能最高的k个节点列表
        """
        all_nodes = []
        
        if 'core' in layers:
            all_nodes.extend(self.core.get_topk(k))
        if 'dynamic' in layers:
            all_nodes.extend(self.dynamic.get_topk(k))
        if 'short_term' in layers:
            all_nodes.extend(self.short_term.get_topk(k))
        
        # 去重并按势能排序
        unique_nodes = {node.id: node for node in all_nodes}
        sorted_nodes = sorted(unique_nodes.values(), key=lambda x: x.potency, reverse=True)
        return sorted_nodes[:k]
    
    def get_repressed_candidates(self, k: int = 3) -> List[NegationNode]:
        """获取逆袭候选：从各层取势能最高的节点，合并后返回前 k 个"""
        all_nodes = []
        all_nodes.extend(self.core.get_topk(k))
        all_nodes.extend(self.dynamic.get_topk(k))
        all_nodes.extend(self.short_term.get_topk(k))
        # 按势能降序排序
        all_nodes.sort(key=lambda x: x.potency, reverse=True)
        # 去重（基于节点ID）
        seen = set()
        unique_nodes = []
        for node in all_nodes:
            if node.id not in seen:
                seen.add(node.id)
                unique_nodes.append(node)
            if len(unique_nodes) >= k:
                break
        return unique_nodes[:k]
    
    def clear(self, keep_protected=True):
        """
        清空操作
        
        Args:
            keep_protected: 是否保留受保护节点
        """
        if keep_protected:
            self.dynamic.clear(keep_protected=False)  # 动态层清空
            self.short_term.clear()
            # 核心层不清空
        else:
            self.core.clear()
            self.dynamic.clear()
            self.short_term.clear()
        
        # 重置节点到层的映射
        if not keep_protected:
            self.node_to_layer = {}
        
        # 更新总势能缓存
        self._total_potency = self.core.total_potency() + self.dynamic.total_potency() + self.short_term.total_potency()
    
    def __len__(self):
        """
        返回所有层节点总数
        
        Returns:
            所有层节点总数
        """
        return len(self.core) + len(self.dynamic) + len(self.short_term)
    
    def protect_node(self, node_id):
        """
        保护指定的节点，防止被遗忘操作删除
        
        Args:
            node_id: 节点ID
        
        Returns:
            bool: 是否成功保护
        """
        # 查找节点所在的层
        layer = self._find_node_layer(node_id)
        if layer:
            if node_id in layer.nodes:
                layer.nodes[node_id].protected = True
                self.logger.info(f"Protected node: {node_id}")
                return True
        return False
    
    def unprotect_node(self, node_id):
        """
        取消保护指定的节点
        
        Args:
            node_id: 节点ID
        
        Returns:
            bool: 是否成功取消保护
        """
        # 查找节点所在的层
        layer = self._find_node_layer(node_id)
        if layer:
            if node_id in layer.nodes:
                layer.nodes[node_id].protected = False
                self.logger.info(f"Unprotected node: {node_id}")
                return True
        return False
    
    def _find_node_layer(self, node_id):
        """
        查找节点所在的层
        
        Args:
            node_id: 节点ID
        
        Returns:
            NegationLayer or None: 节点所在的层
        """
        # 首先尝试从映射中查找
        layer_name = self.node_to_layer.get(node_id)
        if layer_name == 'core':
            return self.core
        elif layer_name == 'dynamic':
            return self.dynamic
        elif layer_name == 'short_term':
            return self.short_term
        
        # 如果映射中没有，遍历所有层查找
        for layer in [self.core, self.dynamic, self.short_term]:
            if node_id in layer.nodes:
                # 更新映射
                self.node_to_layer[node_id] = layer.name
                return layer
        
        return None
    
    def detect_repressed_content(self, threshold_potency=0.5, threshold_days=7):
        """
        检测长期未访问但高势能的节点（压抑内容）
        
        Args:
            threshold_potency: 势能阈值
            threshold_days: 未访问天数阈值
        
        Returns:
            压抑内容节点列表
        """
        import time
        now = time.time()
        repressed = []
        
        for layer in [self.core, self.dynamic, self.short_term]:
            for node in layer.nodes.values():
                # 计算未访问时间（秒）
                time_since_access = now - node.last_accessed
                # 检查是否超过阈值天数且势能高于阈值
                if time_since_access > threshold_days * 86400 and node.potency > threshold_potency:
                    repressed.append(node)
        
        return repressed
    
    def add_negative_consequence(self, text, potency_increment=0.1):
        """
        为与给定文本相关的否定节点增加势能
        
        Args:
            text: 文本内容
            potency_increment: 势能增加量
        """
        if not text:
            return
        
        # 查找与文本相关的节点
        related_nodes = []
        for layer in [self.core, self.dynamic, self.short_term]:
            for node_id, node in layer.nodes.items():
                # 简单实现：检查节点描述是否包含文本中的关键词
                # 实际实现中应该使用更复杂的相似度计算
                if isinstance(node.description, str) and text in node.description:
                    related_nodes.append((node_id, layer))
                elif isinstance(node.description, str):
                    # 检查文本是否包含节点描述中的关键词
                    node_words = node.description.split()
                    for word in node_words:
                        if word in text:
                            related_nodes.append((node_id, layer))
                            break
        
        # 为相关节点增加势能
        for node_id, layer in related_nodes:
            layer.update_potency(node_id, potency_increment)
            self.logger.debug(f"Added negative consequence to node {node_id} in {layer.name} layer: +{potency_increment} potency")
        
        # 如果没有找到相关节点，创建一个新的否定节点
        if not related_nodes:
            description = f"negative_consequence_{text[:50]}"
            node_id = self.add_negation(description, layer='dynamic', initial_potency=potency_increment)
            self.logger.debug(f"Created new negative consequence node {node_id} for text: {text[:50]}")
    
    def to_dict(self):
        """
        将否定图序列化为字典
        
        Returns:
            包含所有层和节点信息的字典
        """
        graph_dict = {
            "core": self._layer_to_dict(self.core),
            "dynamic": self._layer_to_dict(self.dynamic),
            "short_term": self._layer_to_dict(self.short_term),
            "node_to_layer": self.node_to_layer
        }
        return graph_dict
    
    def _layer_to_dict(self, layer):
        """
        将单个层序列化为字典
        
        Args:
            layer: NegationLayer实例
            
        Returns:
            层的字典表示
        """
        layer_dict = {
            "name": layer.name,
            "decay_rate": layer.decay_rate,
            "prune_threshold": layer.prune_threshold,
            "max_nodes": layer.max_nodes,
            "protected": layer.protected,
            "ttl_steps": layer.ttl_steps,
            "next_id_counter": layer.next_id_counter,
            "current_step": layer.current_step,
            "nodes": {}
        }
        
        for node_id, node in layer.nodes.items():
            node_dict = {
                "id": node.id,
                "description": node.description,
                "potency": node.potency,
                "parent_id": node.parent_id,
                "created_step": node.created_step,
                "protected": node.protected
            }
            layer_dict["nodes"][node_id] = node_dict
        
        return layer_dict
    
    def from_dict(self, graph_dict):
        """
        从字典恢复否定图
        
        Args:
            graph_dict: 包含否定图信息的字典
        """
        # 恢复各层
        if "core" in graph_dict:
            self._layer_from_dict(self.core, graph_dict["core"])
        if "dynamic" in graph_dict:
            self._layer_from_dict(self.dynamic, graph_dict["dynamic"])
        if "short_term" in graph_dict:
            self._layer_from_dict(self.short_term, graph_dict["short_term"])
        
        # 恢复节点到层的映射
        if "node_to_layer" in graph_dict:
            self.node_to_layer = graph_dict["node_to_layer"]
    
    def _layer_from_dict(self, layer, layer_dict):
        """
        从字典恢复单个层
        
        Args:
            layer: NegationLayer实例
            layer_dict: 层的字典表示
        """
        # 恢复层的属性
        layer.next_id_counter = layer_dict.get("next_id_counter", 0)
        layer.current_step = layer_dict.get("current_step", 0)
        
        # 恢复节点
        if "nodes" in layer_dict:
            layer.nodes = {}
            for node_id, node_dict in layer_dict["nodes"].items():
                node = NegationNode(
                    node_id=node_dict["id"],
                    description=node_dict["description"],
                    potency=node_dict["potency"],
                    parent_id=node_dict["parent_id"],
                    created_step=node_dict["created_step"]
                )
                node.protected = node_dict["protected"]
                layer.nodes[node_id] = node



