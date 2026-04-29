#!/usr/bin/env python3
"""
语义记忆 (Semantic Memory)
哲学对应：《论存在》第3.1.3节，非我反哺的固化知识。
功能：存储事实性知识（图结构+嵌入向量），支持嵌入检索，置信度衰减与剪枝。
主要类：SemanticMemory
"""

import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger('semantic_memory')


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        a: 第一个向量
        b: 第二个向量
    
    Returns:
        余弦相似度值
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SemanticMemory:
    """
    语义记忆类
    """
    
    def __init__(self, config):
        """
        初始化语义记忆
        
        Args:
            config: 配置字典
        """
        self.max_capacity = config.get('memory.semantic_max_capacity', 10000)
        self.decay_rate = config.get('memory.semantic_decay_rate', 0.999)
        self.graph = nx.DiGraph()  # 有向图存储知识三元组
        self.knowledge_store = {}  # 保留原有键值存储，用于兼容旧接口
        self.logger = get_logger('semantic_memory')
    
    def store(self, key, value, embedding, confidence=1.0):
        """
        存储事实性知识
        
        Args:
            key: 知识的键
            value: 知识的值
            embedding: 知识的嵌入向量
            confidence: 置信度
        """
        if len(self.knowledge_store) >= self.max_capacity:
            self._evict_lowest_confidence()
        
        self.knowledge_store[key] = {
            'value': value,
            'embedding': embedding,
            'confidence': confidence,
            'last_accessed': time.time()
        }
        self.logger.debug(f"Stored knowledge: {key} -> {value} with confidence {confidence}")
    
    def retrieve(self, query_embedding, k=5, min_confidence=0.3):
        """
        根据嵌入向量检索知识
        
        Args:
            query_embedding: 查询嵌入向量
            k: 返回前k个结果
            min_confidence: 最小置信度阈值
        
        Returns:
            排序后的结果列表，每个元素为 (key, value, confidence, similarity)
        """
        results = []
        
        for key, item in self.knowledge_store.items():
            if item['confidence'] >= min_confidence:
                similarity = cosine_similarity(query_embedding, item['embedding'])
                results.append((key, item['value'], item['confidence'], similarity))
        
        # 按相似度排序，返回前k个
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:k]
    
    def decay(self):
        """
        所有条目的置信度衰减
        """
        for key in self.knowledge_store:
            # 受保护的条目不衰减
            if not self.knowledge_store[key].get('protected', False):
                self.knowledge_store[key]['confidence'] *= self.decay_rate
                self.knowledge_store[key]['last_accessed'] = time.time()
        
        self.logger.debug(f"Decayed all non-protected entries by factor {self.decay_rate}")
    
    def prune(self, threshold=0.1):
        """
        修剪低置信度条目
        
        Args:
            threshold: 置信度阈值
        """
        to_delete = [k for k, v in self.knowledge_store.items() 
                   if not v.get('protected', False) and v['confidence'] < threshold]
        
        for k in to_delete:
            del self.knowledge_store[k]
        
        self.logger.info(f"Pruned {len(to_delete)} non-protected entries with confidence < {threshold}")
    
    def _evict_lowest_confidence(self):
        """
        当达到容量上限时，删除置信度最低的非受保护条目
        """
        if not self.knowledge_store:
            return
        
        # 找出置信度最低的非受保护条目
        non_protected_entries = {k: v for k, v in self.knowledge_store.items() 
                               if not v.get('protected', False)}
        
        if non_protected_entries:
            # 从非受保护条目中找出置信度最低的
            lowest_key = min(non_protected_entries, key=lambda k: non_protected_entries[k]['confidence'])
            del self.knowledge_store[lowest_key]
            self.logger.debug(f"Evicted lowest confidence non-protected entry: {lowest_key}")
        else:
            # 如果所有条目都受保护，则删除置信度最低的条目
            lowest_key = min(self.knowledge_store, key=lambda k: self.knowledge_store[k]['confidence'])
            del self.knowledge_store[lowest_key]
            self.logger.debug(f"Evicted lowest confidence entry (all protected): {lowest_key}")
    
    def __len__(self):
        """
        返回存储的条目数量
        
        Returns:
            条目数量
        """
        return len(self.knowledge_store)
    
    def protect_key(self, key):
        """
        保护指定的知识条目，防止被遗忘操作删除
        
        Args:
            key: 知识条目的键
        
        Returns:
            bool: 是否成功保护
        """
        if key in self.knowledge_store:
            self.knowledge_store[key]['protected'] = True
            self.logger.info(f"Protected knowledge: {key}")
            return True
        return False
    
    def unprotect_key(self, key):
        """
        取消保护指定的知识条目
        
        Args:
            key: 知识条目的键
        
        Returns:
            bool: 是否成功取消保护
        """
        if key in self.knowledge_store:
            self.knowledge_store[key]['protected'] = False
            self.logger.info(f"Unprotected knowledge: {key}")
            return True
        return False
    
    def add_fact(self, subject: str, relation: str, object: str, confidence: float = 1.0):
        """
        添加三元组事实到图结构
        
        Args:
            subject: 主语
            relation: 关系
            object: 宾语
            confidence: 置信度
        """
        # 检查图的大小，超过容量则剪枝
        if self.graph.number_of_edges() >= self.max_capacity:
            self._prune_graph()
        
        # 添加边，存储置信度和时间戳
        self.graph.add_edge(subject, object, relation=relation, confidence=confidence, last_accessed=time.time())
        self.logger.debug(f"Added fact: {subject} {relation} {object} with confidence {confidence}")
    
    def query_fact(self, subject: str = None, relation: str = None, object: str = None) -> List[Tuple]:
        """
        查询三元组事实
        
        Args:
            subject: 主语（可选）
            relation: 关系（可选）
            object: 宾语（可选）
        
        Returns:
            匹配的三元组列表，每个元素为 (subject, relation, object, confidence)
        """
        results = []
        
        # 遍历所有边
        for u, v, data in self.graph.edges(data=True):
            # 检查是否匹配查询条件
            if subject and u != subject:
                continue
            if relation and data.get('relation') != relation:
                continue
            if object and v != object:
                continue
            
            # 添加匹配的结果
            results.append((u, data.get('relation'), v, data.get('confidence', 1.0)))
        
        # 按置信度排序
        results.sort(key=lambda x: x[3], reverse=True)
        return results
    
    def load_conceptnet(self, data_path: str):
        """
        从精简版 ConceptNet 文件加载常识
        
        Args:
            data_path: 数据文件路径
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析 CSV 格式：subject,relation,object
                    parts = line.split(',')
                    if len(parts) >= 3:
                        subject = parts[0].strip()
                        relation = parts[1].strip()
                        object = parts[2].strip()
                        confidence = float(parts[3]) if len(parts) > 3 else 1.0
                        self.add_fact(subject, relation, object, confidence)
            
            self.logger.info(f"Loaded ConceptNet data from {data_path}, current graph size: {self.graph.number_of_edges()}")
        except Exception as e:
            self.logger.error(f"Failed to load ConceptNet data: {e}")
    
    def _prune_graph(self, threshold=0.1):
        """
        修剪图中低置信度的边
        
        Args:
            threshold: 置信度阈值
        """
        to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            confidence = data.get('confidence', 1.0)
            if confidence < threshold:
                to_remove.append((u, v))
        
        for u, v in to_remove:
            self.graph.remove_edge(u, v)
        
        self.logger.info(f"Pruned {len(to_remove)} edges with confidence < {threshold}")
