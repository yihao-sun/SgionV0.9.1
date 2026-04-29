#!/usr/bin/env python3
"""
潜在存在空间 (Latent Presence Space, LPS)
哲学对应：《论存在》第3章，存在本身作为一切可能性的总和。
功能：维护高维向量场（FAISS索引），支持相似性查询和低势能采样（勇气机制）。
主要类：LPS
"""

import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger
from utils.config_loader import Config
import time
import os
import logging

# 抑制 sentence_transformers 的 INFO 日志
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class LPS:
    def __init__(self, config=None):
        """
        初始化潜在存在空间
        
        Args:
            config: 配置对象
        """
        import os
        
        # 获取项目根目录的绝对路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # 确保 project_root 是 ExistenceEngine 目录
        if not project_root.endswith('ExistenceEngine'):
            project_root = os.path.join(project_root, 'ExistenceEngine')
        
        self.config = config or Config()
        self.max_capacity = self.config.get('lps.max_capacity', 100000)
        self.default_potency = self.config.get('lps.default_potency', 0.5)
        self.query_k = self.config.get('lps.query_k', 10)
        self.low_potency_threshold = self.config.get('lps.low_potency_threshold', 0.2)
        
        # 初始化日志记录器
        self.logger = get_logger('lps')
        
        # 初始化嵌入模型（使用支持中文的多语言模型）
        import os
        # 完全禁用Hugging Face连接
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(project_root, 'model_cache')
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        from sentence_transformers import SentenceTransformer
        
        # 强制使用本地多语言模型
        model_path = os.path.join(project_root, 'models', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.logger.info(f"尝试加载本地多语言模型: {model_path}")
        self.logger.info(f"模型路径是否存在: {os.path.exists(model_path)}")
        
        try:
            # 直接加载本地模型，指定使用本地文件
            self.encoder = SentenceTransformer(model_path, device='cpu', use_auth_token=False)
            self.logger.info("本地多语言模型加载成功！")
        except Exception as e:
            self.logger.error(f"加载本地多语言模型失败: {e}")
            # 作为备选，尝试使用本地英文模型
            en_model_path = os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')
            self.logger.info(f"尝试加载本地英文模型: {en_model_path}")
            self.logger.info(f"模型路径是否存在: {os.path.exists(en_model_path)}")
            try:
                self.encoder = SentenceTransformer(en_model_path, device='cpu')
                self.logger.info("本地英文模型加载成功！")
            except Exception as e2:
                self.logger.error(f"加载本地英文模型失败: {e2}")
                # 抛出最终错误
                raise
        
        # 确保encoder有device属性
        if not hasattr(self.encoder, 'device'):
            # 临时添加device属性
            self.encoder.device = 'cpu'
        
        # 优先从配置读取，否则自动检测
        self.d_model = self.config.get('lps.d_model', None)
        if self.d_model is None:
            self.d_model = self.encoder.get_sentence_embedding_dimension()  # 384
        
        # FAISS索引：使用HNSWFlat（高效近似最近邻搜索）
        # HNSW参数：M=32（每个节点的连接数），efConstruction=200（构建时探索深度）
        self.index = faiss.IndexHNSWFlat(self.d_model, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64  # 默认检索深度，可动态调整
        
        # 存储元数据和嵌入向量
        self.metadata = []  # 每个元素: {'id', 'text', 'potency', 'last_accessed'}
        self.embeddings = []  # 存储嵌入向量，与metadata对应
        self.id_counter = 0
        
        self.logger = get_logger('lps')
    
    def _normalize(self, vec):
        """
        归一化向量
        
        Args:
            vec: 输入向量
        
        Returns:
            归一化后的向量
        """
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def add(self, text, embedding=None, potency=None, tags=None):
        """
        添加新可能性，返回id
        
        Args:
            text: 文本描述
            embedding: 嵌入向量（可选）
            potency: 势能值（可选）
            tags: 结构化标签（可选）
        
        Returns:
            节点ID
        """
        if embedding is None:
            embedding = self.encoder.encode([text])[0]
        embedding = self._normalize(embedding).astype(np.float32).reshape(1, -1)
        
        if potency is None:
            potency = self.default_potency
        
        # 处理标签
        if tags is None:
            tags = {}
        
        # 自动附加当前结构坐标作为房间标签
        if hasattr(self, 'engine') and self.engine:
            coord = self.engine.structural_coordinator.get_current_coordinate()
            tags['subjective_room'] = coord.as_tarot_code()
            tags['subjective_major'] = coord.major
            # 客观分类由 ObjectiveClassifier 在调用前填入 tags['objective_room']
        
        node_id = self.id_counter
        self.id_counter += 1
        
        self.index.add(embedding)
        self.metadata.append({ 
            'id': node_id, 
            'text': text, 
            'potency': potency, 
            'last_accessed': time.time(),
            'embedding': embedding[0],  # 存储嵌入向量
            'tags': tags   # 使用处理后的标签
        })
        self.embeddings.append(embedding[0])
        
        self.logger.debug(f"Added possibility: {text[:50]}... id={node_id}, potency={potency}")
        return node_id
    
    def query(self, query_vec, k=None, min_potency=None, max_potency=None, add_noise=True):
        """
        返回k个最相似的可能性，每个包含 (embedding, id, potency)
        
        Args:
            query_vec: 查询向量
            k: 返回数量
            min_potency: 最小势能阈值
            max_potency: 最大势能阈值
            add_noise: 是否添加噪声
        
        Returns:
            相似可能性列表
        """
        # 动态调整检索深度（仅适用于 HNSW 索引）
        if hasattr(self, 'engine') and self.engine and hasattr(self.index, 'hnsw'):
            l_inst = self.engine.fse._l_inst
            # 执着时深入搜索（范围 16~128）
            ef = max(16, min(128, int(64 * (1 + l_inst))))
            self.index.hnsw.efSearch = ef
        
        if k is None:
            k = self.query_k
        if min_potency is None:
            min_potency = self.low_potency_threshold
        
        # 归一化查询向量
        if isinstance(query_vec, np.ndarray):
            q = self._normalize(query_vec.astype(np.float32)).reshape(1, -1)
        else:
            q = self._normalize(np.array(query_vec).astype(np.float32)).reshape(1, -1)
        
        distances, indices = self.index.search(q, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # 检查索引是否超出范围
            if idx >= len(self.metadata) or idx >= len(self.embeddings):
                continue
            meta = self.metadata[idx]
            if meta['potency'] < min_potency:
                continue
            if max_potency is not None and meta['potency'] > max_potency:
                continue
            # 获取存储的嵌入
            embedding = meta.get('embedding', self.embeddings[idx])
            result = { 
                'id': meta['id'], 
                'text': meta['text'], 
                'potency': meta['potency'], 
                'distance': float(dist),
                'embedding': embedding
            }
            results.append(result)
        # 添加噪声（势能扰动）
        if add_noise:
            for r in results:
                r['potency'] *= (1 + np.random.normal(0, 0.05))
        return results
    
    def sample_low_potency(self, query_vec, threshold=None):
        """
        返回势能低于threshold且与query最相似的一个可能性
        
        Args:
            query_vec: 查询向量
            threshold: 势能阈值
        
        Returns:
            低势能可能性
        """
        if threshold is None:
            threshold = self.low_potency_threshold
        # 获取所有低于阈值的节点
        low_nodes = [i for i, meta in enumerate(self.metadata) if meta['potency'] < threshold]
        
        # 如果没有低势能节点，从所有节点中随机选一个
        if not low_nodes and self.metadata:
            import random
            low_nodes = [random.randint(0, len(self.metadata) - 1)]
        
        if not low_nodes:
            return None
        
        # 计算与查询向量的相似度
        if query_vec is not None:
            q = self._normalize(np.array(query_vec).astype(np.float32))
            best_node = None
            best_sim = -1
            for i in low_nodes:
                embedding = self.embeddings[i]
                sim = np.dot(q, embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_node = i
            if best_node is not None:
                meta = self.metadata[best_node]
                return {
                    'id': meta['id'],
                    'text': meta['text'],
                    'potency': meta['potency'],
                    'embedding': self.embeddings[best_node]
                }
        # 简化实现：随机返回一个低势能节点
        import random
        idx = random.choice(low_nodes)
        meta = self.metadata[idx]
        return {
            'id': meta['id'],
            'text': meta['text'],
            'potency': meta['potency'],
            'embedding': self.embeddings[idx]
        }
    
    def prune(self, threshold=None):
        """
        删除势能低于阈值的节点
        
        Args:
            threshold: 势能阈值
        """
        if threshold is None:
            threshold = self.low_potency_threshold
        to_remove = [i for i, meta in enumerate(self.metadata) if meta['potency'] < threshold]
        # 注意：FAISS索引删除需要重建，此处先简单标记，实际需重建索引
        # 阶段2简化：记录日志，暂不实际删除，避免复杂度
        self.logger.info(f"Prune would remove {len(to_remove)} nodes with potency < {threshold}")
        # 可选：重建索引（性能代价高，先不做）
    
    def update_potency(self, node_id, delta):
        """
        更新节点势能
        
        Args:
            node_id: 节点ID
            delta: 势能变化量
        """
        for meta in self.metadata:
            if meta['id'] == node_id:
                meta['potency'] = max(0.0, meta['potency'] + delta)
                meta['last_accessed'] = time.time()
                self.logger.debug(f"Updated potency for id={node_id}: {meta['potency']:.3f}")
                return
    
    def add_if_new(self, text, embedding=None, similarity_threshold=0.9, potency=0.5, tags=None):
        """
        仅当与现有条目相似度低于阈值时才添加
        
        Args:
            text: 文本描述
            embedding: 嵌入向量（可选）
            similarity_threshold: 相似度阈值
            potency: 势能值（可选）
            tags: 结构化标签（可选）
        
        Returns:
            节点ID或None
        """
        if embedding is None:
            embedding = self.encoder.encode([text])[0]
        # 查询相似条目
        results = self.query(embedding, k=1)
        if results and results[0]['distance'] > similarity_threshold:
            self.logger.debug(f"Skipping duplicate: {text[:50]} (sim={results[0]['distance']:.3f})")
            return None
        return self.add(text, embedding, potency, tags)
    
    def query_by_tag(self, min_potency: float = 0.3, **kwargs) -> List[Dict]:
        """
        通过结构化标签精确检索知识条目。
        返回势能降序排列的结果列表。
        
        Args:
            min_potency: 最小势能阈值
            **kwargs: 标签键值对，如 type='semantic', entity='X', relation='Y' 等
        """
        results = []
        for meta in self.metadata:
            if meta['potency'] < min_potency:
                continue
            tags = meta.get('tags', {})
            # 检查所有提供的标签
            match = True
            for key, value in kwargs.items():
                if tags.get(key) != value:
                    match = False
                    break
            if match:
                results.append({
                    'id': meta['id'],
                    'text': meta['text'],
                    'potency': meta['potency'],
                    'tags': tags,
                    'embedding': meta.get('embedding')
                })
        results.sort(key=lambda x: x['potency'], reverse=True)
        return results
    
    def __getstate__(self):
        """
        序列化状态
        
        Returns:
            可序列化的状态字典
        """
        # 保存除了 FAISS 索引之外的所有属性
        state = self.__dict__.copy()
        # 将 FAISS 索引转换为字节流
        import faiss
        index_bytes = faiss.serialize_index(self.index)
        state['index'] = index_bytes
        # 保存嵌入向量为列表
        state['embeddings'] = [emb.tolist() for emb in self.embeddings]
        return state
    
    def __setstate__(self, state):
        """
        反序列化状态
        
        Args:
            state: 可序列化的状态字典
        """
        # 恢复除了 FAISS 索引之外的所有属性
        self.__dict__.update(state)
        # 从字节流恢复 FAISS 索引
        import faiss
        self.index = faiss.deserialize_index(state['index'])
        # 恢复嵌入向量为 numpy 数组
        self.embeddings = [np.array(emb) for emb in state['embeddings']]
    
    def _convert_dict_keys_to_str(self, d):
        """递归将字典的所有键转换为字符串类型"""
        if not isinstance(d, dict):
            return d
        return {str(key): self._convert_dict_keys_to_str(value) for key, value in d.items()}
    
    def save(self, path: str):
        """
        保存索引和元数据
        
        Args:
            path: 保存路径（不含扩展名）
        """
        import pandas as pd
        # 保存 FAISS 索引
        faiss.write_index(self.index, f"{path}.index")
        # 处理元数据，确保所有字典的键都是字符串类型
        processed_metadata = []
        for meta in self.metadata:
            processed_meta = meta.copy()
            # 递归处理所有字典类型的字段
            for key, value in processed_meta.items():
                if isinstance(value, dict):
                    processed_meta[key] = self._convert_dict_keys_to_str(value)
            processed_metadata.append(processed_meta)
        # 保存元数据为 Parquet（高效列式存储）
        df = pd.DataFrame(processed_metadata)
        df.to_parquet(f"{path}.parquet", index=False)
    
    def load(self, path: str):
        """
        加载索引和元数据
        
        Args:
            path: 加载路径（不含扩展名）
        """
        import pandas as pd
        self.index = faiss.read_index(f"{path}.index")
        self.metadata = pd.read_parquet(f"{path}.parquet").to_dict('records')
        # 重建 embeddings 列表
        self.embeddings = [np.array(m['embedding']) for m in self.metadata]
        # 重建 id_counter
        if self.metadata:
            self.id_counter = max(m['id'] for m in self.metadata) + 1
        else:
            self.id_counter = 0
