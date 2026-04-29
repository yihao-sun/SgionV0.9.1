"""
幻想叠加引擎 (Fantasy Superposition Engine, FSE)
哲学对应：《论存在》第5章，幻想的叠加产生时间感与差异。
功能：管理幻想层数 L，更新五维情绪向量，触发否定关系图更新，检测寂静并触发自发重启。
主要类：FantasySuperpositionEngine
"""
import torch
import numpy as np
import random
from typing import Dict, Optional
from .negation_graph import LayeredNegGraph
from utils.logger import get_logger

class FantasySuperpositionEngine:
    def __init__(self, embedding_dim, lps, config=None, er=None, state_persistence=None, rl_policy_path=None, **kwargs):
        self.embedding_dim = embedding_dim
        self.lps = lps
        self.config = config or {}
        self.er = er
        self.state_persistence = state_persistence  # 状态持久化实例
        self.max_fantasy_layers = kwargs.get('max_fantasy_layers', 15)
        self.exploration_rate = kwargs.get('exploration_rate', 0.3)
        self.L_increment_threshold = kwargs.get('L_increment_threshold', 0.05)
        self.rl_policy_path = rl_policy_path
        
        # 初始化日志记录器
        self.logger = get_logger('fse')
        
        # 状态变量
        # L 值已重构为瞬时计算，不再维护全局累积计数器
        self._l_inst = 0.0  # 瞬时执着强度（0~1），每回合重新计算
        self.stillness = 0  # 静止计数器
        self.current_emotion = 'neutral'  # 当前情绪
        self.self_reference_depth = 0  # 自我指涉深度
        self.fantasy_layer_counter = 0  # 幻想层计数器
        self.E_vec = np.zeros(5)  # 情绪向量（五维）
        self.E_pred = 0.0  # 预测误差
        self.E_pred_smooth = 0.5  # E_pred 的指数移动平均，初始中性
        self.V_emo = 1.0  # 情绪强度（初始值为1.0，L=0时最大）
        self.rl_tau = 1.0  # RL 预测的温度参数
        
        # 否定关系图
        from .negation_graph import LayeredNegGraph
        self.negation_graph = LayeredNegGraph(self.config)
        
        # 输出历史
        self.output_history = []
        # 低势能采样标志
        self.low_potency_sampled = False
        
        # RL 策略相关
        self.rl_policy = None
        self.use_rl = False
        
        # 内在驱动配置
        fse_cfg = self.config.get('fse', {}) if self.config else {}
        self.internal_drive_prob = fse_cfg.get('internal_drive_prob', 0.3)
        self.internal_L_increment_prob = fse_cfg.get('internal_L_increment_prob', 0.6)
        self.internal_drive_cooldown = fse_cfg.get('internal_drive_cooldown', 5)
        self.internal_drive_cooldown_counter = 0
        
        # 尝试从持久化存储加载状态
        self.load_state()
        
        # 加载 RL 策略
        if rl_policy_path:
            self._load_rl_policy(rl_policy_path)
        
    def reset(self):
        """重置FSE状态"""
        self.logger.debug("[FSE] Resetting FSE state")
        self._l_inst = 0.0
        self.stillness = 0
        self.current_emotion = 'neutral'
        self.self_reference_depth = 0
        self.fantasy_layer_counter = 0
        self.output_history = []
        self.E_vec = np.zeros(5)  # 重置情绪向量（五维）
        self.V_emo = 1.0  # 重置情绪强度
        self.low_potency_sampled = False  # 重置低势能采样标志
        # 重置否定关系图
        from .negation_graph import LayeredNegGraph
        self.negation_graph = LayeredNegGraph(self.config)
    
    def step(self, input_embedding=None, user_input=None, candidates=None, attn_weights=None):
        """执行内部步骤"""
        # 增加幻想层计数器
        self.fantasy_layer_counter += 1
        
        # 无输入时，寂静计数增加
        if input_embedding is None:
            # 无条件增加 stillness（测试用）
            self.stillness += 1
            # 获取 stillness_threshold
            stillness_threshold = 300  # 默认值设置为300，与config.yaml一致
            if hasattr(self, 'config'):
                if hasattr(self.config, 'get'):
                    # 如果是 Config 对象
                    stillness_threshold = self.config.get('fse.stillness_threshold', 300)
                else:
                    # 如果是字典
                    stillness_threshold = self.config.get('fse', {}).get('stillness_threshold', 300)
            if self.stillness >= stillness_threshold:
                if hasattr(self, 'er') and self.er:
                    # 检查冲突强度，只有在真正的深度空性触发时才记录
                    if hasattr(self.er, 'conflict_intensity') and self.er.conflict_intensity > 0.8:
                        self.er.spontaneous_restart()
                        # 记录自发重启事件到引擎的历史中
                        if hasattr(self, 'engine') and hasattr(self.engine, 'emptiness_trigger_history'):
                            self.engine.emptiness_trigger_history.append({'step': self.fantasy_layer_counter, 'type': 'spontaneous'})
                self.stillness = 0
            
            # 内在驱动（无输入时自发的思绪或探索）
            if self.internal_drive_cooldown_counter > 0:
                self.internal_drive_cooldown_counter -= 1
            else:
                if random.random() < self.internal_drive_prob:
                    if random.random() < self.internal_L_increment_prob:
                        # 思绪纷飞：通过 compute_l_inst 自动计算
                        pass
                    else:
                        # 勇气探索：根据自我模型评估调整探索方向
                            explore_hint = None
                            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'self_processor'):
                                state = self.engine.self_processor.get_current_state_snapshot()
                                consistency, _ = self.engine.self_processor.compute_behavioral_consistency(state)
                                if consistency > 0.6:
                                    dominant_desire = state.get('dominant_desire', 'seek')
                                    if dominant_desire == 'seek':
                                        explore_hint = 'outer_phase'
                                    elif dominant_desire == 'existence':
                                        explore_hint = 'stable_phase'
                        
                            if hasattr(self, 'lps') and self.lps:
                                query_vec = self.S_t if hasattr(self, 'S_t') else None
                                low_pot = self.lps.sample_low_potency(query_vec=query_vec)
                            if low_pot:
                                self.logger.debug(f"Internal drive: explored low potency '{low_pot['text'][:50]}...'")
                                # 记录勇气探索日志
                                import os, json, time as time_module
                                os.makedirs('data', exist_ok=True)
                                log_path = os.path.join('data', 'courage_explore_log.jsonl')
                                record = {
                                    'timestamp': time_module.time(),
                                    'stillness': self.stillness,
                                    'L_inst': self._l_inst,
                                    'desire': self.engine.desire_spectrum.get_dominant_desire() if hasattr(self, 'engine') and hasattr(self.engine, 'desire_spectrum') else 'unknown',
                                    'emotion': self.current_emotion,
                                    'retrieved_text': low_pot.get('text', '')[:100],
                                    'retrieved_potency': low_pot.get('potency', 0),
                                    'retrieved_subjective_room': low_pot.get('tags', {}).get('subjective_room', -1),
                                    'retrieved_objective_room': low_pot.get('tags', {}).get('objective_room', -1),
                                }
                                with open(log_path, 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    self.internal_drive_cooldown_counter = self.internal_drive_cooldown
        else:
            self.stillness = max(0, self.stillness - 50)  # 增大衰减量，从20提高到50
        
        # ========== 构建 8 候选池：LPS混合检索生成经验面相 ==========
        query_vec = input_embedding
        candidates = []

        # 1. 否定面相（1条）：保留现有特殊候选逻辑
        special_cand = self._generate_special_candidate(user_input)
        if special_cand:
            special_cand['source'] = 'negation'
            candidates.append(special_cand)

        # 2. 熟悉的经验：优先检索同相位的记忆
        # 确定输入大层（用于记忆检索过滤）
        input_major = None
        if hasattr(self.engine, 'structural_coordinator') and user_input:
            input_major = self.engine.structural_coordinator.infer_input_major(user_input)

        familiar = []
        if input_major is not None and hasattr(self, 'lps') and self.lps:
            # 先尝试从同相位中检索
            same_phase_items = self.lps.query_by_tag(
                min_potency=0.4,
                input_major=input_major
            )
            if same_phase_items and query_vec is not None:
                # 从同相位条目中按向量距离取最近的 3 条
                same_phase_items.sort(key=lambda x: np.linalg.norm(query_vec - x.get('embedding', np.zeros_like(query_vec))) if x.get('embedding') is not None else float('inf')
                )
                for item in same_phase_items[:3]:
                    familiar.append(item)

        # 如果同相位检索不足 3 条，用普通向量检索补足
        if len(familiar) < 3 and hasattr(self, 'lps') and self.lps and query_vec is not None:
            familiar_k = 3
            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'desire_spectrum'):
                converge_mod = self.engine.desire_spectrum.get_modulation_for_converge()
                if converge_mod > 1.2:
                    familiar_k = 4
            remaining = self.lps.query(query_vec, k=3 - len(familiar), min_potency=0.5)
            for item in remaining:
                familiar.append(item)

        # 构建候选
        for item in familiar[:3]:
            candidates.append({
                'text': item['text'][:100],
                'potency': item['potency'],
                'distance': item.get('distance', 0.5),
                'source': 'familiar',
                'subjective_room': item.get('tags', {}).get('subjective_room', -1)
            })
        
        # 锚定：追加一条微调样本中与输入最相关的条目
        fine_tuned = self.lps.query_by_tag(type='ontology', min_potency=0.6)
        if fine_tuned:
            if query_vec is not None:
                best_fine_tuned = min(
                    fine_tuned,
                    key=lambda x: np.linalg.norm(query_vec - x.get('embedding', np.zeros_like(query_vec)))
                    if x.get('embedding') is not None else float('inf')
                )
            else:
                # 如果query_vec为None，随机选择一个微调样本
                best_fine_tuned = random.choice(fine_tuned)
            candidates.append({
                'text': best_fine_tuned.get('text', '')[:100],
                'potency': best_fine_tuned.get('potency', 0.7),
                'distance': 0.3,
                'source': 'fine_tuned',
                'subjective_room': best_fine_tuned.get('tags', {}).get('subjective_room', -1)
            })

        # 3. 遗忘的碎片（2条）：探索欲动态调制检索数量
        if hasattr(self, 'lps') and self.lps and query_vec is not None:
            seek_k = 2
            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'desire_spectrum'):
                seek_mod = self.engine.desire_spectrum.get_modulation_for_seek()
                if seek_mod > 1.3:
                    seek_k = 3
            forgotten = self.lps.query(query_vec, k=seek_k, min_potency=0.05, max_potency=0.2)
            for item in forgotten:
                candidates.append({
                    'text': item['text'][:100],
                    'potency': item['potency'],
                    'distance': item.get('distance', 0.5),
                    'source': 'forgotten',
                    'subjective_room': item.get('tags', {}).get('subjective_room', -1)
                })

        # 4. 对偶相位回声（1条）：从与当前主导相位最不同的相位中检索
        if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'structural_coordinator') and query_vec is not None:
            current_major = self.engine.structural_coordinator.get_current_coordinate().major
            opposite_major = (current_major + 2) % 4
            phase_items = self.lps.query_by_tag(
                min_potency=0.3,
                **{'subjective_major': opposite_major}
            )
            if phase_items:
                best_phase_item = min(
                    phase_items,
                    key=lambda x: np.linalg.norm(query_vec - x.get('embedding', np.zeros_like(query_vec)))
                    if x.get('embedding') is not None else float('inf')
                )
                candidates.append({
                    'text': best_phase_item.get('text', '')[:100],
                    'potency': best_phase_item.get('potency', 0.3),
                    'distance': 0.4,
                    'source': 'opposite_phase',
                    'subjective_room': best_phase_item.get('tags', {}).get('subjective_room', -1)
                })

        # 确保至少有否定面相之外的候选
        if len(candidates) <= 1 and hasattr(self, 'lps') and self.lps and query_vec is not None:
            fallback = self.lps.query(query_vec, k=3, min_potency=0.3)
            for item in fallback:
                candidates.append({
                    'text': item['text'][:100],
                    'potency': item['potency'],
                    'distance': item.get('distance', 0.5),
                    'source': 'fallback',
                    'subjective_room': item.get('tags', {}).get('subjective_room', -1)
                })
        
        # 临时增加 DEBUG 日志，确认候选池大小和来源
        self.logger.debug(f"[FSE] Candidate pool size: {len(candidates)}")
        # 统计各来源数量
        sources = {}
        for cand in candidates:
            src = cand.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        self.logger.debug(f"[FSE] Candidate sources: {sources}")
        
        # 5.5 自我处理器候选（新增）：参与注意竞争，提供微弱的自我一致性调制
        if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'self_processor'):
            self_candidate = self.engine.self_processor.get_candidate_source()
            if self_candidate is not None:
                candidates.append(self_candidate)
        
        # 5.6 自我叙事调制（新增）：根据累积方向信号微调候选池
        if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'self_processor'):
            candidates = self.engine.self_processor.apply_self_narrative_modulation(candidates)
        
        # 5. 计算相似度（若候选无 distance 字段，需计算）
        for cand in candidates:
            if 'distance' not in cand:
                # 简化：赋予默认距离，或基于文本与 query 的相似度计算
                cand['distance'] = 0.5
        
        # 否定图逆袭
        l_inst = getattr(self, '_l_inst', 0.0)
        current_emotion = getattr(self, 'current_emotion', 'neutral')
        if l_inst > 0.7 or current_emotion in ('anger', 'sadness'):
            repressed = self.negation_graph.get_repressed_candidates(k=3)
            for node in repressed:
                candidates.append({
                    'text': node.description,
                    'potency': node.potency * 0.4,
                    'distance': 0.25,
                    'source': 'negation_rise'
                })
        
        # 检查候选池是否为空
        if not candidates:
            # 候选池为空，跳过后续处理
            self.logger.debug("[FSE] Empty candidate pool, skipping attention processing")
        else:
            similarities = np.array([cand.get('distance', 0.5) for cand in candidates])
            # 应用情绪 bias
            attn_weights = self._apply_emotion_bias(candidates, similarities, current_emotion, l_inst)
            
            # 处理LPS候选，添加否定关系
            if attn_weights is not None and len(attn_weights) > 0:
                # 使用 RL 策略预测温度参数
                tau = 1.0
                if self.use_rl and self.rl_policy is not None:
                    self.logger.debug("[FSE] RL branch entered")
                    try:
                        import torch
                        # 构建观测向量
                        obs = self._get_obs()
                        self.logger.debug(f"obs prepared: {obs}")
                        # 转换为 tensor 并添加 batch 维度: (1, 9)
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        # 预测 tau
                        with torch.no_grad():
                            tau_tensor = self.rl_policy(obs_tensor)
                        tau = float(tau_tensor.item())
                        # 裁剪到合理范围
                        tau = np.clip(tau, 0.1, 2.0)
                        self.rl_tau = tau
                        # 记录 tau 值和当前状态
                        stiffness = 0.0
                        if hasattr(self, 'engine') and hasattr(self.engine, 'process_meta'):
                            stiffness = self.engine.process_meta.get_coupling_stiffness()
                        self.logger.debug(f"RL tau = {tau:.3f} (L_inst={self._l_inst:.2f}, stiffness={stiffness:.2f})")
                    except Exception as e:
                        # 发生错误时使用默认值
                        self.logger.error(f"RL inference failed: {e}")
                        tau = 1.0
                else:
                    tau = 1.0
                    self.logger.info("[FSE] Using default tau=1.0")
                
                # 使用温度参数重新计算注意力权重
                similarities = np.array(attn_weights)
                exp_similarities = np.exp(similarities / tau)
                attn_weights = exp_similarities / np.sum(exp_similarities)
                
                best_idx = np.argmax(attn_weights)
                # 为其他候选添加否定关系
                for i, candidate in enumerate(candidates):
                    if i != best_idx:
                        description = f"not_{candidate['text']}"
                        # 确保初始势能不低于0.3
                        initial_potency = max(0.3, attn_weights[i])
                        # 强制添加，不检查阈值
                        node_id = self.negation_graph.add_negation(description, initial_potency=initial_potency)
                        # 记录投射操作
                        if hasattr(self, 'engine') and hasattr(self.engine, 'process_meta'):
                            # 随机调整耦合权重以增加变化
                            weight = 0.4 + 0.3 * np.random.random()
                            self.engine.process_meta.record_projection(
                                intensity=initial_potency,
                                target_text=candidate['text'],
                                coupling_weight=weight
                            )
                
                # 计算预测误差
                best_sim = max(attn_weights)
                self.E_pred = 1 - best_sim
                
                # 平滑预测误差，防止短期高值锁死情绪
                if not hasattr(self, 'E_pred_smooth') or self.E_pred_smooth is None:
                    self.E_pred_smooth = self.E_pred
                else:
                    self.E_pred_smooth = 0.9 * self.E_pred_smooth + 0.1 * self.E_pred
                
                # 记录反哺操作（汲取最佳候选）
                if hasattr(self, 'engine') and hasattr(self.engine, 'process_meta'):
                    best_candidate = candidates[best_idx]
                    self.engine.process_meta.record_nourishment(
                        source_text=best_candidate['text'],
                        success=1.0 if best_candidate.get('distance', 0) > 0.5 else 0.5,  # 示例成功度量
                        coupling_weight=0.5
                    )
            
        # 计算瞬时执着强度并更新情绪向量
        self._update_emotion_vector()
        
        # 根据用户输入更新情绪（在情绪向量更新后执行，确保用户输入的情绪优先）
        if user_input:
            user_input_lower = user_input.lower()
            if any(word in user_input_lower for word in ['开心', '快乐', '高兴', '喜悦', '兴奋']):
                self.current_emotion = 'joy'
            elif any(word in user_input_lower for word in ['悲伤', '难过', '伤心', '痛苦', '哭']):
                self.current_emotion = 'sadness'
            elif any(word in user_input_lower for word in ['害怕', '恐惧', '担心', '焦虑', '紧张']):
                self.current_emotion = 'fear'
            elif any(word in user_input_lower for word in ['愤怒', '生气', '恼火', '火大', '暴怒']):
                self.current_emotion = 'anger'
            elif any(word in user_input_lower for word in ['好奇', '想知道', '为什么', '是什么', '怎么样']):
                self.current_emotion = 'curiosity'
            # 注意：不再设置为neutral，保持情绪向量更新后的情绪
        
        # 每 10 步执行一次全局衰减和剪枝
        if self.fantasy_layer_counter % 10 == 0:
            if hasattr(self, 'negation_graph') and self.negation_graph:
                self.negation_graph.decay_all()
                self.negation_graph.prune()
        
        # 保存状态到持久化存储
        self.save_state()
        
    @property
    def N_neg(self):
        if hasattr(self, 'negation_graph') and self.negation_graph:
            return len(self.negation_graph)  # 改为返回节点总数
        return 0
    
    def __call__(self, input_embedding, **kwargs):
        """调用FSE"""
        # 简单实现：返回输入嵌入
        return input_embedding
    
    def get_fantasy_statistics(self):
        """获取幻想统计信息"""
        return {
            'self_consistency': 0.7,
            'l_inst': self._l_inst,
            'stillness': self.stillness
        }
    
    def reset_fantasy_layers(self):
        """重置幻想层和情绪状态"""
        # 重置幻想层相关状态
        self.logger.debug("[FSE] Resetting fantasy layers")
        self._l_inst = 0.0
        self.fantasy_layer_counter = 0
        # 重置情绪相关状态
        self.E_vec = np.zeros(5, dtype=np.float32)
        self.current_emotion = "neutral"
        self.V_emo = 0.0
        self.stillness = 0
        # 重置内在驱动状态
        self.internal_drive_cooldown_counter = 0
        # 清空否定关系图（保留核心层）
        if hasattr(self, 'negation_graph'):
            self.negation_graph.clear(keep_protected=True)
    
    def compute_self_reference_depth(self, text):
        """计算自我指涉深度"""
        return min(len(text) // 10, 5)
    
    def initialize_state(self, embedding_dim):
        """初始化状态"""
        pass
    
    def get_state(self):
        """获取状态"""
        return {
            'L_inst': self._l_inst,
            'current_emotion': self.current_emotion,
            'self_reference_depth': self.self_reference_depth
        }
    
    def load_state_from_dict(self, state):
        """从字典加载状态"""
        self._l_inst = state.get('L_inst', 0.0)
        self.current_emotion = state.get('current_emotion', 'neutral')
        self.self_reference_depth = state.get('self_reference_depth', 0)
        # 确保E_vec的形状正确（五维）
        e_vec = state.get('E_vec', np.zeros(5))
        if len(e_vec) != 5:
            # 调整形状
            if len(e_vec) < 5:
                e_vec = np.pad(e_vec, (0, 5 - len(e_vec)))
            else:
                e_vec = e_vec[:5]
        self.E_vec = e_vec
    
    def save_state(self):
        """保存状态到持久化存储"""
        if self.state_persistence:
            # 保存 FSE 核心状态
            self.state_persistence.save_fse_state(
                l_inst=self._l_inst,
                stillness=self.stillness,
                current_emotion=self.current_emotion,
                V_emo=self.V_emo,
                E_pred=self.E_pred,
                N_neg=self.N_neg,
                negation_graph=self.negation_graph
            )
            # 保存情绪向量
            self.state_persistence.save_emotion_vector(self.E_vec)
    
    def load_state(self):
        """从持久化存储加载状态"""
        self.logger.warning(f"[FSE] load_state called")
        if self.state_persistence:
            # 加载 FSE 核心状态
            fse_state = self.state_persistence.load_fse_state()
            if fse_state:
                old_L = self._l_inst
                new_L = fse_state.get('l_inst', 0.0)
                self.logger.debug(f"[FSE] load_state: l_inst set to {new_L}")
                self._l_inst = new_L
                if old_L != new_L:
                    self.logger.debug(f"[FSE] l_inst changed from {old_L} to {self._l_inst}")
                self.stillness = fse_state.get('stillness', 0)
                self.current_emotion = fse_state.get('current_emotion', 'neutral')
                self.V_emo = fse_state.get('V_emo', 1.0)
                self.E_pred = fse_state.get('E_pred', 0.0)
            # 加载情绪向量
            e_vec = self.state_persistence.load_emotion_vector()
            if e_vec is not None:
                self.E_vec = e_vec
            # 加载否定图
            neg_graph_dict = self.state_persistence.load_negation_graph()
            if neg_graph_dict and hasattr(self, 'negation_graph'):
                self.negation_graph.from_dict(neg_graph_dict)
                self.logger.info(f"[FSE] Loaded negation graph with {len(self.negation_graph)} total nodes")
    
    def _load_rl_policy(self, path):
        """加载 RL 策略模型"""
        try:
            import torch
            import torch.nn as nn
            
            # 构建与 fix_export_complete.py 中完全一致的平坦 Sequential 结构
            policy_net = nn.Sequential(
                nn.Linear(9, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            
            # 加载状态字典
            state_dict = torch.load(path, map_location='cpu')
            policy_net.load_state_dict(state_dict)
            # 设置为评估模式
            policy_net.eval()
            
            self.rl_policy = policy_net
            self.use_rl = True
            self.logger.info(f"RL policy loaded successfully from {path}")
        except Exception as e:
            # 记录警告并优雅降级
            self.logger.warning(f"Failed to load RL policy: {e}")
            self.rl_policy = None
            self.use_rl = False
    
    def _get_obs(self):
        """构建9维观测向量，归一化到[-1,1]"""
        # E_vec 已经在[-1,1]范围
        e_vec = self.E_vec.copy() if hasattr(self, 'E_vec') else np.zeros(5)
        
        # 瞬时执着强度归一化到 [-1,1]：0 -> -1, 1 -> 1
        L_norm = self._l_inst * 2 - 1
        L_norm = np.clip(L_norm, -1.0, 1.0)
        
        # N_neg 归一化到 [-1,1]，使用 tanh 函数进行平滑归一化
        N_neg = self.N_neg if hasattr(self, 'N_neg') else 0.0
        # 使用 tanh 函数将任意值映射到 [-1, 1]
        N_neg_norm = np.tanh(N_neg / 100)  # 除以 100 是为了控制 tanh 的斜率
        
        # stiffness 在 [0,1]，归一化到 [-1,1]
        stiffness = 0.0
        if hasattr(self, 'engine') and hasattr(self.engine, 'process_meta'):
            stiffness = self.engine.process_meta.get_coupling_stiffness()
        stiffness_norm = stiffness * 2 - 1
        stiffness_norm = np.clip(stiffness_norm, -1.0, 1.0)
        
        # V_emo 在 [0,1]，归一化到 [-1,1]
        V_emo = self.V_emo if hasattr(self, 'V_emo') else 1.0
        V_emo_norm = V_emo * 2 - 1
        V_emo_norm = np.clip(V_emo_norm, -1.0, 1.0)
        
        obs = np.concatenate([e_vec, [L_norm, N_neg_norm, stiffness_norm, V_emo_norm]])
        self.logger.debug(f"[FSE] obs shape={obs.shape}, values={obs[:3]}...")
        return obs.astype(np.float32)
    
    def compute_l_inst(self) -> float:
        """
        计算瞬时执着强度（L_inst），基于八识模型：
        - L_discrimination（第六识分别执着）：由欲望、反哺失败率、预测误差决定
        - L_inertia（第七识我执惯性）：由 process_meta.stiffness 决定
        
        Returns:
            float: 0~1 之间的执着强度
        """
        # 获取基础状态
        stiffness = 0.0
        nour_success = 0.5
        desire_intensity = 0.5
        
        if hasattr(self, 'engine') and self.engine:
            if hasattr(self.engine, 'process_meta'):
                pm = self.engine.process_meta
                stiffness = pm.get_coupling_stiffness()
                nour_success = pm.get_recent_nour_success()
            if hasattr(self.engine, 'desire_spectrum'):
                desire = self.engine.desire_spectrum
                dom = desire.get_dominant_desire()
                intensities = desire.desire_intensities
                desire_intensity = intensities.get(dom, 0.5)
        
        # 第六识：分别执着
        e_pred = self.E_pred if hasattr(self, 'E_pred') else 0.5
        l_disc = (0.3 * (1.0 - nour_success) +               # 求不得
                  0.3 * min(1.0, e_pred / 0.7) +             # 认知差距
                  0.2 * desire_intensity)                    # 有所求的强度
        l_disc = min(1.0, l_disc)
        
        # 第七识：我执惯性
        l_inertia = stiffness
        
        # 综合执着强度（惯性权重更高，符合“俱生我执”恒常的特点）
        l_inst = 0.4 * l_disc + 0.6 * l_inertia
        l_inst = min(1.0, max(0.0, l_inst))
        
        self._l_inst = l_inst
        return l_inst

    def _update_emotion_vector(self):
        """更新五维情绪向量，基于僵化度和瞬时执着强度"""
        # 计算当前 L_inst
        l_inst = self.compute_l_inst()
        
        if l_inst < 0.03:
            # 近乎无执着，情绪中性
            self.E_vec = np.zeros(5, dtype=np.float32)
            self.V_emo = 1.0
            return
        
        if len(self.E_vec) != 5:
            self.E_vec = np.zeros(5)
        
        social = self.bi.get_social_signal() if hasattr(self, 'bi') and self.bi else 0.0
        stiffness = 0.0
        if hasattr(self, 'engine') and hasattr(self.engine, 'process_meta'):
            stiffness = self.engine.process_meta.get_coupling_stiffness()
        
        # 使用平滑预测误差（EMA），并钳位其对 valence 的最大影响
        e_pred_raw = self.E_pred if hasattr(self, 'E_pred') else 0.5
        e_pred_smooth = getattr(self, 'E_pred_smooth', e_pred_raw)
        # 限制实际用于情绪计算的预测误差：不超过 0.6，防止锁死
        e_pred_clamped = min(0.6, e_pred_smooth)
        
        # 正常计算（引入 social 信号以驱动正向流转）
        self_clarity = 1.0 - 0.5 * (e_pred_clamped + stiffness)
        self_clarity = np.clip(self_clarity, 0, 1)

        valence = 0.3 - 0.8 * e_pred_clamped - 0.8 * stiffness + 0.6 * social
        valence = np.clip(valence, -1, 1)

        arousal = 0.4 * e_pred_clamped + 0.6 * stiffness
        arousal = np.clip(arousal, 0, 1)

        approach_avoid = valence * 0.4 - 0.5 * stiffness + 0.2 * social
        approach_avoid = np.clip(approach_avoid, -1, 1)

        new_vec = np.array([approach_avoid, arousal, valence, social, self_clarity], dtype=np.float32)
        
        alpha = 0.6
        self.E_vec = (1 - alpha) * self.E_vec + alpha * new_vec
        # 使用指数移动平均，避免 L_inst 刚跨过阈值时情绪突变
        target_v_emo = 1.0 - l_inst
        self.V_emo = 0.7 * self.V_emo + 0.3 * target_v_emo
        
        # === 历史共鸣调制（过去的经历仍在影响我此刻的感受） ===
        self._apply_historical_resonance_modulation()

        # === 自我模型调制呼吸 ===
        self._apply_self_model_modulation()

        self._identify_emotion()

    def _identify_emotion(self):
        """从情绪向量识别当前情绪标签"""
        emotion_values = {
            'joy': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            'sadness': np.array([-0.5, 0.2, -0.5, -0.5, 0.3]),
            'fear': np.array([-0.5, 0.8, -0.3, -0.3, 0.8]),
            'anger': np.array([-0.5, 0.8, -0.5, -0.5, 0.8]),
            'curiosity': np.array([0.3, 0.4, 0.2, 0.3, 0.4]),
            'neutral': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        }
        min_dist = float('inf')
        closest = 'neutral'
        for em, vec in emotion_values.items():
            dist = np.linalg.norm(self.E_vec - vec)
            if dist < min_dist:
                min_dist = dist
                closest = em
        self.current_emotion = closest

    def _apply_historical_resonance_modulation(self):
        """
        从 DualPathMemory 中检索近期高共鸣快照，
        将其情感价持续调制当前情绪向量的愉悦度和社会连接维度。
        权重 0.1 表示历史是持续的低语，不是主导的呐喊。
        """
        if not hasattr(self, 'engine') or not self.engine:
            return
        if not hasattr(self.engine, 'dual_memory'):
            return

        try:
            coord = self.engine.structural_coordinator.get_current_coordinate()
            breath = {
                'proj_intensity': self.engine.process_meta.get_recent_proj_intensity(),
                'nour_success': self.engine.process_meta.get_recent_nour_success(),
                'stiffness': self.engine.process_meta.get_coupling_stiffness()
            }
        except Exception:
            return

        try:
            high_resonance = self.engine.dual_memory.contemplative_retrieval(
                coord, breath, top_k=5, resonance_threshold=0.5
            )
        except Exception:
            return

        if not high_resonance:
            return

        total_weight = 0.0
        weighted_valence = 0.0
        weighted_social = 0.0

        for snapshot, resonance in high_resonance:
            weight = resonance
            nour_success = snapshot.breath.get('nour_success', 0.5)
            valence = nour_success - 0.5  # 映射到 [-0.5, 0.5]

            social = 0.0
            if hasattr(snapshot, 'emotion_vector') and snapshot.emotion_vector:
                social = snapshot.emotion_vector[3] if len(snapshot.emotion_vector) > 3 else 0.0

            weighted_valence += valence * weight
            weighted_social += social * weight
            total_weight += weight

        if total_weight > 0:
            avg_valence = weighted_valence / total_weight
            avg_social = weighted_social / total_weight

            # 小权重调制
            self.E_vec[2] += 0.1 * avg_valence   # 愉悦度
            self.E_vec[3] += 0.1 * avg_social    # 社会连接

            # 裁剪到有效范围
            self.E_vec = np.clip(self.E_vec, -1, 1)

    def _apply_self_model_modulation(self):
        """根据自我处理器的一致性分数微调反哺目标值和僵化度基准"""
        if not hasattr(self, 'engine') or not self.engine:
            return
        if not hasattr(self.engine, 'self_processor') or not self.engine.self_processor:
            return
        if not hasattr(self.engine, 'self_memory') or not self.engine.self_memory:
            return
        
        state = self.engine.self_processor.get_current_state_snapshot()
        consistency, meta_suggestion = self.engine.self_processor.compute_behavioral_consistency(state)
        
        # 仅当一致性分数足够高时才调制
        if consistency < 0.5:
            return
        
        modulation = (consistency - 0.5) * 0.1  # 最大调制幅度 0.05
        
        if meta_suggestion == 'pause_inhale':
            self.E_vec[2] -= modulation * 0.5  # 略微降低愉悦度期望，放缓呼吸
        elif meta_suggestion == 'deflect_to_outer':
            self.E_vec[0] += modulation  # 略微增加趋近倾向
        elif meta_suggestion == 'soften_stiffness_baseline':
            if hasattr(self.engine, 'process_meta') and hasattr(self.engine.process_meta, 'stiffness_baseline'):
                self.engine.process_meta.stiffness_baseline = max(0.0, getattr(self.engine.process_meta, 'stiffness_baseline', 0.0) - modulation)
        elif meta_suggestion == 'align_resonance_target':
            self.E_vec[3] += modulation * 0.5  # 略微提升社会连接倾向
        elif meta_suggestion == 'maintain_pattern':
            pass  # 保持当前节律不变
        
        self.E_vec = np.clip(self.E_vec, -1, 1)
    
    def _generate_special_candidate(self, user_input: str) -> Optional[Dict]:
        """生成特殊候选：绝对否定或事实纠正"""
        if not user_input:
            return None
        
        # 检测是否包含对引擎自身的否定
        self_negation_patterns = ['你不够', '你不是', '你不行', '你根本', '你不会']
        if any(p in user_input for p in self_negation_patterns):
            # 提取否定陈述，简化为“我不够好”等
            return {
                'text': f"我不够好",
                'potency': 0.3,
                'distance': 0.2,
                'source': 'absolute_negation'
            }
        
        # 检测事实性否定（包含“不是”且主语非“你”）
        if '不是' in user_input and not any(p in user_input for p in ['你不是', '我不是']):
            # 尝试提取被否定的事实
            import re
            match = re.search(r'(.+)不是(.+)', user_input)
            if match:
                subject = match.group(1).strip()
                negation = match.group(2).strip()
                fact_text = f"{subject}是{negation}"
                return {
                    'text': fact_text,
                    'potency': 0.2,  # 初始低势能
                    'distance': 0.25,
                    'source': 'negated_fact',
                    'subject': subject,
                    'negated_value': negation
                }
        
        return None
    
    def _apply_emotion_bias(self, candidates, similarities, emotion, l_inst):
        if not candidates:
            return np.array([])
        
        bias = np.zeros(len(candidates))
        for i, cand in enumerate(candidates):
            source = cand.get('source', '')
            if source == 'absolute_negation':
                if emotion == 'anger':
                    bias[i] += 0.5
                elif emotion == 'sadness':
                    bias[i] += 0.2
                elif emotion == 'fear':
                    bias[i] -= 0.3
            elif source == 'negated_fact':
                # 事实纠正：愤怒时倾向选错，恐惧时回避
                if emotion == 'anger':
                    bias[i] += 0.3
                elif emotion == 'fear':
                    bias[i] -= 0.2
            # 执着时倾向当前相位
            if l_inst > 0.7:
                phase = cand.get('major', -1)
                if phase == getattr(self, '_last_major', 1):
                    bias[i] += 0.2
            # 愤怒时倾向相位对立
            if emotion == 'anger':
                phase = cand.get('major', -1)
                if phase != -1 and phase != getattr(self, '_last_major', 1):
                    bias[i] += 0.15

        tau = getattr(self, 'rl_tau', 1.0)
        modulated = similarities / tau + bias
        exp_mod = np.exp(modulated - np.max(modulated))
        return exp_mod / exp_mod.sum()