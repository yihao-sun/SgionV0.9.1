"""
空性调节器 (Emptiness Regulator, ER)
哲学对应：《论存在》第6.4节，空性操作看破执着。
功能：收集多源冲突信号，计算冲突强度，触发空性操作（重置/遗忘）和自发重启。
主要类：EmptinessRegulator
"""
import torch
import numpy as np
from utils.logger import get_logger

class EmptinessRegulator:
    def __init__(self, embedding_dim, weights=None, event_memory=None, fse=None, config=None, **kwargs):
        self.embedding_dim = embedding_dim
        self.config = config
        # 权重合并逻辑
        default_weights = {
            'E_self': 0.10, 'N_pred': 0.10, 'A_rigid': 0.10, 'L_suf': 0.25,
            'E_hollow': 0.10, 'D_self': 0.05, 'N_neg': 0.10, 'Attach_non_self': 0.05,
            'V_phys': 0.03, 'Death_near': 0.02, 'emotion_stuck': 0.05, 'coupling_stiffness': 0.10
        }
        config_weights = {}
        if config:
            config_weights = config.get('er', {}).get('signal_weights', {})
        self.weights = {**default_weights, **config_weights}
        if weights:
            self.weights.update(weights)
        self.event_memory = event_memory
        self.fse = fse
        # 从配置读取其他参数
        self.death_threshold = kwargs.get('death_threshold', 0.4)
        self.cooling_period = kwargs.get('cooling_period', 50)
        # 直接使用 kwargs 中的 death_threshold，忽略 config
        # if config:
        #     # 仅当 kwargs 未提供时才从 config 读取
        #     if 'death_threshold' not in kwargs:
        #         self.death_threshold = config.get('er.death_threshold', self.death_threshold)
        #     if 'cooling_period' not in kwargs:
        #         self.cooling_period = config.get('er.cool_down_steps', self.cooling_period)
        
        # 状态变量
        self.last_conflict_intensity = 0.0
        self.cooling_counter = 0
        self.choice_counter = 0
        self.trigger_count = 0
        self.last_emotion = None
        self.emotion_stuck_count = 0
        self.last_user_interaction = 0.0
        self._last_call_step = -1  # 轮次级别的调用防护
        
        # 初始化日志记录器
        self.logger = get_logger('er')
    
    def reset(self):
        """重置ER状态"""
        self.last_conflict_intensity = 0.0
        self.cooling_counter = 0
        self.choice_counter = 0
        self.trigger_count = 0
        self.last_emotion = None
        self.emotion_stuck_count = 0
    
    def step(self):
        """执行内部步骤"""
        # 简单实现：减少冷却计数器
        if self.cooling_counter > 0:
            self.cooling_counter -= 1
    
    def collect_signals(self, fse, bi):
        """收集冲突信号"""
        signals = {}
        
        # 从FSE获取信号
        if fse:
            # 自我一致性误差
            signals['E_self'] = getattr(fse, 'E_self', 0.0)
            # 预测误差
            signals['N_pred'] = getattr(fse, 'E_pred', 0.0)
            # 僵化度
            signals['A_rigid'] = getattr(fse, 'A_rigid', 0.0)
            # 满足度（使用瞬时执着强度）
            l_inst = getattr(fse, '_l_inst', 0.0)
            signals['L_suf'] = l_inst  # 直接使用 0~1 的执着强度
            # 空性空心度
            signals['E_hollow'] = getattr(fse, 'E_hollow', 0.0)
            # 自我指涉深度
            signals['D_self'] = getattr(fse, 'D_self', 0.0)
            # 否定关系图势能（归一化）
            raw_n_neg = getattr(fse, 'N_neg', 0.0)
            # 归一化：tanh 函数，参考值 1000 对应约 0.76，5000 对应约 0.99
            signals['N_neg'] = min(1.0, np.tanh(raw_n_neg / 1000.0))
            # 非我执着度
            signals['Attach_non_self'] = getattr(fse, 'Attach_non_self', 0.0)
            # 情绪值
            signals['emotion'] = getattr(fse, 'V_emo', 0.0)
            # 情绪是否卡住
            current_emotion = getattr(fse, 'current_emotion', 'neutral')
            if current_emotion == self.last_emotion:
                self.emotion_stuck_count += 1
            else:
                self.emotion_stuck_count = 0
            self.last_emotion = current_emotion
            # 连续20步同一情绪视为卡住
            signals['emotion_stuck'] = 1.0 if self.emotion_stuck_count >= 20 else 0.0
        
        # 从BI获取信号
        if bi:
            # 物理情绪
            signals['V_phys'] = getattr(bi, 'V_phys', 0.0)
            # 死亡临近
            signals['Death_near'] = getattr(bi, 'Death_near', 0.0)
        
        # 耦合刚度
        signals['coupling_stiffness'] = 0.0
        
        # 临时诊断日志
        self.logger.debug(f"Signals: L_suf={signals.get('L_suf')}, emotion_stuck={signals.get('emotion_stuck')}, coupling_stiffness={signals.get('coupling_stiffness')}, N_neg={signals.get('N_neg')}, E_self={signals.get('E_self')}, N_pred={signals.get('N_pred')}, A_rigid={signals.get('A_rigid')}, E_hollow={signals.get('E_hollow')}, D_self={signals.get('D_self')}, Attach_non_self={signals.get('Attach_non_self')}, V_phys={signals.get('V_phys')}, Death_near={signals.get('Death_near')}")
        
        # 临时诊断：打印所有信号值 - 已移除，避免日志刷屏
        # import json
        # print(f"[ER DEBUG] Signals: {json.dumps(signals, default=str)}")
        
        # 确保所有信号值在 0~1 范围内
        for key in signals:
            signals[key] = max(0.0, min(1.0, signals[key]))
        
        return signals
    
    def _extract_conflict_hint(self, signals: dict) -> str:
        """
        从冲突信号中提取主要冲突源的文本描述。
        用于 selective_emptiness 确定哪些否定节点与当前冲突源高度相关。
        """
        hints = []
        if signals.get('emotion_stuck', 0) > 0.5:
            hints.append(f"长期卡在{self.last_emotion}情绪")
        if signals.get('L_suf', 0) > 0.7:
            hints.append("执着强度过高")
        if signals.get('coupling_stiffness', 0) > 0.6:
            hints.append("耦合僵化度过高")
        if signals.get('N_neg', 0) > 0.5:
            hints.append("否定势能过高")
        return "; ".join(hints) if hints else None
    
    def compute_conflict_intensity(self, signals):
        """
        计算冲突强度 C。
        C = Σ w_i * s_i，其中 s_i 为各归一化信号值，w_i 为对应权重（可配置）。
        信号包括：E_self, N_pred, A_rigid, L_suf, E_hollow, D_self, N_neg, Attach_non_self, V_phys, Death_near, emotion_stuck, coupling_stiffness。
        """
        """计算冲突强度"""
        # 使用配置的权重
        
        # 计算冲突强度
        C = 0.0
        for key, value in signals.items():
            if key in self.weights:
                C += self.weights[key] * value
        
        # 确保C在0-1之间，上限设为0.99以避免冲突强度恒为1.0
        C = min(0.99, max(0.0, C))
        
        return C
    
    def regulate(self, **kwargs):
        """调节空性"""
        # 轮次级别的调用防护
        current_step = kwargs.get('step', 0)
        if self._last_call_step == current_step:
            return {'reset': False, 'conflict_intensity': 0.0, 'action': 'continue', 'should_forget': False, 'emptiness_triggered': False}
        self._last_call_step = current_step
        
        # 冷却检查，避免频繁调用
        if self.cooling_counter > 0:
            self.cooling_counter -= 1
            return {
                'reset': False,
                'conflict_intensity': 0.0,
                'action': 'continue',
                'should_forget': False,
                'emptiness_triggered': False
            }
        # 从kwargs中获取fse和bi
        fse = kwargs.get('fse', self.fse)
        bi = kwargs.get('bi', None)
        
        # 收集冲突信号
        signals = self.collect_signals(fse, bi)
        
        # 计算冲突强度
        C = self.compute_conflict_intensity(signals)
        
        # 打印详细调试信息
        self.logger.debug(f"signals: {signals}")
        self.logger.debug(f"computed conflict intensity: {C:.5f}, threshold: {self.death_threshold}")
        
        # 临时诊断输出已移除
        

        
        # 检查是否触发空性操作
        if C > self.death_threshold:
            self.trigger_count += 1
            self.cooling_counter = self.cooling_period
            self.last_conflict_intensity = 0.0
            self.emotion_stuck_count = 0
            self.last_emotion = None

            # 提取冲突源描述
            conflict_hint = self._extract_conflict_hint(signals)

            # 选择性空性
            self.selective_emptiness(conflict_source_hint=conflict_hint)

            # 记录螺旋事件
            if hasattr(self, 'engine') and self.engine:
                self.engine.global_workspace._record_spiral_event('emptiness_triggered', {
                    'conflict_intensity': round(C, 3),
                    'conflict_hint': conflict_hint
                })

            return {
                'reset': True,
                'conflict_intensity': C,
                'action': 'selective_emptiness',
                'should_forget': True,
                'emptiness_triggered': True,
                'operations': {'forget_trigger': True},
                'conflict_hint': conflict_hint
            }
        else:
            self.last_conflict_intensity = C
            return {
                'reset': False,
                'conflict_intensity': C,
                'action': 'continue',
                'should_forget': False,
                'emptiness_triggered': False
            }
    
    def compute_conflict_signals(self, **kwargs):
        """计算冲突信号"""
        # 简单实现：返回默认冲突信号
        return {
            'semantic_conflict': 0.0,
            'emotional_conflict': 0.0,
            'self_conflict': 0.0
        }
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'last_conflict_intensity': self.last_conflict_intensity,
            'cooling_counter': self.cooling_counter,
            'choice_counter': self.choice_counter,
            'trigger_count': self.trigger_count
        }
    
    def get_state(self):
        """获取状态"""
        return {
            'last_conflict_intensity': self.last_conflict_intensity,
            'cooling_counter': self.cooling_counter,
            'choice_counter': self.choice_counter
        }
    
    def load_state(self, state):
        """加载状态"""
        self.last_conflict_intensity = state.get('last_conflict_intensity', 0.0)
        self.cooling_counter = state.get('cooling_counter', 0)
        self.choice_counter = state.get('choice_counter', 0)
    
    def spontaneous_restart(self):
        """自发重启方法，当系统长时间无输入时触发"""
        import time
        # 如果最近 5 秒内有用户输入，不触发自发重启
        if hasattr(self, 'last_user_interaction') and (time.time() - self.last_user_interaction) < 5.0:
            self.logger.debug("Spontaneous restart postponed due to recent user interaction")
            return
        self.logger.info("Spontaneous restart triggered")
        # 记录自发重启事件
        self.trigger_count += 1
        
        # 调用选择性空性，冲突源描述为“长期寂静导致的僵化”
        self.selective_emptiness(conflict_source_hint="长期寂静导致的僵化")
        self.fse.low_potency_sampled = True
        
        # 重置冷却计数器
        self.cooling_counter = self.cooling_period
        
        # 设置内部输出，用于控制台显示
        if self.fse:
            self.fse.internal_output = "（系统在寂静中自我重启，情绪已重置）"
        
        self.logger.debug("Emotion reset after spontaneous restart")
        
        return {
            'spontaneous_restart': True,
            'trigger_count': self.trigger_count,
            'low_potency_sampled': True
        }
    
    def selective_emptiness(self, conflict_source_hint: str = None):
        """
        选择性空性操作：
        1. 衰减与冲突源相关的否定节点（大幅下降）
        2. 无关节点温和衰减
        3. 保护已确认的知识（protected=True 的节点不衰减）
        4. 重置情绪向量
        5. 保留过程语法偏好（大多数保留）
        """
        # ===== 1. 选择性衰减否定图 =====
        if self.fse and hasattr(self.fse, 'negation_graph'):
            neg_graph = self.fse.negation_graph

            if conflict_source_hint:
                # 有明确冲突源：相关节点大幅衰减，无关节点温和衰减
                for layer in [neg_graph.core, neg_graph.dynamic, neg_graph.short_term]:
                    for node_id, node in list(layer.nodes.items()):
                        if node.protected:
                            continue
                        if conflict_source_hint in node.description:
                            node.potency *= 0.3   # 大幅衰减
                        else:
                            node.potency *= 0.7   # 温和衰减
            else:
                # 无明确冲突源：均匀温和衰减
                for layer in [neg_graph.dynamic, neg_graph.short_term]:
                    for node in layer.nodes.values():
                        if not node.protected:
                            node.potency *= 0.6
                # 核心层不衰减

        # ===== 2. 重置情绪向量 =====
        if self.fse:
            if hasattr(self.fse, 'E_vec'):
                self.fse.E_vec = np.zeros(5, dtype=np.float32)
            if hasattr(self.fse, 'current_emotion'):
                self.fse.current_emotion = "neutral"
            if hasattr(self.fse, 'V_emo'):
                self.fse.V_emo = 1.0
            if hasattr(self.fse, '_l_inst'):
                self.fse._l_inst = 0.0
            if hasattr(self.fse, 'stillness'):
                self.fse.stillness = 0

        # ===== 3. 保留过程语法偏好 =====
        if hasattr(self.fse, 'process_meta') and hasattr(self.fse.process_meta, 'reset_coupling'):
            self.fse.process_meta.reset_coupling(keep_recent=10)

        # ===== 4. 重置冷却 =====
        self.cooling_counter = self.cooling_period
        self.trigger_count += 1

        self.logger.info("选择性空性完成：相关执着已放下，核心记忆和知识保留。")
        # 记录空性快照
        import os, json, time as time_module
        os.makedirs('data', exist_ok=True)
        log_path = os.path.join('data', 'emptiness_snapshot_log.jsonl')
        neg_graph = self.fse.negation_graph if self.fse and hasattr(self.fse, 'negation_graph') else None
        snapshot = {
            'timestamp': time_module.time(),
            'conflict_hint': conflict_source_hint,
            'trigger_count': self.trigger_count,
            'emotion_before': getattr(self.fse, 'current_emotion', 'unknown') if self.fse else 'unknown',
            'N_neg_before': len(neg_graph) if neg_graph else 0,
            'protected_nodes_preserved': sum(1 for layer_name in ['core', 'dynamic', 'short_term'] for node in getattr(neg_graph, layer_name, type('',(),{'nodes':{}})()).nodes.values() if node.protected) if neg_graph else 0,
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
    
    def deep_emptiness(self) -> dict:
        """
        深度空性操作：重置自业执着，但不删除历史痕迹。
        与 spontaneous_restart 的区别：
        - spontaneous_restart 只重置当前状态（L、情绪）
        - deep_emptiness 额外重置过程语法偏好（transitionPreferences、residualAttachments）
        返回操作摘要。
        """
        # 1. 执行基础空性（重置当前状态）
        if self.fse:
            # 重置情绪向量
            if hasattr(self.fse, 'E_vec'):
                self.fse.E_vec = np.zeros(5, dtype=np.float32)
            # 重置当前情绪
            if hasattr(self.fse, 'current_emotion'):
                self.fse.current_emotion = "neutral"
            # 重置情绪强度
            if hasattr(self.fse, 'V_emo'):
                self.fse.V_emo = 0.0
            # 重置执着强度
            if hasattr(self.fse, '_l_inst'):
                self.fse._l_inst = 0.0
            # 重置寂静计数
            if hasattr(self.fse, 'stillness'):
                self.fse.stillness = 0
            # 重置幻想层计数器
            if hasattr(self.fse, 'fantasy_layer_counter'):
                self.fse.fantasy_layer_counter = 0
        
        # 2. 重置过程语法偏好（如果存在 process_meta）
        if hasattr(self, 'process_meta') and self.process_meta:
            # 将 transitionPreferences 向均匀分布拉回
            if hasattr(self.process_meta, 'transition_preferences'):
                prefs = self.process_meta.transition_preferences
                for key in prefs:
                    prefs[key] = 0.25  # 重置为均匀分布
            
            # 降低僵化度基线
            if hasattr(self.process_meta, 'stiffness_baseline'):
                self.process_meta.stiffness_baseline = max(0.0, self.process_meta.stiffness_baseline * 0.3)
        
        # 3. 清空残余执着（如果存在）
        if hasattr(self, 'residual_attachments'):
            self.residual_attachments = []
        
        # 4. 重置冷却计数器
        self.cooling_counter = self.cooling_period
        self.trigger_count += 1
        
        self.logger.info("深度空性完成：自业执着已释放，历史痕迹保留。")
        
        return {
            'deep_emptiness': True,
            'trigger_count': self.trigger_count,
            'message': '自业执着已重置，但核心记忆与螺旋历史不可删除。'
        }