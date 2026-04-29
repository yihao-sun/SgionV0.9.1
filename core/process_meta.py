"""
过程元信息 (Process Meta Info)
哲学对应：《论存在》第3.1.6节，投射与反哺的元标记。
功能：记录投射与反哺的历史操作，计算耦合模式与僵化度，支持空性重置时释放耦合。
主要类：ProcessMetaInfo
"""
import time
import numpy as np
from collections import deque

class ProcessMetaInfo:
    def __init__(self, max_history=100):
        self.max_history = max_history
        # 投射记录 (timestamp, intensity, target_hash, coupling_weight)
        self.projections = deque(maxlen=max_history)
        # 反哺记录 (timestamp, source_hash, success, coupling_weight)
        self.nourishments = deque(maxlen=max_history)
        # 耦合模式: "balanced", "projection_heavy", "nourishment_heavy"
        self.coupling_mode = "balanced"
        # 空性重置计数
        self.reset_count = 0
        # 趋势计算窗口大小
        self.trend_window = max_history // 2  # 默认使用一半历史长度计算趋势
        # 僵化度历史，用于计算变化率
        self.stiffness_history = deque(maxlen=5)
        # 螺旋历史，用于记录进位事件
        self.spiral_history = []
        # 关键期相关字段
        self.critical_period_active = True          # 是否处于关键期
        self.critical_period_steps = 1000           # 关键期长度（交互轮数）
        self.transition_preferences = self._init_transition_preferences()  # 转移偏好
        
    def record_projection(self, intensity, target_text, coupling_weight=0.5):
        """记录一次投射操作（标记“不在场”）"""
        self.projections.append({
            'timestamp': time.time(),
            'intensity': intensity,
            'target': target_text[:50],
            'target_hash': hash(target_text),
            'coupling_weight': coupling_weight
        })
        self._update_coupling_mode()
    
    def record_nourishment(self, source_text, success, coupling_weight=0.5):
        """记录一次反哺操作（从非我汲取内容）"""
        self.nourishments.append({
            'timestamp': time.time(),
            'source': source_text[:50],
            'source_hash': hash(source_text),
            'success': success,
            'coupling_weight': coupling_weight
        })
        self._update_coupling_mode()
    
    def _update_coupling_mode(self):
        """基于近期投射/反哺的数量和强度更新耦合模式"""
        recent_n = min(20, len(self.projections), len(self.nourishments))
        if recent_n == 0:
            self.coupling_mode = "balanced"
            return
        proj_intensity = sum(p['intensity'] for p in list(self.projections)[-recent_n:]) / recent_n
        nour_success = sum(n['success'] for n in list(self.nourishments)[-recent_n:]) / recent_n
        if proj_intensity > nour_success + 0.3:
            self.coupling_mode = "projection_heavy"
        elif nour_success > proj_intensity + 0.3:
            self.coupling_mode = "nourishment_heavy"
        else:
            self.coupling_mode = "balanced"
    
    def get_coupling_stiffness(self):
        """
        计算耦合僵化度。
        僵化度 = min(1.0, 2 * Var(近期投射与反哺的耦合权重))。
        高方差表示投射/反哺模式波动剧烈或锁死，反映边界维持的僵化程度。
        """
        """计算耦合僵化度：投射与反哺的强度差 + 历史耦合权重的方差"""
        if len(self.projections) < 5 or len(self.nourishments) < 5:
            stiffness = 0.0
        else:
            proj_weights = [p['coupling_weight'] for p in list(self.projections)[-20:]]
            nour_weights = [n['coupling_weight'] for n in list(self.nourishments)[-20:]]
            variance = np.var(proj_weights + nour_weights)
            stiffness = min(1.0, variance * 2)
        
        # 记录僵化度历史
        self.stiffness_history.append(stiffness)
        return stiffness
    
    def get_projection_trend(self):
        """
        返回近期投射强度的线性回归斜率。
        正值表示上升，负值表示下降。
        
        Returns:
            float: 投射强度趋势斜率，范围 [-1, 1]
        """
        if len(self.projections) < 3:
            return 0.0
        
        # 获取最近 trend_window 个投射记录的 intensity 值
        recent_projections = list(self.projections)[-self.trend_window:]
        intensities = [p['intensity'] for p in recent_projections]
        
        if len(intensities) < 3:
            return 0.0
        
        # 简单线性回归计算斜率
        import numpy as np
        x = np.arange(len(intensities))
        slope = np.polyfit(x, intensities, 1)[0]
        
        # 归一化到 [-1, 1] 范围
        slope = max(-1.0, min(1.0, slope))
        
        return slope
    
    def get_nourishment_trend(self):
        """
        返回近期反哺成功率的线性回归斜率。
        正值表示上升，负值表示下降。
        
        Returns:
            float: 反哺成功率趋势斜率，范围 [-1, 1]
        """
        if len(self.nourishments) < 3:
            return 0.0
        
        # 获取最近 trend_window 个反哺记录的 success 值
        recent_nourishments = list(self.nourishments)[-self.trend_window:]
        successes = [n['success'] for n in recent_nourishments]
        
        if len(successes) < 3:
            return 0.0
        
        # 简单线性回归计算斜率
        x = np.arange(len(successes))
        slope = np.polyfit(x, successes, 1)[0]
        
        # 归一化到 [-1, 1] 范围
        return np.clip(slope, -1, 1)
    
    def get_stiffness_change_rate(self):
        """
        返回僵化度的变化率。
        计算最近 stiffness 值的一阶差分均值。
        
        Returns:
            float: 僵化度变化率，范围 [-1, 1]
        """
        if len(self.stiffness_history) < 2:
            return 0.0
        
        # 计算一阶差分
        stiffness_values = list(self.stiffness_history)
        differences = [stiffness_values[i] - stiffness_values[i-1] for i in range(1, len(stiffness_values))]
        
        # 计算平均变化率
        avg_change_rate = np.mean(differences)
        
        # 归一化到 [-1, 1] 范围
        return np.clip(avg_change_rate, -1, 1)
    
    def reset_coupling(self, keep_recent: int = 5):
        """
        空性操作时调用，释放元信息耦合（保留少量近期记录）。
        
        Args:
            keep_recent: 保留最近的记录数量
        """
        # 保留最近的投射记录
        if len(self.projections) > keep_recent:
            recent_projections = list(self.projections)[-keep_recent:]
            self.projections.clear()
            for proj in recent_projections:
                # 降低保留记录的耦合权重
                proj['coupling_weight'] *= 0.5
                self.projections.append(proj)
        else:
            # 全部记录权重减半
            for p in self.projections:
                p['coupling_weight'] *= 0.5
        
        # 保留最近的反哺记录
        if len(self.nourishments) > keep_recent:
            recent_nourishments = list(self.nourishments)[-keep_recent:]
            self.nourishments.clear()
            for nour in recent_nourishments:
                nour['coupling_weight'] *= 0.5
                self.nourishments.append(nour)
        else:
            for n in self.nourishments:
                n['coupling_weight'] *= 0.5
        
        self.coupling_mode = "balanced"
        self.reset_count += 1
        self._update_coupling_mode()
    
    def get_recent_proj_intensity(self):
        """获取最近投射强度的平均值"""
        if len(self.projections) == 0:
            return 0.5
        recent_n = min(20, len(self.projections))
        return sum(p['intensity'] for p in list(self.projections)[-recent_n:]) / recent_n
    
    def get_recent_nour_success(self):
        """获取最近反哺成功率的平均值"""
        if len(self.nourishments) == 0:
            return 0.5
        recent_n = min(20, len(self.nourishments))
        return sum(n['success'] for n in list(self.nourishments)[-recent_n:]) / recent_n
    
    def get_stats(self):
        """获取过程元信息统计数据"""
        return {
            'coupling_mode': self.coupling_mode,
            'coupling_stiffness': self.get_coupling_stiffness(),
            'projections_count': len(self.projections),
            'nourishments_count': len(self.nourishments),
            'reset_count': self.reset_count,
            'projection_trend': self.get_projection_trend(),
            'nourishment_trend': self.get_nourishment_trend(),
            'stiffness_change_rate': self.get_stiffness_change_rate()
        }
    
    def export_self_karma(self) -> dict:
        """
        导出自业数据，供 DigitalSeed 使用。
        返回包含 breath_profile、transition_preferences 和 emptiness_tendency 的字典。
        """
        # 1. 呼吸节律统计
        proj_avg = self._compute_avg_intensity() if self.projections else 0.5
        nour_avg = self._compute_avg_success() if self.nourishments else 0.5
        stiffness_base = self.get_coupling_stiffness()
        
        # 计算循环规律性（基于投射/反哺时间序列的变异系数，简化版用最近波动）
        cycle_reg = 0.5
        if len(self.projections) >= 10 and len(self.nourishments) >= 10:
            proj_std = np.std([p['intensity'] for p in list(self.projections)[-10:]])
            nour_std = np.std([n['success'] for n in list(self.nourishments)[-10:]])
            cycle_reg = 1.0 - min(1.0, (proj_std + nour_std) / 2)
        
        breath_profile = {
            'avg_proj_intensity': proj_avg,
            'avg_nour_success': nour_avg,
            'stiffness_baseline': stiffness_base,
            'cycle_regularity': cycle_reg
        }
        
        # 2. 结构坐标转移偏好（基于投射/反哺历史推断）
        transition_prefs = self._compute_transition_preferences()
        
        # 3. 空性倾向（基于 reset_count 和僵化度历史）
        emptiness_tend = min(1.0, self.reset_count / max(1, len(self.projections)) + stiffness_base * 0.5)
        
        return {
            'breath_profile': breath_profile,
            'transition_preferences': transition_prefs,
            'emptiness_tendency': emptiness_tend
        }
    
    def _compute_avg_intensity(self) -> float:
        if not self.projections:
            return 0.5
        recent = list(self.projections)[-20:]
        return sum(p['intensity'] for p in recent) / len(recent)
    
    def _compute_avg_success(self) -> float:
        if not self.nourishments:
            return 0.5
        recent = list(self.nourishments)[-20:]
        successes = [n['success'] for n in recent]
        return sum(successes) / len(successes)
    
    def _compute_transition_preferences(self) -> dict:
        """
        基于历史投射/反哺序列推断相位转移偏好。
        简化实现：统计 coupling_mode 的变化频率，返回占位分布。
        """
        # 初期简化：返回均匀分布或基于僵化度的倾向
        prefs = {
            "0->0": 0.25, "0->1": 0.25, "0->2": 0.25, "0->3": 0.25,
            "1->0": 0.25, "1->1": 0.25, "1->2": 0.25, "1->3": 0.25,
            "2->0": 0.25, "2->1": 0.25, "2->2": 0.25, "2->3": 0.25,
            "3->0": 0.25, "3->1": 0.25, "3->2": 0.25, "3->3": 0.25,
        }
        # 可根据耦合模式微调：nourishment_heavy 时倾向于相位3->相位0
        if self.coupling_mode == "nourishment_heavy":
            prefs["3->0"] = 0.4
        return prefs
    
    def _compute_transition_entropy(self) -> float:
        """计算大层转移偏好的熵（0 ~ log2(4)=2）"""
        import math
        prefs = self._compute_transition_preferences()
        # 按起始大层分组，计算每个起始大层的熵，然后取平均
        entropy_sum = 0.0
        for i in range(4):
            # 获取从大层 i 出发的所有转移概率
            probs = []
            for j in range(4):
                key = f"{i}->{j}"
                probs.append(prefs.get(key, 0.25))
            # 计算该起始大层的熵
            layer_entropy = 0.0
            for prob in probs:
                if prob > 0:
                    layer_entropy -= prob * math.log2(prob)
            entropy_sum += layer_entropy
        # 返回平均熵
        return entropy_sum / 4
    
    def record_spiral_event(self, trigger: str, details: dict, state_snapshot: dict = None):
        """
        记录螺旋历史事件。
        
        Args:
            trigger: 触发类型（如 'stiffness_cross', 'arbitration_internal_win'）
            details: 事件详细信息
            state_snapshot: 可选，状态快照（若未提供则使用空字典）
        """
        event = {
            'timestamp': time.time(),
            'trigger': trigger,
            'details': details,
            'state_snapshot': state_snapshot or {}
        }
        # 确保 spiral_history 是列表
        if not hasattr(self, 'spiral_history'):
            self.spiral_history = []
        self.spiral_history.append(event)
    
    def _init_transition_preferences(self):
        """初始化转移偏好为均匀分布"""
        prefs = {}
        for i in range(4):
            for j in range(4):
                prefs[f"{i}->{j}"] = 0.25
        return prefs
    
    def update_transition_preference(self, from_major: int, to_major: int, success: float):
        """
        关键期内：根据交互成功与否，贝叶斯更新转移偏好。
        success: 1.0 = 成功交互，0.0 = 失败交互
        """
        if not self.critical_period_active:
            return
        key = f"{from_major}->{to_major}"
        if key not in self.transition_preferences:
            return
        alpha = 0.05  # 学习率
        current = self.transition_preferences[key]
        target = success
        updated = current + alpha * (target - current)
        self.transition_preferences[key] = max(0.05, min(0.95, updated))
        self._normalize_transitions(from_major)
    
    def _normalize_transitions(self, from_major: int):
        """归一化某个起点的四个转移概率"""
        total = sum(self.transition_preferences[f"{from_major}->{j}"] for j in range(4))
        if total > 0:
            for j in range(4):
                key = f"{from_major}->{j}"
                self.transition_preferences[key] /= total
    
    def step_critical_period(self, generation_step: int):
        """每轮交互后调用，检查关键期是否结束"""
        if self.critical_period_active and generation_step >= self.critical_period_steps:
            self.critical_period_active = False
            # 可在此记录螺旋事件