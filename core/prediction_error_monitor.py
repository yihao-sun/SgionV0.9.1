"""
预测误差监控器 (Prediction Error Monitor)
哲学对应：预测误差是过程元信息的核心节拍器。
功能：根据 FSE.E_pred 动态调制感知注意力、僵化度累积和空性阈值。
"""

from utils.logger import get_logger


class PredictionErrorMonitor:
    """预测误差监控器，驱动感知系统的注意力分配与僵化度调节"""
    
    def __init__(self, fse=None, process_meta=None, er_module=None):
        self.fse = fse
        self.process_meta = process_meta
        self.er = er_module
        self.logger = get_logger('prediction_error_monitor')
        
        # 状态跟踪
        self.low_error_streak = 0          # 连续低预测误差计数
        self.high_error_streak = 0         # 连续高预测误差计数
        self.original_death_threshold = None  # 保存原始空性阈值
        
        # 配置参数
        self.high_error_threshold = 0.5
        self.low_error_threshold = 0.2
        self.streak_threshold = 3          # 连续多少轮触发累积效应
        
        # 感知注意力权重（供 GlobalWorkspace 读取）
        self.attention_weights = {
            'novelty': 0.5,      # 新颖信号敏感度
            'familiar': 0.3,     # 熟悉信号敏感度
            'resonance': 0.2     # 共鸣信号敏感度
        }
    
    def step(self) -> dict:
        """
        每轮交互后调用，根据当前 E_pred 更新内部状态并返回调制建议。
        返回字典包含：
        - high_importance: bool，是否应提高当前交互的存储权重
        - threshold_adjusted: bool，空性阈值是否被临时调整
        - stiffness_increased: bool，僵化度是否因持续低误差而增加
        - attention_weights: dict，当前的感知注意力权重
        """
        if not self.fse:
            return {}
        
        e_pred = getattr(self.fse, 'E_pred', 0.5)
        result = {'e_pred': e_pred}
        
        # 高预测误差：唤醒，降低空性阈值，增加新颖敏感度
        if e_pred > self.high_error_threshold:
            self.high_error_streak += 1
            self.low_error_streak = 0
            
            # 临时降低 ER 的 death_threshold
            if self.er:
                if self.original_death_threshold is None:
                    self.original_death_threshold = getattr(self.er, 'death_threshold', 0.5)
                # 降低阈值，使空性操作更容易触发（允许快速重新理解）
                new_threshold = max(0.2, self.original_death_threshold * 0.7)
                self.er.death_threshold = new_threshold
                result['threshold_adjusted'] = True
                self.logger.debug(f"高预测误差: E_pred={e_pred:.3f}, death_threshold: {self.original_death_threshold:.2f} -> {new_threshold:.2f}")
            
            # 提高新颖敏感度
            self.attention_weights['novelty'] = min(0.8, 0.5 + 0.1 * self.high_error_streak)
            self.attention_weights['familiar'] = max(0.1, 0.3 - 0.05 * self.high_error_streak)
            
            result['high_importance'] = True
            result['attention_weights'] = self.attention_weights.copy()
        
        # 持续低预测误差：累积僵化度
        elif e_pred < self.low_error_threshold:
            self.low_error_streak += 1
            self.high_error_streak = 0
            
            if self.low_error_streak >= self.streak_threshold:
                # 逐渐提高耦合僵化度（通过 process_meta 的耦合模式）
                if self.process_meta:
                    self.process_meta.coupling_mode = "projection_heavy"
                    result['stiffness_increased'] = True
                    self.logger.debug(f"持续低预测误差: streak={self.low_error_streak}, 触发惯性累积")
            
            # 降低新颖敏感度，提高熟悉敏感度
            self.attention_weights['novelty'] = max(0.2, 0.5 - 0.05 * self.low_error_streak)
            self.attention_weights['familiar'] = min(0.6, 0.3 + 0.05 * self.low_error_streak)
            
            result['attention_weights'] = self.attention_weights.copy()
        
        # 中等预测误差：逐渐恢复
        else:
            if self.high_error_streak > 0:
                self.high_error_streak = max(0, self.high_error_streak - 1)
            if self.low_error_streak > 0:
                self.low_error_streak = max(0, self.low_error_streak - 1)
            
            # 恢复原始空性阈值
            if self.er and self.original_death_threshold is not None:
                self.er.death_threshold = self.original_death_threshold
                self.original_death_threshold = None
                result['threshold_adjusted'] = False
            
            # 注意力权重回归基线
            self.attention_weights = {'novelty': 0.5, 'familiar': 0.3, 'resonance': 0.2}
            result['attention_weights'] = self.attention_weights.copy()
        
        return result
    
    def get_stats(self) -> dict:
        """获取当前监控状态，供 /stats 命令使用"""
        return {
            'E_pred': getattr(self.fse, 'E_pred', 0.5) if self.fse else 0.5,
            'low_error_streak': self.low_error_streak,
            'high_error_streak': self.high_error_streak,
            'attention_weights': self.attention_weights.copy()
        }
    
    def reset(self):
        """重置状态"""
        self.low_error_streak = 0
        self.high_error_streak = 0
        self.attention_weights = {'novelty': 0.5, 'familiar': 0.3, 'resonance': 0.2}
        if self.er and self.original_death_threshold is not None:
            self.er.death_threshold = self.original_death_threshold
            self.original_death_threshold = None