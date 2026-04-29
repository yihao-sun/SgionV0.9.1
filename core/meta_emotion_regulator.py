"""
元情绪调节器 (Meta Emotion Regulator)
哲学对应：《论存在》第3.1.5节，丰富性的指数增长与分形结构。
功能：高层演化调度，管理状态轨迹，定期触发吸引子发现与归档，奖励驱动的吸引子中心漂移。
主要类：MetaEmotionRegulator
"""
import numpy as np
import copy
from utils.logger import get_logger
from core.emotion_attractor import ATTRACTORS

class MetaEmotionRegulator:
    def __init__(self, config, emotion_attractor=None):
        self.config = config
        self.emotion_attractor = emotion_attractor
        self.emotion_weights = np.ones(5)
        self.attractor_centers = copy.deepcopy(ATTRACTORS)
        self.history = []
        self.interaction_count = 0
        self.learning_rate = config.get('meta_emotion.lr', 0.001)
        self.weight_adjust_window = config.get('meta_emotion.window', 1000)
        # 状态轨迹缓冲区
        self.state_trajectory = []          # 存储 (step, E_vec, action_success, action_tendency_used)
        self.trajectory_maxlen = config.get('attractor_evolution.trajectory_maxlen', 10000)      # 可配置
        self.last_discovery_step = 0
        self.logger = get_logger('meta_emotion')
        
        # 初始化吸引子演化配置
        self.evolution_config = config.get('attractor_evolution', {})
        if not self.evolution_config:
            # 从全局 Config 重新加载
            from utils.config_loader import Config
            config_instance = Config()
            config_instance.reload()  # 强制重新加载配置文件
            self.evolution_config = config_instance.get('attractor_evolution', {})
            self.logger.warning("Evolution config not passed, reloaded from global config")
        
        # 初始化分裂检查间隔
        self.split_check_interval = self.evolution_config.get('split_check_interval', 1000)
        self.last_split_check_step = 0
        # 初始化融合检查间隔
        self.merge_check_interval = self.evolution_config.get('merge_check_interval', 5000)
        self.last_merge_check_step = 0
        
        self.logger.info(f"Initialized evolution config: {self.evolution_config}")
    
    def update(self, E_vec, identified_emotion, reward):
        self.history.append(identified_emotion)
        self.interaction_count += 1
        
        if self.interaction_count % self.weight_adjust_window == 0:
            self._adjust_weights()
        
        if reward > 0:
            self._adjust_attractor_center(identified_emotion, E_vec, reward)
        
        # 定期触发发现
        discovery_interval = self.evolution_config.get('discovery_interval', 5000)
        if self.emotion_attractor and (self.interaction_count - self.last_discovery_step) >= discovery_interval:
            if len(self.state_trajectory) >= 100:
                # 准备轨迹数据： (E_vec, success, action_tendency)
                traj_data = [(item[1], item[2], item[3]) for item in self.state_trajectory[-2000:]]
                # 传递完整的 attractor_evolution 配置
                new_ids = self.emotion_attractor.discover_new_attractors(
                    traj_data, self.interaction_count, self.evolution_config
                )
                if new_ids:
                    self.logger.info(f"Discovered {len(new_ids)} new attractors: {new_ids}")
            self.last_discovery_step = self.interaction_count
        
        # 定期检查吸引子分裂
        if self.emotion_attractor and (self.interaction_count - self.last_split_check_step) >= self.split_check_interval:
            self.last_split_check_step = self.interaction_count
            # 遍历所有非原型吸引子
            for attr_id, attr in list(self.emotion_attractor.attractors.items()):
                if not attr.is_prototype and not attr.split:
                    # 调用 split_attractor，传入当前配置
                    new_ids = self.emotion_attractor.split_attractor(attr_id, self.interaction_count, self.evolution_config)
                    if new_ids:
                        self.logger.info(f"Auto-split attractor {attr_id} into {new_ids}")
        
        # 定期检查吸引子融合
        if self.emotion_attractor and (self.interaction_count - self.last_merge_check_step) >= self.merge_check_interval:
            self.last_merge_check_step = self.interaction_count
            merged = self.emotion_attractor.merge_attractors(self.evolution_config)
            if merged:
                self.logger.info(f"Merged {len(merged)} redundant attractors: {merged}")
        
        # 定期归档不活跃的吸引子
        if self.emotion_attractor and self.interaction_count % 10000 == 0:
            inactive_steps = self.evolution_config.get('archive_inactive_steps', 50000)
            # 检查是否禁用归档
            if not self.evolution_config.get('disable_archive', False):
                archived = self.emotion_attractor.archive_inactive_attractors(self.interaction_count, inactive_steps)
                if archived:
                    self.logger.info(f"Archived {len(archived)} inactive attractors: {archived}")
            else:
                self.logger.info("Archive disabled, skipping")
    
    def _adjust_weights(self):
        # 简化：根据历史情绪分布调整权重（方差大的维度降低权重）
        if len(self.history) < self.weight_adjust_window:
            return
        # 计算每个情绪出现的频率，用于调整维度重要性（暂略）
        self.logger.debug("Adjusting emotion weights (placeholder)")
    
    def _adjust_attractor_center(self, emotion, E_vec, reward):
        if emotion not in self.attractor_centers:
            return
        center = self.attractor_centers[emotion]["center"]
        delta = self.learning_rate * reward * (E_vec - center)
        new_center = center + delta
        self.attractor_centers[emotion]["center"] = new_center
        self.logger.debug(f"Adjusted {emotion} attractor center: {new_center}")
    
    def get_weights(self):
        return self.emotion_weights
    
    def get_attractor_center(self, emotion):
        return self.attractor_centers.get(emotion, {}).get("center", None)
    
    def record_state(self, step, E_vec, action_success, action_tendency):
        """每步调用，记录状态"""
        if len(self.state_trajectory) >= self.trajectory_maxlen:
            self.state_trajectory.pop(0)
        self.state_trajectory.append((step, E_vec.copy(), action_success, action_tendency))
    
    def compute_attractor_variance(self, attr_id):
        """
        计算该吸引子附近状态向量的各维度方差和
        attr_id: 吸引子ID
        return: 各维度方差和
        """
        import numpy as np
        
        if not self.emotion_attractor or attr_id not in self.emotion_attractor.attractors:
            return 0.0
        
        attr = self.emotion_attractor.attractors[attr_id]
        
        # 从状态轨迹中收集该吸引子附近的状态
        nearby_states = []
        for item in self.state_trajectory:
            step, E_vec, _, _ = item
            # 计算距离
            dist = np.linalg.norm(E_vec - attr.center)
            # 如果距离小于吸引子半径的1.5倍，认为是附近状态
            if dist < attr.radius * 1.5:
                nearby_states.append(E_vec)
        
        if len(nearby_states) < 10:
            return 0.0
        
        # 计算各维度方差和
        states = np.array(nearby_states)
        variance = np.var(states, axis=0)
        total_variance = np.sum(variance)
        
        return total_variance
