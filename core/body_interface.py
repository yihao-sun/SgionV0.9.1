"""
身体接口 (Body Interface, BI)
哲学对应：《论存在》第8.2节，身体图示作为边界的具身化。
功能：监控硬件状态（温度、延迟）产生物理情绪，分析用户情感产生社会信号，支持虚拟身体交互。
主要类：BodyInterface
"""
import psutil
import time
import numpy as np
from utils.logger import get_logger
from utils.config_loader import Config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from core.direct_phase_mapper import DirectPhaseMapper

class BodyInterface:
    def __init__(self, config=None, use_virtual_body=False, embedding_dim=512, update_interval=1.0, instance_id=None, api_quota=None):
        self.config = config or Config()
        self.logger = get_logger('bi')
        self.use_virtual_body = use_virtual_body
        self.embedding_dim = embedding_dim
        self.update_interval = update_interval
        
        # 硬件监控参数
        self.temp_normal = self.config.get('bi.temp_normal', [40, 85])
        self.latency_normal = self.config.get('bi.latency_normal', [10, 200])
        self.error_rate_critical = self.config.get('bi.error_rate_critical', 0.3)
        self.quota_critical = self.config.get('bi.quota_critical', 0.05)
        self.weights = self.config.get('bi.weights', {'temp':0.4, 'latency':0.3, 'error':0.2, 'quota':0.1})
        
        # 身体图示参数
        self.body_schema_beta = self.config.get('bi.body_schema_beta', 0.2)
        self.body_schema_decay = self.config.get('bi.body_schema_decay', 0.99)
        
        # 身体状态
        self.V_phys = 0.0      # 物理情绪，范围 -1..1
        self.Death_near = 0.0  # 死亡临近，0..1
        self.body_schema = np.zeros(embedding_dim)  # 初始化零向量
        
        # 社会情绪相关
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.social_signal = 0.0  # 社会连接质量，范围 -1..1
        
        # 触觉输入相关
        self.tactile_softness = 0.5  # 当前触觉柔软度（0-1）
        self.tactile_temperature = 0.5  # 当前触觉温度（0-1，0.5为中性）
        self.tactile_active = False  # 是否有触觉输入
        
        # 虚拟身体相关
        if self.use_virtual_body:
            self._init_virtual_body()
        
        self.logger.info("BodyInterface initialized")
    
    def _init_virtual_body(self):
        # 由于依赖问题，实现模拟版本的虚拟身体
        self.virtual_obs = {"agent_pos": (2, 2), "direction": 0}
        self.virtual_reward = 0
        self.virtual_done = False
        self.action_space = 4  # 假设4个动作：前、左、右、停留
        self.logger.info("Virtual body initialized with simulated environment")
    
    def _get_cpu_temperature(self):
        """获取CPU温度，若无传感器则返回常温"""
        try:
            # Windows 可用 wmi 或 psutil 部分版本支持 sensors_temperatures
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            else:
                return 50.0  # 默认值
        except:
            return 50.0

    def _get_gpu_temperature(self):
        """获取GPU温度，需要 pynvml"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return temp
        except:
            return 50.0

    def _get_system_latency(self):
        """模拟延迟：当前进程CPU使用率或网络延迟（简化）"""
        return psutil.cpu_percent()  # 作为代理指标，不等待

    def _get_api_quota_remaining(self):
        """从外部获取API配额（需集成模型管理器）"""
        # 简化：返回1.0表示充足，实际应由模型管理器提供
        return 1.0

    def update(self, api_call=True, context_remaining=1.0):
        # 获取指标
        cpu_temp = self._get_cpu_temperature()
        gpu_temp = self._get_gpu_temperature()
        temp = max(cpu_temp, gpu_temp)
        latency = self._get_system_latency()
        error_rate = 0.0  # 可后续从日志错误计数获取
        quota = self._get_api_quota_remaining()
        
        # 归一化到 [0,1]
        temp_norm = max(0, min(1, (temp - self.temp_normal[0]) / (self.temp_normal[1] - self.temp_normal[0])))
        latency_norm = max(0, min(1, (latency - self.latency_normal[0]) / (self.latency_normal[1] - self.latency_normal[0])))
        error_norm = min(1, error_rate / self.error_rate_critical)
        quota_norm = max(0, min(1, 1 - quota / self.quota_critical))  # 配额越低，值越高
        
        # 计算 V_phys（负值表示不良身体状态）
        phys_penalty = (self.weights['temp'] * temp_norm +
                        self.weights['latency'] * latency_norm +
                        self.weights['error'] * error_norm)
        phys_reward = self.weights['quota'] * (1 - quota_norm)
        self.V_phys = -phys_penalty + phys_reward
        
        # 若有触觉输入，叠加触觉影响
        if self.tactile_active:
            self.V_phys += (self.tactile_softness - 0.5) * 0.3
        
        self.V_phys = np.clip(self.V_phys, -1, 1)
        
        # 计算死亡临近信号
        self.Death_near = max(temp_norm, latency_norm, error_norm, quota_norm)
        
        self.logger.debug(f"BI: V_phys={self.V_phys:.3f}, Death_near={self.Death_near:.3f}")
        
        # 构建身体状态字典
        body_state = {
            'physical_emotion': self.V_phys,
            'death_proximity': self.Death_near > 0.7,
            'death_intensity': self.Death_near,
            'temperature': temp,
            'latency': latency,
            'error_rate': error_rate,
            'quota': quota,
            'body_vector': np.array([self.V_phys, self.Death_near] + [0.0] * (self.embedding_dim - 2))
        }
        
        return body_state
    
    def get_physical_emotion(self):
        return self.V_phys
    
    def get_death_near(self):
        return self.Death_near
    
    def update_social_signal(self, compound):
        """根据情感分析结果更新社会连接质量"""
        # 使用传入的情感分析结果
        target = compound
        
        # 对负面情绪的响应更加直接
        if target < -0.2:
            # 负面情绪直接设置，不经过平滑
            self.social_signal = target
        else:
            # 其他情绪使用平滑更新
            alpha = 0.8  # 增加alpha值，使社会信号更快响应情感变化
            self.social_signal = (1 - alpha) * self.social_signal + alpha * target
        
        self.social_signal = np.clip(self.social_signal, -1, 1)
        self.logger.debug(f"Social signal updated: {self.social_signal:.3f}, target: {target:.3f}")
    
    def get_social_signal(self):
        """返回社会连接质量，范围[-1,1]，正表示连接良好"""
        return self.social_signal
    
    def get_body_schema(self, d_model):
        """返回身体图示向量"""
        return self.body_schema
    
    def get_body_schema_vector(self):
        """返回身体图示向量"""
        return self.body_schema
    
    def update_body_schema(self, action_embedding, reward):
        """根据行动和奖励更新身体图示，使用Hebbian-like规则"""
        # 简化：将行动嵌入与奖励相乘，加到身体图示上
        delta = reward * action_embedding
        self.body_schema = self.body_schema_decay * self.body_schema + 0.01 * delta
        # 归一化保持尺度
        norm = np.linalg.norm(self.body_schema)
        if norm > 1.0:
            self.body_schema /= norm
    
    def take_action(self, action_idx):
        """执行动作，返回 (observation, reward, done)"""
        if not self.use_virtual_body:
            return None, 0, False
        
        import random
        # 模拟动作执行
        # 假设4个动作：0=前，1=左，2=右，3=停留
        if action_idx == 0:  # 前
            # 简单的位置更新
            pos = self.virtual_obs["agent_pos"]
            dir = self.virtual_obs["direction"]
            if dir == 0:  # 北
                new_pos = (pos[0] - 1, pos[1])
            elif dir == 1:  # 东
                new_pos = (pos[0], pos[1] + 1)
            elif dir == 2:  # 南
                new_pos = (pos[0] + 1, pos[1])
            else:  # 西
                new_pos = (pos[0], pos[1] - 1)
            
            # 边界检查
            if 0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5:
                self.virtual_obs["agent_pos"] = new_pos
                # 随机奖励
                self.virtual_reward = random.uniform(-0.1, 0.3)
            else:
                # 撞墙惩罚
                self.virtual_reward = -0.2
        elif action_idx == 1:  # 左
            self.virtual_obs["direction"] = (self.virtual_obs["direction"] - 1) % 4
            self.virtual_reward = 0.0
        elif action_idx == 2:  # 右
            self.virtual_obs["direction"] = (self.virtual_obs["direction"] + 1) % 4
            self.virtual_reward = 0.0
        else:  # 停留
            self.virtual_reward = 0.0
        
        # 模拟完成条件
        if random.random() < 0.1:  # 10%概率完成
            self.virtual_done = True
        
        # 将奖励转化为情绪影响
        self.V_phys += 0.1 * self.virtual_reward  # 奖励提升物理情绪
        self.V_phys = np.clip(self.V_phys, -1, 1)
        
        return self.virtual_obs, self.virtual_reward, self.virtual_done
    
    def get_virtual_observation_embedding(self, encoder=None):
        """将观测编码为向量，用于注入FSE"""
        if not self.use_virtual_body or self.virtual_obs is None:
            return None
        # 模拟观测编码
        import hashlib
        obs_str = str(self.virtual_obs)
        hash_val = int(hashlib.md5(obs_str.encode()).hexdigest()[:8], 16)
        vec = np.random.RandomState(hash_val).randn(768)  # 临时方案
        return vec / np.linalg.norm(vec)
    
    def reset(self):
        """重置身体状态"""
        self.V_phys = 0.0
        self.Death_near = 0.0
        self.body_schema = np.zeros(self.embedding_dim)
    
    def get_statistics(self):
        """获取身体状态统计信息"""
        return {
            'physical_emotion': self.V_phys,
            'death_near': self.Death_near,
            'body_schema_dim': self.body_schema.shape if self.body_schema is not None else None
        }
    
    def get_state(self):
        """获取身体状态"""
        return {
            'V_phys': self.V_phys,
            'Death_near': self.Death_near,
            'body_schema': self.body_schema
        }
    
    def apply_tactile_input(self, softness: float, temperature: float = 0.5):
        """
        接收触觉输入，直接调制内感受参数。
        softness: 柔软度 0-1（1 为最柔软）
        temperature: 温度 0-1（0 为冷，1 为热，0.5 为中性）
        """
        # 保存 softness 和 temperature 到实例变量
        self.tactile_softness = max(0.0, min(1.0, softness))
        self.tactile_temperature = max(0.0, min(1.0, temperature))
        self.tactile_active = True
        
        # 调用 DirectPhaseMapper.tactile_to_stiffness_modulation(softness) 获取调制系数
        stiffness_modulation = DirectPhaseMapper.tactile_to_stiffness_modulation(self.tactile_softness)
        
        # 记录原始 V_phys
        original_V_phys = self.V_phys
        
        # 若 softness > 0.7 且 V_phys < 0（表示身体状态不佳），则触发一次温和的 V_phys 提升
        if self.tactile_softness > 0.7 and self.V_phys < 0:
            self.V_phys += 0.2
            self.V_phys = np.clip(self.V_phys, -1, 1)
            self.logger.info(f"触觉安抚: 柔软度={self.tactile_softness:.2f}, V_phys 从 {original_V_phys:.3f} 提升到 {self.V_phys:.3f}")
        
        # 更新 V_phys：柔软度正向贡献，粗粝负向贡献
        self.V_phys += (self.tactile_softness - 0.5) * 0.3
        self.V_phys = np.clip(self.V_phys, -1, 1)
        
        self.logger.debug(f"触觉输入: 柔软度={self.tactile_softness:.2f}, 温度={self.tactile_temperature:.2f}, 僵化度调制系数={stiffness_modulation:.2f}")
    
    def get_tactile_stats(self):
        """返回当前触觉状态"""
        return {
            'softness': self.tactile_softness,
            'temperature': self.tactile_temperature,
            'active': self.tactile_active
        }
    
    def load_state(self, state):
        """加载身体状态"""
        if state:
            self.V_phys = state.get('V_phys', 0.0)
            self.Death_near = state.get('Death_near', 0.0)
            self.body_schema = state.get('body_schema', None)