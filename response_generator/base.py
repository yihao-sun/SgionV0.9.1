from abc import ABC, abstractmethod
import numpy as np

class ResponseGenerator(ABC):

    @abstractmethod
    def generate(self, user_input: str, S_t: np.ndarray, V_emo: float, L: int = 0, D_self: float = 0, C: float = 0, emotion: str = None, social_signal: float = 0.0) -> str:

        """生成响应文本
        
        Args:
            user_input: 用户输入文本
            S_t: 引擎状态向量
            V_emo: 情绪值
            L: 幻想层数
            D_self: 自我指涉深度
            C: 冲突强度
            emotion: 当前主导情绪
            social_signal: 社会连接信号
            
        Returns:
            生成的响应文本
        """

        pass
