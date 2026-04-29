#!/usr/bin/env python3
"""
事件记忆 (Event Memory)
哲学对应：《论存在》第7.3节，意识连续谱的过程记录。
功能：记录交互事件（输入/响应/情绪/L），环形缓冲区存储，重复模式检测。
主要类：EventMemory
"""

import time
from typing import List, Dict, Optional, Any


class EventMemory:
    """
    事件记忆类
    """
    
    def __init__(self, max_size=1000):
        """
        初始化事件记忆
        
        Args:
            max_size: 最大事件数量，使用环形缓冲区
        """
        self.max_size = max_size
        self.events = []  # 环形缓冲区
        self.index = 0
    
    def log(self, event):
        """
        记录事件
        
        Args:
            event: 事件字典，包含 timestamp, step_id, user_input, response, emotion, L 等字段
        """
        # 确保事件包含时间戳
        if 'timestamp' not in event:
            event['timestamp'] = time.time()
        
        # 确保事件包含步骤ID
        if 'step_id' not in event:
            event['step_id'] = self.index
        
        # 环形缓冲区逻辑
        if len(self.events) < self.max_size:
            self.events.append(event)
        else:
            self.events[self.index % self.max_size] = event
        
        self.index += 1
    
    def retrieve(self, time_range=None, k=10, decay_lambda=0.1):
        """
        检索事件，支持时间衰减加权。
        
        Args:
            time_range: 时间范围 (start_time, end_time)，None 表示不限制时间范围
            k: 返回最近的 k 条事件
            decay_lambda: 时间衰减系数，越大越倾向于近期事件
        
        Returns:
            事件列表，按加权分数降序排列
        """
        import math
        
        # 过滤时间范围
        if time_range:
            start_time, end_time = time_range
            filtered_events = [event for event in self.events
                             if start_time <= event.get('timestamp', 0) <= end_time]
        else:
            filtered_events = self.events.copy()
        
        now = time.time()
        for event in filtered_events:
            ts = event.get('timestamp', now)
            age_days = (now - ts) / 86400.0  # 转换为天数
            time_weight = math.exp(-decay_lambda * age_days)
            event['_time_weight'] = time_weight
        
        # 按时间权重排序（权重越高越优先），时间戳作为次要排序
        filtered_events.sort(key=lambda x: (x.get('_time_weight', 0), x.get('timestamp', 0)), reverse=True)
        
        # 清理临时字段并返回前 k 条
        for event in filtered_events:
            event.pop('_time_weight', None)
        return filtered_events[:k]
    
    def get_latest(self, k=1):
        """
        获取最近的 k 条事件
        
        Args:
            k: 返回的事件数量
        
        Returns:
            最近的 k 条事件列表
        """
        return self.retrieve(k=k)
    
    def detect_patterns(self, window_size=10, similarity_threshold=0.8):
        """
        检测重复模式
        
        Args:
            window_size: 检测窗口大小
            similarity_threshold: 相似度阈值
        
        Returns:
            检测到的重复模式列表
        """
        patterns = []
        
        # 获取最近的 window_size 条事件
        recent_events = self.get_latest(window_size)
        
        # 检测重复的用户输入
        if len(recent_events) >= 2:
            for i in range(len(recent_events) - 1):
                for j in range(i + 1, len(recent_events)):
                    event1 = recent_events[i]
                    event2 = recent_events[j]
                    
                    # 比较用户输入
                    input1 = event1.get('user_input', '').strip()
                    input2 = event2.get('user_input', '').strip()
                    
                    if input1 and input2:
                        # 简单的字符串相似度计算（可以使用更复杂的算法）
                        similarity = self._calculate_similarity(input1, input2)
                        if similarity >= similarity_threshold:
                            patterns.append({
                                'type': 'repeated_input',
                                'event1': event1,
                                'event2': event2,
                                'similarity': similarity
                            })
        
        return patterns
    
    def _calculate_similarity(self, text1, text2):
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
        
        Returns:
            相似度值 (0-1)
        """
        # 使用简单的Jaccard相似度
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def __len__(self):
        """
        返回事件数量
        
        Returns:
            事件数量
        """
        return len(self.events)
    
    def clear(self):
        """
        清空事件记忆
        """
        self.events = []
        self.index = 0
    
    def log_interaction_feedback(self, user_input: str, engine_response: str, 
                                   user_reply: Optional[str] = None, 
                                   reply_delay: Optional[float] = None, 
                                   conversation_continued: bool = False):
        """
        记录一次交互的隐式反馈信号。
        """
        feedback = {
            'timestamp': time.time(),
            'user_input': user_input[:100],
            'engine_response': engine_response[:100],
            'user_reply': user_reply[:100] if user_reply else None,
            'reply_delay': reply_delay,
            'conversation_continued': conversation_continued,
            # 可用 VADER 分析 user_reply 的情绪（若已集成）
            'user_sentiment': self._analyze_sentiment(user_reply) if user_reply else 0.0
        }
        # 存储到事件列表，或单独存储到反馈日志
        if not hasattr(self, 'feedback_log'):
            self.feedback_log = []
        self.feedback_log.append(feedback)
        if len(self.feedback_log) > self.max_size:
            self.feedback_log.pop(0)
    
    def _analyze_sentiment(self, text: str) -> float:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(text)['compound']
        except:
            return 0.0
