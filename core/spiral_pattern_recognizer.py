"""
螺旋历史模式识别器 (Spiral Pattern Recognizer)
功能：从 spiral_history 序列中挖掘频繁模式，识别引擎的行为主题。
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter


class SpiralPatternRecognizer:
    def __init__(self, process_meta, config: Optional[Dict] = None):
        self.process_meta = process_meta
        self.config = config or {}
        
        # 可配置参数
        self.min_support = self.config.get('min_support', 3)
        self.max_pattern_len = self.config.get('max_pattern_len', 5)
        self.window_size = self.config.get('window_size', 50)
        
        # 缓存的模式识别结果
        self.active_patterns: List[Dict] = []
        
        # 事件到符号的映射规则
        self.event_to_symbol = {
            'stiffness_cross': 'S',
            'arbitration_internal_win': 'A',
            'intent_switch': 'I',
            'emptiness_triggered': 'E'
        }
    
    def _events_to_sequence(self, events: List[Dict]) -> List[str]:
        """将事件列表转换为符号序列"""
        symbols = []
        for event in events:
            trigger = event.get('trigger', '')
            symbol = self.event_to_symbol.get(trigger, 'U')
            symbols.append(symbol)
        return symbols
    
    def _find_frequent_subsequences(self, symbols: List[str]) -> List[Tuple[Tuple[str, ...], int]]:
        """
        在符号序列中寻找频繁子序列。
        返回: [((symbol_tuple), count), ...] 按出现次数降序排列
        """
        n = len(symbols)
        pattern_counts = defaultdict(int)
        
        for length in range(2, min(self.max_pattern_len + 1, n + 1)):
            for i in range(n - length + 1):
                pattern = tuple(symbols[i:i+length])
                pattern_counts[pattern] += 1
        
        frequent = [(p, c) for p, c in pattern_counts.items() if c >= self.min_support]
        frequent.sort(key=lambda x: x[1], reverse=True)
        return frequent
    
    def _interpret_pattern(self, pattern: Tuple[str, ...]) -> Tuple[str, str]:
        """
        将符号模式解释为人类可读的主题标签和描述。
        返回: (theme_name, description)
        """
        pattern_str = ''.join(pattern)
        
        if 'S' in pattern_str and 'E' in pattern_str:
            if pattern_str.index('S') < pattern_str.index('E'):
                return ('stiffness_emptiness_cycle', '僵化上升后触发空性')
            else:
                return ('emptiness_stiffness_relief', '空性后僵化度下降')
        
        if pattern_str.count('S') >= 2:
            return ('stiffness_oscillation', '僵化度反复波动')
        
        if pattern_str.count('A') >= 2:
            return ('internal_arbitration_dominance', '内部需求多次胜出')
        
        if 'I' in pattern_str and 'E' in pattern_str:
            return ('intent_switch_with_emptiness', '意图切换伴随空性')
        
        if pattern_str.count('I') >= 2:
            return ('frequent_intent_switch', '意图频繁切换')
        
        return ('unknown_pattern', f'模式: {pattern_str}')
    
    def extract_patterns(self) -> List[Dict]:
        """
        提取当前活跃的主题模式。
        
        Returns:
            模式列表，每个模式包含：
            - theme: 主题标签
            - description: 描述
            - pattern_symbols: 符号序列
            - frequency: 出现次数
            - avg_interval: 平均间隔（事件数）
            - dominant_emotion: 模式伴随的最常见情绪
        """
        if not self.process_meta:
            return []
        
        spiral_history = getattr(self.process_meta, 'spiral_history', [])
        if len(spiral_history) < self.min_support * 2:
            return []
        
        recent_events = spiral_history[-self.window_size:]
        symbols = self._events_to_sequence(recent_events)
        
        if len(symbols) < 4:
            return []
        
        frequent = self._find_frequent_subsequences(symbols)
        
        patterns = []
        for pattern_tuple, count in frequent:
            theme, desc = self._interpret_pattern(pattern_tuple)
            
            pattern_len = len(pattern_tuple)
            positions = []
            for i in range(len(symbols) - pattern_len + 1):
                if tuple(symbols[i:i+pattern_len]) == pattern_tuple:
                    positions.append(i)
            
            avg_interval = 0
            if len(positions) > 1:
                intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                avg_interval = sum(intervals) / len(intervals)
            
            emotions = []
            for pos in positions:
                if pos < len(recent_events):
                    snapshot = recent_events[pos].get('state_snapshot', {})
                    em = snapshot.get('emotion', 'neutral')
                    emotions.append(em)
            dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else 'neutral'
            
            patterns.append({
                'theme': theme,
                'description': desc,
                'pattern_symbols': ''.join(pattern_tuple),
                'frequency': count,
                'avg_interval': round(avg_interval, 2),
                'dominant_emotion': dominant_emotion,
                'last_seen': recent_events[-1].get('timestamp', 0) if recent_events else 0
            })
        
        self.active_patterns = patterns
        return patterns
    
    def get_active_themes(self) -> List[str]:
        """返回当前活跃主题的标签列表（去重）"""
        if not self.active_patterns:
            self.extract_patterns()
        return list(set(p['theme'] for p in self.active_patterns if p['theme'] != 'unknown_pattern'))
    
    def get_theme_stats(self) -> Dict:
        """返回主题统计信息，供 /themes 命令使用"""
        patterns = self.extract_patterns()
        return {
            'active_themes': self.get_active_themes(),
            'patterns': patterns[:5],
            'total_events_analyzed': len(getattr(self.process_meta, 'spiral_history', [])),
            'window_size': self.window_size
        }