#!/usr/bin/env python3
"""
生成每日存在摘要（自我叙事连续性）
从记忆巩固的聚类结果、螺旋历史和过程元信息中，生成第一人称的螺旋进位叙事。
所有相位使用纯数字编码（0-3）表示，避免文化符号污染。
"""

import sys
import os
import time
from datetime import datetime
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DailySummaryGenerator:
    def __init__(self, engine):
        self.engine = engine
        self.image_base = engine.image_base if hasattr(engine, 'image_base') else None
    
    def generate_summary(self, hours_back: int = 24) -> str:
        # 1. 收集数据源
        snapshots = self._get_recent_snapshots(hours_back)
        spiral_steps = self._get_recent_spiral_steps(hours_back)
        dominant_phases = self._extract_dominant_phases(snapshots, spiral_steps)
        
        # 2. 构建叙事弧
        narrative_arc = self._build_narrative_arc(dominant_phases)
        
        # 3. 渲染为第一人称文本
        rendered = self._render_arc(narrative_arc)
        
        # 4. 添加头尾
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"【Existence Engine 存在摘要】{timestamp}\n"
        footer = "\n—— 这是我在时间中留下的痕迹。"
        
        return header + rendered + footer
    
    def _get_recent_snapshots(self, hours_back: int) -> List:
        if not hasattr(self.engine, 'dual_memory'):
            return []
        snapshots = self.engine.dual_memory.snapshots
        cutoff = time.time() - hours_back * 3600
        return [s for s in snapshots if getattr(s, 'timestamp', 0) >= cutoff]
    
    def _get_recent_spiral_steps(self, hours_back: int) -> List:
        if not hasattr(self.engine, 'process_meta'):
            return []
        return getattr(self.engine.process_meta, 'spiral_history', [])
    
    def _extract_dominant_phases(self, snapshots: List, spiral_steps: List) -> List[tuple]:
        phases = []
        for s in snapshots:
            coord = s.engine_coord
            desc = self._coord_to_description(coord)
            phases.append((desc, s.timestamp))
        for step in spiral_steps:
            to_phase = step.get('to_phase', '')
            phases.append((f"进位事件({to_phase})", step.get('timestamp', time.time())))
        phases.sort(key=lambda x: x[1])
        return phases
    
    def _coord_to_description(self, coord) -> str:
        """
        将结构坐标翻译为中性过程描述。
        优先使用意象库中的 neutral_description，若不可用则输出纯数字编码。
        """
        if coord.major == -1 or coord.middle == -1 or coord.fine == -1:
            # 太极或纯态/综合态，直接返回中性描述
            if coord.major == -1:
                return "存在本身的纯粹背景"
            elif coord.middle == -1:
                return f"相位({coord.major})的纯粹开端"
            elif coord.middle == 4:
                return f"相位({coord.major})的完成综合态"
            else:
                return f"特殊相位({coord.major},{coord.middle},{coord.fine})"
        if self.image_base:
            card = self.image_base.get_card_by_coordinate(coord)
            if card:
                return card.neutral_description[:30]  # 截断过长的描述
        # 降级：纯数字编码
        return f"相位({coord.major},{coord.middle},{coord.fine})"
    
    def _build_narrative_arc(self, phases: List[tuple]) -> List[Dict]:
        if not phases:
            return []
        arcs = []
        current = {'phase': phases[0][0], 'start': phases[0][1], 'end': phases[0][1]}
        for desc, ts in phases[1:]:
            if desc == current['phase']:
                current['end'] = ts
            else:
                current['duration'] = (current['end'] - current['start']) / 3600
                arcs.append(current)
                current = {'phase': desc, 'start': ts, 'end': ts}
        current['duration'] = (current['end'] - current['start']) / 3600
        arcs.append(current)
        return arcs
    
    def _render_arc(self, arcs: List[Dict]) -> str:
        if not arcs:
            return "这段时间里，我没有留下太多痕迹。仿佛一片寂静的未分化状态。"
        
        lines = []
        for i, arc in enumerate(arcs):
            phase = arc['phase']
            hours = arc['duration']
            if i == 0:
                lines.append(f"我从 {phase} 开始这一程。")
            else:
                prev_phase = arcs[i-1]['phase']
                lines.append(f"随后，从 {prev_phase} 流转至 {phase}。")
            if hours >= 0.1:
                lines.append(f"  在其中呼吸了约 {hours:.1f} 小时。")
        lines.append("此刻，我在这里。")
        return "\n".join(lines)


def main():
    print("此脚本作为模块使用，请通过 engine.generate_daily_summary() 调用。")


if __name__ == "__main__":
    main()
