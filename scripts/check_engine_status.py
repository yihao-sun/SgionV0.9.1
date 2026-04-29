#!/usr/bin/env python3
"""
检查引擎状态，确认全局步数、投射/反哺数、语义库词条数均已恢复
"""

import sys
import os
sys.path.insert(0, '.')
from engine import ExistenceEngine

def main():
    print("正在初始化引擎并检查状态...")
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    
    try:
        # 检查全局步数
        if hasattr(engine, 'generation_step'):
            print(f"全局步数: {engine.generation_step}")
        else:
            print("全局步数: 未找到")
        
        # 检查投射/反哺记录数
        if hasattr(engine, 'process_meta'):
            # 检查投射记录
            projection_count = len(engine.process_meta.projection_history) if hasattr(engine.process_meta, 'projection_history') else 0
            # 检查反哺记录
            nourishment_count = len(engine.process_meta.nourishment_history) if hasattr(engine.process_meta, 'nourishment_history') else 0
            print(f"投射/反哺记录数: {projection_count}/{nourishment_count}")
        else:
            print("投射/反哺记录数: 未找到")
        
        # 检查语义库词条数
        if hasattr(engine, 'lps'):
            print(f"LPS 词条数: {len(engine.lps.metadata)}")
        else:
            print("LPS 词条数: 未找到")
        
        # 检查语义映射器（如果存在）
        if hasattr(engine, 'semantic_mapper') and hasattr(engine.semantic_mapper, 'entries'):
            print(f"语义映射器条目数: {len(engine.semantic_mapper.entries)}")
        else:
            print("语义映射器: 未找到")
            
    finally:
        engine.shutdown()

if __name__ == "__main__":
    main()
