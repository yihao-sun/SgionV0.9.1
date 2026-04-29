#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吸引子分裂测试脚本
"""

import sys
sys.path.append('.')
from engine import ExistenceEngine
import time
import numpy as np

def main():
    engine = ExistenceEngine(vocab_size=10000)
    # 强制交替两种极端情绪状态
    for step in range(2000):
        if step % 100 < 50:
            # 模拟恐惧：高 arousal，负 valence，低 social
            engine.fse.E_vec = np.array([-0.8, 0.9, -0.8, -0.6, 0.3])
        else:
            # 模拟快乐：正 approach，低 arousal，高 valence，高 social
            engine.fse.E_vec = np.array([0.7, 0.2, 0.8, 0.7, 0.8])
        
        # 强制识别情绪（更新访问计数）
        engine.fse.current_emotion, _, _ = engine.fse.emotion_attractor.identify(
            engine.fse.E_vec, engine.fse.emotion_weights, step=step
        )
        # 模拟行动成功（随机）
        engine.fse.meta_emotion.record_state(step, engine.fse.E_vec, np.random.rand() > 0.5, engine.fse.current_emotion)
        
        if step % 500 == 0:
            print(f"Step {step}, emotion: {engine.fse.current_emotion}")
    
    # 手动触发分裂检测
    for attr_id, attr in engine.fse.emotion_attractor.attractors.items():
        if not attr.is_prototype and attr.visit_count > 500:
            # 计算该吸引子的方差
            if hasattr(engine.fse.meta_emotion, 'compute_attractor_variance'):
                variance = engine.fse.meta_emotion.compute_attractor_variance(attr_id)
                print(f"Attractor {attr_id}: visit_count={attr.visit_count}, variance={variance:.3f}")
                if variance > 0.5:
                    new_ids = engine.fse.emotion_attractor.split_attractor(attr_id, step, engine.fse.meta_emotion.evolution_config)
                    if new_ids:
                        print(f"Split attractor {attr_id} into {new_ids}")
            else:
                # 如果没有 compute_attractor_variance 方法，直接检查状态历史的方差
                if len(attr.state_history) > 0:
                    states = np.array(attr.state_history)
                    variance = np.var(states, axis=0)
                    total_variance = np.sum(variance)
                    print(f"Attractor {attr_id}: visit_count={attr.visit_count}, total_variance={total_variance:.3f}")
                    if total_variance > 0.5:
                        new_ids = engine.fse.emotion_attractor.split_attractor(attr_id, step, engine.fse.meta_emotion.evolution_config)
                        if new_ids:
                            print(f"Split attractor {attr_id} into {new_ids}")

if __name__ == "__main__":
    main()
