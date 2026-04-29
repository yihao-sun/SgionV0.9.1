#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制分裂测试脚本
通过在极端情绪状态间反复震荡，创造高方差以触发吸引子分裂
"""

import sys
sys.path.append('.')
from engine import ExistenceEngine
import numpy as np
import time

def main():
    engine = ExistenceEngine(vocab_size=10000)
    # 强制在高方差区域反复震荡
    for step in range(5000):
        if step % 50 < 25:
            # 极端恐惧
            engine.fse.E_vec = np.array([-0.9, 0.95, -0.9, -0.8, 0.1])
        else:
            # 极端快乐
            engine.fse.E_vec = np.array([0.9, 0.1, 0.9, 0.9, 0.9])
        
        # 强制识别
        engine.fse.current_emotion, _, _ = engine.fse.emotion_attractor.identify(
            engine.fse.E_vec, engine.fse.emotion_weights, step=step
        )
        # 模拟高成功率（奖励）
        success = 0.9 if step % 50 < 25 else 0.7
        engine.fse.meta_emotion.record_state(step, engine.fse.E_vec, success, engine.fse.current_emotion)
        
        if step % 1000 == 0:
            print(f"Step {step}, emotion: {engine.fse.current_emotion}")
    
    # 强制检查分裂
    for attr_id, attr in engine.fse.emotion_attractor.attractors.items():
        if not attr.is_prototype and attr.visit_count > 200:
            variance = engine.fse.meta_emotion.compute_attractor_variance(attr_id)
            if variance > 0.5:
                new_ids = engine.fse.emotion_attractor.split_attractor(attr_id, step, {})
                if new_ids:
                    print(f"✅ Split attractor {attr_id} into {new_ids}")

if __name__ == "__main__":
    main()
