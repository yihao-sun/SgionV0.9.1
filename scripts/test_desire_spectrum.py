#!/usr/bin/env python3
"""测试欲望光谱模块"""

import sys
sys.path.append('.')
from engine import ExistenceEngine


def main():
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    print("引擎初始化成功！")
    print("\n1. 测试欲望光谱初始化状态：")
    desire_stats = engine.desire_spectrum.get_stats()
    print(f"主导欲望: {desire_stats['dominant_desire']}")
    print("欲望强度:")
    for desire, intensity in desire_stats['intensities'].items():
        print(f"  {desire}: {intensity}")
    print("感知敏感度:")
    for sense, sensitivity in desire_stats['sensitivity'].items():
        print(f"  {sense}: {sensitivity}")
    
    print("\n2. 测试辅助方法：")
    print(f"get_dominant_desire(): {engine.desire_spectrum.get_dominant_desire()}")
    print(f"get_sensitivity('novelty'): {engine.desire_spectrum.get_sensitivity('novelty'):.2f}")
    print(f"get_sensitivity('resonance'): {engine.desire_spectrum.get_sensitivity('resonance'):.2f}")
    print(f"should_seek_tactile(): {engine.desire_spectrum.should_seek_tactile()}")
    print(f"should_seek_novelty(): {engine.desire_spectrum.should_seek_novelty()}")
    print(f"should_seek_resonance(): {engine.desire_spectrum.should_seek_resonance()}")
    
    print("\n3. 测试 update 方法：")
    update_result = engine.desire_spectrum.update()
    print(f"更新结果 - 主导欲望: {update_result['dominant_desire']}")
    print("欲望强度:")
    for desire, intensity in update_result['intensities'].items():
        print(f"  {desire}: {intensity:.2f}")
    print("感知敏感度:")
    for sense, sensitivity in update_result['sensitivity'].items():
        print(f"  {sense}: {sensitivity:.2f}")
    
    print("\n4. 测试 step 方法：")
    step_result = engine.desire_spectrum.step()
    print(f"Step 结果 - 主导欲望: {step_result['dominant_desire']}")
    
    print("\n5. 测试重置方法：")
    engine.desire_spectrum.reset()
    reset_stats = engine.desire_spectrum.get_stats()
    print(f"重置后主导欲望: {reset_stats['dominant_desire']}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
