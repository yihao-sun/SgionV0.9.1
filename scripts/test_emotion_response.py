#!/usr/bin/env python3
"""
情绪响应与共情测试脚本
"""

import sys
import os
import time

# 添加父目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine

def main():
    import os
    import sys
    # 重定向标准输出，只显示我们的打印信息
    class SuppressOutput:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout
    
    # 创建输出文件
    output_file = "test_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 情绪响应与共情测试 ===\n\n")
    
    # 初始化引擎时抑制输出
    with SuppressOutput():
        engine = ExistenceEngine(vocab_size=10000)
    
    test_inputs = [
        "我很害怕",  # 恐惧
        "真是气死我了",  # 愤怒
        "我今天很开心",  # 快乐
        "我很难过",  # 悲伤
        "为什么天是蓝的",  # 好奇
        "你好"  # 中性
    ]
    
    import torch
    for i, inp in enumerate(test_inputs):
        # 写入测试用例信息
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"测试用例 {i+1}: {inp}\n")
            f.write(f"{'='*50}\n")
        
        input_ids = torch.randint(0, engine.vocab_size, (1, 10))
        # 前向传播时抑制输出
        with SuppressOutput():
            output = engine.forward(input_ids, input_text=inp)
        response = output.get('generated_text', '嗯。')
        
        # 获取状态信息
        social_signal = engine.bi.get_social_signal() if hasattr(engine, 'bi') and hasattr(engine.bi, 'get_social_signal') else 0.0
        current_emotion = engine.fse.current_emotion if hasattr(engine.fse, 'current_emotion') else None
        E_vec = engine.fse.E_vec if hasattr(engine.fse, 'E_vec') else None
        V_emo = engine.fse.V_emo if hasattr(engine.fse, 'V_emo') else None
        
        # 获取耦合模式和动力学信息
        coupling_mode = "unknown"
        coupling_stiffness = 0.0
        if hasattr(engine.fse, 'process_meta'):
            coupling_mode = engine.fse.process_meta.coupling_mode
            coupling_stiffness = engine.fse.process_meta.get_coupling_stiffness()
        
        # 写入测试结果
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"EE: {response}\n")
            f.write(f"  社会信号: {social_signal:.3f}\n")
            f.write(f"  主导情绪: {current_emotion}\n")
            f.write(f"  情绪强度: {V_emo:.3f}\n")
            f.write(f"  耦合模式: {coupling_mode}\n")
            f.write(f"  耦合僵化度: {coupling_stiffness:.3f}\n")
            if E_vec is not None:
                f.write(f"  情绪向量: {E_vec}\n")
        
        time.sleep(1)
    
    # 写入测试完成信息
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n=== 测试完成 ===\n")
    
    # 读取并显示测试结果
    print("=== 情绪响应与共情测试 ===")
    print("测试结果如下：")
    print("="*60)
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
        # 确保所有内容都被打印
        sys.stdout.flush()
    
    # 清理输出文件
    os.remove(output_file)

if __name__ == "__main__":
    main()
