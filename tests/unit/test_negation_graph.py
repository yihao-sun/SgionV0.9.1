import sys
import os

# 添加父目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine

# 创建引擎实例
engine = ExistenceEngine(vocab_size=10000)

print("=== 测试否定关系图增长 ===")

# 初始状态
print("\n初始状态:")
if hasattr(engine.fse, 'negation_graph'):
    print(f"否定关系图节点数: {len(engine.fse.negation_graph)}")
else:
    print("否定关系图不存在")

# 第一次输入
print("\n第一次输入: 苹果是水果")
response = engine.step("苹果是水果")
print(f"响应: {response}")
if hasattr(engine.fse, 'negation_graph'):
    print(f"否定关系图节点数: {len(engine.fse.negation_graph)}")

# 第二次输入
print("\n第二次输入: 苹果是水果")
response = engine.step("苹果是水果")
print(f"响应: {response}")
if hasattr(engine.fse, 'negation_graph'):
    print(f"否定关系图节点数: {len(engine.fse.negation_graph)}")

# 第三次输入
print("\n第三次输入: 苹果是水果")
response = engine.step("苹果是水果")
print(f"响应: {response}")
if hasattr(engine.fse, 'negation_graph'):
    print(f"否定关系图节点数: {len(engine.fse.negation_graph)}")

print("\n=== 测试完成 ===")
