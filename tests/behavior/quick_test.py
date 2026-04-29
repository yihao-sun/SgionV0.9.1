#!/usr/bin/env python3
"""
快速测试修复效果
"""

import sys
sys.path.insert(0, 'c:\\Users\\85971\\ExistenceEngine')

from core.negation_graph import LayeredNegGraph
import yaml

# 加载配置
with open('c:\\Users\\85971\\ExistenceEngine\\config\\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("=== 快速测试分层遗忘机制 ===")

# 创建分层否定关系图
graph = LayeredNegGraph(config)

# 添加短期层节点
print("\n1. 添加5个短期层节点")
for i in range(5):
    node_id = graph.add_negation(f"短期节点{i}", layer='short_term', initial_potency=2.0)
    print(f"  添加: {node_id}")

print(f"短期层节点数: {len(graph.short_term)}")

# 添加动态层节点
print("\n2. 添加5个动态层节点")
for i in range(5):
    node_id = graph.add_negation(f"动态节点{i}", layer='dynamic', initial_potency=2.0)
    print(f"  添加: {node_id}")

print(f"动态层节点数: {len(graph.dynamic)}")

# 模拟1200步
print("\n3. 模拟1200步衰减")
for step in range(1200):
    graph.decay_all()
    if step % 200 == 0:
        print(f"  步数 {step}: 短期层={len(graph.short_term)}, 动态层={len(graph.dynamic)}")

print(f"\n最终状态:")
print(f"  短期层节点数: {len(graph.short_term)}")
print(f"  动态层节点数: {len(graph.dynamic)}")

# 验证短期层是否为空
if len(graph.short_term) == 0:
    print("\n✅ 测试通过: 短期层节点已消失")
else:
    print(f"\n❌ 测试失败: 短期层仍有 {len(graph.short_term)} 个节点")
