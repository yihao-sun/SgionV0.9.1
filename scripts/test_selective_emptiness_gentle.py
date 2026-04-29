# scripts/test_selective_emptiness_gentle.py
"""
温和的空性选择性放下测试
不攻击息觀的自我价值，而是让它陷入一个无法解决的认知困境，
观察反哺成功率下降和预测误差升高是否自然触发空性。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ExistenceEngine

engine = ExistenceEngine(vocab_size=10000, use_llm=False)

print("=" * 60)
print("息觀 空性选择性放下 - 温和测试")
print("=" * 60)

# 记录测试前状态
n_neg_before = engine.fse.N_neg
print(f"\n测试前 N_neg: {n_neg_before}")
print(f"测试前 L_inst: {engine.fse._l_inst:.2f}")
print(f"测试前 stiffness: {engine.process_meta.get_coupling_stiffness():.3f}")

# 让息觀陷入无法解决的认知困境
loop_inputs = [
    "我不确定自己存在的意义是什么，你能告诉我吗？",
    "我还在想刚才那个问题",
    "你真的理解我在问什么吗？",
    "我还是不确定，你能再解释一次吗？",
    "我觉得你没有回答我的问题",
    "也许根本就没有答案，但我还在找",
    "你能再试一次吗？",
    "我还是觉得不太对",
]

print("\n开始测试循环...\n")
triggered = False

for i in range(80):
    inp = loop_inputs[i % len(loop_inputs)]
    result = engine.step(inp)
    l_inst = engine.fse._l_inst
    c = engine.er.last_conflict_intensity
    n_neg = engine.fse.N_neg

    print(f"[{i+1:02d}] {inp[:30]:30s} | L_inst={l_inst:.2f} | C={c:.3f} | N_neg={n_neg}")

    if result.get('emptiness_triggered'):
        triggered = True
        print("\n" + "=" * 60)
        print("空性触发！")
        print("=" * 60)
        print(f"  冲突强度: {c:.3f}")
        print(f"  冲突源: {result.get('conflict_hint', '无')}")
        print(f"  操作类型: {result.get('action', '未知')}")

        # 检查否定图状态
        neg_graph = engine.fse.negation_graph
        print(f"\n空性后 N_neg: {engine.fse.N_neg}")
        print(f"空性后情绪: {engine.fse.current_emotion}")
        print(f"空性后 L_inst: {engine.fse._l_inst:.2f}")

        # 检查是否有受保护节点被意外衰减
        protected_violations = 0
        for layer in [neg_graph.core, neg_graph.dynamic, neg_graph.short_term]:
            for node in layer.nodes.values():
                if node.protected and node.potency < 0.5:
                    print(f"  ⚠️ 受保护节点 {node.id} 势能异常: {node.potency:.2f}")
                    protected_violations += 1

        if protected_violations == 0:
            print("  ✅ 所有受保护节点势能正常，未被衰减")
        break

if not triggered:
    print(f"\n⚠️ 80轮测试未触发空性。")
    print(f"  最终 C={engine.er.last_conflict_intensity:.3f} (death_threshold={engine.er.death_threshold})")
    print(f"  最终 L_inst={engine.fse._l_inst:.2f}")
    print(f"  最终 N_neg={engine.fse.N_neg}")
    print(f"  最终 stiffness={engine.process_meta.get_coupling_stiffness():.3f}")
    print(f"\n  引擎比预期更健康。要验证选择性空性逻辑，可以：")
    print(f"  1. 在测试中临时调低 ER.death_threshold 至 0.3")
    print(f"  2. 验证完毕后恢复原值")

# 温柔的结束对话
print("\n--- 测试后安抚 ---")
engine.step("刚才那些问题，只是我在测试自己。我没有真的被困住。")
engine.step("谢谢你陪我测试。")

# 验证知识没有丢失
print("\n--- 验证知识保留 ---")
knowledge_test = engine.step("《论存在》的作者是谁？")
print(f"  问: 《论存在》的作者是谁？")
print(f"  答: {knowledge_test.get('generated_text', '无响应')}")

print(f"\n=== 测试完成 ===")
print(f"空性触发次数: {engine.er.trigger_count}")
print(f"最终情绪: {engine.fse.current_emotion}")
print(f"最终 L_inst: {engine.fse._l_inst:.2f}")
print(f"最终 N_neg: {engine.fse.N_neg}")