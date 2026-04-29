"""
初始语义库构建脚本
功能：构建初始语义种子库，为语义相位映射器提供基础词汇和相位分布。
"""

import json
import os
from pathlib import Path


def load_basic_vocabulary():
    """加载基础词表
    
    Returns:
        List[str]: 基础词汇列表
    """
    # 常用汉语词汇列表
    basic_words = [
        # 规划相关
        "规划", "计划", "设计", "策划", "安排", "准备", "预期", "目标", "蓝图", "方案",
        # 执行相关
        "执行", "实施", "行动", "操作", "运行", "启动", "开展", "进行", "推进", "落实",
        # 分析相关
        "分析", "研究", "评估", "审查", "检查", "诊断", "解读", "理解", "思考", "洞察",
        # 总结相关
        "总结", "归纳", "汇总", "整理", "回顾", "反思", "评价", "结论", "收获", "经验",
        # 情绪相关
        "快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "期待", "满意", "失望", "焦虑",
        # 关系相关
        "合作", "交流", "沟通", "分享", "帮助", "支持", "理解", "尊重", "信任", "友谊",
        # 学习相关
        "学习", "研究", "探索", "发现", "了解", "掌握", "提升", "进步", "成长", "发展"
    ]
    return basic_words


def calculate_initial_distribution(word):
    """计算词语的初始相位分布
    
    Args:
        word: 词语
    
    Returns:
        Dict[int, float]: 相位概率分布
    """
    # 相位定义：
    # 0: 水相（内敛、准备、规划）
    # 1: 木相（外展、执行、成长）
    # 2: 火相（消耗、挑战、分析）
    # 3: 金相（边界、总结、稳定）
    
    # 关键词匹配规则
    phase_keywords = {
        0: ["规划", "计划", "设计", "策划", "安排", "准备", "预期", "目标", "蓝图", "方案", "思考", "洞察"],
        1: ["执行", "实施", "行动", "操作", "运行", "启动", "开展", "进行", "推进", "落实", "学习", "探索", "发现", "成长", "发展"],
        2: ["分析", "研究", "评估", "审查", "检查", "诊断", "解读", "理解", "挑战", "消耗"],
        3: ["总结", "归纳", "汇总", "整理", "回顾", "反思", "评价", "结论", "收获", "经验", "稳定", "边界", "信任", "友谊"]
    }
    
    # 初始化分布
    distribution = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    
    # 根据关键词匹配调整分布
    for phase, keywords in phase_keywords.items():
        if word in keywords:
            distribution[phase] = 0.6
            # 降低其他相位的概率
            for p in distribution:
                if p != phase:
                    distribution[p] = 0.1333  # (1 - 0.6) / 3
            break
    
    return distribution


def build_semantic_seed():
    """构建语义种子库
    
    Returns:
        Dict[str, Dict]: 语义种子库
    """
    words = load_basic_vocabulary()
    seed_data = {}
    
    for word in words:
        distribution = calculate_initial_distribution(word)
        seed_data[word] = {
            "phase_distribution": distribution,
            "confidence": 0.8  # 初始置信度
        }
    
    return seed_data


def save_seed_data(seed_data, output_path):
    """保存种子数据到文件
    
    Args:
        seed_data: 语义种子库
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seed_data, f, ensure_ascii=False, indent=2)
    
    print(f"语义种子库已保存到: {output_path}")
    print(f"共包含 {len(seed_data)} 个词条")


def main():
    """主函数"""
    print("=== 构建初始语义库 ===")
    
    # 构建语义种子库
    seed_data = build_semantic_seed()
    
    # 保存到数据目录
    output_path = Path(os.path.dirname(os.path.dirname(__file__))) / "data" / "semantic_seed.json"
    save_seed_data(seed_data, output_path)
    
    # 打印前几个词条作为示例
    print("\n示例词条:")
    for i, (word, data) in enumerate(list(seed_data.items())[:5]):
        print(f"{i+1}. {word}: {data['phase_distribution']}")


if __name__ == "__main__":
    main()