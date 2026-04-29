#!/usr/bin/env python3
"""
知识蒸馏脚本：从 Qwen 模型批量提取常识，注入 LPS。
"""

import sys
sys.path.insert(0, '.')
import re
import time
from engine import ExistenceEngine

# 提示词模板
TEMPLATES = {
    'geography': "请用简洁的陈述句列举你知道的国家及其首都，每行一条，格式为\"[国家]的首都是[首都]\"。\n例如：\n中国的首都是北京。\n法国的首都是巴黎。\n...",
    'china': "请用简洁的陈述句列举中国的基本信息，包括首都、人口、面积等，每行一条，格式为\"[信息类别]是[信息值]\"。\n例如：\n中国的首都是北京。\n中国的人口约14亿。\n...",
    'science': "请用简洁的陈述句列举基础科学事实，每行一条，格式为\"[概念]是[定义或值]\"。\n例如：\n水的沸点是100摄氏度。\n光速是每秒约30万公里。\n...",
    'math': "请用简洁的陈述句列举小学数学知识，每行一条，格式为\"[算式]等于[结果]\"或\"[概念]是指[定义]\"。\n例如：\n1加1等于2。\n乘法口诀：二二得四。\n...",
    'identity': "请用简洁的陈述句描述你的身份和创造者，每行一条，格式为\"[实体]是[描述]\"。\n例如：\n我是Existence Engine。\n我的创造者是太翊豪与DeepSeek。\n...",
}

def extract_triplets(text: str) -> list:
    """从回答中提取三元组"""
    triplets = []
    # 模式1：X的首都是Y
    pattern_capital = re.compile(r'(.+?)的首都是(.+?)[。\n]')
    # 模式2：X是Y
    pattern_is = re.compile(r'(.+?)是(.+?)[。\n]')
    # 模式3：X等于Y
    pattern_equal = re.compile(r'(.+?)等于(.+?)[。\n]')
    # 模式4：我是X / 我的创造者是X
    pattern_identity = re.compile(r'(我|我的创造者)是(.+?)[。\n]')
    
    for match in pattern_capital.finditer(text):
        triplets.append((match.group(1).strip(), '首都', match.group(2).strip()))
    for match in pattern_is.finditer(text):
        triplets.append((match.group(1).strip(), '是', match.group(2).strip()))
    for match in pattern_equal.finditer(text):
        triplets.append((match.group(1).strip(), '等于', match.group(2).strip()))
    for match in pattern_identity.finditer(text):
        subj = "我" if match.group(1) == "我" else "我的创造者"
        triplets.append((subj, '是', match.group(2).strip()))
    return triplets

def main():
    engine = ExistenceEngine(vocab_size=10000, use_llm=True)
    
    for domain, prompt in TEMPLATES.items():
        print(f"蒸馏领域: {domain}")
        # 调用左脑 LLM 生成回答
        response = engine.response_generator._generate_with_llm(
            prompt, engine.fse, intent='KNOWLEDGE_QUERY', temperature=0.3
        )
        # 提取三元组
        triplets = extract_triplets(response)
        print(f"提取到 {len(triplets)} 个三元组")
        
        # 注入 LPS
        for subj, rel, obj in triplets:
            text = f"{subj}{rel}{obj}"
            tags = {
                'type': 'distilled',
                'domain': domain,
                'entity': subj,
                'relation': rel,
                'value': obj,
                'source': 'distilled'
            }
            # 编码并存入 LPS
            embedding = engine.lps.encoder.encode([text])[0] if engine.lps.encoder else None
            node_id = engine.lps.add(text, embedding, potency=0.7, tags=tags)
            print(f"  [+] {text}")
        
        time.sleep(1)  # 避免过快请求
    
    engine.lps.save("data/lps_seed")
    print("蒸馏完成，LPS 已保存。")

if __name__ == '__main__':
    main()
