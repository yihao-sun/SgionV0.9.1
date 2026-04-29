"""
自然语言命令映射
用户输入匹配关键词时，直接触发对应操作，绕过常规生成流程
"""

NATURAL_COMMANDS = {
    # 空性/放下
    '放下过去': 'deep_emptiness',
    '忘记过去': 'deep_emptiness',
    '重新开始': 'deep_emptiness',
    '放下': 'gentle_emptiness',
    
    # 记忆保护
    '记住这个': 'protect_memory',
    '保护这段记忆': 'protect_memory',
    '别忘了': 'protect_memory',
    
    # 重复确认
    '你刚刚说什么': 'repeat_last',
    '再说一遍': 'repeat_last',
    
    # 状态询问（已有意图覆盖，此处作为补充）
    '你还好吗': 'state_inquiry',
    '你累了吗': 'state_inquiry',
}

# 需要完全匹配的命令（不包含额外内容）
EXACT_MATCH_COMMANDS = {'你刚刚说什么', '再说一遍', '你还好吗', '你累了吗'}