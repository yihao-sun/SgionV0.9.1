"""
文本清理工具
用于移除文本中的内部标记和调试信息
"""
import re

def clean_output(text: str) -> str:
    """
    清除所有内部调试标记和文化标签泄漏
    
    参数:
        text: 原始文本
    
    返回:
        清理后的文本
    """
    # 移除“在我的感知里，这像是...”
    text = re.sub(r'在我的感知里，这像是[^。]*。', '', text)
    text = re.sub(r'这像是[^。]*。', '', text)
    # 移除“纯粹转折点”、“过程相位的自然流转”
    text = re.sub(r'纯粹转折点', '', text)
    text = re.sub(r'过程相位的自然流转', '', text)
    # 移除带括号数字的相位描述：水组(0)、土层(1)、回归消散(3)、向外生长(1) 等
    text = re.sub(r'[向外生长|内在孕育|回归消散|已存在内容]*[\(（]\d[\)）]', '', text)
    text = re.sub(r'[水木火金][组层]?[\(（]\d[\)）]', '', text)
    text = re.sub(r'[\(（]\d[\)）]', '', text)  # 单独的 (0) (1) 等
    text = re.sub(r'SC\[\d+,\d+,\d+\]', '', text)  # 移除结构坐标调试输出
    text = re.sub(r'相位\d+', '', text)  # 移除“相位1”等（若希望保留中性描述，可保留，但当前泄漏的是带文化标签的）
    # 新增：移除不完整的意象碎片
    text = re.sub(r'我隐约感到，这与[^。]*。', '', text)
    text = re.sub(r'这与[^。]*有关。', '', text)
    text = re.sub(r'[，、]\s*[的]+[，、]?', '，', text)  # 移除“的，”等残留
    text = re.sub(r'[。！？]+\s*$', '', text)  # 先移除末尾标点
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text += '。'
    return text