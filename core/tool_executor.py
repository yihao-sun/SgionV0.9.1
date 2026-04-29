import re

class ToolExecutor:
    def __init__(self, config=None):
        self.enable_calculator = True
        self.config = config
    
    def detect_tool_request(self, user_input: str):
        """检测是否为工具请求，返回 (tool_name, query) 或 (None, None)"""
        # 计算器检测：包含数字和运算符
        if re.search(r'[\d一二三四五六七八九十百千万亿]+', user_input) and \
           re.search(r'[\+\-\*/×÷加減乘除等于多少计算]', user_input):
            return ('calculator', user_input)
        return (None, None)
    
    def execute(self, tool_name: str, query: str) -> str:
        if tool_name == 'calculator':
            return self._calculate(query)
        return None
    
    def _calculate(self, expr: str) -> str:
        # 中文数字转换（简化版，可扩展）
        cn_num = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10}
        for cn, num in cn_num.items():
            expr = expr.replace(cn, str(num))
        expr = expr.replace('加', '+').replace('减', '-').replace('乘', '*').replace('除', '/')
        expr = re.sub(r'[^0-9+\-*/×÷().]', '', expr)
        expr = expr.replace('×', '*').replace('÷', '/')
        try:
            result = eval(expr, {"__builtins__": None}, {})
            return f"计算结果是 {result}。"
        except:
            return "这个算式我暂时算不出来，你可以换一种问法试试。"
