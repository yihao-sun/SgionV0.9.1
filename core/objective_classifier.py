class ObjectiveClassifier:
    def __init__(self, mapping_path="data/objective_classification.json"):
        import json
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
    
    def classify(self, user_input: str, generated_text: str = None) -> int:
        """返回六十四卦 room_id (0-63)"""
        # 初期基于关键词匹配
        for keyword, info in self.mapping.items():
            if keyword in user_input or (generated_text and keyword in generated_text):
                return (info['inner'] << 3) | info['outer']
        # 默认：坤+坤（绝对耦合）
        return 0