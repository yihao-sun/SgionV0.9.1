"""
感知模块 (Perception Module)
功能：加载意图分类器，提取 ContextVector，替代关键词匹配。
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from typing import Dict, List

class PerceptionModule:
    def __init__(self, model_path="models/intent_classifier"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.intent_mapping = {}
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        mapping_path = os.path.join(self.model_path, "intent_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.intent_mapping = json.load(f)

    def predict_intent(self, text: str) -> Dict:
        if not self.model:
            return {"intent": "GENERAL_CHAT", "confidence": 0.0}
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            intent = self.intent_mapping.get("id_to_intent", {}).get(str(pred_id), "GENERAL_CHAT")
            confidence = probs[pred_id].item()
        return {"intent": intent, "confidence": confidence}

    def predict_phase_from_image(self, image_path: str) -> List[float]:
        """从图像预测相位分布（长度4列表），模型未训练时返回均匀分布"""
        # 预留接口，待模型训练后实现
        return [0.25, 0.25, 0.25, 0.25]

    def predict_phase_from_audio(self, audio_path: str) -> List[float]:
        """从音频预测相位分布（长度4列表），模型未训练时返回均匀分布"""
        return [0.25, 0.25, 0.25, 0.25]
