#!/usr/bin/env python3
"""测试图像和音频相位分类器"""

import torch
from transformers import CLIPProcessor, Wav2Vec2Processor
from scripts.train_image_phase_classifier import CLIPPhaseClassifier
from scripts.train_audio_phase_classifier import Wav2Vec2PhaseClassifier
from PIL import Image
import librosa
import argparse

PHASE_NAMES = {0: "水", 1: "木", 2: "火", 3: "金"}

def test_image_classifier(model_path, image_path):
    """测试图像相位分类器"""
    # 加载处理器和模型
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPPhaseClassifier()
    model.load_state_dict(torch.load(f"{model_path}/image_phase_model.pt"))
    model.eval()
    
    # 处理图像
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # 预测相位
    with torch.no_grad():
        logits = model(inputs["pixel_values"])
        pred = torch.argmax(logits, dim=-1).item()
    
    print(f"图像预测相位: {pred} ({PHASE_NAMES[pred]})")
    return pred

def test_audio_classifier(model_path, audio_path):
    """测试音频相位分类器"""
    # 加载处理器和模型
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2PhaseClassifier()
    model.load_state_dict(torch.load(f"{model_path}/audio_phase_model.pt"))
    model.eval()
    
    # 处理音频
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    
    # 预测相位
    with torch.no_grad():
        logits = model(inputs["input_values"])
        pred = torch.argmax(logits, dim=-1).item()
    
    print(f"音频预测相位: {pred} ({PHASE_NAMES[pred]})")
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_model", default="models/image_phase_classifier")
    parser.add_argument("--audio_model", default="models/audio_phase_classifier")
    parser.add_argument("--image", help="测试图像路径")
    parser.add_argument("--audio", help="测试音频路径")
    args = parser.parse_args()
    
    if args.image:
        test_image_classifier(args.image_model, args.image)
    if args.audio:
        test_audio_classifier(args.audio_model, args.audio)

if __name__ == "__main__":
    main()
