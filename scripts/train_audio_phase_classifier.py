#!/usr/bin/env python3
"""
训练音频→相位分类器
使用 Wav2Vec2 编码器 + 分类头，输出 4 类相位（0-3）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import json
import os
import argparse

class AudioPhaseDataset(Dataset):
    def __init__(self, data_dir, annotation_file, processor, max_length=16000*10):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)  # [{"audio": "xxx.wav", "phase": 0}, ...]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        audio, sr = librosa.load(os.path.join(self.data_dir, item["audio"]), sr=16000)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "label": torch.tensor(item["phase"], dtype=torch.long)
        }

class Wav2Vec2PhaseClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=4):
        super().__init__()
        self.audio_model = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Linear(self.audio_model.config.hidden_size, num_classes)
    
    def forward(self, input_values):
        outputs = self.audio_model(input_values=input_values)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(pooled)
        return logits

def train(args):
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    dataset = AudioPhaseDataset(args.data_dir, args.annotation, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: {
        "input_values": torch.stack([x["input_values"] for x in b]),
        "label": torch.stack([x["label"] for x in b])
    })
    
    model = Wav2Vec2PhaseClassifier(model_name=args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "audio_phase_model.pt"))
    processor.save_pretrained(args.output_dir)
    print(f"模型已保存至 {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--model_name", default="facebook/wav2vec2-base")
    parser.add_argument("--output_dir", default="models/audio_phase_classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    train(args)
