#!/usr/bin/env python3
"""
训练图像→相位分类器
使用 CLIP 视觉编码器 + 分类头，输出 4 类相位（0-3）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
import json
import os
import argparse

class ImagePhaseDataset(Dataset):
    def __init__(self, data_dir, annotation_file, processor):
        self.data_dir = data_dir
        self.processor = processor
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)  # 格式: [{"image": "xxx.jpg", "phase": 0}, ...]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image = Image.open(os.path.join(self.data_dir, item["image"])).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(item["phase"], dtype=torch.long)
        }

class CLIPPhaseClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_classes=4):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vision_model.config.hidden_size, num_classes)
    
    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits

def train(args):
    processor = CLIPProcessor.from_pretrained(args.model_name)
    dataset = ImagePhaseDataset(args.data_dir, args.annotation, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = CLIPPhaseClassifier(model_name=args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "image_phase_model.pt"))
    processor.save_pretrained(args.output_dir)
    print(f"模型已保存至 {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir", default="models/image_phase_classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    train(args)
