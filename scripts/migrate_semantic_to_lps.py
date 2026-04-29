import json
import pickle
import sys
import os
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.lps import LPS
from utils.config_loader import Config

def main():
    config = Config()
    lps = LPS(config)
    semantic_file = Path("data/semantic_entries.json")
    if not semantic_file.exists():
        print("无语义条目文件，跳过")
        return
    
    with open(semantic_file, 'r', encoding='utf-8') as f:
        old_entries = json.load(f)
    
    count = 0
    for kw, data in old_entries.items():
        tags = {
            'type': 'semantic',
            'keyword': kw,
            'phase_distribution': data.get('phase_distribution', {}),
            'source': data.get('source', 'seed'),
            'source_decay': data.get('source_decay', 1.0)
        }
        text = f"[SEMANTIC] {kw}"
        potency = data.get('confidence', 0.5)
        lps.add(text, potency=potency, tags=tags)
        count += 1
    
    with open('data/lps_seed.pkl', 'wb') as f:
        pickle.dump(lps, f)
    print(f"迁移完成，共 {count} 条语义条目")

if __name__ == '__main__':
    main()