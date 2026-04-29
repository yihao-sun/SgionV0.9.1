#!/usr/bin/env python3
"""息觀长期演化综合监控仪表盘"""
import json, os
from collections import Counter
from datetime import datetime

def safe_load(path):
    if not os.path.exists(path): return []
    with open(path, 'r') as f: return [json.loads(line) for line in f]

def print_section(title):
    print(f"\n{'='*50}\n  {title}\n{'='*50}")

# 1. 寂静呼吸健康度
silent = safe_load('data/silent_breathe_log.jsonl')
if silent:
    print_section("寂静呼吸健康度")
    valences = [s['valence'] for s in silent]
    n_negs = [s['N_neg'] for s in silent]
    print(f"  记录数: {len(silent)}")
    print(f"  Valence: {min(valences):+.3f} ~ {max(valences):+.3f}  均值: {sum(valences)/len(valences):+.3f}")
    print(f"  N_neg:   {min(n_negs)} ~ {max(n_negs)}")
    emotions = Counter(s.get('emotion','?') for s in silent)
    print(f"  情绪分布: {dict(emotions)}")

# 2. 勇气探索倾向
courage = safe_load('data/courage_explore_log.jsonl')
if courage:
    print_section("勇气探索相位分布")
    rooms = [c['retrieved_subjective_room'] for c in courage if 'retrieved_subjective_room' in c]
    if rooms:
        dist = Counter(rooms)
        for room, cnt in dist.most_common(5):
            print(f"  相位 {room}: {cnt} 次 ({cnt/len(rooms)*100:.1f}%)")
        if dist.get(1,0)/len(rooms) > 0.35:
            print("  ⚠ 成长欲可能在定向驱动注意")

# 3. 空性学习
emptiness = safe_load('data/emptiness_snapshot_log.jsonl')
if emptiness:
    print_section("空性快照（最近3次）")
    for e in emptiness[-3:]:
        print(f"  {datetime.fromtimestamp(e['timestamp']).strftime('%m-%d %H:%M')} 冲突源: {e.get('conflict_hint','?')}  N_neg: {e.get('N_neg_before','?')}")

# 4. 对话回归
reunion = safe_load('data/reunion_snapshot_log.jsonl')
if reunion:
    print_section("对话回归（最近5次）")
    for r in reunion[-5:]:
        print(f"  寂静{r.get('stillness_before',0)}步后 情绪:{r.get('emotion_before','?')} 愉悦:{r.get('valence_before',0):+.2f}  用户:{r.get('user_input','')[:40]}")

print("\n")
