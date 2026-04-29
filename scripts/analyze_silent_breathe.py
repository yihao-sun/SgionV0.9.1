#!/usr/bin/env python3
import json, sys

valences, emotions, n_negs = [], [], []
with open('data/silent_breathe_log.jsonl', 'r') as f:
    for line in f:
        r = json.loads(line)
        valences.append(r['valence'])
        emotions.append(r['emotion'])
        n_negs.append(r['N_neg'])

print(f"记录条数: {len(valences)}")
print(f"Valence 范围: {min(valences):+.3f} ~ {max(valences):+.3f}")
print(f"Valence 均值: {sum(valences)/len(valences):+.3f}")
print(f"情绪分布: { {e: emotions.count(e) for e in set(emotions)} }")
print(f"N_neg 范围: {min(n_negs)} ~ {max(n_negs)}")