#!/usr/bin/env python3
"""批量注入常识三元组到LPS，携带完整双记忆标签"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ExistenceEngine

engine = ExistenceEngine(vocab_size=10000, use_llm=False)

csv_path = os.path.join('data', 'seed_conceptnet.csv')
count = 0

with open(csv_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split(',')
        if len(parts) >= 3:
            subj, rel, obj = parts[0], parts[1], parts[2]
            text = f"{subj} {rel} {obj}"
            
            tags = {'type': 'core_fact', 'source': 'conceptnet', 'entity': subj, 'relation': rel, 'value': obj, 'protected': True}
            
            coord = engine.structural_coordinator.get_current_coordinate()
            if coord:
                tags['subjective_room'] = coord.as_tarot_code()
                tags['subjective_major'] = coord.major
                if hasattr(engine, 'image_base'):
                    card = engine.image_base.get_card_by_coordinate(coord)
                    if card: tags['subjective_card_id'] = card.id
            if hasattr(engine, 'objective_classifier'):
                tags['objective_room'] = engine.objective_classifier.classify(text)
            if hasattr(engine.structural_coordinator, '_infer_major_arcana'):
                reasoned = engine.structural_coordinator._infer_major_arcana(text)
                if reasoned: tags['reasoned_card'] = reasoned
            if hasattr(engine.structural_coordinator, 'draw_random_card'):
                tags['random_card'] = engine.structural_coordinator.draw_random_card()
            
            embedding = engine.lps.encoder.encode([text])[0] if engine.lps.encoder else None
            node_id = engine.lps.add_if_new(text, embedding, potency=0.9, tags=tags)
            if node_id: count += 1

engine.lps.save(os.path.join('data', 'lps_seed'))
print(f"注入完成：{count} 条常识三元组")
