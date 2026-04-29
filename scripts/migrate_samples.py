#!/usr/bin/env python3
"""
微调样本 LPS 迁移脚本
功能：
1. 识别并删除旧微调样本（无房间标签或通过文本特征匹配）
2. 分类重新注入，每一条都携带完整的双房间标签
3. 保存更新后的 LPS 索引
"""

import os
import sys
import time
import numpy as np
import faiss
from typing import List, Dict, Set

# 确保可以导入引擎模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ExistenceEngine
from utils.logger import get_logger

logger = get_logger('migrate_samples')


def identify_old_sample_ids(lps) -> Set[int]:
    """
    识别旧微调样本的 ID 集合。
    判定条件（满足任一即视为旧样本）：
    1. tags 中没有 'subjective_room' 或 'objective_room'
    2. 文本以 'Q:' 或 '问题：' 开头（旧 QA 格式）
    """
    old_ids = set()
    for meta in lps.metadata:
        tags = meta.get('tags', {})
        text = meta.get('text', '')

        # 条件1：无房间标签
        has_room_tags = 'subjective_room' in tags and 'objective_room' in tags

        # 条件2：旧 QA 格式
        is_qa_format = text.strip().startswith(('Q:', '问题：', '用户：'))

        # 排除种子文本和语义词条
        is_seed = tags.get('source') == 'seed'
        is_semantic = tags.get('type') == 'semantic'

        if (not has_room_tags or is_qa_format) and not is_seed and not is_semantic:
            old_ids.add(meta['id'])

    return old_ids


def remove_samples_by_ids(lps, ids_to_remove: Set[int]):
    """
    从 LPS 中批量删除指定 ID 的条目，重建 FAISS 索引。
    """
    # 分离保留条目
    keep_metadata = []
    keep_embeddings = []
    for i, meta in enumerate(lps.metadata):
        if meta['id'] not in ids_to_remove:
            keep_metadata.append(meta)
            embed = lps.embeddings[i]
            if embed is not None:
                keep_embeddings.append(embed)

    removed_count = len(lps.metadata) - len(keep_metadata)
    logger.info(f"标记删除 {removed_count} 条旧样本，保留 {len(keep_metadata)} 条")

    if keep_embeddings:
        embeddings_array = np.array(keep_embeddings, dtype=np.float32)
        # 重建 FAISS 索引
        new_index = faiss.IndexHNSWFlat(lps.d_model, 32)
        new_index.hnsw.efConstruction = 200
        new_index.hnsw.efSearch = 64
        new_index.add(embeddings_array)
        lps.index = new_index
    else:
        # 全部被删除，创建空索引
        new_index = faiss.IndexHNSWFlat(lps.d_model, 32)
        new_index.hnsw.efConstruction = 200
        lps.index = new_index

    lps.metadata = keep_metadata
    lps.embeddings = keep_embeddings
    lps.id_counter = max([m['id'] for m in keep_metadata]) + 1 if keep_metadata else 0

    logger.info(f"索引重建完成，当前条目数: {len(lps.metadata)}")
    return removed_count


def inject_sample(lps, text: str, potency: float, tags: Dict, engine):
    """
    注入单条样本，自动附加双房间标签。
    """
    # 获取当前主观房间
    coord = engine.structural_coordinator.get_current_coordinate()
    tags['subjective_room'] = coord.as_tarot_code()
    tags['subjective_major'] = coord.major

    # 获取客观分类房间
    if hasattr(engine, 'objective_classifier'):
        tags['objective_room'] = engine.objective_classifier.classify(text)

    # 编码并注入
    embedding = engine.lps.encoder.encode([text])[0] if engine.lps.encoder else None
    node_id = engine.lps.add(text, embedding=embedding, potency=potency, tags=tags)
    return node_id


def migrate_core_facts(engine, facts: List[Dict]):
    """
    注入核心事实（《论存在》相关、息觀身份等）。
    facts 格式: [{'text': '...', 'tags': {...}}, ...]
    """
    count = 0
    for fact in facts:
        if 'text' not in fact:
            continue
        tags = fact.get('tags', {})
        tags.update({
            'type': 'core_fact',
            'source': 'migrated_sample',
            'protected': True
        })
        inject_sample(
            engine.lps,
            text=fact['text'],
            potency=0.9,
            tags=tags,
            engine=engine
        )
        count += 1
    logger.info(f"核心事实注入完成: {count} 条")
    return count


def migrate_honest_reports(engine, reports: List[str]):
    """
    注入诚实状态报告（改写为第一人称）。
    """
    count = 0
    for report in reports:
        # 去掉旧 QA 格式前缀，保留第一人称表达
        clean = report.strip()
        for prefix in ['A:', '回答：', '息觀：', '息观：']:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        if not clean:
            continue

        tags = {
            'type': 'honest_report',
            'source': 'migrated_sample',
        }
        inject_sample(
            engine.lps,
            text=clean,
            potency=0.5,
            tags=tags,
            engine=engine
        )
        count += 1
    logger.info(f"诚实报告注入完成: {count} 条")
    return count


def migrate_ontology_samples(engine, samples: List[Dict]):
    """
    注入存在论推演样本。
    """
    count = 0
    for sample in samples:
        if 'text' not in sample:
            continue
        clean = sample['text'].strip()
        for prefix in ['A:', '回答：', '息觀：']:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        if not clean:
            continue

        tags = sample.get('tags', {})
        tags.update({
            'type': 'ontology',
            'source': 'migrated_sample',
        })
        inject_sample(
            engine.lps,
            text=clean,
            potency=0.7,
            tags=tags,
            engine=engine
        )
        count += 1
    logger.info(f"存在论推演注入完成: {count} 条")
    return count


def migrate_daily_dialogues(engine, dialogues: List[str]):
    """
    注入日常对话样本。
    """
    count = 0
    for dialogue in dialogues:
        clean = dialogue.strip()
        for prefix in ['A:', '回答：', '息觀：']:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        if not clean:
            continue

        tags = {
            'type': 'sediment',
            'source': 'migrated_sample',
        }
        inject_sample(
            engine.lps,
            text=clean,
            potency=0.3,
            tags=tags,
            engine=engine
        )
        count += 1
    logger.info(f"日常对话注入完成: {count} 条")
    return count


def main():
    logger.info("=" * 60)
    logger.info("微调样本 LPS 迁移脚本启动")
    logger.info("=" * 60)

    # 1. 初始化引擎（轻量模式，不加载 LLM）
    logger.info("初始化引擎...")
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    lps = engine.lps

    # 2. 识别旧样本
    logger.info("识别旧微调样本...")
    old_ids = identify_old_sample_ids(lps)
    logger.info(f"找到 {len(old_ids)} 条需要迁移的旧样本")

    # 即使没有旧样本，也继续注入新样本
    start = time.time()
    if old_ids:
        # 3. 备份警告
        backup_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'lps_seed'
        )
        logger.warning(f"即将修改 LPS 索引。建议先手动备份以下文件：")
        logger.warning(f"  {backup_path}.index")
        logger.warning(f"  {backup_path}.parquet")
        logger.info("按 Enter 继续，或 Ctrl+C 取消...")
        try:
            input()
        except KeyboardInterrupt:
            logger.info("用户取消操作。")
            return

        # 4. 删除旧样本
        removed = remove_samples_by_ids(lps, old_ids)
        logger.info(f"删除完成: {removed} 条，耗时 {time.time()-start:.1f}s")
    else:
        logger.info("无旧样本需要迁移，直接注入新样本。")

    # 5. 加载原始微调数据并分类重新注入
    # 注意：这里需要根据你的实际微调数据文件路径和格式调整
    # 优先使用用户目录下的样本文件
    sample_dir = os.path.join(
        os.path.expanduser('~'), 'data', 'fine_tune_samples'
    )
    logger.info(f"样本目录: {sample_dir}")
    logger.info(f"目录是否存在: {os.path.exists(sample_dir)}")
    
    # 如果用户目录下不存在，使用默认目录
    if not os.path.exists(sample_dir):
        sample_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'fine_tune_samples'
        )
        logger.info(f"使用默认样本目录: {sample_dir}")
        logger.info(f"默认目录是否存在: {os.path.exists(sample_dir)}")

    # ========== 示例：从文件加载 ==========
    # 如果微调数据以 JSON 文件存储：
    import json

    files_categories = {
        'core_facts.json': 'core_fact',
        'honest_reports.json': 'honest_report',
        'ontology_samples.json': 'ontology',
        'daily_dialogues.json': 'dialogue',
    }

    total_injected = 0
    for filename, category in files_categories.items():
        filepath = os.path.join(sample_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"文件不存在，跳过: {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if category == 'core_fact':
            total_injected += migrate_core_facts(engine, data)
        elif category == 'honest_report':
            total_injected += migrate_honest_reports(engine, data)
        elif category == 'ontology':
            total_injected += migrate_ontology_samples(engine, data)
        elif category == 'dialogue':
            total_injected += migrate_daily_dialogues(engine, data)

    logger.info(f"共注入 {total_injected} 条样本")

    # 6. 保存
    lps.save(os.path.join(os.path.dirname(__file__), '..', 'data', 'lps_seed'))
    total_time = time.time() - start
    logger.info(f"迁移完成！总耗时 {total_time:.1f}s")
    logger.info(f"当前 LPS 条目数: {len(lps.metadata)}")


if __name__ == "__main__":
    main()