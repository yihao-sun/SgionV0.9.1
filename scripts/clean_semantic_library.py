#!/usr/bin/env python3
"""
语义库清理脚本
移除 keyword 为 None 的无效语义条目，重建 LPS 索引。
"""

import sys
import os
import shutil
import time
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ExistenceEngine


def main():
    print("=" * 50)
    print("息觀 语义库清理工具")
    print("=" * 50)

    # 1. 备份确认
    backup_dir = os.path.join('data', 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    index_path = os.path.join('data', 'lps_seed.index')
    parquet_path = os.path.join('data', 'lps_seed.parquet')

    backup_index = os.path.join(backup_dir, f'lps_seed_{timestamp}.index')
    backup_parquet = os.path.join(backup_dir, f'lps_seed_{timestamp}.parquet')

    print(f"\n备份文件到: {backup_dir}/")
    if os.path.exists(index_path):
        shutil.copy2(index_path, backup_index)
        print(f"  OK {index_path} -> {backup_index}")
    if os.path.exists(parquet_path):
        shutil.copy2(parquet_path, backup_parquet)
        print(f"  OK {parquet_path} -> {backup_parquet}")
    print("备份完成。")

    # 2. 初始化引擎
    print("\n初始化引擎...")
    engine = ExistenceEngine(vocab_size=10000, use_llm=False)
    lps = engine.lps

    # 3. 统计无效条目
    total = len(lps.metadata)
    valid_entries = []
    invalid_ids = []

    for i, meta in enumerate(lps.metadata):
        tags = meta.get('tags', {})
        if tags.get('type') == 'semantic':
            keyword = tags.get('keyword')
            if keyword is None or keyword == '' or keyword == 'None':
                invalid_ids.append(meta['id'])
            else:
                valid_entries.append((i, meta))
        else:
            valid_entries.append((i, meta))

    invalid_count = len(invalid_ids)
    semantic_total = sum(1 for m in lps.metadata if m.get('tags', {}).get('type') == 'semantic')
    valid_semantic = semantic_total - invalid_count

    print(f"\n当前状态:")
    print(f"  LPS 总条目: {total}")
    print(f"  语义库总条目: {semantic_total}")
    print(f"  无效条目 (keyword=None): {invalid_count}")
    print(f"  有效语义条目: {valid_semantic}")

    if invalid_count == 0:
        print("\n没有需要清理的无效条目。退出。")
        return

    # 4. 确认操作
    print(f"\n将删除 {invalid_count} 条无效语义条目，重建 LPS 索引。")
    confirm = input("确认操作？输入 yes 继续: ").strip().lower()
    if confirm != 'yes':
        print("操作已取消。")
        return

    # 5. 重建索引
    print("\n重建 LPS 索引...")
    start = time.time()

    # 提取保留的条目
    new_metadata = []
    new_embeddings = []
    for i, meta in valid_entries:
        new_metadata.append(meta)
        emb = lps.embeddings[i]
        if emb is not None:
            new_embeddings.append(emb)

    # 重建 FAISS 索引
    if new_embeddings:
        embeddings_array = np.array(new_embeddings, dtype=np.float32)
        new_index = faiss.IndexHNSWFlat(lps.d_model, 32)
        new_index.hnsw.efConstruction = 200
        new_index.hnsw.efSearch = 64
        new_index.add(embeddings_array)
        lps.index = new_index

    lps.metadata = new_metadata
    lps.embeddings = new_embeddings
    lps.id_counter = max([m['id'] for m in new_metadata]) + 1 if new_metadata else 0

    elapsed = time.time() - start

    # 6. 保存
    lps.save(os.path.join('data', 'lps_seed'))

    print(f"\n清理完成！耗时 {elapsed:.1f}s")
    print(f"  删除条目: {invalid_count}")
    print(f"  保留条目: {len(new_metadata)}")
    print(f"  备份文件: {backup_dir}/")
    print("\n建议重启引擎以加载更新后的索引。")


if __name__ == "__main__":
    main()
