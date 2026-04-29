import pytest
import os
from core.structural_coordinator import StructuralCoordinate
from core.image_base import ImageBase
from core.dual_path_memory import DualPathMemory

def test_snapshot_storage_and_retrieval():
    # 获取正确的文件路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'tarot_cards.json')
    image_base = ImageBase(data_path=data_path)
    memory = DualPathMemory(image_base=image_base)
    coord1 = StructuralCoordinate(1, 2, 1)
    breath1 = {'proj_intensity': 0.7, 'nour_success': 0.3, 'stiffness': 0.2}
    memory.store_snapshot(coord1, coord1, breath1, "测试快照1")

    coord2 = StructuralCoordinate(1, 2, 2)
    breath2 = {'proj_intensity': 0.8, 'nour_success': 0.4, 'stiffness': 0.1}
    memory.store_snapshot(coord2, coord2, breath2, "测试快照2")

    # 检索与 coord1 高度相似的快照
    results = memory.contemplative_retrieval(coord1, breath1, top_k=5)
    assert len(results) >= 1
    # 第一个结果应是与 coord1 完全匹配的快照（共鸣度最高）
    assert results[0][0].summary == "测试快照1"