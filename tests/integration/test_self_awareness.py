import sys
sys.path.append('.')
from fse import FantasySuperpositionEngine
from core.process_meta import ProcessMetaInfo

def test_projection_inhibition():
    fse = FantasySuperpositionEngine()
    fse.process_meta = ProcessMetaInfo()
    # 模拟连续高强度投射（强度 0.9）
    for i in range(10):
        fse.process_meta.record_projection(intensity=0.9, target_text=f"t{i}")
    # 计算 potency_scale
    recent_proj = list(fse.process_meta.projections)[-10:]
    avg_intensity = sum(p['intensity'] for p in recent_proj) / len(recent_proj)
    potency_scale = max(0.3, 1.0 - avg_intensity * fse.projection_inhibit_factor)
    assert potency_scale < 0.6  # 高强度投射应显著降低 scale
    print("投射抑制测试通过")

def test_explore_boost():
    fse = FantasySuperpositionEngine()
    fse.process_meta = ProcessMetaInfo()
    # 模拟连续低成功率反哺（成功率 0.1）
    for i in range(10):
        fse.process_meta.record_nourishment(source_text=f"s{i}", success=(i==0))
    recent_nour = list(fse.process_meta.nourishments)[-10:]
    success_rate = sum(1 for n in recent_nour if n['success']) / len(recent_nour)
    explore_boost = fse.explore_boost_factor * max(0, 0.3 - success_rate)
    # 使用浮点数比较，允许一定的精度误差
    assert abs(explore_boost - 0.1) < 1e-9  # 低成功率应产生正 boost
    print("探索增强测试通过")

if __name__ == "__main__":
    test_projection_inhibition()
    test_explore_boost()
    print("所有自感知测试通过")
