import sys
sys.path.append('.')
from core.process_meta import ProcessMetaInfo

def test_process_meta():
    p = ProcessMetaInfo(max_history=50)
    
    # 模拟投射
    for i in range(10):
        p.record_projection(intensity=0.8, target_text=f"target_{i}")
    # 模拟反哺
    for i in range(10):
        p.record_nourishment(source_text=f"source_{i}", success=(i%2==0))
    
    print("初始耦合模式:", p.coupling_mode)
    print("僵化度:", p.get_coupling_stiffness())
    
    # 重置
    p.reset_coupling(keep_recent=5)
    print("重置后耦合模式:", p.coupling_mode)
    print("重置后投射记录数:", len(p.projections))
    print("重置后反哺记录数:", len(p.nourishments))
    print("重置次数:", p.reset_count)
    
    assert p.coupling_mode == "balanced"
    assert len(p.projections) <= 5
    print("测试通过")

if __name__ == "__main__":
    test_process_meta()
