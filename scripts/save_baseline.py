import json
import os
import sys
from datetime import datetime

# 添加父目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine
from tests.behavior.runner import BehaviorRunner

def _convert_to_json_serializable(obj):
    """将对象转换为JSON可序列化的类型"""
    import numpy as np
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def save_baseline():
    """运行完整测试套件并保存基线结果"""
    engine = ExistenceEngine(vocab_size=10000)
    runner = BehaviorRunner(engine)
    
    test_suite_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'behavior', 'philosophy_test_cases.json')
    results = runner.run_suite(test_suite_path)
    
    # 转换结果为JSON可序列化类型
    results_serializable = _convert_to_json_serializable(results)
    
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "philosophy_test_cases",
        "results": results_serializable,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r['passed']),
            "failed": sum(1 for r in results if not r['passed']),
            "consistency_score": sum(1 for r in results if r['passed']) / len(results) if results else 0
        }
    }
    
    baseline_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'baseline_results.json')
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    
    print(f"基线结果已保存到: {baseline_path}")
    print(f"总测试数: {baseline['summary']['total']}")
    print(f"通过测试数: {baseline['summary']['passed']}")
    print(f"一致性评分: {baseline['summary']['consistency_score']:.2%}")
    
    return baseline

if __name__ == "__main__":
    save_baseline()
