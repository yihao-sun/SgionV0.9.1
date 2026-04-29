import json
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from datetime import datetime

# 添加父目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine
from tests.behavior.runner import BehaviorRunner

def plot_metric_trend(metric, results):
    """生成指标趋势图"""
    test_names = [r['name'] for r in results]
    metric_values = []
    
    for r in results:
        if metric in r['metrics']:
            metric_values.append(r['metrics'][metric])
        else:
            metric_values.append(0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_names, metric_values, marker='o', linestyle='-')
    plt.title(f'{metric} Trend Across Tests')
    plt.xlabel('Test Case')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{metric}_trend.png')
    plt.close()

def main():
    # 创建引擎实例
    engine = ExistenceEngine(vocab_size=10000)
    # 创建测试运行器
    runner = BehaviorRunner(engine)
    
    # 运行完整测试套件（包含15项测试）
    test_suite_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'behavior', 'test_cases.json')
    results = runner.run_suite(test_suite_path)
    
    # 生成报告
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    consistency_score = passed / total if total > 0 else 0
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "consistency_score": consistency_score,
        "details": results
    }
    
    # 转换 numpy 类型为原生 Python 类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 转换报告中的 numpy 类型
    report = convert_numpy_types(report)
    
    # 保存报告到 JSON 文件
    report_path = os.path.join(os.path.dirname(__file__), '..', 'philosophy_report.json')
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成指标趋势图
    for metric in ['L', 'V_emo', 'N_neg']:
        plot_metric_trend(metric, results)
    
    # 打印结果
    print(f"哲学一致性报告生成完成！")
    print(f"总测试数: {total}")
    print(f"通过测试数: {passed}")
    print(f"一致性评分: {consistency_score:.2%}")
    print(f"报告保存路径: {report_path}")

if __name__ == "__main__":
    main()