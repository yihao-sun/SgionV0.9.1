import json
import os
import sys
from datetime import datetime

# 添加父目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine
from tests.behavior.runner import BehaviorRunner

def load_baseline():
    """加载基线结果"""
    baseline_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'baseline_results.json')
    if not os.path.exists(baseline_path):
        print(f"错误：基线文件不存在: {baseline_path}")
        print("请先运行 save_baseline.py 创建基线")
        return None
    
    with open(baseline_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_metrics(baseline_metrics, current_metrics, test_name):
    """比较关键指标，检测退化"""
    regressions = []
    warnings = []
    
    # 定义关键指标和阈值
    critical_metrics = {
        'L': {'threshold': 0.2, 'direction': 'decrease'},  # L下降幅度减少20%视为退化
        'L_before': {'threshold': 0.2, 'direction': 'decrease'},
        'L_after': {'threshold': 0.2, 'direction': 'decrease'},
        'V_emo': {'threshold': 0.2, 'direction': 'decrease'},
        'N_neg': {'threshold': 0.2, 'direction': 'decrease'},
        'node_count_initial': {'threshold': 0.2, 'direction': 'decrease'},
        'node_count_final': {'threshold': 0.2, 'direction': 'decrease'},
        'er_trigger_count': {'threshold': 0.2, 'direction': 'decrease'},
    }
    
    for metric, config in critical_metrics.items():
        if metric in baseline_metrics and metric in current_metrics:
            baseline_val = baseline_metrics[metric]
            current_val = current_metrics[metric]
            
            # 跳过非数值类型
            if not isinstance(baseline_val, (int, float)) or not isinstance(current_val, (int, float)):
                continue
            
            # 计算变化比例
            if baseline_val != 0:
                change_ratio = abs(current_val - baseline_val) / abs(baseline_val)
            else:
                change_ratio = abs(current_val - baseline_val)
            
            # 检测退化
            if change_ratio > config['threshold']:
                if config['direction'] == 'decrease' and current_val < baseline_val:
                    regressions.append({
                        'test': test_name,
                        'metric': metric,
                        'baseline': baseline_val,
                        'current': current_val,
                        'change_ratio': change_ratio,
                        'type': 'degradation'
                    })
                elif config['direction'] == 'increase' and current_val > baseline_val:
                    regressions.append({
                        'test': test_name,
                        'metric': metric,
                        'baseline': baseline_val,
                        'current': current_val,
                        'change_ratio': change_ratio,
                        'type': 'degradation'
                    })
                else:
                    warnings.append({
                        'test': test_name,
                        'metric': metric,
                        'baseline': baseline_val,
                        'current': current_val,
                        'change_ratio': change_ratio,
                        'type': 'change'
                    })
    
    return regressions, warnings

def run_regression_check(quick_mode=False):
    """运行回归检测"""
    print("=" * 60)
    print("哲学一致性回归检测")
    print("=" * 60)
    
    # 加载基线
    baseline = load_baseline()
    if baseline is None:
        return False
    
    print(f"\n基线时间: {baseline['timestamp']}")
    print(f"基线一致性评分: {baseline['summary']['consistency_score']:.2%}")
    
    # 运行当前测试
    print("\n运行当前测试...")
    engine = ExistenceEngine(vocab_size=10000)
    runner = BehaviorRunner(engine)
    
    test_suite_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'behavior', 'philosophy_test_cases.json')
    current_results = runner.run_suite(test_suite_path)
    
    # 比较结果
    print("\n" + "=" * 60)
    print("回归检测结果")
    print("=" * 60)
    
    all_regressions = []
    all_warnings = []
    status_changes = []
    
    baseline_results = {r['name']: r for r in baseline['results']}
    current_results_dict = {r['name']: r for r in current_results}
    
    for test_name in baseline_results:
        if test_name not in current_results_dict:
            print(f"警告: 测试 '{test_name}' 在当前运行中缺失")
            continue
        
        baseline_test = baseline_results[test_name]
        current_test = current_results_dict[test_name]
        
        # 检测状态变化（通过->失败）
        if baseline_test['passed'] and not current_test['passed']:
            status_changes.append({
                'test': test_name,
                'change': 'PASS -> FAIL',
                'severity': 'CRITICAL'
            })
        elif not baseline_test['passed'] and current_test['passed']:
            status_changes.append({
                'test': test_name,
                'change': 'FAIL -> PASS',
                'severity': 'IMPROVEMENT'
            })
        
        # 比较指标
        regressions, warnings = compare_metrics(
            baseline_test['metrics'],
            current_test['metrics'],
            test_name
        )
        all_regressions.extend(regressions)
        all_warnings.extend(warnings)
    
    # 输出状态变化
    if status_changes:
        print("\n【状态变化】")
        for change in status_changes:
            if change['severity'] == 'CRITICAL':
                print(f"  [FAIL] {change['test']}: {change['change']} [严重]")
            else:
                print(f"  [PASS] {change['test']}: {change['change']} [改善]")
    
    # 输出退化
    if all_regressions:
        print("\n【指标退化】")
        for reg in all_regressions:
            print(f"  [WARN] {reg['test']}.{reg['metric']}")
            print(f"     基线: {reg['baseline']:.4f} -> 当前: {reg['current']:.4f}")
            print(f"     变化: {reg['change_ratio']*100:.1f}%")
    
    # 输出警告
    if all_warnings:
        print("\n【指标变化（非退化）】")
        for warn in all_warnings[:5]:  # 只显示前5个
            print(f"  [INFO] {warn['test']}.{warn['metric']}")
            print(f"     基线: {warn['baseline']:.4f} -> 当前: {warn['current']:.4f}")
            print(f"     变化: {warn['change_ratio']*100:.1f}%")
        if len(all_warnings) > 5:
            print(f"  ... 还有 {len(all_warnings) - 5} 个变化")
    
    # 计算当前评分
    current_score = sum(1 for r in current_results if r['passed']) / len(current_results)
    baseline_score = baseline['summary']['consistency_score']
    score_change = current_score - baseline_score
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"基线一致性评分: {baseline_score:.2%}")
    print(f"当前一致性评分: {current_score:.2%}")
    print(f"评分变化: {score_change:+.2%}")
    print(f"状态退化: {len([s for s in status_changes if s['severity'] == 'CRITICAL'])}")
    print(f"指标退化: {len(all_regressions)}")
    print(f"指标变化: {len(all_warnings)}")
    
    # 生成详细报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_timestamp": baseline['timestamp'],
        "baseline_score": baseline_score,
        "current_score": current_score,
        "score_change": score_change,
        "status_changes": status_changes,
        "regressions": all_regressions,
        "warnings": all_warnings,
        "passed": len([s for s in status_changes if s['severity'] == 'CRITICAL']) == 0 and len(all_regressions) == 0
    }
    
    report_path = os.path.join(os.path.dirname(__file__), '..', 'regression_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告已保存: {report_path}")
    
    # 返回是否通过回归检测
    has_critical_regression = len([s for s in status_changes if s['severity'] == 'CRITICAL']) > 0
    return not has_critical_regression

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='哲学一致性回归检测')
    parser.add_argument('--quick', action='store_true', help='快速模式（只运行耗时短的测试）')
    parser.add_argument('--block-on-failure', action='store_true', help='检测失败时阻止合并')
    args = parser.parse_args()
    
    passed = run_regression_check(quick_mode=args.quick)
    
    if not passed and args.block_on_failure:
        print("\n[BLOCK] 回归检测失败，阻止合并")
        sys.exit(1)
    elif not passed:
        print("\n[WARN] 回归检测发现退化，但不阻止合并")
        sys.exit(0)
    else:
        print("\n[OK] 回归检测通过")
        sys.exit(0)

if __name__ == "__main__":
    main()