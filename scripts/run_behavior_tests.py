import sys
import os
import argparse

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import ExistenceEngine
from tests.behavior.runner import BehaviorRunner

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default=None, help='Specific test name to run')
    args = parser.parse_args()
    
    engine = ExistenceEngine(vocab_size=10000)
    runner = BehaviorRunner(engine)
    if args.test:
        runner.run_suite("tests/behavior/test_cases.json", [args.test])
    else:
        runner.run_suite("tests/behavior/test_cases.json")
