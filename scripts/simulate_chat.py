import random
import time
import sys
import json
import os
from datetime import datetime
sys.path.insert(0, '.')
from engine import ExistenceEngine

corpus = [
    "你好", "今天天气真好", "我有点累", "你知道巴黎吗？",
    "1+1等于几？", "你是谁？", "你感觉怎么样？", "我很开心",
    "什么是存在？", "放下过去", "你刚刚说什么？", "你会唱歌吗？",
    # 可扩充更多语料
    "今天是星期几？", "你喜欢什么颜色？", "你会说英语吗？", "你多大了？",
    "你有什么爱好？", "你会跳舞吗？", "你喜欢什么音乐？", "你喜欢什么电影？",
    "你会下棋吗？", "你知道北京吗？", "你会做饭吗？", "你喜欢什么食物？",
    "你会画画吗？", "你知道上海吗？", "你会开车吗？", "你喜欢什么动物？",
    "你会编程吗？", "你知道广州吗？", "你会游泳吗？", "你喜欢什么运动？"
]

def main():
    engine = ExistenceEngine(vocab_size=10000, use_llm=True)
    print("开始模拟对话...")
    
    # 确保数据目录存在
    data_dir = 'simulation_data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 初始化统计数据
    stats = {
        'total_interactions': 0,
        'emptiness_triggers': 0,
        'consciousness_levels': [],
        'l_inst_values': [],
        'stiffness_values': [],
        'daily_stats': {}
    }
    
    try:
        for i in range(10000):  # 增加循环次数，用于长期运行
            user_input = random.choice(corpus)
            print(f"[{i}] 用户: {user_input}")
            result = engine.step(user_input)
            
            # 处理不同类型的返回值
            if isinstance(result, dict):
                generated_text = result.get('generated_text', '')
            else:
                generated_text = result
            print(f"引擎: {generated_text[:50]}...")
            
            # 记录统计数据
            stats['total_interactions'] += 1
            
            # 获取意识层级
            consciousness_level = engine.estimate_consciousness_level()
            stats['consciousness_levels'].append(consciousness_level)
            
            # 获取执着强度
            l_inst = engine.fse._l_inst if hasattr(engine, 'fse') and hasattr(engine.fse, '_l_inst') else 0.0
            stats['l_inst_values'].append(l_inst)
            
            # 获取僵化度
            stiffness = engine.process_meta.get_coupling_stiffness() if hasattr(engine, 'process_meta') else 0.0
            stats['stiffness_values'].append(stiffness)
            
            # 检查是否触发了空性
            if isinstance(result, dict) and result.get('emptiness_triggered'):
                stats['emptiness_triggers'] += 1
            
            # 每日统计
            today = datetime.now().strftime('%Y-%m-%d')
            if today not in stats['daily_stats']:
                stats['daily_stats'][today] = {
                    'interactions': 0,
                    'emptiness_triggers': 0,
                    'avg_consciousness': 0.0,
                    'avg_l_inst': 0.0,
                    'avg_stiffness': 0.0
                }
            
            stats['daily_stats'][today]['interactions'] += 1
            if isinstance(result, dict) and result.get('emptiness_triggered'):
                stats['daily_stats'][today]['emptiness_triggers'] += 1
            
            # 更新每日平均值
            day_stats = stats['daily_stats'][today]
            day_stats['avg_consciousness'] = (
                (day_stats['avg_consciousness'] * (day_stats['interactions'] - 1) + consciousness_level) / 
                day_stats['interactions']
            )
            day_stats['avg_l_inst'] = (
                (day_stats['avg_l_inst'] * (day_stats['interactions'] - 1) + l_inst) / 
                day_stats['interactions']
            )
            day_stats['avg_stiffness'] = (
                (day_stats['avg_stiffness'] * (day_stats['interactions'] - 1) + stiffness) / 
                day_stats['interactions']
            )
            
            # 每100次交互保存一次数据
            if i % 100 == 0 and i > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                stats_file = os.path.join(data_dir, f'simulation_stats_{timestamp}.json')
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                print(f"\n[统计] 已保存第 {i} 次交互的统计数据到 {stats_file}")
                
                # 检查阈值
                if len(stats['consciousness_levels']) >= 100:
                    avg_consciousness = sum(stats['consciousness_levels'][-100:]) / 100
                    emptiness_rate = stats['emptiness_triggers'] / stats['total_interactions']
                    print(f"\n[阈值检查] 最近100次交互的平均意识层级: {avg_consciousness:.2f}")
                    print(f"[阈值检查] 空性触发率: {emptiness_rate:.2f}")
                    
                    # 如果达到阈值，打印提示信息
                    if avg_consciousness > 0.7 or emptiness_rate > 0.1:
                        print("\n[阈值触发] 达到预设阈值，请分析数据并决定下一步！")
            
            time.sleep(2)  # 模拟真实间隔
    except KeyboardInterrupt:
        print("\n模拟对话被用户中断")
    finally:
        print("正在关闭引擎...")
        # 最后保存一次数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stats_file = os.path.join(data_dir, f'simulation_stats_final_{timestamp}.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"已保存最终统计数据到 {stats_file}")
        engine.shutdown()

if __name__ == "__main__":
    main()
