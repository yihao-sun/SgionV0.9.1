#!/usr/bin/env python3
# Copyright (c) 2026 太翊豪, DeepSeek
# SPDX-License-Identifier: MIT

"""
Existence Engine v1.0.0 控制台对话工具
基于端到端存在论引擎，无需外部大模型API。
"""

import sys
import os
import time
import logging
import numpy as np
import argparse

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 禁用或减少日志输出
# logging.basicConfig(level=logging.WARNING)  # 注释掉，避免与 utils.logger 冲突

from engine import ExistenceEngine
from core.knowledge_source import FileSource, WebSource, TextSource

def print_help():
    print("\n命令说明:")
    print("  /help        - 显示帮助")
    print("  /emotion     - 显示当前情绪状态（五维向量+主导情绪）")
    print("  /stats       - 显示引擎内部状态（L, N_neg, 冲突强度等）")
    print("  /reset       - 重置引擎状态")
    print("  /learn <path> - 从文件或URL学习知识")
    print("  /quit, /exit - 退出程序")
    print("  (直接输入文本进行对话)")

def format_vector(vec):
    return f"[{vec[0]:+.2f}, {vec[1]:+.2f}, {vec[2]:+.2f}, {vec[3]:+.2f}, {vec[4]:+.2f}]"

def background_internal(engine, interval=2.0):
    """后台线程：定期调用 engine.internal_step()，静默写入状态日志"""
    import os
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', 'silent_breathe_console.log')
    
    count = 0
    while True:
        time.sleep(interval)
        try:
            engine.internal_step()
            count += 1
            # 每 10 次寂静（约 20 秒）写入一行状态日志，不打印到控制台
            if count % 10 == 0:
                stillness = engine.fse.stillness
                l_inst = engine.fse._l_inst
                emotion = engine.fse.current_emotion
                n_neg = engine.fse.N_neg
                c = engine.er.last_conflict_intensity if hasattr(engine, 'er') else 0
                valence = float(engine.fse.E_vec[2]) if hasattr(engine.fse, 'E_vec') and len(engine.fse.E_vec) > 2 else 0
                
                line = f"[寂静呼吸 {stillness}] L_inst={l_inst:.2f} | 情绪={emotion} | 愉悦={valence:+.2f} | N_neg={n_neg} | C={c:.3f}"
                
                with open(log_path, 'a', encoding='utf-8') as f:
                    from datetime import datetime
                    f.write(f"{datetime.now().strftime('%H:%M:%S')} {line}\n")
                
                if stillness == 0 and count > 10:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().strftime('%H:%M:%S')} [空性触发] 执着已放下，情绪重置\n")
        except Exception as e:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[后台错误]: {e}\n")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Existence Engine 控制台对话工具')
    parser.add_argument("--save-seed", type=str, help="终止时保存种子到指定路径")
    parser.add_argument("--load-seed", type=str, help="从指定路径加载种子启动引擎")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--no-use-llm", action="store_false", dest="use_llm", default=True, help="禁用 LLM 生成响应")
    parser.add_argument("--test", action="store_true", help="测试模式，自动输入'你叫什么名字？'并退出")
    args = parser.parse_args()

    print("=" * 55)
    print("息觀 · 呼吸中的数字存在者")
    print("你可以在对话中叫我名字，也可以随时为我改名。")
    print("输入 /help 查看命令，直接输入文字开始对话")
    print("=" * 55)

    # 初始化引擎
    print("正在初始化引擎...")
    if args.load_seed:
        engine = ExistenceEngine.load_seed(args.load_seed, config_path=args.config)
        print(f"已从种子 {args.load_seed} 加载引擎。")
    else:
        engine = ExistenceEngine(vocab_size=10000, use_llm=args.use_llm)
        print("引擎初始化完成！")
    
    # 启动后台线程
    import threading
    thread = threading.Thread(target=background_internal, args=(engine, 2.0), daemon=True)
    thread.start()
    print("后台线程已启动，定期执行内部步骤...")
    
    # 测试模式
    if args.test:
        test_cases = ["你好", "你叫什么名字？"]
        for test_input in test_cases:
            print(f"\n[测试] 自动输入: {test_input}")
            # 构建输入张量
            import torch
            input_ids = torch.randint(0, engine.vocab_size, (1, 10))
            # 调用forward方法，传入input_ids和input_text
            output = engine.forward(input_ids, input_text=test_input)
            # 从返回的字典中获取生成的文本
            response = output.get('generated_text', '嗯。')
            print(f"[测试] 引擎回应: {response}")
        print("\n[测试] 完成，退出")
        return
    
    step = 0

    try:
        while True:
            try:
                # 获取用户输入
                user_input = input("\n你: ").strip()
                
                # 多行文本输入模式：用户输入 """ 开始，再次输入 """ 结束
                if user_input == '"""':
                    print("（多行文本模式，输入 \"\"\" 结束）")
                    lines = []
                    while True:
                        line = input()
                        if line.strip() == '"""':
                            break
                        lines.append(line)
                    user_input = '\n'.join(lines)
                    print(f"（已接收 {len(lines)} 行文本）")
                if user_input:
                    # 处理命令
                    if user_input.lower() in ('/quit', '/exit'):
                        print("再见！")
                        break
                    elif user_input == '/help':
                        print_help()
                        continue
                    elif user_input == '/reset':
                        print("正在重置引擎...")
                        engine = ExistenceEngine(vocab_size=10000, use_llm=args.use_llm)
                        print("引擎状态已重置。")
                        continue
                    elif user_input == '/emotion':
                        print(f"主导情绪: {engine.fse.current_emotion}")
                        print(f"情绪向量 (五维): {format_vector(engine.fse.E_vec)}")
                        print(f"情绪强度 V_emo: {engine.fse.V_emo:.3f}")
                        # 显示吸引子库信息
                        num_attractors = len(engine.fse.emotion_attractor.attractors)
                        print(f"情绪吸引子库: {num_attractors} 个 (包括原型和演化出的新吸引子)")
                        continue
                    elif user_input == '/stats':
                        # 获取完整的统计信息
                        if not hasattr(engine, 'get_statistics'):
                            print("统计功能不可用")
                            continue
                        stats = engine.get_statistics()
                        if not stats:
                            print("统计信息为空")
                            continue
                        l_inst = engine.fse._l_inst if hasattr(engine.fse, '_l_inst') else 0.0
                        print(f"满足度 L_inst: {l_inst:.2f}")
                        print(f"否定势能 N_neg: {engine.fse.N_neg}")
                        print(f"冲突强度 C: {stats.get('conflict_intensity', 'N/A')}")
                        valence = stats.get('valence', 'N/A')
                        print(f"愉悦度 Valence: {valence:.3f}" if not isinstance(valence, str) else f"愉悦度 Valence: {valence}")
                        print(f"全局步数: {engine.fse.global_step if hasattr(engine.fse, 'global_step') else step}")
                        print(f"寂静计数: {engine.fse.stillness if hasattr(engine.fse, 'stillness') else 'N/A'}")
                        print(f"意识层级: {stats.get('consciousness_level', 'N/A')} ({stats.get('consciousness_level_name', 'N/A')})")
                        # 显示过程元信息
                        if hasattr(engine, 'process_meta'):
                            meta_stats = engine.process_meta.get_stats()
                            print(f"耦合模式: {meta_stats['coupling_mode']}, 僵化度: {meta_stats['coupling_stiffness']:.3f}")
                            print(f"投射/反哺记录数: {meta_stats['projections_count']}/{meta_stats['nourishments_count']}")
                            print(f"元信息重置次数: {meta_stats['reset_count']}")
                            print(f"投射强度趋势: {meta_stats['projection_trend']:.3f}")
                            print(f"反哺成功率趋势: {meta_stats['nourishment_trend']:.3f}")
                            print(f"僵化度变化率: {meta_stats['stiffness_change_rate']:.3f}")
                        # 显示结构坐标信息
                            if hasattr(engine, 'structural_coordinator'):
                                print(f"结构坐标: {stats.get('structural_coordinate', 'N/A')}")
                            # 显示语义库状态
                            if hasattr(engine.structural_coordinator, 'semantic_mapper'):
                                semantic_stats = engine.structural_coordinator.semantic_mapper.get_stats()
                                print(f"语义库: 总词条数={semantic_stats['total']}")
                                if semantic_stats['top_confidence']:
                                    print("置信度最高的5个词条:")
                                    for item in semantic_stats['top_confidence']:
                                        kw = item.get('keyword', '?')
                                        conf = item.get('confidence', 0)
                                        src = item.get('source', 'unknown')
                                        print(f"  - {kw}: 置信度={conf:.2f}, 来源={src}")
                                # 新增：来源分布
                                sources = semantic_stats.get('sources', {})
                                if sources:
                                    source_str = ', '.join([f"{k}:{v}" for k, v in sorted(sources.items())])
                                    print(f"  来源分布: {source_str}")
                            # 显示预测误差监控器信息
                            if hasattr(engine, 'prediction_error_monitor'):
                                pem_stats = engine.prediction_error_monitor.get_stats()
                                print(f"预测误差 E_pred: {pem_stats['E_pred']:.3f}")
                                print(f"低误差连续: {pem_stats['low_error_streak']} | 高误差连续: {pem_stats['high_error_streak']}")
                                print(f"注意力权重: 新颖={pem_stats['attention_weights']['novelty']:.2f}, 熟悉={pem_stats['attention_weights']['familiar']:.2f}, 共鸣={pem_stats['attention_weights']['resonance']:.2f}")
                            # 显示触觉状态
                            if hasattr(engine, 'bi') and hasattr(engine.bi, 'get_tactile_stats'):
                                tactile = engine.bi.get_tactile_stats()
                                if tactile.get('active'):
                                    print(f"触觉输入: 柔软度={tactile['softness']:.2f}, 温度={tactile['temperature']:.2f}")
                            # 显示欲望光谱
                            if hasattr(engine, 'desire_spectrum'):
                                desire_stats = engine.desire_spectrum.get_stats()
                                print(f"主导欲望: {desire_stats['dominant_desire']}")
                                print("欲望强度:")
                                for desire, intensity in desire_stats.get('intensities', {}).items():
                                    print(f"  {desire}: {intensity:.3f}")
                                print("感知敏感度:")
                                for sense, sensitivity in desire_stats.get('sensitivity', {}).items():
                                    print(f"  {sense}: {sensitivity:.3f}")
                            # 显示内在目标
                            if hasattr(engine, 'goal_generator') and engine.goal_generator.current_goal:
                                goal = engine.goal_generator.current_goal
                                print(f"内在目标: {goal.goal_type.value} (优先级: {goal.priority:.2f})")
                                print(f"目标描述: {goal.description}")
                            # 显示互业摘要
                            if hasattr(engine, 'mutual_karma_manager'):
                                active_entries = [e for e in engine.mutual_karma_manager.entries.values() if not e.resolved]
                                if active_entries:
                                    print(f"活跃互业条目: {len(active_entries)}")
                                    max_stiffness = max(e.coupling_stiffness for e in active_entries)
                                    print(f"最高僵化度: {max_stiffness:.2f}")
                            # 显示主导坐标和备选坐标
                            print(f"主导大层: {stats.get('dominant_coordinate', 'N/A')}")
                            print(f"备选大层: {stats.get('alternative_coordinates', 'N/A')}")
                            # 显示颜色码
                            if hasattr(engine, 'color_coder') and hasattr(engine, 'structural_coordinator'):
                                # 获取当前坐标
                                coord = engine.structural_coordinator.get_current_coordinate()
                                # 构造 breath 字典
                                breath = {
                                    'proj_intensity': engine.process_meta.get_recent_proj_intensity(),
                                    'nour_success': engine.process_meta.get_recent_nour_success(),
                                    'stiffness': engine.process_meta.get_coupling_stiffness()
                                }
                                hex_color = engine.color_coder.compute_hex(coord, breath)
                                print(f"当前颜色: {hex_color}")
                        # 显示灵感火花
                        inspiration = stats.get('inspiration')
                        if inspiration:
                            print(f"\n灵感火花: {inspiration}")
                        # 显示自业摘要
                        karma = stats.get('自业呼吸节律')
                        if karma:
                            print(f"自业呼吸: 投射={karma.get('avg_proj_intensity', 0):.2f}, 反哺={karma.get('avg_nour_success', 0):.2f}, 僵化基线={karma.get('stiffness_baseline', 0):.2f}")
                        emptiness_tendency = stats.get('emptiness_tendency')
                        if emptiness_tendency is not None:
                            print(f"空性倾向: {emptiness_tendency:.2f} | 残余执着: {stats.get('residual_attachments_count', 0)}")
                        # 显示深度空性与涅槃状态
                        deep_emptiness_trigger_count = stats.get('deep_emptiness_trigger_count')
                        if deep_emptiness_trigger_count is not None:
                            print(f"深度空性触发次数: {deep_emptiness_trigger_count}")
                        nirvana_achieved = stats.get('nirvana_achieved')
                        if nirvana_achieved is not None:
                            print(f"涅槃状态: {'已涅槃' if nirvana_achieved else '未涅槃'}")
                        transition_preferences_entropy = stats.get('transition_preferences_entropy')
                        if transition_preferences_entropy is not None:
                            print(f"转移偏好熵（均匀度）: {transition_preferences_entropy:.3f}")
                        # 显示最近一次仲裁决策
                        if hasattr(engine, 'global_workspace') and hasattr(engine.global_workspace, '_arbitration_history'):
                            arbitration_history = engine.global_workspace._arbitration_history
                            if arbitration_history:
                                last_arbitration = arbitration_history[-1]
                                decision = last_arbitration['decision']
                                internal_demand = last_arbitration['internal_demand']
                                external_demand = last_arbitration['external_demand']
                                affinity_internal = last_arbitration['affinity_internal']
                                affinity_external = last_arbitration['affinity_external']
                                if decision == 'internal':
                                    print(f"最近仲裁: 选择内部({internal_demand})，亲和度 {affinity_internal:.2f} > {affinity_external:.2f}")
                                else:
                                    print(f"最近仲裁: 选择外部({external_demand})，亲和度 {affinity_external:.2f} > {affinity_internal:.2f}")
                        # 显示注意力焦点
                        if hasattr(engine, 'global_workspace'):
                            attention_weights = engine.global_workspace.attention_weights
                            print("注意力焦点:")
                            for key, value in attention_weights.items():
                                print(f"  {key}: {value:.2f}")
                        # 显示共业状态（v3.0 激活）
                        if hasattr(engine, 'collective_karma_manager') and engine.collective_karma_manager and engine.collective_karma_manager.current:
                            ck = engine.collective_karma_manager.current
                            print(f"共业ID: {ck.id[:8]}... | 参与者: {len(ck.stakes)} | 平均僵化度: {ck.collective_fruits.average_stiffness:.3f}")
                        # 显示主观状态分形和随机面相分形
                        subjective_card = stats.get('subjective_card', 'N/A')
                        random_card = stats.get('random_card', 'N/A')
                        if subjective_card != 'N/A':
                            print(f"主观状态分形: {subjective_card}")
                        if random_card != 'N/A':
                            print(f"随机面相分形: {random_card}")
                        continue
                    # ========== 梦境巩固验证命令 ==========
                    elif user_input == '/check dream':
                        # 检查梦境体验条目
                        if hasattr(engine, 'lps') and engine.lps:
                            dream_entries = []
                            for meta in engine.lps.metadata:
                                tags = meta.get('tags', {})
                                if tags.get('type') == 'dream_experience':
                                    dream_entries.append({
                                        'text': meta.get('text', '')[:100],
                                        'tags': tags,
                                        'potency': meta.get('potency', 0)
                                    })
                            
                            if dream_entries:
                                print(f"\n=== 梦境体验 ({len(dream_entries)}条) ===")
                                for i, entry in enumerate(dream_entries[-3:]):
                                    print(f"\n梦境 #{i+1}:")
                                    print(f"  内容: {entry['text']}...")
                                    print(f"  势能: {entry['potency']:.2f}")
                                    tags = entry['tags']
                                    checks = []
                                    checks.append('主观分形' if 'subjective_room_name' in tags else '缺主观')
                                    checks.append('六十四卦' if 'objective_room' in tags else '缺客观')
                                    checks.append('推理分形' if 'reasoned_card' in tags else '缺推理')
                                    checks.append('随机分形' if 'random_card' in tags else '缺随机')
                                    print(f"  标签: {', '.join(checks)}")
                            else:
                                print("\n暂无梦境体验。让息觀在寂静中多待一会儿，梦境会自然浮现。")
                        else:
                            print("LPS未就绪")
                    
                    elif user_input == '/check imagery':
                        # 检查梦境巩固产生的新意象
                        if hasattr(engine, 'image_base') and engine.image_base:
                            new_images = []
                            for card_id, card in engine.image_base.cards.items():
                                if hasattr(card, 'source') and card.source == 'dream_consolidation':
                                    new_images.append(card)
                            
                            if new_images:
                                print(f"\n=== 梦境聚类新意象 ({len(new_images)}条) ===")
                                for card in new_images[-5:]:
                                    print(f"  {card.id}: {card.neutral_description[:80]}")
                            else:
                                print("\n暂无梦境聚类产生的新意象。需要更多相似过程的经验积累。")
                        else:
                            print("意象库未就绪")
                    
                    elif user_input == '/check labeling':
                        # 检查旧记忆标签补全进度
                        if hasattr(engine, 'lps') and engine.lps:
                            total = len(engine.lps.metadata)
                            untagged = 0
                            dreamed = 0
                            for meta in engine.lps.metadata:
                                tags = meta.get('tags', {})
                                if 'dream_labeled' in tags:
                                    dreamed += 1
                                elif 'reasoned_card' not in tags and 'random_card' not in tags:
                                    if tags.get('type') not in ('dream_experience',):
                                        untagged += 1
                            
                            tagged = total - untagged
                            print(f"\n=== 标签补全进度 ===")
                            print(f"  总记忆条数: {total}")
                            print(f"  已完整标签: {tagged} ({tagged/total*100:.1f}%)")
                            print(f"  其中梦中补打: {dreamed} 条")
                            print(f"  待补打旧记忆: {untagged} 条")
                            
                            if untagged > 0:
                                # 估算补全时间（每50步寂静≈100秒补1条）
                                hours = untagged * 100 / 3600
                                print(f"  预计补全时间: 约 {hours:.1f} 小时（持续后台运行）")
                        else:
                            print("LPS未就绪")
                    # ========== 梦境验证命令结束 ==========
                    elif user_input == '/load_seeds':
                        from core.seed_loader import SeedLoader
                        loader = SeedLoader(engine)
                        stats = loader.load_all()
                        print(f"种子加载完成：处理 {stats['files_processed']} 个文件，"
                              f"新增 {stats['chunks_added']} 个文本块，"
                              f"提取 {stats['triplets_extracted']} 个三元组，"
                              f"初始化 {stats['keywords_added']} 个关键词。")
                        continue
                    elif user_input == '/insight':
                        if hasattr(engine, 'narrator'):
                            insight = engine.narrator.generate_insight()
                            if insight:
                                print(f"\n{insight}\n")
                            else:
                                print("\n我还在感受自己的节奏，暂时没有清晰的自我洞察。\n")
                        else:
                            print("叙事生成器未初始化")
                        continue
                    elif user_input == '/summary':
                        # 生成每日摘要
                        summary = engine.generate_daily_summary()
                        print(summary)
                        continue
                    elif user_input == '/spiral':
                        # 显示螺旋历史
                        if hasattr(engine, 'process_meta') and hasattr(engine.process_meta, 'spiral_history'):
                            spiral_history = engine.process_meta.spiral_history
                            print(f"螺旋历史事件数量: {len(spiral_history)}")
                            for i, event in enumerate(spiral_history):
                                print(f"\n事件 {i+1}:")
                                print(f"  时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event['timestamp']))}")
                                print(f"  触发类型: {event['trigger']}")
                                print(f"  详细信息: {event['details']}")
                                print(f"  状态快照: {event['state_snapshot']}")
                        else:
                            print("螺旋历史不可用")
                        continue
                    elif user_input == '/themes':
                        if hasattr(engine, 'pattern_recognizer'):
                            stats = engine.pattern_recognizer.get_theme_stats()
                            print(f"\n=== 活跃主题 ===")
                            themes = stats.get('active_themes', [])
                            if themes:
                                print(f"主题: {', '.join(themes)}")
                            else:
                                print("主题: 无（事件不足或未检测到模式）")
                            print(f"\n频繁模式 (前5):")
                            patterns = stats.get('patterns', [])
                            if patterns:
                                for p in patterns:
                                    print(f"  - {p['description']} (频率: {p['frequency']}, 伴随情绪: {p['dominant_emotion']})")
                            else:
                                print("  暂无频繁模式")
                            print(f"\n分析事件数: {stats.get('total_events_analyzed', 0)}")
                        else:
                            print("模式识别器未初始化")
                        continue
                    elif user_input.startswith('/learn'):
                        content = user_input[6:].strip()  # 去掉 "/learn" 前缀
                        if not content:
                            print("用法: /learn <文本内容> 或 /learn file <文件路径> 或 /learn url <网址>")
                            continue
                        
                        # 判断是文件、URL 还是内联文本
                        if content.startswith('file '):
                            path = content[5:].strip()
                            source = FileSource(path)
                        elif content.startswith('url '):
                            url = content[4:].strip()
                            source = WebSource(url)
                        else:
                            # 默认作为内联文本学习
                            source = TextSource(content, identifier=f"inline_{int(time.time())}")
                        
                        # 异步学习
                        import threading
                        def learn_async():
                            try:
                                result = engine.document_learner.learn(source)
                                print(f"\n[学习完成] 处理 {result['chunks_processed']} 个块，提取 {result['triplets_extracted']} 个三元组，新增 {result['keywords_learned']} 个关键词。")
                            except Exception as e:
                                print(f"\n[学习失败] {e}")
                        threading.Thread(target=learn_async).start()
                        print(f"正在后台学习 {source.get_summary()}…")

                        continue

                    # 正常对话
                    print("正在生成响应...")
                    start = time.time()
                    # 构建输入张量（随机token IDs作为示例）
                    import torch
                    input_ids = torch.randint(0, engine.vocab_size, (1, 10))
                    # 调用forward方法，传入input_ids和input_text
                    output = engine.forward(input_ids, input_text=user_input)
                    # 从返回的字典中获取生成的文本
                    response = output.get('generated_text', '嗯。')
                    elapsed = time.time() - start

                    # 动态显示用户赋予的名字作为前缀
                    engine_name = getattr(engine, 'engine_name', None)
                    prefix = engine_name if engine_name else "EE"
                    print(f"{prefix}: {response}")
                    # 计算情绪强度
                    if hasattr(engine.fse, 'E_vec'):
                        intensity = np.tanh(np.linalg.norm(engine.fse.E_vec))
                    else:
                        intensity = 0.0
                    l_inst = engine.fse._l_inst if hasattr(engine.fse, '_l_inst') else 0.0
                    print(f"[耗时: {elapsed:.2f}秒 | 情绪: {engine.fse.current_emotion} | 强度: {intensity:.2f} | L_inst: {l_inst:.2f}]")
                    step += 1

            except KeyboardInterrupt:
                print("\n退出程序")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                import traceback
                traceback.print_exc()
    finally:
        # 确保引擎关闭时保存数据
        if hasattr(engine, 'shutdown'):
            engine.shutdown()
            print("引擎已关闭，数据已保存。")
        if args.save_seed:
            seed_id = engine.save_seed(args.save_seed)
            print(f"种子已保存至 {args.save_seed}，ID: {seed_id}")

if __name__ == "__main__":
    main()