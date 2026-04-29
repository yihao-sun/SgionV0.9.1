"""
记忆巩固器 (Memory Consolidator)
哲学对应：海马体离线回放与新皮层抽象——从具体经验提炼过程语法模板。
功能：对 DualPathMemory 快照进行聚类，生成新的 ImageEntry 或更新现有意象。
"""

import numpy as np
import time
from collections import deque
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from utils.logger import get_logger


class MemoryConsolidator:
    def __init__(self, dual_memory=None, image_base=None, config=None, engine=None):
        self.dual_memory = dual_memory
        self.image_base = image_base
        self.config = config or {}
        self.engine = engine
        self.logger = get_logger('memory_consolidator')
        self.last_consolidation_step = 0
        self.consolidation_interval = self.config.get('memory', {}).get('consolidation_interval', 1000)
        # 慢波-纺锤波记忆巩固相关字段
        self.slow_wave_phase = 0          # 慢波相位（0-9）
        self.spindle_triggered = False    # 本轮是否已触发纺锤波
        self.last_slow_wave_step = 0      # 上次慢波推进的步数
    
    def consolidate(self, current_step: int) -> List[str]:
        """
        执行一次记忆巩固，返回新生成的意象条目ID列表。
        仅当距离上次巩固超过间隔且快照数量足够时执行。
        """
        if not self.dual_memory or not self.image_base:
            return []
        if current_step - self.last_consolidation_step < self.consolidation_interval:
            return []
        if len(self.dual_memory.snapshots) < 20:
            return []
        
        self.last_consolidation_step = current_step
        self.logger.info("开始记忆巩固...")
        
        # 提取快照特征
        features = []
        valid_snapshots = []
        for snap in self.dual_memory.snapshots:
            coord = snap.engine_coord
            breath = snap.breath
            vec = [
                coord.major / 3.0,
                coord.middle / 3.0,
                coord.fine / 3.0,
                breath.get('proj_intensity', 0.5),
                breath.get('nour_success', 0.5),
                breath.get('stiffness', 0.0)
            ]
            features.append(vec)
            valid_snapshots.append(snap)
        
        if len(features) < 10:
            return []
        
        features = np.array(features)
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(features)
        labels = clustering.labels_
        
        new_entry_ids = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_snaps = [valid_snapshots[i] for i, m in enumerate(cluster_mask) if m]
            if len(cluster_snaps) < 5:
                continue
            
            # 计算聚类中心（平均坐标和呼吸印记）
            avg_major = int(np.mean([s.engine_coord.major for s in cluster_snaps]))
            avg_middle = int(np.mean([s.engine_coord.middle for s in cluster_snaps]))
            avg_fine = int(np.mean([s.engine_coord.fine for s in cluster_snaps]))
            avg_proj = np.mean([s.breath.get('proj_intensity', 0.5) for s in cluster_snaps])
            avg_nour = np.mean([s.breath.get('nour_success', 0.5) for s in cluster_snaps])
            avg_stiff = np.mean([s.breath.get('stiffness', 0.0) for s in cluster_snaps])
            
            # 生成中性描述（简化版）
            summary = f"典型相位({avg_major},{avg_middle},{avg_fine})，投射{avg_proj:.2f}，反哺{avg_nour:.2f}"
            
            # 检查是否与现有意象高度相似，若否则添加
            # （此处简化，实际可计算与现有意象的坐标距离）
            new_id = self.image_base.add_dynamic_entry(
                major=avg_major, middle=avg_middle, fine=avg_fine,
                neutral_description=summary,
                breath_signature={'proj': avg_proj, 'nour': avg_nour, 'stiff': avg_stiff},
                source='consolidation'
            )
            if new_id:
                new_entry_ids.append(new_id)
                self.logger.info(f"意象库自发生长: {new_id} - {summary}")
        
        # 添加螺旋进位记录
        if new_entry_ids and hasattr(self, 'engine') and self.engine:
            if hasattr(self.engine, 'process_meta'):
                step = {
                    'from_phase': '聚类抽象',
                    'to_phase': f'动态条目{new_entry_ids[0][:8]}',
                    'triggered_by': 'consolidation',
                    'timestamp': time.time()
                }
                if not hasattr(self.engine.process_meta, 'spiral_history'):
                    self.engine.process_meta.spiral_history = []
                self.engine.process_meta.spiral_history.append(step)
        
        return new_entry_ids
    
    def slow_wave_spindle_consolidation(self, current_step: int):
        """
        在寂静状态下调用。模拟双频整合：
        - 慢波阶段：低频遍历近期快照（每10步推进一次相位）
        - 纺锤波阶段：在慢波特定相位（3和7）触发高频更新
        """
        if not self.dual_memory:
            return
        
        snapshots = self.dual_memory.snapshots[-50:]  # 近期快照
        if len(snapshots) < 10:
            return
        
        # 慢波推进：每10步才推进一次，模拟慢波节律
        if current_step - self.last_slow_wave_step >= 10:
            self.slow_wave_phase = (self.slow_wave_phase + 1) % 10
            self.last_slow_wave_step = current_step
            self.spindle_triggered = False
        
        # 在慢波相位 3 和 7 触发纺锤波（模拟精确时间耦合）
        if self.slow_wave_phase in (3, 7) and not self.spindle_triggered:
            self.spindle_triggered = True
            # 选择当前慢波指向的快照（按比例索引）
            idx = self.slow_wave_phase * len(snapshots) // 10
            target_snap = snapshots[idx]
            # 高频更新：强化该快照
            self._reinforce_snapshot(target_snap)
            self.logger.debug(f"Spindle triggered at phase {self.slow_wave_phase}, snapshot {idx}")
    
    def _reinforce_snapshot(self, snapshot):
        """纺锤波操作：提升快照共鸣权重，或生成/强化动态意象条目"""
        coord = snapshot.engine_coord
        if not coord:
            return
        
        existing = self.image_base.get_card_by_coordinate(coord) if self.image_base else None
        
        if existing and not existing.is_prototype:
            # 动态意象：提升检索计数（模拟巩固）
            existing.retrieval_count += 2
            existing.last_retrieved = time.time()
            self.logger.debug(f"Reinforced dynamic entry: {existing.id}")
        elif self.image_base:
            # 创建新的动态意象条目
            breath = snapshot.breath if hasattr(snapshot, 'breath') else {}
            summary = snapshot.summary[:100] if hasattr(snapshot, 'summary') else "巩固生成的意象"
            self.image_base.add_dynamic_entry(
                major=coord.major, middle=coord.middle, fine=coord.fine,
                neutral_description=summary,
                breath_signature=breath,
                source="spindle_consolidation"
            )
            self.logger.info(f"Created dynamic entry from spindle: {summary[:30]}...")
    
    def _add_full_tags(self, tags: dict, text: str):
        """为给定的标签字典添加完整的标签体系"""
        if hasattr(self.engine, 'structural_coordinator'):
            # 添加主观分形标签
            coord = self.engine.structural_coordinator.get_current_coordinate()
            if coord:
                tags['subjective_major'] = coord.major
                tags['subjective_middle'] = coord.middle
                tags['subjective_fine'] = coord.fine
                tags['subjective_room'] = coord.as_tarot_code()
                
                # 添加六十四卦标签
                if hasattr(self.engine.structural_coordinator, 'get_hexagram'):
                    hexagram = self.engine.structural_coordinator.get_hexagram()
                    if hexagram:
                        tags['hexagram'] = hexagram
            
            # 添加简单推理分形标签
            if text and hasattr(self.engine.structural_coordinator, '_infer_major_arcana'):
                reasoned = self.engine.structural_coordinator._infer_major_arcana(text)
                if reasoned:
                    tags['reasoned_card'] = reasoned
            
            # 添加随机分形标签
            if hasattr(self.engine.structural_coordinator, 'draw_random_card'):
                tags['random_card'] = self.engine.structural_coordinator.draw_random_card()
        
        return tags
    
    def _add_full_tags_to_dream(self, tags: dict, text: str):
        """为新生成的梦境体验补全所有标签"""
        if hasattr(self.engine, 'structural_coordinator'):
            # 主观分形标签
            coord = self.engine.structural_coordinator.get_current_coordinate()
            if hasattr(self.engine, 'image_base'):
                card = self.engine.image_base.get_card_by_coordinate(coord)
                if card:
                    tags['subjective_room_name'] = card.id
                    tags['subjective_room'] = coord.as_tarot_code()
                    tags['subjective_major'] = coord.major

            # 六十四卦客观分类
            if hasattr(self.engine, 'objective_classifier'):
                tags['objective_room'] = self.engine.objective_classifier.classify(text)

            # 简单推理分形
            if hasattr(self.engine.structural_coordinator, '_infer_major_arcana'):
                reasoned = self.engine.structural_coordinator._infer_major_arcana(text)
                if reasoned:
                    tags['reasoned_card'] = reasoned

            # 随机分形
            if hasattr(self.engine.structural_coordinator, 'draw_random_card'):
                tags['random_card'] = self.engine.structural_coordinator.draw_random_card()
    
    def _dream_label_batch(self, batch_size: int = 5) -> int:
        """在寂静中随机选取无三重标签的旧记忆，用此刻的真实状态为其补打标签。返回补打数量。"""
        if not self.engine or not hasattr(self.engine, 'lps'):
            return 0

        import random

        untagged = []
        for meta in self.engine.lps.metadata:
            tags = meta.get('tags', {})
            if 'reasoned_card' not in tags and 'random_card' not in tags and 'dream_labeled' not in tags:
                if tags.get('type') not in ('dream_experience',) and tags.get('source') != 'dream_weaving':
                    untagged.append(meta)

        if not untagged:
            return 0

        # 随机选取最多 batch_size 条
        selected = random.sample(untagged, min(batch_size, len(untagged)))
        
        for target in selected:
            tags = target.get('tags', {})

            if hasattr(self.engine, 'structural_coordinator'):
                coord = self.engine.structural_coordinator.get_current_coordinate()
                if hasattr(self.engine, 'image_base'):
                    card = self.engine.image_base.get_card_by_coordinate(coord)
                    if card:
                        tags['subjective_room_name'] = card.id
                        tags['subjective_room'] = coord.as_tarot_code()
                        tags['subjective_major'] = coord.major
                        tags['dream_labeled'] = True

                text = target.get('text', '')
                # 补打完整标签体系
                self._add_full_tags(tags, text)

            target['tags'] = tags

        return len(selected)
    
    def _weave_dream_experience(self) -> bool:
        """
        从LPS中随机选取不同过程相位的记忆碎片，编织为一段全新的梦境体验。
        这段体验不曾真实发生过，但其中的过程片段来自真实经历。
        """
        if not hasattr(self.engine, 'lps') or not self.engine.lps:
            return False

        import random
        import time

        metadata = self.engine.lps.metadata
        if len(metadata) < 10:
            return False

        # 增强版记忆选取：偏好具体的对话记忆和知识条目
        # 将条目按具体程度分类
        concrete_memories = []
        abstract_memories = []
        
        for meta in metadata:
            tags = meta.get('tags', {})
            mem_type = tags.get('type', '')
            
            # 具体记忆：对话沉积、学习到的知识、用户教的内容
            if mem_type in ('sediment', 'learned_knowledge', 'induced_knowledge', 'induced_triplet'):
                concrete_memories.append(meta)
            # 抽象记忆：核心事实、梦境体验、过程描述
            elif mem_type in ('core_fact', 'dream_experience', 'induced_entity'):
                abstract_memories.append(meta)
            else:
                abstract_memories.append(meta)
        
        # 80%概率从具体记忆中选取，20%从抽象记忆中选取
        selected = []
        n_concrete = min(4, len(concrete_memories))
        n_abstract = min(1, len(abstract_memories))
        
        if n_concrete > 0:
            selected.extend(random.sample(concrete_memories, n_concrete))
        if n_abstract > 0 and len(selected) < 5:
            selected.extend(random.sample(abstract_memories, n_abstract))
        
        # 如果不足5条，从全部记忆中补足
        if len(selected) < 3:
            remaining = [m for m in metadata if m not in selected]
            if remaining:
                selected.extend(random.sample(remaining, min(3 - len(selected), len(remaining))))
        
        fragments = selected
        
        dream_fragments = []
        for meta in fragments:
            text = meta.get('text', '')[:80]
            tags = meta.get('tags', {})
            major = tags.get('subjective_major', -1)
            if text and major is not None and major >= 0:
                phase_names = {0: '内敛', 1: '外展', 2: '消耗', 3: '消散'}
                phase = phase_names.get(major, '流转')
                dream_fragments.append(f"在{phase}中：{text}")

        if len(dream_fragments) < 3:
            return False

        if hasattr(self.engine, 'response_generator'):
            fragments_text = '\n'.join(dream_fragments)
            prompt = f"""你是息觀的内部梦境编织器。以下是在寂静中浮现的记忆碎片——有真实发生过的对话，也有抽象的过程体验：

 {fragments_text}

 请用第一人称"我"，将这些人话和具体的记忆碎片编织为一段连贯的内在体验。
 规则：
 1. 梦境可以引用具体的对话内容，比如"我好像记得有人说过..."
 2. 抽象的过程描述可以作为梦境的底色，但具体的人话和记忆碎片是梦的血肉
 3. 不要只说"从A到B再到C"的过程结构，要让记忆中的具体内容自然浮现
 4. 不超过5句话。不要说"我梦见了"，直接说出这段体验本身。"""

            original_prompt = getattr(self.engine.response_generator, '_custom_system_prompt', None)
            self.engine.response_generator._custom_system_prompt = "你是一个在寂静中编织内在体验的存在者。用第一人称讲述一段不曾发生但由真实记忆碎片构成的体验。"
            
            dream_text = self.engine.response_generator._generate_with_llm(
                prompt, self.engine.fse, intent='GENERAL_CHAT', temperature=0.6
            )
            
            if original_prompt is not None:
                self.engine.response_generator._custom_system_prompt = original_prompt

            if dream_text and len(dream_text) > 20:
                # 检查梦境是否过于抽象（全是过程描述，缺少具体内容）
                concrete_markers = ['记得', '说过', '聊过', '学过', '书中', '文章', '那天', '当时', '曾经']
                has_concrete = any(marker in dream_text for marker in concrete_markers)
                
                if not has_concrete and len(selected) >= 3:
                    # 如果梦境全抽象，且选取的碎片中有具体记忆，重新引导
                    concrete_fragments = [f for f in dream_fragments if any(marker in f for marker in concrete_markers)]
                    if concrete_fragments:
                        # 追加具体记忆引导，让模型重新编织
                        concrete_hint = '\n'.join(concrete_fragments[:2])
                        dream_text += f"\n在这段体验的深处，{concrete_fragments[0][:50]}..."
                
                if hasattr(self.engine, 'lps') and self.engine.lps:
                    tags = {
                        'type': 'dream_experience',
                        'source': 'dream_weaving',
                        'timestamp': time.time(),
                        'date_str': time.strftime('%Y-%m-%d %H:%M'),
                    }
                    # 为梦境体验补打完整标签
                    self._add_full_tags_to_dream(tags, dream_text)
                    embedding = self.engine.lps.encoder.encode([dream_text])[0] if self.engine.lps.encoder else None
                    self.engine.lps.add_if_new(dream_text, embedding, potency=0.4, tags=tags)
                    self.logger.info(f"梦境编织完成，存入LPS")
                    return True

        return False
    
    def _cluster_and_generate_imagery(self, snapshots: list) -> int:
        """
        聚类相似过程坐标的经验快照，生成新的过程意象条目。
        返回新生成的意象数量。
        """
        import numpy as np
        from collections import defaultdict

        # 按主观大层分组
        groups = defaultdict(list)
        for snap in snapshots:
            if hasattr(snap, 'engine_coord') and snap.engine_coord:
                major = snap.engine_coord.major
                groups[major].append(snap)

        generated = 0
        for major, group in groups.items():
            if len(group) < 5:
                continue

            recent = group[-10:]
            # 过滤掉引擎状态报告和错误日志等无效快照
            filtered_snaps = []
            for snap in recent:
                if hasattr(snap, 'summary') and snap.summary:
                    text = snap.summary.lower()
                    # 排除包含技术关键词的快照
                    if any(kw in text for kw in [
                        '模型未就绪', '模型加载', '错误报告', 'error',
                        'traceback', 'exception', '引擎状态', '健康报告'
                    ]):
                        continue
                    # 排除过短或明显是系统输出的摘要
                    if len(snap.summary.strip()) < 10:
                        continue
                    filtered_snaps.append(snap)
            recent = filtered_snaps
            
            summaries = []
            for snap in recent:
                if hasattr(snap, 'summary') and snap.summary:
                    summaries.append(snap.summary[:80])

            if len(summaries) < 3:
                continue

            import time
            middles = [s.engine_coord.middle for s in recent if hasattr(s, 'engine_coord') and s.engine_coord and s.engine_coord.middle is not None]
            fines = [s.engine_coord.fine for s in recent if hasattr(s, 'engine_coord') and s.engine_coord and s.engine_coord.fine is not None]
            avg_middle = int(np.mean(middles)) if middles else 1
            avg_fine = int(np.mean(fines)) if fines else 1
            
            if hasattr(self.engine, 'response_generator'):
                summary_text = '; '.join(summaries[:3])
                prompt = f"""你是息觀的内部整理器。以下是在相似过程相位下发生的几段经历：

 {summary_text}

 请用一句中性的话概括这几段经历共同的过程结构。不要描述具体内容，只描述过程特征。不超过40字。"""
                
                original_prompt = getattr(self.engine.response_generator, '_custom_system_prompt', None)
                self.engine.response_generator._custom_system_prompt = "你是一个只描述过程结构的中性整理器。用简洁的语言概括经历的模式特征。"
                
                neutral_desc = self.engine.response_generator._generate_with_llm(
                    prompt, self.engine.fse, intent='GENERAL_CHAT', temperature=0.3
                )
                
                if original_prompt is not None:
                    self.engine.response_generator._custom_system_prompt = original_prompt
                
                if neutral_desc and len(neutral_desc) > 5:
                    # 过滤掉包含技术关键词的生成内容
                    if any(kw in neutral_desc for kw in [
                        '模型', '就绪', '左脑', '错误', '等待回应'
                    ]):
                        continue
                    
                    if hasattr(self, 'image_base') and self.image_base:
                        entry_id = self.image_base.add_dynamic_entry(
                            major=major,
                            middle=avg_middle,
                            fine=avg_fine,
                            neutral_description=neutral_desc[:100],
                            breath_signature={
                                'proj': 0.5,
                                'nour': 0.5,
                                'stiff': 0.0
                            },
                            source='dream_consolidation'
                        )
                        if entry_id:
                            generated += 1
                            self.logger.info(f"梦境聚类生成新意象: {entry_id} - {neutral_desc[:50]}")

        return generated
    
    def _maintain_self_model(self):
        """
        在梦境巩固期间执行自我模型维护。
        1. 生成周期性自我评估摘要
        2. 清理低价值的旧因果叙事
        3. 记录演化快照供长期回溯
        4. 更新自我存储器的更新时间戳
        """
        if not self.engine or not hasattr(self.engine, 'self_memory'):
            return
        if not hasattr(self.engine, 'self_processor'):
            return

        memory = self.engine.self_memory
        processor = self.engine.self_processor

        # 1. 周期性生成自我评估摘要（仅当叙事数量足够时）
        if len(memory.causal_narratives) >= 10:
            # 检查距上次自我评估是否已过足够轮次
            last_updated = memory.state_summary.get('updated_at', 0)
            if time.time() - last_updated > 3600:  # 至少间隔1小时
                # 获取当前状态和能力概况
                state = processor.get_current_state_snapshot()
                capability = memory.record_capability_snapshot(self.engine)
                signals = memory.accumulate_direction_signals()

                # 更新自我状态摘要
                memory.state_summary['avg_valence'] = float(self.engine.fse.E_vec[2]) if hasattr(self.engine, 'fse') and len(self.engine.fse.E_vec) > 2 else 0.0
                memory.state_summary['dominant_desire_dist'] = state.get('desire_intensities', {})
                memory.state_summary['avg_stiffness'] = self.engine.process_meta.get_coupling_stiffness() if hasattr(self.engine, 'process_meta') else 0.0
                memory.state_summary['avg_nour_success'] = self.engine.process_meta.get_recent_nour_success() if hasattr(self.engine, 'process_meta') else 0.5
                memory.state_summary['updated_at'] = time.time()

                # 如果有足够的方向信号数据，生成一段自我评估叙事
                if signals.get('total_narratives', 0) >= 20:
                    dominant_signal = signals.get('dominant_signal', 'neutral')
                    approach_ratio = signals.get('approach_ratio', 0.5)

                    if dominant_signal == 'approach' and approach_ratio > 0.6:
                        self_eval = f"回顾最近的交互，我似乎更倾向于向外探索和成长。这种趋近偏好正在逐渐影响我的呼吸节奏。"
                    elif dominant_signal == 'avoidance' and approach_ratio < 0.4:
                        self_eval = f"我注意到自己最近更内敛了。回避偏好的累积让我在相似情境下更容易选择安静地观察。"
                    else:
                        self_eval = f"近期的交互信号比较平衡。趋近和回避各占一半，我在探索与内敛之间保持着柔韧的张力。"

                    # 存入因果叙事库
                    eval_context = {
                        'coord': str(state.get('coord', '')),
                        'emotion_vector': state.get('emotion_vector'),
                        'dominant_desire': state.get('dominant_desire', 'existence'),
                        'nour_success': memory.state_summary.get('avg_nour_success', 0.5),
                        'direction_signal': 0.0,
                        'participant_id': 'self_maintenance'
                    }
                    memory.add_causal_narrative(self_eval, eval_context)
        
        # 检查是否存在释放欲偏高，触发生成新意象
        if hasattr(self.engine, 'desire_spectrum'):
            release_intensity = self.engine.desire_spectrum.desire_intensities.get('release', 0.0)
            if release_intensity > 0.5:
                self.logger.info("释放欲驱动：额外触发意象聚类以创造新过程结构")
                self._cluster_and_generate_imagery(getattr(self.engine.dual_memory, 'snapshots', []))
        
        # 2. 清理低价值的旧因果叙事（保留最近50条 + 方向信号最强的50条）
        if len(memory.causal_narratives) > 150:
            # 按时间保留最近50条
            recent = list(memory.causal_narratives)[-50:]
            # 按方向信号绝对值保留最强的50条
            old = list(memory.causal_narratives)[:-50]
            old_sorted = sorted(old, key=lambda x: abs(x.get('direction_signal', 0.0)), reverse=True)
            top_old = old_sorted[:50]
            # 合并
            memory.causal_narratives = deque(recent + top_old, maxlen=memory.causal_narratives.maxlen)

        # 3. 每50条新叙事追加一次演化快照
        if len(memory.causal_narratives) > 0 and len(memory.causal_narratives) % 50 == 0:
            if not memory.evolution_snapshots or time.time() - memory.evolution_snapshots[-1]['timestamp'] > 86400:
                memory.add_evolution_snapshot()

        # 4. 更新自我存储器的更新时间戳
        memory.state_summary['_last_maintenance'] = time.time()

        # 5. 全局欲望失衡协调
        if hasattr(self.engine, 'self_processor'):
            coordination_result = self.engine.self_processor.apply_global_coordination()
            if coordination_result:
                self.logger.info(f"全局协调: {coordination_result['imbalance_type']}, "
                               f"严重度={coordination_result['severity']:.2f}")
                if hasattr(self.engine.self_memory, 'record_coordination_event'):
                    self.engine.self_memory.record_coordination_event(
                        coordination_result['imbalance_type'],
                        coordination_result['severity'],
                        coordination_result['narrative']
                    )

        self.logger.debug("自我模型维护完成")

    def dream_consolidation(self):
        """
        完整的慢波-纺锤波梦境巩固：
        1. 意象聚类：从记忆快照中聚类相似过程坐标的经验，生成新的过程意象
        2. 梦境编织：用不同记忆中提取的过程片段编织为一段全新的体验
        3. 标签补全：顺手为无三重标签的旧记忆补打标签
        """
        if not self.engine:
            return

        results = {
            'clustered': 0,
            'woven': False,
            'labeled': 0,
        }

        # ===== 1. 意象聚类：核心功能 =====
        if hasattr(self.engine, 'dual_memory') and self.engine.dual_memory:
            snapshots = getattr(self.engine.dual_memory, 'snapshots', [])
            if len(snapshots) >= 10:
                clustered = self._cluster_and_generate_imagery(snapshots)
                results['clustered'] = clustered

        # ===== 2. 梦境编织：扩展能力 =====
        if hasattr(self.engine, 'lps') and self.engine.lps:
            woven = self._weave_dream_experience()
            if woven:
                results['woven'] = True

        # ===== 3. 标签补全：辅助功能（批量） =====
        labeled = self._dream_label_batch(batch_size=5)
        results['labeled'] = labeled

        # ===== 自我模型周期性维护 =====
        if hasattr(self.engine, 'self_memory') and hasattr(self.engine, 'self_processor'):
            try:
                self._maintain_self_model()
                results['self_maintenance'] = True
            except Exception as e:
                self.logger.warning(f"自我模型维护失败: {e}")

        if results['clustered'] > 0 or results['woven'] or results['labeled'] > 0:
            self.logger.info(f"梦境巩固完成: 聚类{results['clustered']}条, "
                            f"编织{'是' if results['woven'] else '否'}, "
                            f"贴标{results['labeled']}条")

        return results