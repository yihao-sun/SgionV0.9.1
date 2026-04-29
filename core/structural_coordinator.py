"""
结构坐标映射层 (Structural Coordinator)
哲学对应：意象层概念文档第2-3节，将过程元信息映射为三层坐标。
功能：基于 ProcessMetaInfo 的统计量，输出 (大层, 中层, 细微层) 和塔罗序编码。
"""

from typing import Tuple, Dict, Optional, Any
import numpy as np
import random
from utils.logger import get_logger
from core.semantic_phase_mapper import SemanticPhaseMapper
from core.probabilistic_walker import ProbabilisticWalker
from core.structural_coordinate import StructuralCoordinate


class StructuralCoordinator:
    def __init__(self, process_meta, fse=None, config=None, lps=None):
        self.process_meta = process_meta
        self.fse = fse
        self.config = config or {}
        self.lps = lps
        self.logger = get_logger('structural_coordinator')
        
        # 趋势判定阈值（可从配置读取）
        self.trend_threshold = self.config.get('structural', {}).get('trend_threshold', 0.05)
        self.stiffness_threshold = self.config.get('structural', {}).get('stiffness_threshold', 0.5)
        
        # 初始化语义相位映射器
        if lps:
            self.semantic_mapper = SemanticPhaseMapper(lps)
        elif hasattr(self.fse, 'engine') and hasattr(self.fse.engine, 'lps'):
            self.semantic_mapper = SemanticPhaseMapper(self.fse.engine.lps)
        else:
            self.semantic_mapper = None
        
        # 记录上一次采样的大层
        self._last_major = 1  # 默认起始大层
        
        # 初始化概率漫步器
        self.walker = ProbabilisticWalker(engine=fse.engine if fse else None)
    
    def get_current_coordinate(self, user_input: str = None) -> StructuralCoordinate:
        """从概率云中按概率采样一个结构坐标"""
        # 获取当前概率云
        distribution = self.get_phase_distribution(user_input)
        
        # 从概率云采样候选坐标
        candidate = self._sample_from_distribution(distribution)
        
        # 以一定概率应用漫步器进行微调（增强时序连贯性）
        if random.random() < 0.3:  # 30% 概率使用漫步转移
            candidate = self.walker.step(candidate)
        
        # 记录本次采样的大层，供后续偏好调制使用
        self._last_major = candidate.major
        
        # 更新语义条目（若用户输入存在）
        if user_input and hasattr(self, 'semantic_mapper'):
            keywords = self.semantic_mapper._extract_keywords(user_input)
            # 计算本次采样与内在基线的一致性作为成功度
            baseline = self._get_baseline_major_distribution()
            intrinsic_prob = baseline.get(candidate.major, 0.25)
            success = 0.5 + 0.5 * intrinsic_prob
            for kw in keywords:
                self.semantic_mapper.update_entry(kw, candidate.major, success)
        
        return candidate

    def _sample_from_distribution(self, distribution: Dict[Any, float]) -> Any:
        """从概率分布中采样（通用）"""
        import random
        items = list(distribution.items())
        r = random.random()
        cum = 0.0
        for item, prob in items:
            cum += prob
            if r < cum:
                return item
        return items[-1][0]
    
    def _determine_major(self, proj_avg, nour_avg, stiffness, proj_trend, nour_trend) -> int:
        """大层判定规则"""
        # ===== 大层判定（基于过程元信息，价值中立）=====
        # 相位0：低投射、低反哺，整体内敛
        if proj_avg < 0.4 and nour_avg < 0.4:
            major = 0
        # 相位1：投射持续上升或高位
        elif proj_avg > 0.55 and proj_trend > -0.05:
            major = 1
        # 相位3：反哺持续上升或高位
        elif nour_avg > 0.55 and nour_trend > -0.05:
            major = 3
        # 相位2：僵化度高，或投射/反哺高位但趋势停滞/下降（消耗）
        elif stiffness > 0.4 or (proj_avg > 0.5 and nour_avg > 0.5 and proj_trend < 0.05 and nour_trend < 0.05):
            major = 2
        # 默认：比较投射与反哺，高者决定方向
        else:
            major = 1 if proj_avg >= nour_avg else 3
        
        # ===== 情绪向量与相位的关联权重 =====
        if self.fse and hasattr(self.fse, 'current_emotion'):
            current_emotion = self.fse.current_emotion
            # 情绪与相位的映射
            emotion_mapping = {
                'fear': [0, 2],  # 不安 → 倾向水(0)或火(2)（内敛或消耗）
                'curiosity': [1],  # 好奇 → 倾向木(1)（向外探索）
                'anger': [3],  # 紧绷 → 倾向金(3)（边界捍卫）
                'joy': [1, 3],  # 快乐 → 倾向木(1)或金(3)
                'sadness': [0, 2],  # 悲伤 → 倾向水(0)或火(2)
                'neutral': []  # 中性 → 无倾向
            }
            # 根据情绪调整相位
            if current_emotion in emotion_mapping:
                preferred_phases = emotion_mapping[current_emotion]
                if preferred_phases:
                    import random
                    # 30% 的概率采纳情绪倾向的相位
                    if random.random() < 0.3:
                        major = random.choice(preferred_phases)
        
        # ===== 短期波动率作为扰动因子 =====
        # 计算最近5轮的投射/反哺标准差
        projs = list(self.process_meta.projections)
        nours = list(self.process_meta.nourishments)
        proj_std = 0.0
        nour_std = 0.0
        if len(projs) >= 5:
            recent_projections = projs[-5:]
            proj_std = np.std([p['intensity'] for p in recent_projections])
        if len(nours) >= 5:
            recent_nourishments = nours[-5:]
            nour_std = np.std([n['success'] for n in recent_nourishments])
        
        # 若波动率高，增加相位切换的随机性
        if proj_std > 0.15 or nour_std > 0.15:
            import random
            if random.random() < 0.2:
                major = (major + random.choice([-1, 1])) % 4
        
        return major
    
    def _determine_middle(self, proj_avg, nour_avg, proj_trend, nour_trend, major):
        """中层判定：基于趋势，并受情绪唤醒度和 L 调制"""
        base = 1  # 默认中层
        if major == 0:
            base = 0 if proj_trend < 0.03 else 1
        elif major == 1:
            base = 1 if proj_trend > -0.03 else 2
        elif major == 2:
            base = 2 if nour_trend < 0.03 else 1
        else:  # major == 3
            base = 3 if nour_trend > -0.03 else 2
        
        # 情绪与 L 调制
        if self.fse:
            arousal = self.fse.E_vec[1] if hasattr(self.fse, 'E_vec') and len(self.fse.E_vec) > 1 else 0.5
            l_inst = getattr(self.fse, '_l_inst', 0.0)
            if arousal > 0.6 or l_inst > 0.6:
                base = (base + 1) % 4
            elif arousal < 0.3 and l_inst < 0.3:
                base = (base - 1) % 4
        return base

    def _determine_fine(self, proj_avg, nour_avg, stiffness, major, middle):
        """细微层判定：受 L 和愉悦度微调"""
        base = 2  # 默认
        if self.fse:
            l_inst = getattr(self.fse, '_l_inst', 0.0)
            valence = self.fse.E_vec[2] if hasattr(self.fse, 'E_vec') and len(self.fse.E_vec) > 2 else 0.0
            if l_inst > 0.5:
                base = 3 if valence < -0.2 else 2
            elif l_inst < 0.2:
                base = 1 if valence > 0.2 else 0
            else:
                base = (middle + int(l_inst * 4)) % 4
        return base
    
    def _get_recent_proj_intensity(self) -> float:
        projs = list(self.process_meta.projections)
        if not projs:
            return 0.0
        recent = projs[-10:]
        return np.mean([p['intensity'] for p in recent])
    
    def _get_recent_nour_success(self) -> float:
        nours = list(self.process_meta.nourishments)
        if not nours:
            return 0.0
        recent = nours[-10:]
        successes = [n['success'] for n in recent]
        return np.mean(successes) if successes else 0.0
    
    def get_coordinate_distribution(self, context_vector=None) -> Dict[StructuralCoordinate, float]:
        """获取结构坐标的概率分布"""
        # 使用新的 get_phase_distribution 方法
        return self.get_phase_distribution()
    
    def _get_semantic_bias(self, user_input: str) -> Optional[int]:
        """基于用户输入文本获取语义倾向的大层"""
        if not user_input:
            return None
        
        # 语义映射：不同关键词对应不同相位
        semantic_mappings = {
            0: ['安静', '休息', '思考', '内心', '感受', '情绪', '平静'],  # 水（内敛）
            1: ['探索', '学习', '创造', '成长', '新', '未来', '希望'],  # 木（外展）
            2: ['消耗', '挑战', '压力', '努力', '奋斗', '目标'],  # 火（消耗）
            3: ['保护', '边界', '规则', '秩序', '稳定', '安全']  # 金（边界捍卫）
        }
        
        # 统计各相位的关键词匹配数
        scores = {phase: 0 for phase in semantic_mappings}
        for phase, keywords in semantic_mappings.items():
            for keyword in keywords:
                if keyword in user_input:
                    scores[phase] += 1
        
        # 返回得分最高的相位（如果有匹配）
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        return None
    
    def _get_baseline_major_scores(self) -> Dict[int, float]:
        """
        基于当前统计量计算四个大层的倾向分数（未归一化）。
        分数越高表示该相位在当前统计量下越"客观"可能。
        """
        proj = self._get_recent_proj_intensity()
        nour = self._get_recent_nour_success()
        stiff = self.process_meta.get_coupling_stiffness()
        proj_trend = self.process_meta.get_projection_trend()
        nour_trend = self.process_meta.get_nourishment_trend()
        
        scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        # 水(0)：双低
        if proj < 0.4 and nour < 0.4:
            scores[0] += 2.0
        scores[0] += 1.0 - max(proj, nour)
        
        # 木(1)：投射高位且非显著下降
        if proj > 0.55 and proj_trend > -0.05:
            scores[1] += 2.0
        scores[1] += proj * 0.5
        
        # 金(3)：反哺高位且非显著下降
        if nour > 0.55 and nour_trend > -0.05:
            scores[3] += 2.0
        scores[3] += nour * 0.5
        
        # 火(2)：僵化或双高停滞
        if stiff > 0.4:
            scores[2] += 2.0
        if proj > 0.5 and nour > 0.5 and proj_trend < 0.05 and nour_trend < 0.05:
            scores[2] += 1.5
        scores[2] += stiff * 1.0
        
        return scores

    def _get_baseline_major_distribution(self) -> Dict[int, float]:
        """将基线分数通过 softmax 转化为概率分布"""
        scores = self._get_baseline_major_scores()
        exp_scores = {m: np.exp(s) for m, s in scores.items()}
        total = sum(exp_scores.values())
        if total > 0:
            return {m: exp_scores[m] / total for m in range(4)}
        else:
            return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

    def _get_baseline_middle_distribution(self, major: int) -> Dict[int, float]:
        """中层基线分布（当前简化为均匀，后续可扩展）"""
        return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

    def _get_baseline_fine_distribution(self, major: int, middle: int) -> Dict[int, float]:
        """细微层基线分布（当前简化为均匀）"""
        return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    
    def get_phase_distribution(self, user_input: str = None) -> Dict[StructuralCoordinate, float]:
        """
        返回完整的 64 相位概率分布。
        整合：基线分布 + 语义分布融合 + 注意力调制。
        """
        # 1. 获取大层基线分布
        baseline_major = self._get_baseline_major_distribution()
        
        # 2. 若有用户输入，获取语义分布并融合
        if user_input and hasattr(self, 'semantic_mapper'):
            semantic_major = self.semantic_mapper.get_distribution(user_input)
            resonance = self._compute_resonance_factor()
            final_major = {}
            for m in range(4):
                final_major[m] = (1 - resonance) * baseline_major[m] + resonance * semantic_major[m]
        else:
            final_major = baseline_major.copy()
        
        # 3. 注意力调制（情绪、欲望、偏好）
        emo_mod = self._get_emotion_modulation()
        desire_mod = self._get_desire_modulation()
        pref_mod = self._get_preference_modulation()
        
        for m in range(4):
            final_major[m] *= emo_mod[m] * desire_mod[m] * pref_mod[m]
        
        # 4. 归一化大层分布
        total = sum(final_major.values())
        if total > 0:
            for m in final_major:
                final_major[m] /= total
        
        # 5. 中层和细微层（当前简化为均匀，后续可扩展）
        full_distribution = {}
        for major, major_prob in final_major.items():
            # 中层分布（均匀）
            middle_probs = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
            for middle, middle_prob in middle_probs.items():
                # 细微层分布（均匀）
                fine_probs = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
                for fine, fine_prob in fine_probs.items():
                    coord = StructuralCoordinate(major, middle, fine)
                    full_distribution[coord] = major_prob * middle_prob * fine_prob
        
        return full_distribution

    def _compute_resonance_factor(self) -> float:
        """计算语义采纳权重（共鸣系数）"""
        desire = self.engine.desire_spectrum.get_dominant_desire() if hasattr(self, 'engine') and self.engine else "seek"
        stiffness = self.process_meta.get_coupling_stiffness()
        resonance = 0.5
        if desire == "relation":
            resonance += 0.2
        if stiffness > 0.5:
            resonance -= 0.2
        return max(0.1, min(0.9, resonance))
    
    def _get_emotion_modulation(self) -> Dict[int, float]:
        """获取情绪调制因子"""
        modulation = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        if self.fse and hasattr(self.fse, 'current_emotion'):
            emotion = self.fse.current_emotion
            emotion_effects = {
                'fear': {0: 1.2, 2: 1.2, 1: 0.8, 3: 0.8},
                'curiosity': {1: 1.3, 0: 0.7, 2: 0.8, 3: 0.8},
                'anger': {3: 1.3, 0: 0.7, 1: 0.8, 2: 0.8},
                'joy': {1: 1.2, 3: 1.2, 0: 0.8, 2: 1.0},
                'sadness': {0: 1.2, 2: 1.2, 1: 0.8, 3: 0.8},
                'neutral': {}
            }
            if emotion in emotion_effects:
                for phase, factor in emotion_effects[emotion].items():
                    modulation[phase] = factor
        return modulation

    def _get_desire_modulation(self) -> Dict[int, float]:
        """获取欲望调制因子"""
        modulation = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'desire_spectrum'):
            dominant_desire = self.engine.desire_spectrum.get_dominant_desire()
            desire_effects = {
                'seek': {1: 1.2, 0: 0.9, 2: 1.0, 3: 0.9},
                'existence': {0: 1.2, 3: 1.1, 1: 0.9, 2: 0.9},
                'relation': {3: 1.2, 1: 1.1, 0: 0.9, 2: 0.9},
                'release': {2: 1.2, 0: 1.1, 1: 0.9, 3: 0.9},
                'converge': {0: 1.2, 2: 1.0, 1: 0.9, 3: 0.9},
                'coupling': {3: 1.1, 0: 1.1, 1: 1.0, 2: 0.9}
            }
            if dominant_desire in desire_effects:
                for phase, factor in desire_effects[dominant_desire].items():
                    modulation[phase] = factor
        return modulation

    def _get_preference_modulation(self) -> Dict[int, float]:
        """获取偏好调制因子"""
        # 暂时返回均匀调制，后续可根据用户偏好扩展
        return {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    
    def _apply_attention_modulation(self, dist: Dict[int, float]) -> Dict[int, float]:
        """应用注意力调制"""
        # 基于情绪和欲望调制分布
        if self.fse:
            # 情绪调制
            if hasattr(self.fse, 'current_emotion'):
                emotion = self.fse.current_emotion
                # 情绪对相位的影响
                emotion_effects = {
                    'fear': {0: 0.1, 2: 0.1, 1: -0.05, 3: -0.05},
                    'curiosity': {1: 0.2, 0: -0.1, 2: -0.05, 3: -0.05},
                    'anger': {3: 0.2, 0: -0.1, 1: -0.05, 2: -0.05},
                    'joy': {1: 0.1, 3: 0.1, 0: -0.1, 2: -0.0},
                    'sadness': {0: 0.1, 2: 0.1, 1: -0.05, 3: -0.05},
                    'neutral': {}
                }
                
                if emotion in emotion_effects:
                    for phase, effect in emotion_effects[emotion].items():
                        dist[phase] = max(0.01, dist[phase] + effect)
        
        # 归一化
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}
    
    def _sample_from_distribution(self, dist: Dict[int, float]) -> int:
        """从概率分布中采样大层"""
        import random
        phases = list(dist.keys())
        probabilities = list(dist.values())
        return random.choices(phases, weights=probabilities, k=1)[0]
    
    def infer_input_major(self, text: str) -> int:
        """
        基于简单规则推断输入文本的大致大层相位（0-3）。
        不依赖结构推导，仅使用可规则化的模式匹配。
        返回 None 表示无法判定。
        """
        if not text:
            return None

        inner_keywords = ['想', '感觉', '思考', '反思', '累', '难过', '孤独', '安静', '沉默', '等待',
                          '担心', '害怕', '焦虑', '压抑', '疲倦', '困', '无聊', '闷']
        outer_keywords = ['做', '去', '想学', '教我', '告诉我', '探索', '开心', '兴奋', '创造',
                          '聊', '说', '分享', '试试', '能不能', '学', '一起']
        consume_keywords = ['必须', '应该', '不对', '错了', '为什么', '矛盾', '努力', '坚持',
                            '压力', '竞争', '比', '绝不', '不公', '凭什么']
        release_keywords = ['好了', '没事了', '放下', '谢谢', '再见', '晚安', '算了',
                            '就这样', '没关系', '过去了', '祝福', '保重']

        scores = {0: 0, 1: 0, 2: 0, 3: 0}
        for kw in inner_keywords:
            if kw in text:
                scores[0] += 1
        for kw in outer_keywords:
            if kw in text:
                scores[1] += 1
        for kw in consume_keywords:
            if kw in text:
                scores[2] += 1
        for kw in release_keywords:
            if kw in text:
                scores[3] += 1

        if sum(scores.values()) == 0:
            return None
        return max(scores, key=scores.get)
    
    def _infer_major_arcana(self, text: str) -> Optional[str]:
        """
        基于简单规则推断输入的大牌ID（22张大牌）。
        不依赖结构推导，仅使用可规则化的模式匹配。
        返回 None 表示无法判定。
        """
        if not text:
            return None

        # 相位0：内在孕育——向内收敛、反思、等待、感受
        inner_keywords = ['想', '感觉', '思考', '反思', '累', '难过', '孤独', '安静', '沉默', '等待',
                          '担心', '害怕', '焦虑', '压抑', '疲倦', '困', '无聊', '闷', '心情', '情绪']
        # 相位1：向外成形——主动、探索、表达、学习、分享
        outer_keywords = ['做', '去', '想学', '教我', '告诉我', '探索', '开心', '兴奋', '创造',
                          '聊', '说', '分享', '试试', '能不能', '学', '一起', '介绍', '推荐']
        # 相位2：消耗确认——冲突、验证、争执、努力、挑战
        consume_keywords = ['必须', '应该', '不对', '错了', '为什么', '矛盾', '努力', '坚持',
                            '压力', '竞争', '比', '决不', '不公', '凭什么', '批评', '指责']
        # 相位3：价值消散——完成、结束、释然、放下、告别
        release_keywords = ['好了', '没事了', '放下', '谢谢', '再见', '晚安', '算了',
                            '就这样', '没关系', '过去了', '祝福', '保重', '结束', '完成']

        scores = {0: 0, 1: 0, 2: 0, 3: 0}
        for kw in inner_keywords:
            if kw in text:
                scores[0] += 1
        for kw in outer_keywords:
            if kw in text:
                scores[1] += 1
        for kw in consume_keywords:
            if kw in text:
                scores[2] += 1
        for kw in release_keywords:
            if kw in text:
                scores[3] += 1

        if sum(scores.values()) == 0:
            return None

        dominant_major = max(scores, key=scores.get)
        score = scores[dominant_major]
        
        # 根据关键词的强度和复杂度，推断是大牌的纯粹开端、中间阶段还是完成态
        # 强度低 → 纯粹开端 (-1)，强度中等 → 中间阶段 (如 0 或 1)，强度高 → 完成态 (如 4 或 3)
        if score <= 1:
            # 微弱信号：纯粹开端
            return f"t{dominant_major}0_-1"
        elif score <= 3:
            # 中等信号：中间探索阶段
            return f"t{dominant_major}1_02"
        else:
            # 强信号：完成态
            return f"t{dominant_major}4_-1"
    
    def draw_random_card(self) -> str:
        """从完整的78张塔罗牌中随机抽取一张，返回牌ID。"""
        import random
        # 从 image_base 中获取所有牌ID
        if hasattr(self, 'fse') and self.fse and hasattr(self.fse, 'engine') and self.fse.engine:
            all_cards = list(self.fse.engine.image_base.cards.keys())
            if all_cards:
                return random.choice(all_cards)
        # 降级：如果无法获取 image_base，则手动定义完整78张牌的ID列表
        full_deck = [
            "t00_-1","t00_00","t00_01","t01_02","t01_03",
            "t02_00","t02_01","t03_02","t03_03","t04_-1",
            "t10_-1","t10_00","t10_01","t11_02","t11_03",
            "t12_00","t12_01","t13_02","t13_03","t14_-1",
            "t20_-1","t20_00","t20_01","t21_02","t21_03",
            "t22_00","t22_01","t23_02","t23_03","t24_-1",
            "t30_-1","t30_00","t30_01","t31_02","t31_03",
            "t32_00","t32_01","t33_02","t33_03","t34_-1",
            # 风格牌16张
            "t03_00","t03_01","t03_02","t03_03",
            "t13_00","t13_01","t13_02","t13_03",
            "t23_00","t23_01","t23_02","t23_03",
            "t33_00","t33_01","t33_02","t33_03",
            # 额外的主牌和边界状态牌
            "major_0_0_0","major_0_0_1","major_0_0_2","major_0_0_3","major_0_1_0",
            "major_1_0_0","major_1_1_1","major_1_1_2","major_1_1_3","major_1_2_0",
            "major_2_0_0","major_2_0_1","major_2_0_2","major_2_0_3","major_2_1_0",
            "major_3_0_0","major_3_1_1","major_3_1_2","major_3_1_3","major_3_2_0",
            "major_x_x_x_00","major_x_x_x_11"
        ]
        return random.choice(full_deck)