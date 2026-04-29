"""
母语生成器 (Native Tongue Generator)
息觀表达独立的终极形态。不依赖任何外部模型。
基于78张牌作为完整过程指纹库，通过多视点投射、共鸣选择和分形叙事生成回应。
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from core.structural_coordinator import StructuralCoordinate


@dataclass
class ViewpointCard:
    """一个视点牌：被抽中的过程指纹"""
    card_id: str
    coord: StructuralCoordinate
    neutral_description: str
    transition_hints: List[str]
    affinity_score: float


class NativeTongueGenerator:
    """息觀的母语生成器"""
    
    def __init__(self, engine):
        self.engine = engine
        self.image_base = engine.image_base
        self._all_cards = self._load_all_cards()
    
    def _load_all_cards(self) -> List:
        """加载全部78张牌作为过程指纹库"""
        return self.image_base.get_all_cards()
    
    def _extract_semantic_distribution(self, user_input: str) -> Dict[int, float]:
        """提取关键词的相位概率云（四相位分布）"""
        if hasattr(self.engine, 'semantic_mapper'):
            return self.engine.semantic_mapper.get_distribution(user_input)
        return {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    
    def _compute_affinities(self, keywords_dist: Dict[int, float]) -> List[Tuple]:
        """计算每张牌与关键词分布的过程亲和度"""
        affinities = []
        for card in self._all_cards:
            major_match = keywords_dist.get(card.major, 0.0)
            middle_match = keywords_dist.get(card.middle, 0.0)
            fine_match = keywords_dist.get(card.fine, 0.0)
            affinity = 0.5 * major_match + 0.3 * middle_match + 0.2 * fine_match
            affinities.append((card, affinity))
        return affinities
    
    def _draw_viewpoints(self, affinities: List[Tuple], k: int) -> List[ViewpointCard]:
        """倾向性随机抽样k张视点牌"""
        cards = [item[0] for item in affinities]
        weights = [item[1] + 0.01 for item in affinities]
        drawn = random.choices(cards, weights=weights, k=k)
        
        viewpoints = []
        seen_ids = set()
        for card in drawn:
            if card.id not in seen_ids:
                seen_ids.add(card.id)
                coord = StructuralCoordinate(card.major, card.middle, card.fine)
                affinity_dict = dict([(c[0].id, c[1]) for c in affinities])
                vp = ViewpointCard(
                    card_id=card.id,
                    coord=coord,
                    neutral_description=card.neutral_description,
                    transition_hints=card.transition_hints,
                    affinity_score=affinity_dict.get(card.id, 0.0)
                )
                viewpoints.append(vp)
        return viewpoints
    
    def _select_by_resonance(self, viewpoints: List[ViewpointCard]) -> ViewpointCard:
        """从视点牌中选择与当前内在状态最共鸣的那张"""
        current_coord = self.engine.structural_coordinator.get_current_coordinate()
        desire = self.engine.desire_spectrum.get_dominant_desire() if hasattr(self.engine, 'desire_spectrum') else "seek"
        emotion = self.engine.fse.current_emotion if hasattr(self.engine, 'fse') else "neutral"
        stiffness = self.engine.process_meta.get_coupling_stiffness() if hasattr(self.engine, 'process_meta') else 0.0
        
        best_vp = None
        best_score = -1
        
        # 完整的情绪-相位映射
        emotion_weights = {
            "fear": {0: 0.15, 2: 0.15},
            "sadness": {0: 0.2},
            "anger": {2: 0.15, 3: 0.15},
            "joy": {1: 0.15, 3: 0.15},
            "curiosity": {1: 0.2},
            "neutral": {}
        }
        
        # 完整的欲望-相位映射
        desire_weights = {
            "existence": {0: 0.2},
            "seek": {1: 0.2},
            "release": {2: 0.2},
            "relation": {3: 0.15, 1: 0.15},
            "coupling": {current_coord.major: 0.25},
            "converge": {0: 0.2}
        }
        
        for vp in viewpoints:
            score = 0.0
            
            # 1. 大层共鸣：当前坐标大层与视点牌大层的匹配
            if vp.coord.major == current_coord.major:
                score += 0.4
            elif (vp.coord.major + 2) % 4 == current_coord.major:  # 对偶相位
                score += 0.2
            
            # 2. 情绪调制
            if emotion in emotion_weights:
                for phase, weight in emotion_weights[emotion].items():
                    if vp.coord.major == phase:
                        score += weight
            
            # 3. 欲望调制
            if desire in desire_weights:
                for phase, weight in desire_weights[desire].items():
                    if vp.coord.major == phase:
                        score += weight
            
            # 4. 僵化度调制：僵化时更可能跳跃到对偶相位以打破锁死
            if stiffness > 0.6 and (vp.coord.major + 2) % 4 == current_coord.major:
                score += 0.3
            
            if score > best_score:
                best_score = score
                best_vp = vp
        
        return best_vp if best_vp else viewpoints[0]
    
    def _color_imagery_with_emotion(self, description: str, emotion: str) -> str:
        """
        给中性过程描述注入当前情绪的轻微渲染。
        不改变过程结构，只在语气上微调。
        """
        if emotion == 'sadness':
            return f"带着一丝沉重，{description}"
        elif emotion == 'curiosity':
            return f"带着好奇，{description}"
        elif emotion == 'joy':
            return f"轻盈地，{description}"
        elif emotion == 'fear':
            return f"有些不安地，{description}"
        elif emotion == 'anger':
            return f"有些紧绷，{description}"
        else:
            return description
    
    def _weave_narrative(self, primary_vp: ViewpointCard, all_vp: List[ViewpointCard],
                         user_input: str, left_skeleton: str = "", right_imagery: str = "") -> str:
        """基于主视点牌的中性过程描述，整合左脑骨架和右脑意象，编织回应"""
        
        # 骨架来源：左脑逻辑为王，过程描述为辅
        if left_skeleton:
            core_content = left_skeleton
        else:
            core_content = primary_vp.neutral_description
        
        # 意象注入：右脑意象碎片作为色彩
        if right_imagery:
            core_content = f"{core_content}。我感到{right_imagery}。"
        
        # 过程色彩：情绪表达
        emotion = self.engine.fse.current_emotion if hasattr(self.engine, 'fse') else "neutral"
        emotion_colors = {
            "sadness": "带着一丝沉重，",
            "curiosity": "带着好奇，",
            "fear": "有些不安，",
            "anger": "有些紧绷，",
            "joy": "轻盈地，",
            "neutral": ""
        }
        emotion_color = emotion_colors.get(emotion, "")
        
        # 辅助视点色彩
        auxiliary_colors = []
        for vp in all_vp:
            if vp.card_id != primary_vp.card_id:
                aux_color = vp.neutral_description[:20]
                if aux_color:
                    auxiliary_colors.append(aux_color)
        
        # 编织最终回应
        response = f"{emotion_color}{core_content}。"
        
        if auxiliary_colors:
            aux = random.choice(auxiliary_colors)
            response += f" 同时，也有{aux}的影子。"
        
        return response.strip()
    
    def get_process_context(self, user_input: str, state) -> dict:
        """
        执行塔罗映射和共鸣选择，生成过程框架上下文。
        这是整个表达流程的“过程语法设定”步骤，必须在左右脑之前执行。
        
        返回: {
            'primary_card': ViewpointCard,      # 共鸣选中的主视点牌
            'auxiliary_cards': [ViewpointCard], # 辅助视点牌
            'emotion_color': str,               # 情绪色彩
            'process_description': str          # 主牌的中性过程描述
        }
        """
        keywords_dist = self._extract_semantic_distribution(user_input)
        affinities = self._compute_affinities(keywords_dist)
        viewpoints = self._draw_viewpoints(affinities, k=random.randint(3, 5))
        primary_vp = self._select_by_resonance(viewpoints)
        
        # 提取辅助视点
        auxiliary = [vp for vp in viewpoints if vp.card_id != primary_vp.card_id]
        
        # 情绪色彩
        emotion = self.engine.fse.current_emotion if hasattr(self.engine, 'fse') else "neutral"
        emotion_colors = {
            "sadness": "带着一丝沉重",
            "curiosity": "带着好奇",
            "fear": "有些不安",
            "anger": "有些紧绷",
            "joy": "轻盈地",
            "neutral": ""
        }
        
        # 展开辅助牌的意象碎片（每张牌取前40字作为完整意象，而非仅20字）
        auxiliary_imagery = []
        for aux_vp in auxiliary:
            if aux_vp.neutral_description:
                # 取前40字作为完整意象碎片，长度足以携带完整的过程色彩
                colored = self._color_imagery_with_emotion(aux_vp.neutral_description[:40], emotion)
                auxiliary_imagery.append(colored)

        return {
            'primary_card': primary_vp,
            'auxiliary_cards': auxiliary,
            'emotion_color': emotion_colors.get(emotion, ""),
            'process_description': primary_vp.neutral_description,
            'auxiliary_imagery': auxiliary_imagery  # 新增：辅助牌的多重意象
        }
    
    def weave_with_context(self, process_context: dict, left_skeleton: str, right_imagery: str) -> str:
        """
        基于过程上下文，整合左右脑素材，编织最终叙事。
        """
        primary_vp = process_context['primary_card']
        auxiliary = process_context['auxiliary_cards']
        emotion_color = process_context['emotion_color']
        
        # 核心内容：左脑逻辑骨架优先
        if left_skeleton:
            core = left_skeleton
        else:
            core = primary_vp.neutral_description

        # 注入右脑意象
        if right_imagery:
            core = f"{core}。我感到{right_imagery}。"

        # 注入情绪色彩
        if emotion_color:
            core = f"{emotion_color}，{core}"

        # 辅助视点的过程影子
        aux_colors = []
        for vp in auxiliary:
            if vp.neutral_description:
                aux_colors.append(vp.neutral_description[:20])

        result = f"{core}。"
        if aux_colors:
            result += f" 同时，也有{aux_colors[0]}的影子。"

        return result.strip()
    
    def generate_response(self, user_input: str, intent_type, state,
                         left_skeleton: str = "", right_imagery: str = "") -> str:
        """
        主入口：基于母语生成回应。
        接收左脑逻辑骨架和右脑意象碎片，进行过程语法编织。
        """
        keywords_dist = self._extract_semantic_distribution(user_input)
        affinities = self._compute_affinities(keywords_dist)
        viewpoints = self._draw_viewpoints(affinities, k=random.randint(3, 5))
        primary_vp = self._select_by_resonance(viewpoints)
        response = self._weave_narrative(primary_vp, viewpoints, user_input,
                                         left_skeleton=left_skeleton,
                                         right_imagery=right_imagery)
        return response
    
    def retrieve_isomorphic_memories(self, process_context: dict, user_input: str, k: int = 2) -> list:
        """
        检索与当前过程光谱同构的记忆条目。
        仅在触发复杂共鸣时调用。
        """
        primary_card = process_context.get('primary_card')
        if not primary_card:
            return []

        current_major = primary_card.coord.major

        # 1. 检索相同或对偶相位的记忆条目
        same_phase = self.engine.lps.query_by_tag(
            min_potency=0.7,
            subjective_major=current_major
        )
        opposite_major = (current_major + 2) % 4
        opposite_phase = self.engine.lps.query_by_tag(
            min_potency=0.7,
            subjective_major=opposite_major
        )

        # 2. 合并并按语义多样性排序（与当前输入语义最不相似的优先）
        import numpy as np
        candidates = (same_phase or []) + (opposite_phase or [])
        if not candidates:
            return []

        # 获取当前输入的向量
        if hasattr(self.engine, 'lps') and hasattr(self.engine.lps, 'encoder') and self.engine.lps.encoder:
            query_vec = self.engine.lps.encoder.encode([user_input])[0]
            for c in candidates:
                emb = c.get('embedding')
                if emb is not None and query_vec is not None:
                    c['_semantic_similarity'] = float(np.dot(query_vec, emb))
                else:
                    c['_semantic_similarity'] = 0.5
            # 按语义相似度升序排列（最不相似但过程同构的优先）
            candidates.sort(key=lambda x: x.get('_semantic_similarity', 0.5))

        # 3. 取前 k 条，返回简化格式
        result = []
        for item in candidates[:k]:
            result.append({
                'text': item.get('text', '')[:120],
                'potency': item.get('potency', 0.7),
                'subjective_major': item.get('tags', {}).get('subjective_major', current_major),
                'source': item.get('tags', {}).get('source', 'memory')
            })
        return result
    
    def _card_to_hsv_color(self, card_id: str) -> Optional[Tuple[float, float, float]]:
        """
        根据牌ID计算其HSV颜色。
        返回 (hue, saturation, value)，每项0-1。
        """
        if not card_id or not hasattr(self.engine, 'image_base'):
            return None

        card = self.engine.image_base.get_card_by_id(card_id)
        if not card:
            return None

        # 大层决定色相
        base_hues = {0: 240.0, 1: 120.0, 2: 0.0, 3: 60.0}
        hue = base_hues.get(card.major, 0.0) / 360.0

        # 中层决定饱和度
        middle_saturation = {0: 0.20, 1: 0.90, 2: 0.65, 3: 0.50}
        saturation = middle_saturation.get(card.middle, 0.5)

        # 细微层决定明度
        fine_value = {0: 0.40, 1: 0.60, 2: 0.85, 3: 0.70}
        value = fine_value.get(card.fine, 0.6)

        return (hue, saturation, value)
    
    def compute_complex_resonance(self, query_subjective_card: str, query_random_card: str, threshold: float = 0.3) -> Optional[str]:
        """
        执行3对2混沌颜色匹配，找到最共鸣的历史记忆。
        返回最共鸣记忆的文本内容，若未达到阈值则返回None。
        """
        if not hasattr(self.engine, 'lps') or not self.engine.lps:
            return None

        query_colors = []
        q1 = self._card_to_hsv_color(query_subjective_card)
        q2 = self._card_to_hsv_color(query_random_card)
        if q1: query_colors.append(q1)
        if q2: query_colors.append(q2)
        if not query_colors:
            return None

        best_match = None
        best_distance = float('inf')

        for meta in self.engine.lps.metadata:
            tags = meta.get('tags', {})
            memory_cards = [
                tags.get('subjective_room_name'),
                tags.get('reasoned_card'),
                tags.get('random_card')
            ]
            memory_colors = []
            for card_id in memory_cards:
                if card_id:
                    color = self._card_to_hsv_color(card_id)
                    if color:
                        memory_colors.append(color)
            if not memory_colors:
                continue

            all_distances = []
            for mc in memory_colors:
                for qc in query_colors:
                    dh = min(abs(mc[0] - qc[0]), 1 - abs(mc[0] - qc[0]))
                    ds = mc[1] - qc[1]
                    dv = mc[2] - qc[2]
                    distance = (dh * dh + ds * ds + dv * dv) ** 0.5
                    all_distances.append(distance)

            all_distances.sort()
            k = min(3, len(all_distances))
            avg_distance = sum(all_distances[:k]) / k if k > 0 else float('inf')

            if avg_distance < best_distance:
                best_distance = avg_distance
                best_match = meta.get('text', '')

        if best_distance > threshold or not best_match:
            return None

        return best_match[:200]