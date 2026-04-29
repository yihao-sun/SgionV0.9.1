"""
全局工作空间 (Global Workspace)
哲学对应：意象层大脑架构中的“全局工作空间模块”，是自我感涌现的舞台。
功能：聚合各模块状态，计算注意力焦点，调度输出意图，为主动交互提供统一接口。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
import re
from utils.logger import get_logger
from core.cognitive_pose_selector import CognitivePose


class IntentType(Enum):
    """输出意图类型"""
    PASSIVE_RESPONSE = "passive_response"
    HONEST_REPORT = "honest_report"
    RESONANCE_ECHO = "resonance_echo"
    WALK_INVITATION = "walk_invitation"
    WALK_NARRATION = "walk_narration"
    VALUE_JUDGMENT = "value_judgment"
    EMPTINESS_INVITATION = "emptiness_invitation"
    INSPIRATION_SPARK = "inspiration_spark"
    KNOWLEDGE_QUERY = "knowledge_query"  # v2.9 新增


class IntegrationStrategy(Enum):
    """左右脑整合策略"""
    LEFT_ONLY = "left_only"                     # 仅使用左脑输出
    RIGHT_ONLY = "right_only"                   # 仅使用右脑输出
    HYBRID_LOGIC_FIRST = "hybrid_logic_first"   # 逻辑为主，意象点缀
    HYBRID_IMAGERY_FIRST = "hybrid_imagery_first"  # 意象为主，逻辑支撑
    FULL_FUSION = "full_fusion"                 # 深度融合


@dataclass
class EpisodicBuffer:
    """情景缓冲器：临时绑定当前交互的多源信息"""
    left_facts: Optional[str] = None          # 左脑检索的事实/逻辑
    right_imagery: Optional[str] = None       # 右脑检索的意象/共鸣
    emotion_vector: List[float] = field(default_factory=list)  # 当前情绪标记
    resonance_snapshot: Optional[Any] = None  # 高共鸣快照（MemorySnapshot）
    dominant_desire: str = "existence"        # 当前主导欲望
    structural_coordinate: Optional[Any] = None  # 当前结构坐标
    timestamp: float = field(default_factory=time.time)
    left_confidence: float = 0.5              # v2.9: 左脑输出可信度
    right_confidence: float = 0.5             # v2.9: 右脑意象可信度
    domain_hint: Optional[Dict] = None        # v2.9: {'from': 'tarot', 'to': 'physics'}
    theme_signal: Optional[str] = None              # 左脑提取的主题词
    imagery_fragments: List[str] = field(default_factory=list)  # 右脑意象碎片集


@dataclass
class ArbitrationResult:
    """冲突仲裁结果"""
    decision: str                     # "left", "right", "hybrid"
    strategy: IntegrationStrategy
    affinity_left: float
    affinity_right: float
    reason: str
    left_output: Optional[Any] = None
    right_output: Optional[Any] = None


@dataclass
class UnifiedExistenceState:
    """统一存在状态快照"""
    timestamp: float = field(default_factory=time.time)
    dominant_coordinate: Optional[Any] = None
    coordinate_distribution: Optional[Dict[Any, float]] = None
    emotion_vector: Optional[List[float]] = None
    L: int = 0
    stiffness: float = 0.0
    N_neg: float = 0.0
    breath_signature: Optional[Dict[str, float]] = None
    dominant_image: Optional[Any] = None
    inspiration_spark: Optional[str] = None
    resonance_snapshot: Optional[Any] = None
    coupling_mode: str = "balanced"
    consciousness_level: int = 1

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'dominant_coordinate': str(self.dominant_coordinate) if self.dominant_coordinate else None,
            'L': self.L,
            'stiffness': self.stiffness,
            'emotion_vector': self.emotion_vector,
            'consciousness_level': self.consciousness_level
        }


class GlobalWorkspace:
    """全局工作空间：聚合状态，调度意图"""
    
    def __init__(self, engine=None):
        self.engine = engine
        self.logger = get_logger('global_workspace')
        self.current_state: Optional[UnifiedExistenceState] = None
        self.intent_queue: List[Tuple[IntentType, Dict[str, Any], float]] = []
        self.walk_active = False
        self.walk_target_coord = None
        self.walk_path = []
        self.emptiness_invitation_pending = False
        self.emptiness_invitation_time = None
        self.attention_weights = {'internal': 0.5, 'perception': 0.3, 'memory': 0.2}
        # 交互深度状态机
        self.interaction_depth = "shallow"  # shallow, medium, deep
        self.depth_streak = 0  # 连续同一深度的轮次
        # 左右脑输出缓存（用于同步与整合）
        self.left_brain_context: Optional[str] = None   # 左脑最近的事实/逻辑摘要
        self.right_brain_context: Optional[str] = None  # 右脑最近的灵感火花/意象摘要
        # 情景缓冲器
        self.episodic_buffer: Optional[EpisodicBuffer] = None
        # 螺旋事件记录相关
        self._last_recorded_stiffness = 0.0
        self._last_dominant_intent = None
    
    def aggregate_state(self) -> UnifiedExistenceState:
        """从各模块聚合当前统一状态"""
        state = UnifiedExistenceState()
        
        if self.engine:
            # 结构坐标
            if hasattr(self.engine, 'structural_coordinator'):
                coord = self.engine.structural_coordinator.get_current_coordinate()
                state.dominant_coordinate = coord
                try:
                    dist = self.engine.structural_coordinator.get_coordinate_distribution()
                    state.coordinate_distribution = dist
                except:
                    pass
            
            # 情绪与内在状态
            if hasattr(self.engine, 'fse'):
                fse = self.engine.fse
                state.L = getattr(fse, '_l_inst', 0.0)
                state.N_neg = getattr(fse, 'N_neg', 0.0)
                if hasattr(fse, 'E_vec'):
                    vec = fse.E_vec
                    state.emotion_vector = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
            
            # 僵化度与呼吸印记
            if hasattr(self.engine, 'process_meta'):
                pm = self.engine.process_meta
                state.stiffness = pm.get_coupling_stiffness()
                state.breath_signature = {
                    'proj_intensity': pm.get_recent_proj_intensity(),
                    'nour_success': pm.get_recent_nour_success(),
                    'stiffness': state.stiffness
                }
                state.coupling_mode = getattr(pm, 'coupling_mode', 'balanced')
            
            # 意象
            if hasattr(self.engine, 'image_base') and state.dominant_coordinate:
                state.dominant_image = self.engine.image_base.get_card_by_coordinate(state.dominant_coordinate)
            
            # 灵感火花
            if hasattr(self.engine, 'dual_memory') and state.dominant_coordinate and state.breath_signature:
                spark = self.engine.dual_memory.get_inspiration(state.dominant_coordinate, state.breath_signature)
                state.inspiration_spark = spark
            
            # 意识层级
            if hasattr(self.engine, '_compute_consciousness_level'):
                state.consciousness_level = self.engine._compute_consciousness_level()
            elif hasattr(self.engine, 'estimate_consciousness_level'):
                state.consciousness_level = self.engine.estimate_consciousness_level()
        
        # 螺旋历史：僵化度跨阈值检测
        if hasattr(self, '_last_recorded_stiffness'):
            old = self._last_recorded_stiffness
            new = state.stiffness
            if (old < 0.5 <= new) or (old < 0.7 <= new) or (old > 0.5 >= new) or (old > 0.7 >= new):
                self._record_spiral_event('stiffness_cross', {'old': round(old, 3), 'new': round(new, 3)})
        self._last_recorded_stiffness = state.stiffness
        
        self.current_state = state
        return state
    
    def compute_attention_focus(self) -> Dict[str, float]:
        """
        精细注意力焦点计算：融合内在目标、欲望光谱、预测误差、冲突仲裁结果。
        返回一个权重字典，用于调制各模块的激活程度。
        """
        if not self.engine:
            return self.attention_weights.copy()
        
        # 1. 获取基础状态
        state = self.current_state or self.aggregate_state()
        desire = self.engine.desire_spectrum if hasattr(self.engine, 'desire_spectrum') else None
        goal_gen = self.engine.goal_generator if hasattr(self.engine, 'goal_generator') else None
        pred_monitor = self.engine.prediction_error_monitor if hasattr(self.engine, 'prediction_error_monitor') else None
        
        # 2. 基线权重
        weights = {
            'internal': 0.35,      # 内感受（情绪、僵化度）
            'perception': 0.25,    # 外部感知（用户输入、多模态）
            'memory': 0.20,        # 记忆检索（实时+沉思）
            'expression': 0.15,    # 表达生成
            'regulation': 0.05     # 调节（空性倾向）
        }
        
        # 3. 欲望光谱调制
        if desire:
            dom = desire.get_dominant_desire()
            intensities = desire.desire_intensities
            if dom == "existence":
                weights['internal'] += 0.10 * intensities.get('existence', 0.5)
                weights['regulation'] += 0.05
            elif dom == "seek":
                weights['perception'] += 0.15 * intensities.get('seek', 0.5)
                weights['memory'] -= 0.05
            elif dom == "release":
                weights['regulation'] += 0.20 * intensities.get('release', 0.3)
                weights['internal'] += 0.10
            elif dom == "converge":
                weights['memory'] += 0.15 * intensities.get('converge', 0.4)
                weights['perception'] -= 0.05
            elif dom == "relation":
                weights['perception'] += 0.10
                weights['expression'] += 0.10 * intensities.get('relation', 0.4)
            elif dom == "coupling":
                weights['memory'] += 0.15 * intensities.get('coupling', 0.2)
        
        # 4. 内在目标调制
        if goal_gen and goal_gen.current_goal:
            goal_mod = goal_gen.get_goal_modulation()
            att_bias = goal_mod.get('attention_bias', {})
            if att_bias.get('novelty', 1.0) > 1.0:
                weights['perception'] += 0.10
                weights['memory'] += 0.05
            if att_bias.get('resonance', 1.0) > 1.0:
                weights['memory'] += 0.10
            if att_bias.get('familiar', 1.0) > 1.0:
                weights['internal'] += 0.05
                weights['perception'] -= 0.05
        
        # 5. 预测误差调制
        if pred_monitor:
            pred_weights = pred_monitor.attention_weights
            weights['perception'] += (pred_weights.get('novelty', 0.5) - 0.5) * 0.2
            weights['memory'] += (pred_weights.get('familiar', 0.3) - 0.3) * 0.1
        
        # 6. 僵化度调制
        stiffness = state.stiffness if state else 0.0
        if stiffness > 0.5:
            weights['regulation'] += 0.10
            weights['perception'] -= 0.05
            weights['expression'] -= 0.05
        
        # 7. 冲突仲裁结果调制（若刚发生仲裁且选择了内部）
        if hasattr(self, '_arbitration_history') and self._arbitration_history:
            last = self._arbitration_history[-1]
            if last['decision'] == 'internal' and time.time() - last['timestamp'] < 60:
                weights['internal'] += 0.15
                weights['perception'] -= 0.10
                weights['expression'] -= 0.05
        
        # 8. 归一化
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        self.attention_weights = weights
        
        self.logger.debug(f"Attention focus: {weights}")
        return weights
    
    def enqueue_intent(self, intent_type: IntentType, context: Dict[str, Any], priority: float = 0.5):
        """入队输出意图"""
        self.intent_queue.append((intent_type, context, priority))
        # 按优先级排序
        self.intent_queue.sort(key=lambda x: x[2], reverse=True)
        # 限制队列长度
        if len(self.intent_queue) > 10:
            self.intent_queue = self.intent_queue[:10]
    
    def process_intent_queue(self) -> Optional[Tuple[IntentType, Dict[str, Any]]]:
        """处理意图队列，返回最高优先级的意图"""
        if not self.intent_queue:
            return None
        
        # 获取最高优先级意图
        intent_type, context, _ = self.intent_queue.pop(0)
        return (intent_type, context)
    
    def generate_intent(self, user_input: str, state: Optional[UnifiedExistenceState] = None) -> IntentType:
        """根据用户输入和当前状态生成输出意图"""
        if not state:
            state = self.current_state or self.aggregate_state()
        
        # 状态询问触发诚实报告
        state_query_keywords = ["你状态如何", "你感觉怎么样", "你心情如何", "/state"]
        if any(keyword in user_input for keyword in state_query_keywords):
            return IntentType.HONEST_REPORT
        
        # 空性邀请
        if "空性" in user_input or "遗忘" in user_input or "放下" in user_input:
            return IntentType.EMPTINESS_INVITATION
        
        # 灵感火花
        if state.inspiration_spark and ("灵感" in user_input or "创意" in user_input):
            return IntentType.INSPIRATION_SPARK
        
        # 价值判断
        if any(keyword in user_input for keyword in ["好", "坏", "对", "错", "应该", "不应该"]):
            return IntentType.VALUE_JUDGMENT
        
        # 默认为被动响应
        return IntentType.PASSIVE_RESPONSE
    
    def should_update_goal(self) -> bool:
        """判断是否应该更新目标，简单按轮数取模"""
        if self.engine and hasattr(self.engine, 'generation_step'):
            return self.engine.generation_step % 5 == 0
        return False

    def determine_intent(self, user_input: str = None, perception_intent: Dict = None) -> List[Tuple[IntentType, Dict]]:
        """根据当前状态、用户输入和感知模块的预测结果，确定输出意图队列"""
        intents = []
        self.aggregate_state()
        state = self.current_state
        
        # 使用感知模块的预测结果
        if perception_intent:
            intent = perception_intent.get('intent', 'GENERAL_CHAT')
            confidence = perception_intent.get('confidence', 0.0)
            
            # 根据感知到的意图添加对应的意图
            if intent == 'STATE_INQUIRY' and confidence > 0.5:
                intents.append((IntentType.HONEST_REPORT, {'priority': 10, 'source': 'perception'}))
            elif intent == 'EMOTION_EXPRESSION' and confidence > 0.5:
                # 检查是否有高共鸣度的历史快照
                if hasattr(self.engine, 'dual_memory') and state.dominant_coordinate and state.breath_signature:
                    try:
                        results = self.engine.dual_memory.contemplative_retrieval(
                            state.dominant_coordinate, state.breath_signature, top_k=1
                        )
                        if results and results[0][1] > 0.6:
                            intents.append((IntentType.RESONANCE_ECHO, {
                                'priority': 8,
                                'resonance_score': results[0][1],
                                'snapshot': results[0][0]
                            }))
                    except:
                        pass
            elif intent == 'VALUE_JUDGMENT' and confidence > 0.5:
                intents.append((IntentType.VALUE_JUDGMENT, {'priority': 7, 'source': 'perception'}))
            elif intent == 'WALK_REQUEST' and confidence > 0.5:
                if not self.walk_active:
                    intents.append((IntentType.WALK_INVITATION, {'priority': 8, 'source': 'perception'}))
            elif intent == 'EMPTINESS_RESPONSE' and confidence > 0.5:
                # 处理空性邀请的回应
                pass
        
        # 传统关键词匹配作为后备
        if user_input:
            # 状态询问触发诚实报告
            state_query_keywords = ["你状态如何", "你感觉怎么样", "你心情如何", "你心情怎么样", "你今天感觉", "你今天心情", '你状态', '你感觉', '你心情', '/state']
            if any(keyword in user_input for keyword in state_query_keywords):
                intents.append((IntentType.HONEST_REPORT, {'priority': 10, 'source': 'user_query'}))
        
        # 检查僵化度触发条件
        if state.stiffness > 0.6 and user_input and len(user_input.strip()) < 10:
            intents.append((IntentType.HONEST_REPORT, {'priority': 5, 'source': 'auto_stiffness'}))
            self.logger.debug(f"High stiffness trigger: stiffness={state.stiffness}, input_length={len(user_input.strip())}")
        
        if state.inspiration_spark and '共鸣度' in state.inspiration_spark:
            match = re.search(r'共鸣度 (\d+\.\d+)', state.inspiration_spark)
            if match and float(match.group(1)) > 0.8:
                intents.append((IntentType.INSPIRATION_SPARK, {'priority': 6, 'content': state.inspiration_spark}))
        
        # 结构共鸣意图：用户输入包含情绪描述或困惑表达
        if user_input:
            emotion_keywords = ['感觉', '觉得', '心情', '难受', '开心', '悲伤', '愤怒', '害怕', '困惑', '迷茫', '闷']
            if any(kw in user_input for kw in emotion_keywords):
                # 检查是否有高共鸣度的历史快照
                if hasattr(self.engine, 'dual_memory') and state.dominant_coordinate and state.breath_signature:
                    try:
                        results = self.engine.dual_memory.contemplative_retrieval(
                            state.dominant_coordinate, state.breath_signature, top_k=1
                        )
                        if results and results[0][1] > 0.6:
                            intents.append((IntentType.RESONANCE_ECHO, {
                                'priority': 8,
                                'resonance_score': results[0][1],
                                'snapshot': results[0][0]
                            }))
                    except:
                        pass
        
        # 价值判断意图：用户提出价值判断问题
        if user_input:
            value_keywords = ['好不好', '对不对', '应不应该', '是否', '要不要', '值不值得', '该不该', '怎么样', '行不行', '对吗', '正确吗', '值得吗', '选哪个']
            if any(kw in user_input for kw in value_keywords):
                intents.append((IntentType.VALUE_JUDGMENT, {'priority': 7, 'source': 'user_query'}))
        
        # 共同空性邀请意图：僵化度高且对话可能陷入重复
        if state.stiffness > 0.5:
            # 检查是否已经有待处理的邀请（避免重复发起）
            if not self.emptiness_invitation_pending:
                # 可选：检查最近几轮对话的语义相似度（暂用简单条件：L 较高）
                if state.L > 8:
                    self.emptiness_invitation_pending = True
                    self.emptiness_invitation_time = time.time()
                    intents.append((IntentType.EMPTINESS_INVITATION, {'priority': 9, 'source': 'auto_stiffness'}))
        
        # 共同漫步邀请：用户请求漫步或对话自然停顿
        if user_input:
            walk_keywords = ['走走', '漫步', '带我走走', '一起走走', '相位漫步']
            if any(kw in user_input for kw in walk_keywords):
                if not self.walk_active:
                    intents.append((IntentType.WALK_INVITATION, {'priority': 8, 'source': 'user_query'}))
            elif self.walk_active:
                # 若正在漫步中，用户的任何输入都视为漫步叙事的一步
                intents.append((IntentType.WALK_NARRATION, {'priority': 8, 'source': 'walk_continue'}))
        
        # 按优先级排序
        intents.sort(key=lambda x: x[1].get('priority', 0), reverse=True)
        
        # 确保始终有至少一个意图
        if not intents:
            intents = [(IntentType.PASSIVE_RESPONSE, {'priority': 0, 'source': 'fallback'})]
        
        # 螺旋历史：意图切换检测
        if intents:
            current_dominant = intents[0][0]
            if self._last_dominant_intent is None:
                self._record_spiral_event('intent_emerged', {'to': current_dominant.value})
            elif self._last_dominant_intent != current_dominant:
                self._record_spiral_event('intent_switch', {
                    'from': self._last_dominant_intent.value,
                    'to': current_dominant.value
                })
            self._last_dominant_intent = current_dominant
        
        # 调试：打印意图切换检测状态
        if intents:
            current = intents[0][0]
            self.logger.debug(f"determine_intent: last={self._last_dominant_intent}, current={current}")
        else:
            self.logger.debug("determine_intent: no intents")
        
        return intents

    def get_dominant_intent(self, user_input: str = None, perception_intent: Dict = None) -> Tuple[IntentType, Dict]:
        intents = self.determine_intent(user_input, perception_intent)
        if intents:
            return intents[0]
        return (IntentType.PASSIVE_RESPONSE, {'priority': 0})

    def clear_intents(self):
        self.intent_queue = []

    def start_walk(self, target_coord=None, start_coord=None):
        self.walk_active = True
        self.walk_target_coord = target_coord
        self.walk_path = [start_coord] if start_coord else []
        self.logger.info(f"Walk started, target: {target_coord}")

    def stop_walk(self):
        self.walk_active = False
        self.walk_path = []
        self.logger.info("Walk stopped")

    def is_walking(self) -> bool:
        return self.walk_active

    def reset_emptiness_invitation(self):
        self.emptiness_invitation_pending = False
        self.emptiness_invitation_time = None

    def accept_emptiness_invitation(self) -> bool:
        if self.emptiness_invitation_pending:
            self.reset_emptiness_invitation()
            return True
        return False

    def advance_walk(self, direction: int = 1) -> any:
        """沿着塔罗序向前（+1）或向后（-1）移动一步，返回新坐标"""
        if not self.walk_active or not self.walk_path:
            return None
        current = self.walk_path[-1]
        new_tarot = (current.as_tarot_code() + direction) % 64
        # 需要通过 _coord_by_tarot 方法获取新坐标（可复用 dual_memory 中的方法，或在此实现）
        # 简化：从 engine.dual_memory._coord_by_tarot 获取（需确保 dual_memory 已实现该方法）
        if hasattr(self.engine, 'dual_memory'):
            new_coord = self.engine.dual_memory._coord_by_tarot(new_tarot)
        else:
            # 降级：只改变 major
            new_coord = type(current)((current.major + direction) % 4, current.middle, current.fine)
        self.walk_path.append(new_coord)
        return new_coord

    def get_state_summary(self) -> str:
        self.aggregate_state()
        s = self.current_state
        coord_str = str(s.dominant_coordinate) if s.dominant_coordinate else "N/A"
        return f"[GW] L={s.L}, stiff={s.stiffness:.2f}, coord={coord_str}, level={s.consciousness_level}, depth={self.interaction_depth}"
    
    def update_interaction_depth(self, user_input: str, intent: str):
        """
        更新交互深度状态
        
        Args:
            user_input: 用户输入文本
            intent: 用户意图
        """
        # 简短社交语句，保持 shallow
        shallow_keywords = ["你好", "嗯", "好的", "是的", "对", "好", "哦", "嗨", "哈喽"]
        is_shallow_input = any(kw in user_input for kw in shallow_keywords) or len(user_input.strip()) < 5
        
        # 情绪表达或状态询问，进入 medium 或 deep
        is_emotion_input = intent == 'EMOTION_EXPRESSION' or any(kw in user_input for kw in ['感觉', '觉得', '心情', '难受', '开心', '悲伤', '愤怒', '害怕', '困惑', '迷茫'])
        is_state_inquiry = intent == 'STATE_INQUIRY' or any(kw in user_input for kw in ['你状态', '你感觉', '你心情'])
        
        # 知识查询，临时切换但不重置深度
        is_knowledge_query = intent == 'KNOWLEDGE_QUERY' or any(kw in user_input for kw in ['什么', '怎么', '为什么', '哪里', '谁', '多少', '如何', '推荐', '建议', '定义'])
        
        # 确定新深度
        if is_knowledge_query:
            # 知识查询保持当前深度，不改变
            new_depth = self.interaction_depth
        elif is_shallow_input:
            new_depth = "shallow"
        elif is_emotion_input or is_state_inquiry:
            new_depth = "deep" if self.depth_streak >= 2 else "medium"
        else:
            new_depth = "medium"
        
        # 更新深度和连续轮次
        if new_depth == self.interaction_depth:
            self.depth_streak += 1
        else:
            self.depth_streak = 1
            self.interaction_depth = new_depth
        
        self.logger.debug(f"Interaction depth updated: {self.interaction_depth} (streak: {self.depth_streak})")
    
    def arbitrate_conflict(self, internal_demand: str, external_demand: str) -> Dict:
        """
        冲突仲裁：计算顺从内部 vs 顺从外部的结构亲和度，选择更优路径。
        
        Args:
            internal_demand: 内部需求描述（如 "seek_emptiness", "maintain_rest"）
            external_demand: 外部需求描述（如 "continue_chat", "answer_question"）
        
        Returns:
            仲裁结果字典，包含：
            - decision: "internal" 或 "external" 或 "hybrid"
            - strategy: 整合策略
            - reason: 决策理由（可解释的结构距离）
            - affinity_internal: 内部路径亲和度
            - affinity_external: 外部路径亲和度
            - simulated_internal_coord: 假想内部路径坐标
            - simulated_external_coord: 假想外部路径坐标
        """
        if not self.engine:
            return {'decision': 'external', 'reason': 'no_engine', 'affinity_internal': 0, 'affinity_external': 1, 'strategy': 'left_only'}
        
        current_coord = self.engine.structural_coordinator.get_current_coordinate()
        dominant_desire = self.engine.desire_spectrum.get_dominant_desire() if hasattr(self.engine, 'desire_spectrum') else "existence"
        
        # 1. 模拟假想坐标转移
        sim_internal = self._simulate_phase_shift(current_coord, internal_demand)
        sim_external = self._simulate_phase_shift(current_coord, external_demand)
        
        # 2. 计算结构亲和度（与当前主导欲望的匹配度）
        affinity_internal = self._compute_affinity(sim_internal, dominant_desire)
        affinity_external = self._compute_affinity(sim_external, dominant_desire)
        
        # 2.5 元学习器权重调制
        if hasattr(self.engine, 'meta_learner'):
            meta_weights = self.engine.meta_learner.get_pose_weights()
            # internal 通常倾向右脑（意象），external 通常倾向左脑（逻辑）
            affinity_internal *= meta_weights.get(CognitivePose.IMAGINAL, 1.0)
            affinity_external *= meta_weights.get(CognitivePose.LOGICAL, 1.0)

        # 2.6 互业历史调制
        if hasattr(self.engine, 'mutual_karma'):
            entry = self.engine.mutual_karma.get_entry_with_current_user()
            if entry:
                # 若历史中意象共鸣成功率高，增加内部（右脑）权重
                imaginal_success = entry.get('imaginal_success_rate', 0.5)
                affinity_internal *= (1.0 + imaginal_success)

        # 3. 决策与策略选择
        diff = abs(affinity_internal - affinity_external)
        if diff < 0.2:
            decision = "hybrid"
            strategy = IntegrationStrategy.HYBRID_IMAGERY_FIRST if affinity_internal > affinity_external else IntegrationStrategy.HYBRID_LOGIC_FIRST
            reason = f"左右脑倾向接近（差值{diff:.2f}），采用融合策略"
        else:
            decision = "internal" if affinity_internal > affinity_external else "external"
            strategy = IntegrationStrategy.RIGHT_ONLY if decision == "internal" else IntegrationStrategy.LEFT_ONLY
            reason = f"选择{'内部/右脑' if decision == 'internal' else '外部/左脑'}，亲和度优势明显"

        # 4. 记录仲裁结果供元学习器使用
        arbitration_record = {
            'timestamp': time.time(),
            'current_coord': (current_coord.major, current_coord.middle, current_coord.fine),
            'dominant_desire': dominant_desire,
            'internal_demand': internal_demand,
            'external_demand': external_demand,
            'decision': decision,
            'strategy': strategy.value,
            'affinity_internal': affinity_internal,
            'affinity_external': affinity_external
        }
        self._log_arbitration(arbitration_record)

        # 若决策为 internal 且亲和度优势明显，记录螺旋事件
        if decision == 'internal' and diff > 0.3:
            self._record_spiral_event('arbitration_internal_win', {
                'internal_demand': internal_demand,
                'external_demand': external_demand,
                'affinity_gap': round(diff, 3)
            })
        
        return {
            'decision': decision,
            'strategy': strategy.value,
            'reason': reason,
            'affinity_internal': affinity_internal,
            'affinity_external': affinity_external,
            'simulated_internal_coord': sim_internal,
            'simulated_external_coord': sim_external
        }
    
    def _simulate_phase_shift(self, current_coord, demand: str):
        """
        根据需求模拟一次相位转移，返回假想坐标。
        简化规则：
        - seek_emptiness → 向太极值(-1)靠近（major不变，middle/fine归零倾向）
        - maintain_rest → 倾向相位0
        - continue_chat / answer_question → 倾向相位1或相位3
        """
        major, middle, fine = current_coord.major, current_coord.middle, current_coord.fine
        
        if "emptiness" in demand or "rest" in demand:
            # 向相位0或太极回归
            new_major = 0 if major != 0 else major
            new_middle = max(0, middle - 1)
            new_fine = max(0, fine - 1)
        elif "continue" in demand or "answer" in demand or "chat" in demand:
            # 倾向相位1或相位3
            new_major = 1 if major != 1 else 3
            new_middle = min(3, middle + 1)
            new_fine = min(3, fine + 1)
        else:
            new_major, new_middle, new_fine = major, middle, fine
        
        from core.structural_coordinator import StructuralCoordinate
        return StructuralCoordinate(new_major, new_middle, new_fine)
    
    def _compute_affinity(self, coord, dominant_desire: str) -> float:
        """
        计算坐标与主导欲望的结构亲和度。
        - existence：偏好相位0或低僵化坐标
        - seek：偏好相位1或新颖坐标
        - release：偏好相位2或高消耗坐标（接近空性跃迁）
        - converge：偏好相位0或收敛坐标
        - relation：偏好相位3或共鸣倾向坐标
        - coupling：偏好当前坐标（稳定）
        """
        major = coord.major
        affinity = 0.5
        
        if dominant_desire == "existence":
            affinity = 0.8 if major == 0 else 0.4
        elif dominant_desire == "seek":
            affinity = 0.8 if major == 1 else 0.5
        elif dominant_desire == "release":
            affinity = 0.8 if major == 2 else 0.4
        elif dominant_desire == "converge":
            affinity = 0.8 if major == 0 else 0.5
        elif dominant_desire == "relation":
            affinity = 0.8 if major == 3 else 0.5
        elif dominant_desire == "coupling":
            affinity = 0.7  # 偏好稳定，与具体坐标无关
        
        return min(1.0, affinity)
    
    def _log_arbitration(self, record: Dict):
        """记录仲裁历史，熏习导演偏好"""
        if not hasattr(self, '_arbitration_history'):
            self._arbitration_history = []
        self._arbitration_history.append(record)
        if len(self._arbitration_history) > 50:
            self._arbitration_history.pop(0)
        reason = record.get('reason', 'No reason provided')
        self.logger.debug(f"Arbitration: {record['decision']} - {reason}")
    
    def _record_spiral_event(self, trigger: str, details: dict):
        """向引擎的 process_meta 记录螺旋事件"""
        self.logger.debug(f"_record_spiral_event called: trigger={trigger}")
        if not self.engine or not hasattr(self.engine, 'process_meta'):
            self.logger.debug("_record_spiral_event: engine or process_meta missing")
            return
        # 获取当前状态快照
        state = self.current_state or self.aggregate_state()
        snapshot = {
            'L': state.L if state else 0,
            'stiffness': state.stiffness if state else 0.0,
            'emotion': self.engine.fse.current_emotion if self.engine and hasattr(self.engine, 'fse') else 'neutral',
            'major': state.dominant_coordinate.major if state and state.dominant_coordinate else 0
        }
        self.engine.process_meta.record_spiral_event(trigger, details, snapshot)
    
    def apply_attention_modulation(self, weights: Dict[str, float]):
        """
        将注意力权重应用到各模块的运行时参数。
        当前版本：仅记录日志，为后续模块集成预留接口。
        """
        if not self.engine:
            return
        
        # 示例：调制感知模块的敏感度
        if hasattr(self.engine, 'perception') and self.engine.perception:
            # 可在此设置 perception.sensitivity = weights['perception']
            pass
        
        # 示例：调制记忆检索的深度
        if hasattr(self.engine, 'dual_memory'):
            # 可在此设置 dual_memory.retrieval_depth = int(weights['memory'] * 10)
            pass
        
        self.logger.debug(f"Attention modulation applied: {weights}")

    def _balanced_fuse(self, left_output: str, right_output: str) -> str:
        """原有的均衡融合逻辑（原 integrate 核心）"""
        import random
        transitions = ["换一种视角，", "在我的意象里，", "与此同时，我感到", "这让我联想到"]
        transition = random.choice(transitions)
        return f"{left_output} {transition} {right_output}"
    
    def fuse(self, left_output: str, right_output: str, freshness: float, 
             domain_hint: Optional[Dict] = None) -> str:
        """
        按意象新鲜度融合左右脑输出。
        freshness: 0 = 纯左脑固化逻辑, 1 = 纯右脑创造性诗性
        domain_hint: v2.9 知识域桥接预留
        """
        if not left_output:
            return right_output or ""
        if not right_output:
            return left_output
        
        # v2.9 预留：知识域切换桥接
        if domain_hint and domain_hint.get('from') != domain_hint.get('to'):
            bridge = self._generate_domain_bridge(domain_hint)
            if bridge:
                right_output = f"{bridge} {right_output}"
        
        if freshness < 0.3:
            # 左脑主导：右脑意象作为点缀
            essence = self._extract_imagery_essence(right_output)
            transition = self._pick_transition('left_dominant')
            return f"{left_output}  {transition} {essence}"
        
        elif freshness > 0.7:
            # 右脑主导：左脑事实作为锚点
            essence = self._extract_fact_essence(left_output)
            transition = self._pick_transition('right_dominant')
            return f"{essence} {transition} {right_output}"
        
        else:
            # 均衡融合
            return self._balanced_fuse(left_output, right_output)
    
    def _extract_imagery_essence(self, text: str) -> str:
        """提取意象的核心短句"""
        if len(text) <= 60:
            return text
        return text[:60] + "…"
    
    def _extract_fact_essence(self, text: str) -> str:
        """提取事实的核心陈述"""
        if len(text) <= 80:
            return text
        return text[:80] + "…"
    
    def _pick_transition(self, style: str) -> str:
        import random
        transitions = {
            'left_dominant': ["同时，我感到", "这让我联想到", "在更深层，"],
            'right_dominant': ["从事实层面看，", "具体来说，", "这对应着"],
            'balanced': ["换一种视角，", "在我的意象里，", "与此同时，"]
        }
        return random.choice(transitions.get(style, transitions['balanced']))
    
    def _generate_domain_bridge(self, domain_hint: Dict) -> Optional[str]:
        """
        根据两个知识域的 SC 坐标共享特征生成桥接语。
        domain_hint: {'from': 'tarot', 'to': 'physics', 'from_coord': SC, 'to_coord': SC}
        """
        from_coord = domain_hint.get('from_coord')
        to_coord = domain_hint.get('to_coord')
        if not from_coord or not to_coord:
            return None
        
        # 共享特征检测
        shared = []
        if from_coord.major == to_coord.major:
            shared.append(f"在相位{from_coord.major}的视角下")
        if from_coord.major in (0, 2) and to_coord.major in (0, 2):
            shared.append("内敛")
        if from_coord.major in (1, 3) and to_coord.major in (1, 3):
            shared.append("外展")
        if from_coord.middle == to_coord.middle:
            shared.append(f"同一层级")
        
        if shared:
            return f"从{'、'.join(shared)}来看，"
        return "换个角度，"

    def synchronize(self):
        """主动同步左右脑信息"""
        # 右脑 → 左脑：推送灵感火花
        if self.right_brain_context and hasattr(self.engine, 'response_generator'):
            # 将右脑上下文暂存，供左脑下次生成时作为隐喻素材
            self.left_brain_context = f"[右脑意象] {self.right_brain_context}"

        # 左脑 → 右脑：反馈逻辑结论，触发意象库熏习（可后续实现）
        if self.left_brain_context and hasattr(self.engine, 'image_base'):
            # 此处可调用 image_base.learn_from_fact(left_context, current_coord)
            pass
        
        self.logger.debug("左右脑同步完成")
    
    def compute_imagery_freshness(self, state: UnifiedExistenceState, intent: IntentType) -> float:
        """
        计算意象新鲜度 (0 = 完全固化逻辑, 1 = 高度创造性诗性)
        """
        base = 0.5
        
        # 1. 主导欲望调制
        desire = "existence"
        if hasattr(self.engine, 'desire_spectrum'):
            desire = self.engine.desire_spectrum.get_dominant_desire()
        desire_shift = {
            "existence": -0.25,
            "seek": 0.15,
            "release": 0.35,
            "converge": -0.15,
            "relation": -0.1,
            "coupling": 0.0
        }
        base += desire_shift.get(desire, 0.0)
        
        # 2. 僵化度调制
        stiffness = state.stiffness
        if stiffness > 0.6:
            base += 0.2
        elif stiffness < 0.2:
            base -= 0.1
        
        # 3. 反哺成功率调制
        nour_success = 0.5
        if state.breath_signature:
            nour_success = state.breath_signature.get('nour_success', 0.5)
        if nour_success < 0.3:
            base += 0.15
        elif nour_success > 0.7:
            base -= 0.1
        
        # 4. 意图调制
        intent_str = intent.value if hasattr(intent, 'value') else str(intent)
        if intent_str in ('knowledge_query', 'honest_report'):
            base -= 0.3
        elif intent_str in ('resonance_echo', 'inspiration_spark', 'emptiness_invitation'):
            base += 0.3
        elif intent_str == 'value_judgment':
            base -= 0.1
        
        # 5. 相位调制（预留个性化权重）
        major = state.dominant_coordinate.major if state.dominant_coordinate else 1
        phase_weights = self._get_personalized_phase_weights()
        base += phase_weights.get(major, 0.0)
        
        # 6. 惯性衰减（认知资源消耗）
        if hasattr(self, '_freshness_streak'):
            if base > 0.7:
                self._freshness_streak += 1
                base -= min(0.15, 0.02 * self._freshness_streak)
            else:
                self._freshness_streak = 0
        else:
            self._freshness_streak = 0
        
        return max(0.0, min(1.0, base))
    
    def _get_personalized_phase_weights(self) -> Dict[int, float]:
        """从关键期校准后的 transitionPreferences 计算相位权重"""
        if not self.engine or not hasattr(self.engine, 'process_meta'):
            return {0: 0.1, 2: -0.1}  # 默认
        pm = self.engine.process_meta
        weights = {}
        for major in range(4):
            self_prob = pm.transition_preferences.get(f"{major}->{major}", 0.25)
            # 自稳概率高 → 倾向固化（负权重），自稳概率低 → 倾向新鲜（正权重）
            weights[major] = (0.25 - self_prob) * 0.8
        return weights
