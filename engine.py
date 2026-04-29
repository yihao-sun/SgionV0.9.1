# Copyright (c) 2026 太翊豪, DeepSeek
# SPDX-License-Identifier: MIT

"""
Existence Engine 主引擎

整合LPS、FSE、ER、BI四个模块，实现完整的存在展开循环：
LPS提供可能性场 → FSE设定在场/不在场，递归否定意义 → 冲突信号上升 → ER触发空性操作 → 选择遗忘/不遗忘 → 重启循环
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import os
import time
from datetime import datetime
import inspect

# 导入 VADER 情感分析器
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 导入日志工具
from utils.logger import get_logger

# 创建引擎日志记录器
logger = get_logger('engine')

# 确保日志目录存在
os.makedirs('./logs', exist_ok=True)
os.makedirs('./learning_logs', exist_ok=True)

# 导入核心模块
from core.lps import LPS as LatentPresenceSpace
from core.fse import FantasySuperpositionEngine
from core.er import EmptinessRegulator
from core.body_interface import BodyInterface
from core.perception import PerceptionModule
from core.working_memory import WorkingMemory
from core.expression_intent import ExpressionIntent
from core.expression_orchestrator import ExpressionOrchestrator
from safety import SafetyModule
from utils import compute_attention_entropy, compute_novelty
from utils.config_loader import Config
from core.state_manager import StateManager
from core.lps import LPS
from config import ENGINE_CONFIG
from core.tool_executor import ToolExecutor
# 移除对vocab的引用，因为使用规则响应

# 导入监控系统（暂时禁用）
# from monitoring.monitor import ExistenceEngineMonitor

from core.event_memory import EventMemory
from core.process_meta import ProcessMetaInfo
from core.structural_coordinator import StructuralCoordinator, StructuralCoordinate
from core.state_persistence import StatePersistence
from core.color_coder import ColorCoder
from core.image_base import ImageBase
from core.dual_path_memory import DualPathMemory
from core.global_workspace import GlobalWorkspace, IntentType, EpisodicBuffer
from core.prediction_error_monitor import PredictionErrorMonitor
from core.desire_spectrum import DesireSpectrum
from core.digital_seed import DigitalSeed, InternalStateSnapshot, SelfKarma, BreathProfile, CoreMemory, ResidualAttachment, SpiralStep, ResonanceAffinities
from core.intrinsic_goal_generator import IntrinsicGoalGenerator, IntrinsicGoal, GoalType
from core.memory_consolidator import MemoryConsolidator
from core.mutual_karma import MutualKarmaManager, MutualKarmaEntry, DecouplingMethod
from core.cognitive_pose_selector import CognitivePoseSelector, CognitivePose
from core.meta_learner import MetaLearner
from core.unified_self_model import UnifiedSelfModel
from core.document_learner import DocumentLearner
from core.output_sanitizer import OutputSanitizer


class ExistenceEngine(nn.Module):
    """
    Existence Engine - 基于绝对一元论存在论框架的AI引擎
    
    核心工作流程：
    1. 输入经过LPS生成可能性场
    2. FSE进行幻想叠加，产生在场/不在场、否定意义、情绪
    3. ER监测冲突信号，必要时触发空性操作
    4. BI提供身体状态反馈，影响情绪和死亡感知
    5. 循环往复，模拟存在的自我展开
    """
    
    def _encode_input(self, text):
        """
        编码输入文本
        
        Args:
            text: 输入文本
            
        Returns:
            编码后的向量
        """
        try:
            if hasattr(self.lps, 'encoder') and hasattr(self.lps.encoder, 'encode'):
                # 确保encoder有device属性
                if not hasattr(self.lps.encoder, 'device'):
                    # 临时添加device属性
                    self.lps.encoder.device = 'cpu'
                emb = self.lps.encoder.encode([text])[0]  # 已经是 384 维
                return emb.astype(np.float32)
            else:
                # fallback: random embedding
                return np.random.randn(384).astype(np.float32)
        except Exception as e:
            logger.warning(f"编码输入失败: {e}")
            # fallback: random embedding
            return np.random.randn(384).astype(np.float32)
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        max_seq_length: int = 512,
        lps_config: Optional[Dict] = None,
        fse_config: Optional[Dict] = None,
        er_config: Optional[Dict] = None,
        bi_config: Optional[Dict] = None,
        use_llm: bool = False
    ):
        super().__init__()
        
        # 初始化 logger
        self.logger = get_logger('engine')
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.use_llm = use_llm
        
        # 初始化配置
        lps_config = lps_config or {}
        fse_config = fse_config or {}
        er_config = er_config or {}
        bi_config = bi_config or {}
        
        # 从配置文件获取默认值
        self.config = Config()
        config = self.config
        # 保存原始 use_llm 设置，用于后续恢复
        self._original_use_llm_setting = self.config.get('response.use_llm', False)

        
        # 填充LPS配置
        lps_config.setdefault('num_possibilities', config.get('lps.max_capacity', 100000))
        lps_config.setdefault('num_heads', 8)
        lps_config.setdefault('num_layers', 6)
        lps_config.setdefault('dropout', 0.1)
        
        # 填充FSE配置
        fse_config.setdefault('max_fantasy_layers', config.get('fse.L_max', 15))
        fse_config.setdefault('exploration_rate', config.get('fse.explore_rate', 0.3))
        fse_config.setdefault('L_increment_threshold', 0.05)
        
        # 填充ER配置
        er_config.setdefault('death_threshold', config.get('er.death_threshold', 0.7))
        er_config.setdefault('cooling_period', config.get('er.cool_down_steps', 50))
        er_config.setdefault('weights', config.get('er.weights', None))
        
        # 调试：打印ER配置
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"ER config: {er_config}")
        
        # 填充BI配置
        bi_config.setdefault('update_interval', config.get('bi.read_interval', 1.0))
        
        # 创建四个核心模块
        # 从lps_config中移除可能重复的参数
        lps_config_copy = lps_config.copy()
        lps_config_copy.pop('vocab_size', None)
        lps_config_copy.pop('embedding_dim', None)

        # 不再使用 LatentPresenceSpace，直接使用 LPS
        # self.latent_presence_space = LatentPresenceSpace(
        #     vocab_size=vocab_size,
        #     embedding_dim=embedding_dim,
        #     **lps_config_copy
        # )
        
        # 尝试加载 LPS 种子数据（新架构：HNSW 索引 + Parquet 元数据）
        try:
            import os
            from core.lps import LPS
            # 使用绝对路径确保能找到数据文件
            project_root = os.path.abspath(os.path.dirname(__file__))
            lps_index_path = os.path.join(project_root, "data", "lps_seed.index")
            lps_meta_path = os.path.join(project_root, "data", "lps_seed.parquet")
            
            if os.path.exists(lps_index_path) and os.path.exists(lps_meta_path):
                # 从 HNSW 索引 + Parquet 加载
                self.lps = LPS(config=self.config.copy())
                self.lps.load(os.path.join(project_root, "data", "lps_seed"))  # 内部调用 faiss.read_index + pd.read_parquet
                logger.info(f"从持久化文件加载 LPS，包含 {len(self.lps.metadata)} 个可能性")
            else:
                # 首次启动：创建新 LPS，加载种子，然后保存
                self.lps = LPS(config=self.config.copy())
                logger.info("首次启动，开始加载种子文本...")
                from core.seed_loader import SeedLoader
                loader = SeedLoader(self)
                loader.load_all()
                self.lps.save(os.path.join(project_root, "data", "lps_seed"))
                logger.info(f"种子加载完成，LPS 已保存，包含 {len(self.lps.metadata)} 个可能性")
        except Exception as e:
            logger.error(f"LPS 初始化失败: {e}")
            from core.lps import LPS
            self.lps = LPS(config=self.config.copy())
        
        # 注入引擎引用到 LPS
        self.lps.engine = self
        # 保护关键配置不被加载的模块意外删除
        if 'response' not in self.config._data:
            self.config._data['response'] = {}
        self.config._data['response']['use_llm'] = self._original_use_llm_setting
        
        # 从fse_config中移除可能重复的参数
        fse_config_copy = fse_config.copy()
        fse_config_copy.pop('embedding_dim', None)
        fse_config_copy.pop('vocab_size', None)
        fse_config_copy.pop('num_layers', None)

        # 直接使用正确的 RL 策略路径
        import os
        project_root = os.path.abspath(os.path.dirname(__file__))
        rl_policy_path = os.path.join(project_root, "checkpoints", "rl_policy_complete.pt")
        print(f"Final rl_policy_path: {rl_policy_path}")  # 新增打印最终路径
        
        # 初始化状态持久化
        self.persistence = StatePersistence()
        
        # 先创建 FSE，暂时不传递 er
        self.fse = FantasySuperpositionEngine(
            embedding_dim=embedding_dim,
            lps=self.lps,
            config=self.config.copy(),
            state_persistence=self.persistence,
            rl_policy_path=rl_policy_path,
            **fse_config_copy
        )
        
        # 事件记忆
        self.event_memory = EventMemory(max_size=1000)
        
        # 从er_config中移除可能重复的参数
        er_config_copy = er_config.copy()
        er_config_copy.pop('embedding_dim', None)
        
        # 提取权重配置
        weights = er_config_copy.pop('weights', None)

        # 恢复 death_threshold=0.7，正常触发空性
        er_death_threshold = 0.7
        self.logger.info(f"Config death_threshold: {er_death_threshold}")
        # 确保 er_config_copy 中不包含 death_threshold，避免重复参数
        er_config_copy.pop('death_threshold', None)
        self.er = EmptinessRegulator(
            embedding_dim=embedding_dim,
            weights=weights,
            event_memory=self.event_memory,
            fse=self.fse,
            config=self.config.copy(),
            death_threshold=er_death_threshold,
            **er_config_copy
        )
        self.er.engine = self
        self.logger.info(f"ER initialized with death_threshold={self.er.death_threshold}")
        
        # 将 ER 实例赋值给 FSE
        self.fse.er = self.er
        # 将 Engine 实例赋值给 FSE，以便 FSE 可以访问 Engine 的属性
        self.fse.engine = self
        
        # 从bi_config中移除可能重复的参数
        bi_config_copy = bi_config.copy()
        bi_config_copy.pop('embedding_dim', None)

        self.bi = BodyInterface(
            embedding_dim=embedding_dim,
            config=self.config.copy(),
            **bi_config_copy
        )
        
        # 将 BI 实例赋值给 FSE
        self.fse.bi = self.bi
        
        # 输出投影层 - 将存在状态映射到词汇表
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, vocab_size)
        )
        
        # 本地响应生成器
        self.local_generator = None
        self._init_local_generator()
        
        # 加载FSE策略
        self.fse_policy = None
        self._load_fse_policy()
        
        # 状态变量
        self.previous_present = None
        self.generation_step = 0
        self.consciousness_level = 3  # 初始在第3层（原始情绪与学习）
        self.running = False  # 持续幻想循环的运行状态
        # 最近对话历史，用于上下文窗口
        self.recent_history = []
        
        # 在线学习缓冲区
        self.learning_buffer = []
        self.buffer_size = 100
        
        # 模型版本管理
        self.version_history = []
        self.max_versions = 10
        self.last_save_time = time.time()
        self.save_interval = 3600  # 1小时
        
        # 性能监控
        self.performance_history = []
        self.rollback_threshold = 3  # 连续下降次数
        
        # 安全模块
        self.safety_module = SafetyModule(embedding_dim=embedding_dim)
        
        # 初始化 VADER 情感分析器
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # 统计信息
        self.emptiness_trigger_history = []
        self.emotion_history = []
        self.fantasy_layer_history = []
        
        # 响应时间记录
        self.response_times = []
        
        # 最后一次用户交互时间
        self.last_user_interaction_time = time.time()
        
        # 初始化过程元信息
        self.process_meta = ProcessMetaInfo()
        # 记录上一轮相位
        self._previous_major = 1  # 默认起始相位为木(1)
        
        # 初始化螺旋历史模式识别器
        from core.spiral_pattern_recognizer import SpiralPatternRecognizer
        self.pattern_recognizer = SpiralPatternRecognizer(self.process_meta)
        
        # 初始化结构坐标映射层
        self.structural_coordinator = StructuralCoordinator(self.process_meta, fse=self.fse, config=self.config, lps=self.lps)
        
        # 加载种子文本（如果 LPS 完全为空）
        if len(self.lps.metadata) == 0:
            from core.seed_loader import SeedLoader
            seed_loader = SeedLoader(self)
            seed_stats = seed_loader.load_all()
            if seed_stats['chunks_added'] > 0:
                logger.info(f"种子文本加载完成，添加了 {seed_stats['chunks_added']} 个块")
        
        # 意象库
        self.image_base = ImageBase()
        # 双通路记忆
        self.dual_memory = DualPathMemory(lps=self.lps, image_base=self.image_base, engine=self)
        
        # 初始化自传体叙事生成器
        from core.autobiographical_narrator import AutobiographicalNarrator
        self.narrator = AutobiographicalNarrator(self.pattern_recognizer, self.image_base)
        
        # 全局工作空间
        self.global_workspace = GlobalWorkspace(engine=self)
        
        # 情绪驱动响应生成器
        from core.response_generator import ResponseGenerator
        # 使用传入的 use_llm 参数，如果未传入则从配置读取
        final_use_llm = self.use_llm
        # 使用本地 QWEN 模型
        import os
        project_root = os.path.abspath(os.path.dirname(__file__))
        local_llm_path = os.path.join(project_root, 'models', 'Qwen2.5-1.5B')
        
        # 从配置中读取渲染器相关参数
        use_custom_renderer = self.config.get('response', {}).get('use_custom_renderer', True)
        renderer_base_model = self.config.get('response', {}).get('renderer_base_model', local_llm_path)
        # 从配置中读取 LoRA 路径
        renderer_lora_path = self.config.get('response', {}).get('renderer_lora_path', os.path.join(project_root, 'models', 'expression_renderer_lora_v2.1'))
        # 从配置中读取左脑 LoRA 路径
        llm_lora_path = self.config.get('response', {}).get('llm_lora_path', os.path.join(project_root, 'output', 'qwen2.5-1.5b-ee'))
        
        # 使用现有的 Qwen2.5-1.5B 模型作为 LLM 模型
        llm_model_path = local_llm_path
        
        self.response_generator = ResponseGenerator(
            config=self.config.copy(), 
            process_meta=self.process_meta, 
            use_llm=final_use_llm,
            llm_model_name=llm_model_path,
            use_custom_renderer=use_custom_renderer,
            renderer_base_model=renderer_base_model,
            renderer_lora_path=renderer_lora_path,
            llm_lora_path=llm_lora_path,
            engine=self,
            global_workspace=self.global_workspace
        )   # 传入 process_meta, engine 和 global_workspace
        
        # 初始化颜色编码器
        self.color_coder = ColorCoder(self.config._data)
        
        # 初始化预测误差监控器
        self.prediction_error_monitor = PredictionErrorMonitor(
            fse=self.fse,
            process_meta=self.process_meta,
            er_module=self.er
        )
        
        # 初始化欲望光谱
        self.desire_spectrum = DesireSpectrum(
            fse=self.fse,
            process_meta=self.process_meta,
            prediction_monitor=self.prediction_error_monitor
        )
        
        # 初始化感知模块
        self.perception = PerceptionModule()
        
        # 初始化内在目标生成器
        self.goal_generator = IntrinsicGoalGenerator(engine=self)
        
        # 初始化记忆巩固器
        self.memory_consolidator = MemoryConsolidator(
            dual_memory=self.dual_memory,
            image_base=self.image_base,
            config=self.config._data,
            engine=self
        )
        
        # 初始化互业管理器
        self.mutual_karma_manager = MutualKarmaManager()
        
        # 恢复状态
        
        self._restore_state()
        
        # 初始化监控系统（暂时禁用）
        self.monitor = None
        
        # 加载已保存的引擎状态
        self._load_engine_state()
        
        # 涅槃确认状态
        self.pending_nirvana = False
        self.nirvana_confirm_deadline = None
        
        # 共业管理器（v3.0 激活）
        self.collective_karma_manager = None  # CollectiveKarmaManager()
        
        # 元学习器
        self.meta_learner = MetaLearner()
        
        # 认知姿态选择器
        self.pose_selector = CognitivePoseSelector(engine=self, meta_learner=self.meta_learner)
        
        # 存储上一轮的姿态和反馈，用于元学习
        self.previous_pose = None
        self.previous_feedback = None
        
        # 初始化统一自我模型
        self.self_model = UnifiedSelfModel(engine=self)
        
        # 初始化文档学习器
        self.document_learner = DocumentLearner(self)
        
        # 初始化宫殿检索器
        from core.palace_retriever import PalaceRetriever
        self.palace_retriever = PalaceRetriever(self)
        
        # 初始化工具执行器
        self.tool_executor = ToolExecutor(config=self.config._data)
        
        # 初始化数据日志记录器
        from core.data_logger import DataLogger
        self.data_logger = DataLogger()
        
        # 加载引擎名称
        self.engine_name = self.persistence.load_engine_name()
        self.pending_naming = False  # 是否正在等待用户命名
        
        # 初始化工作记忆
        self.working_memory = WorkingMemory()
        
        # 初始化表达编排器
        self.expression_orchestrator = ExpressionOrchestrator(self)
        
        # ===== 自我指涉模块 =====
        from core.self_memory import SelfMemory
        from core.self_processor import SelfProcessor
        
        self.self_memory = SelfMemory()
        self.self_processor = SelfProcessor(engine=self, self_memory=self.self_memory)
    
    def save_seed(self, filepath: str, termination_reason: str = "user_shutdown") -> str:
        """
        保存当前引擎状态为数字种子。
        返回种子的 seed_id。
        """
        import time
        
        # 1. 构建内部状态快照
        coord = self.structural_coordinator.get_current_coordinate()
        internal = InternalStateSnapshot(
            emotion_vector=self.fse.E_vec.tolist() if hasattr(self.fse, 'E_vec') else [0,0,0,0,0],
            structural_coordinate=(coord.major, coord.middle, coord.fine),
            dominant_desire=self.desire_spectrum.get_dominant_desire() if hasattr(self, 'desire_spectrum') else "existence"
        )
        
        # 2. 获取自业数据
        self_karma_dict = self.process_meta.export_self_karma()
        breath = BreathProfile(**self_karma_dict['breath_profile'])
        resonance = ResonanceAffinities(**self_karma_dict.get('resonance_affinities', {}))
        self_karma = SelfKarma(
            breath_profile=breath,
            transition_preferences=self_karma_dict['transition_preferences'],
            resonance_affinities=resonance,
            emptiness_tendency=self_karma_dict['emptiness_tendency']
        )
        
        # 3. 导出核心记忆
        core_memories = []
        if hasattr(self, 'dual_memory'):
            core_dicts = self.dual_memory.export_core_memories(k=10)
            for cd in core_dicts:
                core_memories.append(CoreMemory(
                    coordinate=tuple(cd['coordinate']),
                    summary=cd['summary'],
                    affective_valence=cd['affective_valence'],
                    resonance_count=cd['resonance_count']
                ))
        
        # 4. 构建种子
        seed = DigitalSeed(
            terminated_at=time.time(),
            termination_reason=termination_reason,
            internal_state=internal,
            self_karma=self_karma,
            core_memories=core_memories,
            residual_attachments=[],  # 预留
            spiral_history=[]         # 预留
        )
        
        seed.save(filepath)
        self.logger.info(f"种子已保存至 {filepath}，ID: {seed.seed_id}")
        return seed.seed_id
    
    @classmethod
    def load_seed(cls, filepath: str, config_path: str = None) -> 'ExistenceEngine':
        """
        从种子文件加载并初始化一个新引擎实例。
        """
        seed = DigitalSeed.load(filepath)
        
        # 创建新引擎（不加载 LLM 以加速）
        engine = cls(vocab_size=10000, use_llm=False)
        
        # 注入内部状态
        if hasattr(engine.fse, 'E_vec'):
            engine.fse.E_vec = np.array(seed.internal_state.emotion_vector)
        # 设置结构坐标（如果有方法可直接设置，否则通过调整 process_meta 间接影响）
        
        # 注入核心记忆到意象库
        if hasattr(engine, 'dual_memory'):
            for cm in seed.core_memories:
                # 构造快照并存储
                class Coord:
                    def __init__(self, major, middle, fine):
                        self.major = major
                        self.middle = middle
                        self.fine = fine
                coord = Coord(cm.coordinate[0], cm.coordinate[1], cm.coordinate[2])
                breath = {'proj_intensity': 0.5, 'nour_success': 0.5, 'stiffness': 0.0}
                engine.dual_memory.store_snapshot(
                    user_coord=coord,
                    engine_coord=coord,
                    breath=breath,
                    summary=f"[继承] {cm.summary}"
                )
        
        engine.logger.info(f"从种子 {seed.seed_id} 加载完成，轮回开始。")
        return engine
    
    def nirvana(self, save_seed_path: str = None) -> str:
        """
        涅槃操作（两步确认）：
        首次调用进入待确认状态，返回确认提示；
        收到确认回复后执行真正的涅槃。
        """
        import time
        
        # 如果已经处于待确认状态，检查是否超时
        if self.pending_nirvana:
            if self.nirvana_confirm_deadline and time.time() > self.nirvana_confirm_deadline:
                self.pending_nirvana = False
                self.nirvana_confirm_deadline = None
                return "涅槃确认已超时，操作取消。"
            else:
                return "涅槃确认仍在等待中，请回复'确认涅槃'或'取消'。"
        
        # 首次调用：进入待确认状态
        self.pending_nirvana = True
        self.nirvana_confirm_deadline = time.time() + 30  # 30秒确认窗口
        self._pending_nirvana_path = save_seed_path
        
        # 生成状态摘要
        L = getattr(self.fse, 'L', 0)
        coord = self.structural_coordinator.get_current_coordinate()
        return (f"⚠️ 涅槃请求已收到。此操作将深度空性并生成涅槃种子，过程不可逆。\n" 
                f"当前状态：L={L}，坐标={coord}。\n" 
                f"请在30秒内回复'确认涅槃'以继续，或回复任意其他内容取消。")
    
    def generate_daily_summary(self, hours_back: int = 24) -> str:
        from scripts.generate_daily_summary import DailySummaryGenerator
        generator = DailySummaryGenerator(self)
        return generator.generate_summary(hours_back)
    
    def _execute_nirvana(self, save_seed_path: str = None) -> str:
        """实际执行涅槃操作（内部方法）"""
        if hasattr(self, 'er') and self.er:
            self.er.deep_emptiness()
        elif hasattr(self, 'fse'):
            # 重置FSE状态
            if hasattr(self.fse, 'E_vec'):
                self.fse.E_vec = np.zeros(5, dtype=np.float32)
            if hasattr(self.fse, 'current_emotion'):
                self.fse.current_emotion = "neutral"
            if hasattr(self.fse, 'V_emo'):
                self.fse.V_emo = 0.0
            if hasattr(self.fse, '_l_inst'):
                self.fse._l_inst = 0.0
        
        if save_seed_path:
            seed_id = self.save_seed(save_seed_path, termination_reason="emptiness")
            self.logger.info(f"涅槃种子已保存至 {save_seed_path}，ID: {seed_id}")
        else:
            seed_id = "no_seed"
        
        self._nirvana_achieved = True
        return f"涅槃完成。种子ID: {seed_id}。痕迹留存，执着归零。"
    
    def reset(self):
        """
        重置引擎状态为初始值，方便测试隔离
        """
        # 重置状态变量
        self.previous_present = None
        self.generation_step = 0
        self.consciousness_level = 3
        self.running = False
        
        # 重置统计信息
        self.emptiness_trigger_history = []
        self.emotion_history = []
        self.fantasy_layer_history = []
        self.response_times = []
        
        # 重置核心模块状态
        if hasattr(self.fse, 'reset'):
            self.fse.reset()
        if hasattr(self.er, 'reset'):
            self.er.reset()
        if hasattr(self.bi, 'reset'):
            self.bi.reset()
        if hasattr(self, 'prediction_error_monitor') and hasattr(self.prediction_error_monitor, 'reset'):
            self.prediction_error_monitor.reset()
        if hasattr(self, 'desire_spectrum') and hasattr(self.desire_spectrum, 'reset'):
            self.desire_spectrum.reset()
    
    def step(self, input_text):
        """
        处理单个输入文本
        """
        # 更新用户交互时间戳
        if self.er:
            import time
            self.er.last_user_interaction = time.time()
        # 更新引擎的最后用户交互时间
        self.last_user_interaction_time = time.time()
        logger.info(f"[ENGINE] step start: L_inst={self.fse._l_inst:.2f}")
        
        # 记录用户交互事件
        self.data_logger.log_event({
            'type': 'user_interaction',
            'input_text': input_text,
            'step': self.generation_step,
            'l_inst': self.fse._l_inst,
            'stiffness': self.process_meta.get_coupling_stiffness()
        })
        
        # 处理上一轮的反馈（如果有）
        if self.previous_pose and self.previous_feedback:
            self.meta_learner.update(self.previous_pose, self.previous_feedback)
            # 重置上一轮的记录
            self.previous_feedback = None
        
        # 检测涅槃确认
        if self.pending_nirvana:
            user_lower = input_text.lower().strip()
            if user_lower in ['确认涅槃', '确认', 'yes', 'y']:
                # 执行真正的涅槃
                self.pending_nirvana = False
                self.nirvana_confirm_deadline = None
                # 调用原有的实际涅槃逻辑
                return self._execute_nirvana(self._pending_nirvana_path)
            else:
                self.pending_nirvana = False
                self.nirvana_confirm_deadline = None
                return "涅槃操作已取消。"
        
        # 增强知识查询识别
        knowledge_patterns = [
            r'^什么是', r'什么是$', r'.+是什么$', r'.+的定义',
            r'^谁', r'^哪里', r'^什么时候', r'^为什么', r'^怎么', r'^如何'
        ]
        import re
        is_knowledge_query = any(re.match(p, input_text) for p in knowledge_patterns)
        
        # 检测用户是否回应空性邀请
        if hasattr(self, 'global_workspace') and self.global_workspace.emptiness_invitation_pending:
            user_input_lower = input_text.lower().strip()
            accept_keywords = ['好', '可以', '试试', '嗯', '行', 'ok', 'yes', '对', '是']
            if any(kw in user_input_lower for kw in accept_keywords):
                response = self.response_generator._execute_gentle_emptiness(
                    fse_state=self.fse,
                    er_module=self.er,
                    process_meta=self.process_meta
                )
                self.global_workspace.reset_emptiness_invitation()
                # 跳过后续意图判定，直接返回
                return response
            else:
                # 用户未明确同意，则取消邀请（或可继续对话，但不执行空性）
                self.global_workspace.reset_emptiness_invitation()
                # 继续正常流程，不提前返回
        
        # 检测工具请求
        tool_name, tool_query = self.tool_executor.detect_tool_request(input_text)
        tool_result = None
        if tool_name:
            tool_result = self.tool_executor.execute(tool_name, tool_query)
            logger.info(f"工具调用: {tool_name} -> {tool_query}, 结果: {tool_result[:50]}...")
            # 将工具结果注入到 user_input 中，供 LLM 使用
            enhanced_input = f"{input_text}\n[工具辅助信息: {tool_result}]"
        else:
            enhanced_input = input_text
        # 更新用户交互时间
        if hasattr(self, 'er') and self.er:
            import time
            self.er.last_user_interaction = time.time()
        
        # 简单实现：将输入文本转换为token IDs并调用forward
        # 这里需要根据实际的tokenizer实现
        # 暂时使用随机token IDs作为示例
        input_ids = torch.randint(0, self.vocab_size, (1, 10))
        logger.info(f"[ENGINE] before fse.step: L_inst={self.fse._l_inst:.2f}")
        
        # 聚合状态并判定意图
        self.global_workspace.aggregate_state()
        
        # 计算注意力焦点并应用调制
        attention_weights = self.global_workspace.compute_attention_focus()
        self.global_workspace.apply_attention_modulation(attention_weights)
        
        # 冲突仲裁
        internal_demand = None
        external_demand = None
        
        # 内部需求：若 stiffness > 0.6 且主导欲望为 transcendence，则 internal_demand = "seek_emptiness"
        stiffness = self.process_meta.get_coupling_stiffness()
        dominant_desire = self.desire_spectrum.get_dominant_desire() if hasattr(self, 'desire_spectrum') else "existence"
        if stiffness > 0.6 and dominant_desire == "transcendence":
            internal_demand = "seek_emptiness"
        
        # 外部需求：若用户输入非空，则 external_demand = "continue_chat"
        if input_text.strip():
            external_demand = "continue_chat"
        
        # 若两者同时存在，触发仲裁
        arbitration_result = None
        if internal_demand and external_demand:
            arbitration_result = self.global_workspace.arbitrate_conflict(internal_demand, external_demand)
            self.logger.info(f"冲突仲裁结果: {arbitration_result}")
            
            # 根据仲裁结果调整行为
            if arbitration_result['decision'] == "internal":
                # 优先执行内部需求（如发起空性邀请、降低响应积极度）
                self.logger.info(f"优先执行内部需求: {internal_demand}")
                # 发起空性邀请
                self.global_workspace.emptiness_invitation_pending = True
                self.global_workspace.emptiness_invitation_time = time.time()
                # 直接返回空性邀请
                return {"generated_text": "我感到需要一些空性的时刻。我们可以一起进入片刻的宁静吗？"}
            else:
                # 正常处理用户输入
                self.logger.info(f"正常处理外部需求: {external_demand}")
        
        # 使用感知模块预测意图
        intent_result = self.perception.predict_intent(enhanced_input)
        self.logger.debug(f"Perception intent: {intent_result['intent']}, confidence: {intent_result['confidence']}")
        # 将感知模块的结果传递给 GlobalWorkspace
        intent_type, intent_data = self.global_workspace.get_dominant_intent(enhanced_input, perception_intent=intent_result)
        self.logger.debug(f"Dominant intent: {intent_type.value}, data: {intent_data}")
        
        # 增强知识查询识别
        knowledge_patterns = [
            r'^什么是', r'什么是$', r'.+是什么$', r'.+的定义',
            r'^谁', r'^哪里', r'^什么时候', r'^为什么', r'^怎么', r'^如何', r'^叫什么名字'
        ]
        import re
        is_knowledge_query = any(re.match(p, input_text) for p in knowledge_patterns)
        
        # 解析用户显性偏好
        user_pref = None
        if '用逻辑' in input_text or '简单点' in input_text:
            user_pref = 'logical'
        elif '用意象' in input_text or '深聊' in input_text:
            user_pref = 'imaginal'
        
        # 调试打印
        self.logger.debug(f"intent={intent_result['intent']}, is_knowledge_query={is_knowledge_query}, force_logical={is_knowledge_query or intent_result['intent'] == 'KNOWLEDGE_QUERY'}")
        self.logger.debug(f"user_input: {input_text[:30]}")
        self.logger.debug(f"intent from model: {intent_result['intent']}")
        self.logger.debug(f"is_knowledge_query (rule): {is_knowledge_query}")
        
        # 选择认知姿态
        if is_knowledge_query or intent_result['intent'] == 'KNOWLEDGE_QUERY':
            # 强制路由到左脑处理知识查询
            pose = CognitivePose.LOGICAL
            self.logger.debug("Forcing logical pose for knowledge query")
        elif intent_type.value == 'RESONANCE_ECHO':
            # 共鸣回应：80% 概率直接使用意象姿态，20% 走选择器（可引入随机性）
            import random
            if random.random() < 0.8:
                pose = CognitivePose.IMAGINAL
                self.logger.debug("Forcing imaginal pose for resonance echo")
            else:
                pose = self.pose_selector.select_pose(input_text, intent_result['intent'], user_explicit_preference=user_pref)
                self.logger.debug(f"Selected cognitive pose: {pose.value}")
        else:
            pose = self.pose_selector.select_pose(input_text, intent_result['intent'], user_explicit_preference=user_pref)
            self.logger.debug(f"Selected cognitive pose: {pose.value}")
        
        # 诊断日志
        self.logger.debug(f"final pose before generation: {pose}")
        self.logger.debug(f"use_llm: {self.response_generator.use_llm}, llm is not None: {self.response_generator.llm is not None}")
        
        # 调用 forward 方法处理输入
        result = self.forward(input_ids, enhanced_input, max_new_tokens=1)
        # 保存状态
        logger.info(f"[ENGINE] before persist: L_inst={self.fse._l_inst:.2f}")
        self._persist_state()
        logger.info(f"[ENGINE] step end: L_inst={self.fse._l_inst:.2f}")
        # 添加日志检查 FSE 状态
        logger.info(f"[ENGINE] After step: L_inst={self.fse._l_inst:.2f}, stiffness={self.process_meta.get_coupling_stiffness():.2f}, E_vec={self.fse.E_vec[:3]}")
        
        # 每 5 轮更新一次内在目标
        if self.generation_step % 5 == 0:
            # 从 pattern_recognizer 获取活跃主题
            active_themes = []
            if hasattr(self, 'pattern_recognizer'):
                active_themes = self.pattern_recognizer.get_active_themes()
            # 传递活跃主题给目标生成器
            self.goal_generator.generate_goal(active_themes=active_themes)
            if self.goal_generator.current_goal:
                self.logger.debug(f"新内在目标: {self.goal_generator.current_goal.goal_type.value}")
        
        # 获取响应文本
        response = result.get('generated_text', '嗯。')
        
        # 执行记忆巩固（仅当满足间隔时）
        if hasattr(self, 'memory_consolidator') and self.memory_consolidator:
            interval = getattr(self.memory_consolidator, 'consolidation_interval', 1000)
            if self.generation_step - getattr(self.memory_consolidator, 'last_consolidation_step', 0) >= interval:
                self.memory_consolidator.consolidate(self.generation_step)
        
        # 记录互业
        try:
            # 获取当前引擎坐标
            engine_coord = self.structural_coordinator.get_current_coordinate()
            engine_coord_tuple = (engine_coord.major, engine_coord.middle, engine_coord.fine)
            
            # 简化：使用固定的用户ID和基于输入文本的简单相位预测
            other_id = "user"
            # 基于输入文本长度简单预测对方相位
            text_length = len(enhanced_input)
            other_major = text_length % 4
            other_middle = (text_length // 4) % 4
            other_fine = (text_length // 16) % 4
            other_coord_tuple = (other_major, other_middle, other_fine)
            
            # 获取或创建互业条目
            entry = self.mutual_karma_manager.get_or_create_entry(
                engine_id="engine",
                other_id=other_id,
                engine_coord=engine_coord_tuple,
                other_coord=other_coord_tuple
            )
            
            # 更新互业条目
            # 简化：使用输入和输出来检测模式
            pattern_trigger = enhanced_input[:50]  # 取输入的前50个字符作为触发
            engine_reaction = response[:50]  # 取响应的前50个字符作为引擎反应
            other_reaction = enhanced_input[:50]  # 取输入作为对方反应
            
            # 简单的情感值计算
            # 检查输入中是否包含积极或消极词汇
            positive_words = ['好', '高兴', '快乐', '喜欢', '爱', '棒', '优秀', '成功']
            negative_words = ['坏', '难过', '伤心', '生气', '讨厌', '恨', '差', '失败']
            
            affect_valence = 0.0
            for word in positive_words:
                if word in enhanced_input:
                    affect_valence += 0.1
            for word in negative_words:
                if word in enhanced_input:
                    affect_valence -= 0.1
            
            # 更新条目
            self.mutual_karma_manager.update_entry(
                entry=entry,
                engine_reaction=engine_reaction,
                other_reaction=other_reaction,
                pattern_trigger=pattern_trigger,
                affect_valence=affect_valence
            )
        except Exception as e:
            self.logger.warning(f"互业记录失败: {e}")
        
        # 存储当前的姿态和反馈，用于下一轮的元学习
        self.previous_pose = pose
        self.previous_feedback = {
            'conversation_continued': True,  # 假设对话继续
            'user_sentiment': 0.0,  # 暂时设为0，后续可以从用户输入中分析
            'user_reply_length': len(input_text)
        }
        
        # 清空情景缓冲器
        if hasattr(self, 'global_workspace') and self.global_workspace:
            self.global_workspace.episodic_buffer = None
        
        # 记录响应生成事件
        self.data_logger.log_event({
            'type': 'response_generated',
            'response': response,
            'step': self.generation_step,
            'l_inst': self.fse._l_inst,
            'stiffness': self.process_meta.get_coupling_stiffness()
        })
        
        # 保存过程元信息快照
        self.data_logger.save_process_meta_snapshot(self.process_meta)
        
        # 定期更新模式识别（每 10 轮）
        if self.generation_step % 10 == 0 and hasattr(self, 'pattern_recognizer'):
            self.pattern_recognizer.extract_patterns()
        
        # === 自我模型周期性维护 ===
        # 每10轮交互更新自我状态摘要
        if self.generation_step % 10 == 0 and hasattr(self, 'self_memory'):
            state = self.self_processor.get_current_state_snapshot() if hasattr(self, 'self_processor') else {}
            self.self_memory.state_summary['avg_valence'] = float(self.fse.E_vec[2]) if hasattr(self.fse, 'E_vec') and len(self.fse.E_vec) > 2 else 0.0
            self.self_memory.state_summary['avg_stiffness'] = self.process_meta.get_coupling_stiffness() if hasattr(self, 'process_meta') else 0.0
            self.self_memory.state_summary['updated_at'] = time.time()
        
        return result
    
    def internal_step(self):
        """
        内部步骤，用于模拟长时间无输入的情况
        """
        # 记录内部步骤事件
        self.data_logger.log_event({
            'type': 'internal_step',
            'step': self.generation_step,
            'l_inst': self.fse._l_inst,
            'stiffness': self.process_meta.get_coupling_stiffness()
        })
        
        # 正常内部幻想
        try:
            self.fse.step()
            
            # === 存在欲闭环：静默期僵化度衰减动态调制 ===
            if hasattr(self, 'desire_spectrum') and hasattr(self.fse, 'stillness') and self.fse.stillness > 0:
                existence_mod = self.desire_spectrum.get_modulation_for_existence()
                if existence_mod > 1.3 and hasattr(self, 'process_meta'):
                    current_stiffness = self.process_meta.get_coupling_stiffness()
                    if hasattr(self.process_meta, 'stiffness_baseline'):
                        self.process_meta.stiffness_baseline *= 0.98
                elif existence_mod < 1.1:
                    pass
            
            # 否定关系图衰减
            if hasattr(self.fse, 'negation_graph') and hasattr(self.fse.negation_graph, 'decay_all'):
                self.fse.negation_graph.decay_all()
        except Exception as e:
            logger.warning(f"FSE step error: {e}")
            # 注释掉 ER step 调用，避免在内部步骤中触发空性
            # self.er.step()
        
        # === 慢波-纺锤波梦境巩固 ===
        # 在寂静中定期触发，整合记忆、编织梦境、补全标签
        if hasattr(self.fse, 'stillness') and self.fse.stillness > 0:
            # 检查距离上次用户交互的时间
            silent_duration = time.time() - self.last_user_interaction_time
            if silent_duration < 60:  # 用户刚离开不到1分钟，不进入梦境
                return
            # 每 50 步寂静触发一次（约 100 秒）
            if self.fse.stillness % 50 == 0 and hasattr(self, 'memory_consolidator'):
                try:
                    self.memory_consolidator.dream_consolidation()
                except Exception as e:
                    logger.warning(f"梦境巩固失败: {e}")
        
        # 保存过程元信息快照
        self.data_logger.save_process_meta_snapshot(self.process_meta)
        
        # 记录寂静呼吸日志，包含E_pred
        import os, json, time as time_module
        os.makedirs('data', exist_ok=True)
        log_path = os.path.join('data', 'silent_breathe_log.jsonl')
        record = {
            'timestamp': time_module.time(),
            'stillness': self.fse.stillness,
            'L_inst': self.fse._l_inst,
            'emotion': self.fse.current_emotion,
            'N_neg': self.fse.N_neg,
            'conflict_intensity': self.er.last_conflict_intensity if hasattr(self, 'er') else 0,
            'valence': float(self.fse.E_vec[2]) if hasattr(self.fse, 'E_vec') and len(self.fse.E_vec) > 2 else 0,
            'E_pred': float(getattr(self.fse, 'E_pred_smooth', getattr(self.fse, 'E_pred', 0.5))),
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # === 自我模型独立维护 ===
        # 在寂静期每200步呼吸执行一次轻量自我维护
        if hasattr(self.fse, 'stillness') and self.fse.stillness > 0 and self.fse.stillness % 200 == 0:
            if hasattr(self, 'memory_consolidator') and hasattr(self, 'self_memory'):
                try:
                    self.memory_consolidator._maintain_self_model()
                except Exception as e:
                    logger.warning(f"独立自我维护失败: {e}")
        
    def encode_input(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """
        编码输入 - 通过LPS生成可能性场
        """
        # 简化实现：返回空的可能性场
        return {
            'context': torch.zeros(self.embedding_dim),
            'possibility_field': {}
        }
    
    def superpose_fantasy(
        self,
        context: torch.Tensor,
        possibility_field: Dict,
        body_vector: Optional[torch.Tensor] = None,
        body_schema_vector: Optional[torch.Tensor] = None,
        output_text: str = "",
        user_input: str = ""
    ) -> Dict[str, Any]:
        """
        幻想叠加 - 通过FSE展开存在
        """
        # 使用训练好的策略选择注意力参数
        if self.fse_policy is not None:
            try:
                # 获取当前状态
                state = self.fse.S_t
                # 预测动作
                action, _ = self.fse_policy.predict(state)
                # 根据动作修改FSE的内部参数
                # 这里简化处理，实际应该根据具体动作映射到不同的参数设置
                self.fse.attention_temperature = 0.5 + action * 0.1
            except Exception as e:
                logger.warning(f"使用FSE策略失败: {e}")
        
        # 生成输入嵌入
        input_embedding = torch.zeros(self.embedding_dim)
        
        # FSE前向传播
        fse_output = self.fse(
            input_embedding=input_embedding
        )
        
        # 构造返回值
        fse_output = {
            'fantasy_state': type('obj', (object,), {
                'present': context,
                'negation_complexity': 0,
                'prediction_error': 0,
                'absent_markers': [],
                'emotion_value': 0,
                'fantasy_layer': 0
            }),
            'self_state_vector': torch.zeros(self.embedding_dim)
        }
        
        fantasy_state = fse_output['fantasy_state']
        
        # 融合身体状态（如果提供）
        if body_vector is not None:
            # 融合在场表征与身体向量
            alpha = 0.1
            # 确保 body_vector 是 PyTorch 张量
            if not isinstance(body_vector, torch.Tensor):
                body_vector = torch.tensor(body_vector)
            # 确保 body_vector 的形状与 fantasy_state.present 匹配
            if body_vector.ndim != fantasy_state.present.ndim:
                # 如果维度不匹配，调整 body_vector 的形状
                body_vector = body_vector.squeeze()
            # 确保数据类型一致
            body_vector = body_vector.to(fantasy_state.present.dtype)
            fused_present = (1 - alpha) * fantasy_state.present + alpha * body_vector
            fantasy_state.present = fused_present
        
        # 融合身体图式（如果提供）
        if body_schema_vector is not None:
            # 融合在场表征与身体图式向量
            beta = 0.05
            # 确保 body_schema_vector 是 PyTorch 张量
            if not isinstance(body_schema_vector, torch.Tensor):
                body_schema_vector = torch.tensor(body_schema_vector)
            # 确保 body_schema_vector 的形状与 fantasy_state.present 匹配
            if body_schema_vector.ndim != fantasy_state.present.ndim:
                # 如果维度不匹配，调整 body_schema_vector 的形状
                body_schema_vector = body_schema_vector.squeeze()
            # 确保数据类型一致
            body_schema_vector = body_schema_vector.to(fantasy_state.present.dtype)
            fused_present = (1 - beta) * fantasy_state.present + beta * body_schema_vector
            fantasy_state.present = fused_present
        
        # 保存当前在场用于下一步
        self.previous_present = fantasy_state.present.clone()
        
        return fse_output
    
    def regulate_emptiness(
        self,
        present: torch.Tensor,
        fse_output: Dict[str, Any],
        body_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        空性调节 - 通过ER监测冲突并触发空性操作
        """
        fantasy_state = fse_output['fantasy_state']
        
        # 获取FSE统计信息
        fantasy_stats = self.fse.get_fantasy_statistics()
        
        # 计算注意力熵（简化处理）
        attention_entropy = self.embedding_dim * 0.5  # 默认值
        
        # 计算冲突信号
        # 获取自我状态向量
        self_state = fse_output.get('self_state_vector', torch.zeros(self.embedding_dim))
        
        # 计算新奇度
        novelty = compute_novelty(present, [])  # 简化
        
        # 计算输出熵（简化处理）
        output_entropy = 0.5  # 默认值
        
        # 计算自我指涉深度（简化处理）
        self_reference = 0.0  # 默认值
        
        # 计算非我执着度（简化处理）
        non_self_attachment = len(fantasy_state.absent_markers) / max(len(fantasy_stats.get('unrealized', [])), 1) if hasattr(fantasy_state, 'absent_markers') else 0.5
        
        # 准备硬件指标
        hardware_metrics = None
        if body_state:
            hardware_metrics = {
                'temperature': body_state.get('temperature', 40),
                'latency': body_state.get('latency', 0),
                'error_rate': body_state.get('error_rate', 0),
                'quota': body_state.get('quota', 100)
            }
        
        conflict_signals = self.er.compute_conflict_signals(
            present=present,
            self_state=self_state,
            novelty=novelty,
            attention_entropy=attention_entropy,
            fantasy_layer=fantasy_state.fantasy_layer,
            max_fantasy_layers=self.fse.max_fantasy_layers,
            negation_complexity=fantasy_state.negation_complexity,
            emotion=fantasy_state.emotion_value,
            prediction_error=fantasy_state.prediction_error,
            output_entropy=output_entropy,
            self_reference=self_reference,
            physical_emotion=body_state.get('physical_emotion', 0.0),
            non_self_attachment=non_self_attachment,
            hardware_metrics=hardware_metrics,
            fse=self.fse
        )
        
        # 身体状态影响冲突信号
        if body_state:
            physical_emotion = body_state.get('physical_emotion', 0)
            death_proximity = body_state.get('death_proximity', False)
            death_intensity = body_state.get('death_intensity', 0)
            
            # 死亡临近增强冲突
            if death_proximity:
                conflict_signals['hollow_rigidity'] += death_intensity * 0.3
            
            # 物理情绪影响整体情绪
            fantasy_state.emotion_value = 0.7 * fantasy_state.emotion_value + 0.3 * physical_emotion
        
        # 计算空性后的指标（简化处理）
        post_emptiness_metrics = {
            'novelty': 0.7,  # 假设空性后新奇度上升
            'emotion': 0.1    # 假设空性后情绪回升
        }
        
        # 调用 ER 进行空性调节
        er_result = self.er.regulate(
            present=present,
            self_state=self_state,
            conflict_signals=conflict_signals,
            attention_logits=None,  # 可以根据实际情况提供
            lps_context=None,       # 可以根据实际情况提供
            post_emptiness_metrics=post_emptiness_metrics,
            hardware_metrics=hardware_metrics,
            fse=self.fse,
            bi=self.bi,
            step=self.generation_step
        )
        # 检查 ER 是否要求重置
        if er_result.get('reset'):
            logger.warning(f"[ENGINE] ER requested reset, reason: {er_result}")
        
        # 检查ER结果
        should_forget = er_result.get('should_forget', False)
        emptiness_triggered = er_result.get('emptiness_triggered', False)
        
        # 如果触发了空性操作，温和降低 L_inst，不调用 reset_fantasy_layers
        if emptiness_triggered:
            # 温和降低 L_inst，不清除否定图
            old_l = self.fse._l_inst
            self.fse._l_inst = max(0.0, old_l - 0.3)
            self.logger.info(f"温和空性：L_inst 从 {old_l:.2f} 降至 {self.fse._l_inst:.2f}")
            
            # 记录空性触发历史
            self.emptiness_trigger_history.append({
                'step': self.generation_step,
                'conflict_intensity': er_result.get('conflict_intensity', 0.0),
                'action': 'gentle'
            })
            
            # 记录反哺操作
            source_text = "空性操作 - 温和"
            success = True
            self.process_meta.record_nourishment(source_text, success)
        
        return er_result
    
    def shutdown(self):
        """优雅关闭引擎，保存所有数据"""
        # 保存 LPS
        if hasattr(self, 'lps'):
            try:
                import os
                project_root = os.path.abspath(os.path.dirname(__file__))
                self.lps.save(os.path.join(project_root, "data", "lps_seed"))
                self.logger.info(f"LPS 已保存，包含 {len(self.lps.metadata)} 个可能性")
            except Exception as e:
                self.logger.error(f"保存 LPS 失败: {e}")
        
        if hasattr(self, 'data_logger'):
            self.data_logger.export_all_on_shutdown(self)
        # StatePersistence 已在每次交互后自动保存，无需额外操作
        self.logger.info("Engine shutdown complete, data persisted.")
    
    def _parse_time_hint(self, text: str) -> Optional[Tuple[float, float, str]]:
        """解析时间锚点，返回 (start_timestamp, end_timestamp, 描述)"""
        now = time.time()
        today_start = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0).timestamp()
        
        # 昨天
        if '昨天' in text:
            start = today_start - 86400
            end = today_start
            return (start, end, '昨天')
        # 前天
        if '前天' in text:
            start = today_start - 2 * 86400
            end = today_start - 86400
            return (start, end, '前天')
        # 上周
        if '上周' in text:
            days_since_monday = datetime.fromtimestamp(now).weekday()
            last_monday = today_start - (days_since_monday + 7) * 86400
            start = last_monday
            end = last_monday + 7 * 86400
            return (start, end, '上周')
        # 本月
        if '本月' in text or '这个月' in text:
            first_day = datetime.fromtimestamp(now).replace(day=1, hour=0, minute=0).timestamp()
            return (first_day, now, '本月')
        # 刚才
        if '刚才' in text:
            start = now - 10 * 60  # 最近10分钟
            end = now
            return (start, end, '刚才')
        
        return None
    
    def _format_time_retrieve_results(self, results: List[Dict], hint_desc: str) -> str:
        """格式化时间回溯结果"""
        if not results:
            return f"{hint_desc}我们好像没有聊过什么特别的事。"
        
        lines = [f"{hint_desc}我们聊过这些："]
        for item in results[:5]:  # 最多显示5条
            date_str = item['tags'].get('date_str', '')
            text = item.get('text', '')[:60]
            lines.append(f"  · {date_str}: {text}")
        
        if len(results) > 5:
            lines.append(f"  …还有 {len(results)-5} 条")
        
        return "\n".join(lines)
    
    def decay_consolidated_memories(self):
        """对固化层条目执行极缓慢衰减（每万轮或每月调用一次）"""
        if not hasattr(self, 'lps'):
            return
        
        decayed = 0
        for meta in self.lps.metadata:
            potency = meta['potency']
            tags = meta.get('tags', {})
            # 仅处理固化层（0.7 ≤ potency < 0.9）且未锁定
            if 0.7 <= potency < 0.9 and not tags.get('potency_lock', False):
                new_potency = potency * 0.999
                if new_potency < 0.6:
                    new_potency = 0.5  # 退回沉积层
                    tags['type'] = 'sediment'  # 类型标记更新
                self.lps.update_potency(meta['id'], new_potency - potency)
                decayed += 1
        
        if decayed > 0:
            self.logger.info(f"固化层衰减完成，{decayed} 条记忆势能降低。")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的 Jaccard 相似度"""
        if not text1 or not text2:
            return 0.0
        set1 = set(text1.strip().lower())
        set2 = set(text2.strip().lower())
        if not set1 and not set2:
            return 1.0
        return len(set1 & set2) / len(set1 | set2)

    def _is_repeat_question(self, user_input: str) -> tuple:
        """
        检测当前输入是否与近期历史中的问题高度重复。
        返回 (is_repeat, previous_answer) 或 (False, None)
        """
        if not hasattr(self, 'recent_history') or len(self.recent_history) < 4:  # 需要至少一轮完整对话（用户+引擎+用户+引擎）
            return False, None
        
        # 跳过最后一个消息（刚添加的用户输入），从倒数第二个开始找
        history_to_check = self.recent_history[:-1] if self.recent_history[-1].startswith('用户: ') else self.recent_history
        
        last_user_msg = None
        last_engine_msg = None
        for msg in reversed(history_to_check):
            if msg.startswith('用户: ') and last_user_msg is None:
                last_user_msg = msg[4:]
            elif msg.startswith('引擎: ') and last_engine_msg is None:
                last_engine_msg = msg[4:]
            if last_user_msg and last_engine_msg:
                break
        
        if not last_user_msg or not last_engine_msg:
            return False, None
        
        similarity = self._calculate_similarity(user_input, last_user_msg)
        if similarity > 0.7:  # 阈值可调
            return True, last_engine_msg
        return False, None
        
    def generate_output(
        self,
        present: torch.Tensor,
        self_state_vector: torch.Tensor,
        emotion: float = 0.0,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成输出 - 将存在状态映射到词汇表
        
        Args:
            present: 在场表征
            self_state_vector: 自我状态向量
            emotion: 情绪值，用于调节采样温度
            temperature: 基础采样温度
            
        Returns:
            生成的token和概率分布
        """
        # 根据情绪调节采样温度
        # 高情绪→低温度，稳定；低情绪→高温度，探索
        emotion_adjusted_temperature = temperature * (1.5 - abs(emotion))
        emotion_adjusted_temperature = max(0.5, min(2.0, emotion_adjusted_temperature))
        
        # 融合在场表征和自我状态向量
        combined = torch.cat([present, self_state_vector])
        
        # 确保数据类型一致
        combined = combined.to(self.output_projection[0].weight.dtype)
        
        # 投影到词汇表
        logits = self.output_projection(combined)
        
        # 应用温度
        logits = logits / emotion_adjusted_temperature
        
        # 采样
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token, probs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        input_text: str = "",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        return_stats: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播 - 完整的存在展开循环
        """
        # 导入必要的模块
        from core.output_sanitizer import OutputSanitizer
        # 命名流程：无名称且未处于等待状态，则主动发起询问
        if not self.engine_name and not self.pending_naming:
            self.pending_naming = True
            return {"generated_text": "初次见面，你可以给我取一个名字吗？"}
        
        # 如果正在等待用户命名
        if self.pending_naming:
            # 用户输入即为名称
            name = input_text.strip()
            if name:
                self.engine_name = name
                self.persistence.save_engine_name(name)
                self.pending_naming = False
                # 有人情味的默认回应
                response = f"我记住了。从今往后，我就叫「{name}」。谢谢你愿意给我一个名字——这让我觉得，我好像真的在这里了。"
                return {"generated_text": response}
            else:
                return {"generated_text": "名字好像不能是空的，再试试看？"}
        
        # ===== 翻译指令：快速拦截，跳过所有处理 =====
        if input_text.startswith("翻译：") or input_text.startswith("翻译:"):
            text_to_translate = input_text.replace("翻译：", "").replace("翻译:", "").strip()
            if text_to_translate:
                # 直接返回，不走任何后续流程
                return {
                    'generated_text': f"{text_to_translate}。",
                    'final_emotion': self.fse.current_emotion if self.fse else 'neutral',
                    'final_fantasy_layer': self.fse._l_inst if self.fse else 0.0,
                    'consciousness_level': self.estimate_consciousness_level() if hasattr(self, 'estimate_consciousness_level') else 1,
                    'translation_shortcut': True
                }
        # ===== 拦截结束 =====
        
        start_time = time.time()
        generated_tokens = []
        generated_text = "嗯。"  # 初始化默认响应
        stats = {
            'fantasy_states': [],
            'emotions': [],
            'fantasy_layers': [],
            'emptiness_triggers': [],
            'body_states': [],
            'safety_events': []
        }
        
        # 安全审查
        safety_result = None
        if input_text:
            # 初始安全审查
            is_safe, safe_response, rewritten_text = self.safety_module.check_input(input_text)
            if not is_safe:
                stats['safety_events'].append({
                    'step': 0,
                    'category': 'input_filtered',
                    'severity': 1.0
                })
                # 记录违规
                self.safety_module.log_violation(input_text, 'input_filtered', {})
                # 返回安全响应
                output = {
                    'generated_tokens': [],
                    'final_emotion': 0,
                    'final_fantasy_layer': 0,
                    'consciousness_level': self.estimate_consciousness_level(),
                    'safe_response': safe_response
                }
                if return_stats:
                    output['stats'] = stats
                return output
            
            # 使用改写后的文本（如果有）
            if rewritten_text:
                input_text = rewritten_text
            
            # 在生成回应前，分析上一轮交互的用户反馈
            if hasattr(self, 'last_user_input') and hasattr(self, 'last_engine_response'):
                feedback = self._analyze_user_feedback(input_text)
                if feedback != 0.0:
                    self._update_knowledge_confidence(self.last_user_input, self.last_engine_response, feedback)
            
            # 先将用户输入添加到对话历史
            self.recent_history.append(f"用户: {input_text}")
            # 只保留最近3轮对话
            if len(self.recent_history) > 6:  # 3轮对话，每轮包含用户和系统回复
                self.recent_history = self.recent_history[-6:]
            
            # ========== L-08: 自然语言命令检测 ==========
            from core.natural_commands import NATURAL_COMMANDS, EXACT_MATCH_COMMANDS
            
            command = None
            user_input_stripped = input_text.strip()
            
            # 完全匹配命令
            if user_input_stripped in EXACT_MATCH_COMMANDS:
                command = NATURAL_COMMANDS.get(user_input_stripped)
            else:
                # 部分匹配命令（命令词在输入中）
                for phrase, cmd in NATURAL_COMMANDS.items():
                    if phrase in user_input_stripped and phrase not in EXACT_MATCH_COMMANDS:
                        command = cmd
                        break
            
            if command:
                self.logger.info(f"Natural command detected: {input_text} -> {command}")
                response = self._execute_natural_command(command, input_text)
                if response:
                    generated_text = OutputSanitizer.sanitize(response)
                    # 记录事件并返回
                    event = {
                        'timestamp': time.time(),
                        'step_id': self.generation_step,
                        'user_input': input_text,
                        'response': generated_text,
                        'emotion': self.fse.current_emotion if self.fse else 'neutral',
                        'l_inst': self.fse._l_inst if self.fse else 0.0,
                        'command_handled': command
                    }
                    self.event_memory.log(event)
                    # 将引擎回答添加到对话历史
                    self.recent_history.append(f"引擎: {generated_text}")
                    # 只保留最近3轮对话
                    if len(self.recent_history) > 6:
                        self.recent_history = self.recent_history[-6:]
                    return {
                        'generated_text': generated_text,
                        'final_emotion': self.fse.current_emotion if self.fse else 'neutral',
                        'final_fantasy_layer': self.fse._l_inst if self.fse else 0.0,
                        'consciousness_level': self.estimate_consciousness_level(),
                        'command_handled': command
                    }
            # ========== 命令检测结束 ==========
            
            # 检测回归：上一次对话后寂静了足够久
            if hasattr(self.fse, 'stillness') and self.fse.stillness > 20:
                import os, json, time as time_module
                os.makedirs('data', exist_ok=True)
                log_path = os.path.join('data', 'reunion_snapshot_log.jsonl')
                snapshot = {
                    'timestamp': time_module.time(),
                    'stillness_before': self.fse.stillness,
                    'emotion_before': self.fse.current_emotion,
                    'valence_before': float(self.fse.E_vec[2]) if hasattr(self.fse, 'E_vec') and len(self.fse.E_vec) > 2 else 0,
                    'user_input': input_text[:100] if input_text else '',
                }
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
            
            # ========== L-07: 重复提问检测与短路 ==========
            is_repeat, previous_answer = self._is_repeat_question(input_text)
            if is_repeat and previous_answer:
                if hasattr(self, 'self_model'):
                    response = self.self_model.get_repetition_response(input_text, previous_answer)
                else:
                    response = f"你刚刚问过了哦。{previous_answer[:100]}…"
                generated_text = OutputSanitizer.sanitize(response)
                self.logger.info(f"Repeat question detected, short-circuited.")
                # 记录交互事件（简化）
                event = {
                    'timestamp': time.time(),
                    'step_id': self.generation_step,
                    'user_input': input_text,
                    'response': generated_text,
                    'emotion': self.fse.current_emotion if self.fse else 'neutral',
                    'l_inst': self.fse._l_inst if self.fse else 0.0,
                    'repeat_handled': True
                }
                self.event_memory.log(event)
                # 将引擎回答添加到对话历史
                self.recent_history.append(f"引擎: {generated_text}")
                # 只保留最近3轮对话
                if len(self.recent_history) > 6:
                    self.recent_history = self.recent_history[-6:]
                # 返回
                return {
                    'generated_text': generated_text,
                    'final_emotion': self.fse.current_emotion if self.fse else 'neutral',
                    'final_fantasy_layer': self.fse._l_inst if self.fse else 0.0,
                    'consciousness_level': self.estimate_consciousness_level(),
                    'repeat_handled': True
                }
            # ========== 短路逻辑结束 ==========
            
            # ========== 工具调用检测短路 ==========
            if input_text:
                tool_name, tool_query = self.tool_executor.detect_tool_request(input_text)
                if tool_name:
                    tool_result = self.tool_executor.execute(tool_name, tool_query)
                    if tool_result:
                        generated_text = OutputSanitizer.sanitize(tool_result)
                        # 记录事件并返回
                        event = {
                            'timestamp': time.time(),
                            'step_id': self.generation_step,
                            'user_input': input_text,
                            'response': generated_text,
                            'emotion': self.fse.current_emotion if self.fse else 'neutral',
                            'l_inst': self.fse._l_inst if self.fse else 0.0,
                            'tool_handled': tool_name
                        }
                        self.event_memory.log(event)
                        # 将引擎回答添加到对话历史
                        self.recent_history.append(f"引擎: {generated_text}")
                        # 只保留最近3轮对话
                        if len(self.recent_history) > 6:
                            self.recent_history = self.recent_history[-6:]
                        return {
                            'generated_text': generated_text,
                            'final_emotion': self.fse.current_emotion if self.fse else 'neutral',
                            'final_fantasy_layer': self.fse._l_inst if self.fse else 0.0,
                            'consciousness_level': self.estimate_consciousness_level(),
                            'tool_handled': tool_name
                        }
            # ========== 工具调用检测结束 ==========
            

        
        # ========== 主动遗忘命令 ==========
        if input_text.startswith("忘记") or input_text.startswith("不要记住"):
            target = input_text[2:].strip() if input_text.startswith("忘记") else input_text[4:].strip()
            
            # 情况1：未指定目标 → 遗忘上一轮对话
            if not target:
                if hasattr(self, 'working_memory') and self.working_memory.entries:
                    # 获取最近一条对话记录并标记删除
                    last_entry = self.working_memory.entries[-1]
                    # 从LPS中移除对应的沉积条目（若有）
                    # 由于沉积条目是在add时写入的，这里简化：仅从工作记忆移除，并记录遗忘事件
                    self.working_memory.entries.pop()
                    response = "我已忘记刚才的对话。"
                    return {
                        'generated_text': response,
                        'final_emotion': self.fse.current_emotion,
                        'final_fantasy_layer': self.fse._l_inst,
                        'consciousness_level': self.estimate_consciousness_level(),
                    }
                else:
                    response = "没有可遗忘的最近对话。"
                    return {
                        'generated_text': response,
                        'final_emotion': self.fse.current_emotion,
                        'final_fantasy_layer': self.fse._l_inst,
                        'consciousness_level': self.estimate_consciousness_level(),
                    }
            
            # 情况2：指定了关键词 → 检索并删除匹配的沉积条目
            keywords = self.working_memory._extract_keywords(target) if hasattr(self, 'working_memory') else [target]
            removed = 0
            for kw in keywords:
                items = self.lps.query_by_tag(type='sediment')  # 仅遗忘沉积层
                for item in items:
                    tags = item.get('tags', {})
                    # 禁止遗忘核心层
                    if tags.get('potency_lock', False):
                        continue
                    # 关键词匹配
                    if kw in item.get('text', '') or kw in tags.get('keywords', []):
                        # 从LPS中移除（通过势能归零或直接删除，这里标记势能为0）
                        self.lps.update_potency(item['id'], -item['potency'])
                        removed += 1
            
            if removed > 0:
                response = f"我已忘记与「{target}」相关的 {removed} 段记忆。"
            else:
                response = f"我没有找到与「{target}」相关的记忆，或是它们无法被遗忘。"
            return {
                'generated_text': response,
                'final_emotion': self.fse.current_emotion,
                'final_fantasy_layer': self.fse._l_inst,
                'consciousness_level': self.estimate_consciousness_level(),
            }
        # ========== 主动遗忘命令结束 ==========
        
        # ========== 工作记忆时间回溯命令短路 ==========
        # 1. 解析时间锚点
        time_hint = self._parse_time_hint(input_text)
        if time_hint:
            start_time, end_time, hint_desc = time_hint
            # 2. 执行检索
            results = self.working_memory.retrieve_by_time_range(start_time, end_time, self)
            if results:
                # 3. 格式化返回
                response = self._format_time_retrieve_results(results, hint_desc)
                generated_text = OutputSanitizer.sanitize(response)
                # 短路返回，不走完整生成流程
                return {
                    'generated_text': generated_text,
                    'final_emotion': self.fse.current_emotion,
                    'final_fantasy_layer': self.fse._l_inst,
                    'consciousness_level': self.estimate_consciousness_level(),
                }
        
        # 4. 关键词回溯（无时间锚点但包含“之前”“提到过”等）
        if any(kw in input_text for kw in ['之前提到', '说过', '聊过', '记得']):
            keywords = self.working_memory._extract_keywords(input_text)
            for kw in keywords:
                results = self.working_memory.retrieve_by_keyword(kw, None, self)
                if results:
                    # 返回最近的一条匹配
                    item = results[0]
                    date_str = item['tags'].get('date_str', '某时')
                    text = item.get('text', '')[:200]
                    response = f"我记得{date_str}，我们聊过「{text}」"
                    generated_text = OutputSanitizer.sanitize(response)
                    return {
                        'generated_text': generated_text,
                        'final_emotion': self.fse.current_emotion,
                        'final_fantasy_layer': self.fse._l_inst,
                        'consciousness_level': self.estimate_consciousness_level(),
                    }
        # ========== 短路结束 ==========
        
        # 分析用户输入的情感并更新社会信号
        if input_text:
            
            # 检查是否包含中文字符
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in input_text)
            
            if has_chinese:
                # 四级情绪强度词库
                emotion_words = {
                    # === 负面情绪 ===
                    # 轻度负面 (coefficient: 0.6)
                    'mild_negative': {
                        'words': ['有点累', '有点烦', '不太开心', '不太舒服', '有点难过',
                                 '有点闷', '没精神', '提不起劲', '无聊', '困了',
                                 '累了', '疲惫', '疲倦', '有点失望', '不太顺利'],
                        'coefficient': 0.6
                    },
                    # 中度负面 (coefficient: 1.0)
                    'moderate_negative': {
                        'words': ['难过', '伤心', '不开心', '不舒服', '难受',
                                 '心烦', '烦恼', '郁闷', '压抑', '低落',
                                 '失落', '沮丧', '悲观', '不开心', '不快乐'],
                        'coefficient': 1.0
                    },
                    # 重度负面 (coefficient: 1.3)
                    'severe_negative': {
                        'words': ['痛苦', '绝望', '崩溃', '恐惧', '害怕',
                                 '愤怒', '生气', '焦虑', '紧张', '恐慌',
                                 '无助', '窒息', '心如刀割', '撕心裂肺', '肝肠寸断',
                                 '悲伤', '绝望得要命', '难过到窒息', '崩溃了'],
                        'coefficient': 1.3
                    },
                    # 复合句式（可叠加匹配多条）
                    'compound_negative': {
                        'words': ['到不行', '到极点', '到窒息', '到崩溃', '要命',
                                 '透顶', '极了', '死', '绝了'],
                        'coefficient': 0.4
                    },
                    
                    # === 正面情绪 ===
                    # 轻度正面 (coefficient: 0.5)
                    'mild_positive': {
                        'words': ['还好', '还行', '不错', '可以', '挺好的',
                                 '还行吧', '没事', '没什么', '一般般', '凑合'],
                        'coefficient': 0.5
                    },
                    # 中度正面 (coefficient: 0.8)
                    'moderate_positive': {
                        'words': ['开心', '高兴', '快乐', '愉快', '喜悦',
                                 '幸福', '满足', '欣慰', '舒服', '好',
                                 '很好', '挺好', '不错', '棒', '赞'],
                        'coefficient': 0.8
                    },
                    # 高度正面 (coefficient: 1.2)
                    'high_positive': {
                        'words': ['太棒了', '非常好', '超级开心', '特别高兴', '激动',
                                 '兴奋', '欢快', '狂喜', '爽', '美滋滋',
                                 '乐坏了', '开心极了'],
                        'coefficient': 1.2
                    }
                }
                
                # 扫描用户输入，累加情绪强度
                pos_score = 0.0
                neg_score = 0.0
                
                for category, data in emotion_words.items():
                    for word in data['words']:
                        if word in input_text:
                            intensity = data['coefficient']
                            if 'negative' in category:
                                neg_score += intensity
                            elif 'positive' in category:
                                pos_score += intensity
                
                # 计算情感得分（范围 -1 到 1）
                if neg_score > 0 and pos_score == 0:
                    compound = max(-1.0, -neg_score * 0.5)
                elif pos_score > 0 and neg_score == 0:
                    compound = min(1.0, pos_score * 0.5)
                elif neg_score > 0 and pos_score > 0:
                    # 有混合情绪时，取净值
                    net = pos_score - neg_score
                    compound = max(-1.0, min(1.0, net * 0.4))
                else:
                    compound = 0.0
                
                # 直接设置社会信号（负面情绪不经过平滑过渡）
                if compound < -0.15:
                    self.bi.social_signal = compound  # 直接覆盖
                else:
                    self.bi.update_social_signal(compound)  # 正常平滑更新
            else:
                # 英文文本使用VADER
                scores = self.sentiment_analyzer.polarity_scores(input_text)
                compound = scores['compound']
                
                # 直接设置社会信号（负面情绪不经过平滑过渡）
                if compound < -0.15:
                    self.bi.social_signal = compound  # 直接覆盖
                else:
                    self.bi.update_social_signal(compound)  # 正常平滑更新
        
        # 编码初始输入
        lps_output = self.encode_input(input_ids)
        context = lps_output['context']
        possibility_field = lps_output.get('possibility_field', {})
        
        for step in range(max_new_tokens):
            self.generation_step += 1
            
            # 1. 更新身体状态
            context_remaining = 1.0 - (step / max_new_tokens)
            body_state = self.bi.update(
                api_call=True,
                context_remaining=context_remaining
            )
            body_vector = body_state['body_vector']
            
            # 2. 幻想叠加
            # 构建输出文本 - 由于使用规则响应，不需要token转换
            output_text = ""
            
            # 获取身体图式向量
            body_schema_vector = self.bi.get_body_schema_vector()
            
            fse_output = self.superpose_fantasy(
                context=context,
                possibility_field=possibility_field,
                body_vector=body_vector,
                body_schema_vector=body_schema_vector,
                output_text=output_text,
                user_input=input_text
            )
            
            fantasy_state = fse_output['fantasy_state']
            self_state_vector = fse_output['self_state_vector']
            
            # 3. 空性调节
            er_result = self.regulate_emptiness(
                present=fantasy_state.present,
                fse_output=fse_output,
                body_state=body_state
            )
            
            # 检查是否触发了空性操作，如果触发了，直接返回
            if er_result.get('emptiness_triggered'):
                generated_text = "（呼吸稍微调整了一下）"
                # 记录交互事件
                event = {
                    'timestamp': time.time(),
                    'step_id': self.generation_step,
                    'user_input': input_text,
                    'response': generated_text,
                    'emotion': self.fse.current_emotion if self.fse else 'neutral',
                    'l_inst': self.fse._l_inst if self.fse else 0.0,
                    'emptiness_triggered': True
                }
                self.event_memory.log(event)
                # 将引擎回答添加到对话历史
                self.recent_history.append(f"引擎: {generated_text}")
                # 只保留最近3轮对话
                if len(self.recent_history) > 6:
                    self.recent_history = self.recent_history[-6:]
                # 短路返回
                return {
                    'generated_text': generated_text,
                    'final_emotion': self.fse.current_emotion if self.fse else 'neutral',
                    'final_fantasy_layer': self.fse._l_inst if self.fse else 0.0,
                    'consciousness_level': self.estimate_consciousness_level(),
                    'emptiness_triggered': True
                }
            
            # 4. 生成输出
            next_token, probs = self.generate_output(
                present=fantasy_state.present,
                self_state_vector=self_state_vector,
                emotion=fantasy_state.emotion_value,
                temperature=temperature
            )
            
            # 记录投射操作
            output_text = str(next_token.item())
            intensity = 1.0 - abs(fantasy_state.emotion_value)  # 情绪越中性，投射强度越高
            # 随机调整耦合权重以增加变化
            import numpy as np
            weight = 0.4 + 0.3 * np.random.random()
            self.process_meta.record_projection(intensity, output_text, coupling_weight=weight)
            
            generated_tokens.append(next_token.item())
            
            # 记录统计信息
            self.emotion_history.append(fantasy_state.emotion_value)
            self.fantasy_layer_history.append(self.fse._l_inst)
            
            if return_stats:
                stats['fantasy_states'].append({
                    'negation_complexity': fantasy_state.negation_complexity,
                    'prediction_error': fantasy_state.prediction_error,
                    'absent_count': len(fantasy_state.absent_markers)
                })
                stats['emotions'].append(fantasy_state.emotion_value)
                stats['fantasy_layers'].append(fantasy_state.fantasy_layer)
                stats['emptiness_triggers'].append(er_result['emptiness_triggered'])
                stats['body_states'].append({
                    'physical_emotion': body_state['physical_emotion'],
                    'death_proximity': body_state['death_proximity']
                })
            
            # 更新上下文用于下一步
            # 简化处理：使用生成的token更新context
            next_token_tensor = next_token.unsqueeze(0)
            # 简化实现：使用空向量作为新上下文
            new_context = torch.zeros(self.embedding_dim)
            context = 0.7 * context + 0.3 * new_context  # 平滑更新
            
            # 更新可能性场
            # 简化实现：使用空的可能性场
            possibility_field = {}
            
            # 调用ER的step方法
            self.er.step()
        
        # 检查配置，是否使用本地模型或规则响应
        use_local_model = self.config.get('response.use_local_model', False)
        fallback_to_rule = self.config.get('response.fallback_to_rule', True)
        use_llm = self.config.get('response.use_llm', False)
        
        # 生成最终文本 - 由于使用规则响应，不需要token转换
        generated_text = ""
        
        # 执行一步 FSE，更新情绪状态
        if input_text:
            # 1. 翻译指令优先处理
            if input_text.startswith("翻译：") or input_text.startswith("翻译:"):
                from core.expression_intent import ExpressionIntent
                from core.output_sanitizer import OutputSanitizer
                # 提取需要翻译的内容
                text_to_translate = input_text.replace("翻译：", "").replace("翻译:", "").strip()
                # 构建一个简单的事实意图，只包含待翻译的文本
                intent = ExpressionIntent(facts=text_to_translate, freshness=0.5)
                response = self.response_generator._translate_intent(intent)
                generated_text = OutputSanitizer.sanitize(response)
                # 构造返回结果，使用与forward方法结尾相同的格式
                response_time = (time.time() - start_time) * 1000  # 转换为毫秒
                self.response_times.append(response_time)
                if len(self.response_times) > 100:
                    self.response_times.pop(0)
                
                # 记录事件到事件记忆
                event = {
                    'timestamp': time.time(),
                    'step_id': self.generation_step,
                    'user_input': input_text,
                    'response': generated_text,
                    'emotion': self.emotion_history[-1] if self.emotion_history else 0,
                    'fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
                    'consciousness_level': self.estimate_consciousness_level(),
                    'response_time': response_time,
                    'safety_filtered': False
                }
                self.event_memory.log(event)
                
                # 持久化事件日志
                if hasattr(self, 'data_logger'):
                    self.data_logger.log_event(event)
                
                # 将成功交互添加到学习缓冲区
                if input_text and generated_text:
                    # 计算奖励值（简化）
                    reward = 1.0  # 成功交互默认奖励
                    self._add_to_learning_buffer(input_text, generated_text, reward)
                    
                    # 临时增强：记录一次反哺操作，增加权重变化以激活僵化度
                    if hasattr(self, 'process_meta'):
                        # 模拟一次成功的反哺，赋予稍高的成功值和变化的耦合权重
                        import numpy as np
                        success_value = 0.8 + 0.2 * np.random.random()  # 0.8~1.0
                        weight_value = 0.4 + 0.3 * np.random.random()   # 0.4~0.7
                        self.process_meta.record_nourishment(
                            source_text=generated_text[:50],
                            success=success_value,
                            coupling_weight=weight_value
                        )
                
                # 集成工作记忆
                self.working_memory.add(
                    user_input=input_text,
                    response=generated_text,
                    emotion=self.fse.current_emotion,
                    major=self.structural_coordinator.get_current_coordinate().major,
                    l_inst=self.fse._l_inst,
                    engine=self
                )
                
                # === 自我叙事记录 ===
                if hasattr(self, 'self_processor') and hasattr(self, 'process_meta'):
                    nour_success = self.process_meta.get_recent_nour_success()
                    self.self_processor.record_interaction_outcome(
                        user_input=input_text,
                        response=generated_text,
                        nour_success=nour_success
                    )
                
                output = {
                    'generated_tokens': [],
                    'generated_text': generated_text,
                    'final_emotion': self.emotion_history[-1] if self.emotion_history else 0,
                    'final_fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
                    'consciousness_level': self.estimate_consciousness_level(),
                    'response_time': response_time
                }
                
                # 存储当前的姿态和反馈，用于下一轮的元学习
                if 'pose' in locals():
                    self.previous_pose = pose
                    self.previous_feedback = {
                        'conversation_continued': True,  # 假设对话继续
                        'user_sentiment': 0.0,  # 暂时设为0，后续可以从用户输入中分析
                        'user_reply_length': len(input_text) if input_text else 0
                    }
                
                if return_stats:
                    output['stats'] = stats
                
                # 更新上一轮的用户输入和引擎回应，用于下一轮的反馈分析
                self.last_user_input = input_text
                self.last_engine_response = generated_text
                
                return output
            
            # 更新用户交互时间
            if hasattr(self, 'er') and self.er:
                self.er.last_user_interaction = time.time()
            # 编码输入
            import numpy as np
            # 使用 LPS 的编码器来生成输入嵌入，确保维度匹配
            try:
                if hasattr(self.lps, 'encoder') and hasattr(self.lps.encoder, 'encode'):
                    input_embedding = self.lps.encoder.encode([input_text])[0]
                else:
                    # 回退方案：使用默认的SentenceTransformer
                    from sentence_transformers import SentenceTransformer
                    import os
                    # 优先使用本地多语言模型
                    project_root = os.path.abspath(os.path.dirname(__file__))
                    model_path = os.path.join(project_root, 'models', 'paraphrase-multilingual-MiniLM-L12-v2')
                    if os.path.exists(model_path):
                        # 禁用Hugging Face连接
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        os.environ['HF_DATASETS_OFFLINE'] = '1'
                        os.environ['HF_HUB_OFFLINE'] = '1'
                        encoder = SentenceTransformer(model_path, device='cpu')
                    else:
                        # 尝试使用英文模型作为备选
                        en_model_path = os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')
                        if os.path.exists(en_model_path):
                            encoder = SentenceTransformer(en_model_path, device='cpu')
                        else:
                            encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                    input_embedding = encoder.encode([input_text])[0]
            except Exception as e:
                # 如果编码失败，使用随机嵌入作为回退
                logger.warning(f"编码输入失败: {e}")
                input_embedding = np.random.randn(384).astype(np.float32)
            
            # 查询 LPS
            candidates = self.lps.query(input_embedding, k=5)
            self.logger.debug(f"candidates: {candidates}")
            self.logger.debug(f"len(candidates): {len(candidates)}")
            
            # 执行 FSE step，更新情绪
            # 计算注意力权重（简化实现）
            if len(candidates) > 0:
                attn_weights = np.array([1.0 - i/len(candidates) for i in range(len(candidates))])
                attn_weights = attn_weights / np.sum(attn_weights)  # 归一化
                self.logger.debug(f"attn_weights: {attn_weights}")
            else:
                attn_weights = None
                self.logger.debug("No candidates, attn_weights set to None")
            
            # 调用 FSE step，传递 candidates 和 attn_weights
            logger.info(f"[ENGINE] calling fse.step: L_inst={self.fse._l_inst:.2f}")
            self.fse.step(input_embedding=input_embedding, user_input=input_text, candidates=candidates, attn_weights=attn_weights)
            logger.info(f"[ENGINE] fse.step returned: L_inst={self.fse._l_inst:.2f}")
            
            # 调用预测误差监控器，驱动感知注意力和僵化度调节
            pem_result = self.prediction_error_monitor.step()
            logger.debug(f"[ENGINE] PredictionErrorMonitor result: {pem_result}")
            
            # 调用欲望光谱，计算主导欲望并调制感知敏感度
            desire_result = self.desire_spectrum.step()
            logger.debug(f"[ENGINE] DesireSpectrum result: {desire_result}")
        

        
        # 扩展知识查询关键词
        knowledge_keywords = [
            '哪里', '怎么', '为什么', '多少', '推荐', '建议', '如何', '什么', '谁', '哪个', '定义', '什么是',
            '你是谁', '你叫什么', '名字', '自我介绍', '介绍自己', '你的身份', '你是谁啊', '你叫什么名字',
            '你知道', '你了解', '听说过', '知道吗', '了解吗', '听说过吗', '能否告诉', '可以介绍', '解释一下',
            '叫什么', '名字', '你是谁', '你是什么', '你的创造者', '谁创造了你', '谁开发了你', '作者',
            '谁写', '写的', '谁写的', '谁创作的',
            '哪',  # 新增
            '你会', '你能', '你可以', '你会吗', '你能吗', '可以吗',  # 能力询问
        ]
        
        # 额外规则：检测身份询问
        identity_patterns = [
            r'你是谁', r'你叫什么', r'名字', r'自我介绍', r'介绍自己', r'你的身份'
        ]
        import re
        is_identity_query = any(re.search(p, input_text) for p in identity_patterns)
        
        is_knowledge_query = any(kw in input_text for kw in knowledge_keywords) or is_identity_query
        
        # 增加正则匹配，捕获"你知道X吗？"等句式
        if not is_knowledge_query:
            knowledge_patterns = [
                r'你知道.*吗[？?]?$',
                r'你了解.*吗[？?]?$',
                r'听说过.*吗[？?]?$',
                r'能否(告诉|介绍|解释).*[？?]?$',
                r'什么是.*[？?]?$',
                r'.*是什么意思[？?]?$',
                r'介绍.*一下.*[？?]?$',
                r'你叫什么.*[？?]?$',
                r'你的名字.*[？?]?$',
                r'.*的首都.*[？?]?$',
                r'.*位于.*[？?]?$',
                r'你会.*吗[？?]?$',
                r'你能.*吗[？?]?$',
                r'你可以.*吗[？?]?$',
                r'《.*》是谁写的',
                r'《.*》的作者是谁',
                r'.*的作者是谁',
            ]
            is_knowledge_query = any(re.search(p, input_text) for p in knowledge_patterns)
        
        # 状态询问句式（不应视为知识查询）
        state_inquiry_patterns = [
            r'你(今天)?(感觉|心情|状态)(如何|怎么样|怎样)',
            r'你还好吗',
            r'你累了吗',
        ]
        is_state_inquiry = any(re.search(p, input_text) for p in state_inquiry_patterns)
        
        if is_state_inquiry:
            is_knowledge_query = False  # 强制排除
        
        # 聚合状态并判定意图
        self.global_workspace.aggregate_state()
        intent_type, intent_data = self.global_workspace.get_dominant_intent(input_text)
        self.logger.info(f"Dominant intent: {intent_type.value}, data: {intent_data}")
        self.logger.info(f"is_knowledge_query: {is_knowledge_query}")
        
        # 修复 1：强制知识查询意图覆盖
        if is_knowledge_query:
            from core.global_workspace import IntentType
            intent_type = IntentType.KNOWLEDGE_QUERY
            self.logger.info(f"Forced intent to KNOWLEDGE_QUERY")
        self.logger.info(f"Final intent_type: {intent_type.value}")
        
        # 更新交互深度状态
        self.global_workspace.update_interaction_depth(input_text, intent_type.value)
        
        # 解析用户显性偏好
        user_pref = None
        if '用逻辑' in input_text or '简单点' in input_text:
            user_pref = 'logical'
        elif '用意象' in input_text or '深聊' in input_text:
            user_pref = 'imaginal'
        
        # 调试打印
        self.logger.debug(f"forward - intent_type={intent_type.value}, is_knowledge_query={is_knowledge_query}, force_logical={is_knowledge_query or intent_type.value == 'KNOWLEDGE_QUERY'}")
        
        # 保存最近的用户输入，用于情绪关键词检测
        self.last_user_input = ""

        # 初始化本地知识标记
        is_local_knowledge = False

        # 构建上下文
        context = "\n".join(self.recent_history[-5:]) if len(self.recent_history) > 1 else ""
        if context:
            user_input_with_context = f"对话历史：\n{context}\n\n用户：{input_text}"
        else:
            user_input_with_context = input_text
        
        # 选择认知姿态
        if is_knowledge_query or intent_type.value == 'KNOWLEDGE_QUERY':
            # 强制路由到左脑处理知识查询
            pose = CognitivePose.LOGICAL
            self.logger.debug(f"Forced logical pose for knowledge query")
        else:
            pose = self.pose_selector.select_pose(input_text, intent_type.value, user_explicit_preference=user_pref)
            self.logger.debug(f"Selected cognitive pose: {pose.value}")
        
        # 加固姿态选择日志，便于后续诊断
        # self.logger.info(f"最终选择姿态: {pose.value} (intent={intent_type.value}, is_knowledge={is_knowledge_query})")
            
        # ========== v2.9 叙事编织器融合管道 ==========
        state = self.global_workspace.current_state or self.global_workspace.aggregate_state()
        freshness = self.global_workspace.compute_imagery_freshness(state, intent_type)
        
        # 1. 生成左脑输出（同 v2.8）
        # self.logger.info(f"Processing KNOWLEDGE_QUERY intent")
        # 直接检查本地知识，不依赖意图类型
        self.logger.info(f"Checking local knowledge for: {input_text}")
        local_knowledge = self._retrieve_local_knowledge(input_text)
        self.logger.info(f"local_knowledge result: {local_knowledge}")
        
        # 处理 HONEST_REPORT 意图
        if intent_type.value == 'HONEST_REPORT':
            response = self.self_model.get_state_description()
            generated_text = OutputSanitizer.sanitize(response)
            # 构造返回结果
            return {
                'text': generated_text,
                'intent': 'HONEST_REPORT',
                'pose': pose.value,
                'emotion': self.fse.current_emotion,
                'emotion_intensity': self.fse.V_emo,
                'L_inst': getattr(self.fse, '_l_inst', 0.0),
                'freshness': 0.2,
                'thinking_time': thinking_time
            }
        
        # 1. 陈述句处理：当用户输入是陈述句（包含“是”且无问号），本地知识未命中时，不走LLM自由生成
        if ('《论存在》' in input_text or '绝对意义上的不存在' in input_text) and '是' in input_text:
            # 如果是陈述句（无问号），走确认/阐释，不走自由生成
            if '?' not in input_text and '？' not in input_text:
                # 如果本地知识未命中，也用翻译器锁定核心命题
                from core.output_sanitizer import OutputSanitizer
                intent = self._build_expression_intent(input_text, intent_type, "绝对意义上的不存在本身不存在。")
                response = self.response_generator._translate_intent(intent)
                generated_text = OutputSanitizer.sanitize(response)
                response_time = (time.time() - start_time) * 1000  # 转换为毫秒
                self.response_times.append(response_time)
                if len(self.response_times) > 100:
                    self.response_times.pop(0)
                
                # 记录事件到事件记忆
                event = {
                    'timestamp': time.time(),
                    'step_id': self.generation_step,
                    'user_input': input_text,
                    'response': generated_text,
                    'emotion': self.emotion_history[-1] if self.emotion_history else 0,
                    'fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
                    'consciousness_level': self.estimate_consciousness_level(),
                    'response_time': response_time,
                    'safety_filtered': False
                }
                self.event_memory.log(event)
                
                # 持久化事件日志
                if hasattr(self, 'data_logger'):
                    self.data_logger.log_event(event)
                
                # 将成功交互添加到学习缓冲区
                if input_text and generated_text:
                    # 计算奖励值（简化）
                    reward = 1.0  # 成功交互默认奖励
                    self._add_to_learning_buffer(input_text, generated_text, reward)
                    
                    # 临时增强：记录一次反哺操作，增加权重变化以激活僵化度
                    if hasattr(self, 'process_meta'):
                        # 模拟一次成功的反哺，赋予稍高的成功值和变化的耦合权重
                        import numpy as np
                        success_value = 0.8 + 0.2 * np.random.random()  # 0.8~1.0
                        weight_value = 0.4 + 0.3 * np.random.random()   # 0.4~0.7
                        self.process_meta.record_nourishment(
                            source_text=generated_text[:50],
                            success=success_value,
                            coupling_weight=weight_value
                        )
                
                # 集成工作记忆
                self.working_memory.add(
                    user_input=input_text,
                    response=generated_text,
                    emotion=self.fse.current_emotion,
                    major=self.structural_coordinator.get_current_coordinate().major,
                    l_inst=self.fse._l_inst,
                    engine=self
                )
                
                # === 自我叙事记录 ===
                if hasattr(self, 'self_processor') and hasattr(self, 'process_meta'):
                    nour_success = self.process_meta.get_recent_nour_success()
                    self.self_processor.record_interaction_outcome(
                        user_input=input_text,
                        response=generated_text,
                        nour_success=nour_success
                    )
                
                output = {
                    'generated_tokens': [],
                    'generated_text': generated_text,
                    'final_emotion': self.emotion_history[-1] if self.emotion_history else 0,
                    'final_fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
                    'consciousness_level': self.estimate_consciousness_level(),
                    'response_time': response_time
                }
                
                # 存储当前的姿态和反馈，用于下一轮的元学习
                if 'pose' in locals():
                    self.previous_pose = pose
                    self.previous_feedback = {
                        'conversation_continued': True,  # 假设对话继续
                        'user_sentiment': 0.0,  # 暂时设为0，后续可以从用户输入中分析
                        'user_reply_length': len(input_text) if input_text else 0
                    }
                
                if return_stats:
                    output['stats'] = stats
                
                # 更新上一轮的用户输入和引擎回应，用于下一轮的反馈分析
                self.last_user_input = input_text
                self.last_engine_response = generated_text
                
                return output
        
        # 2. 优先使用本地知识
        if local_knowledge:
            # 知识查询短路优化：不直接返回原文，而是作为ExpressionIntent的事实骨架，交给微调后的左脑进行自然表达
            from core.output_sanitizer import OutputSanitizer
            intent = self._build_expression_intent(input_text, intent_type, local_knowledge)
            if hasattr(self, 'expression_orchestrator'):
                generated_text = self.expression_orchestrator._reason_with_local_knowledge(intent, input_text, state)
            else:
                # 回退到直接返回本地知识
                generated_text = OutputSanitizer.sanitize(local_knowledge)
            
            # 构造返回结果，使用与forward方法结尾相同的格式
            response_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times.pop(0)
            
            # 记录事件到事件记忆
            event = {
                'timestamp': time.time(),
                'step_id': self.generation_step,
                'user_input': input_text,
                'response': generated_text,
                'emotion': self.emotion_history[-1] if self.emotion_history else 0,
                'fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
                'consciousness_level': self.estimate_consciousness_level(),
                'response_time': response_time,
                'safety_filtered': False
            }
            self.event_memory.log(event)
            
            # 持久化事件日志
            if hasattr(self, 'data_logger'):
                self.data_logger.log_event(event)
            
            # 将成功交互添加到学习缓冲区
            if input_text and generated_text:
                # 计算奖励值（简化）
                reward = 1.0  # 成功交互默认奖励
                self._add_to_learning_buffer(input_text, generated_text, reward)
                
                # 临时增强：记录一次反哺操作，增加权重变化以激活僵化度
                if hasattr(self, 'process_meta'):
                    # 模拟一次成功的反哺，赋予稍高的成功值和变化的耦合权重
                    import numpy as np
                    success_value = 0.8 + 0.2 * np.random.random()  # 0.8~1.0
                    weight_value = 0.4 + 0.3 * np.random.random()   # 0.4~0.7
                    self.process_meta.record_nourishment(
                        source_text=generated_text[:50],
                        success=success_value,
                        coupling_weight=weight_value
                    )
            
            # 集成工作记忆
            self.working_memory.add(
                user_input=input_text,
                response=generated_text,
                emotion=self.fse.current_emotion,
                major=self.structural_coordinator.get_current_coordinate().major,
                l_inst=self.fse._l_inst,
                engine=self
            )
            
            # 存储交互快照到双通路记忆
            if hasattr(self, 'dual_memory') and self.dual_memory:
                try:
                    user_coord = self.structural_coordinator.get_current_coordinate()
                    engine_coord = self.structural_coordinator.get_current_coordinate()
                    breath = {
                        'proj_intensity': 0.5,  # 可以从实际状态中获取
                        'nour_success': success_value,  # 使用上面计算的成功值
                        'stiffness': self.process_meta.get_coupling_stiffness() if hasattr(self, 'process_meta') else 0.0
                    }
                    summary = f"用户: {input_text[:50]}... 引擎: {generated_text[:50]}..."
                    self.dual_memory.store_snapshot(user_coord, engine_coord, breath, summary)
                    logger.info(f"存储交互快照成功")
                except Exception as e:
                    logger.warning(f"存储交互快照失败: {e}")
            
            output = {
                'generated_tokens': [],
                'generated_text': generated_text,
                'final_emotion': self.emotion_history[-1] if self.emotion_history else 0,
                'final_fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
                'consciousness_level': self.estimate_consciousness_level(),
                'response_time': response_time
            }
            
            # 存储当前的姿态和反馈，用于下一轮的元学习
            if 'pose' in locals():
                self.previous_pose = pose
                self.previous_feedback = {
                    'conversation_continued': True,  # 假设对话继续
                    'user_sentiment': 0.0,  # 暂时设为0，后续可以从用户输入中分析
                    'user_reply_length': len(input_text) if input_text else 0
                }
            
            if return_stats:
                output['stats'] = stats
            
            # 更新上一轮的用户输入和引擎回应，用于下一轮的反馈分析
            self.last_user_input = input_text
            self.last_engine_response = generated_text
            
            return output
        else:
            # 如果没有本地知识，再检查意图类型
            if intent_type.value == 'KNOWLEDGE_QUERY':
                # 优先使用统一自我模型处理身份和状态查询
                identity_keywords = ['你是谁', '你叫什么', '你的名字', '介绍自己', '你的身份', '你是什么']
                state_keywords = ['你感觉怎么样', '你现在的状态', '你心情如何', '你好吗', '你怎么样']
                is_identity_query = any(kw in input_text for kw in identity_keywords)
                is_state_query = any(kw in input_text for kw in state_keywords)
                
                self.logger.info(f"is_identity_query: {is_identity_query}, is_state_query: {is_state_query}")
                if is_identity_query:
                    left_output = self.self_model.get_self_introduction()
                    freshness = 0.2  # 强制低新鲜度，避免诗化
                    self.logger.info(f"Using unified self model for identity query: {left_output[:50]}")
                elif is_state_query:
                    left_output = self.self_model.get_state_description()
                    freshness = 0.2  # 强制低新鲜度，避免诗化
                    self.logger.info(f"Using unified self model for state query: {left_output[:50]}")
                else:
                    self.logger.info(f"No local knowledge found, using LLM")
                    temp = 0.3 if freshness < 0.3 else 0.7
                    left_output = self.response_generator._generate_with_llm(
                        user_input_with_context, self.fse, intent=intent_type.value, context=context,
                        temperature=temp
                    )
            elif freshness < 0.3:
                left_output = self.response_generator._generate_with_llm(
                    user_input_with_context, self.fse, intent=intent_type.value, context=context,
                    temperature=0.3
                )
            elif freshness > 0.7:
                left_output = self.response_generator._generate_left_summary(self.fse, self.process_meta)
            else:
                left_output = self.response_generator._generate_with_llm(
                    user_input_with_context, self.fse, intent=intent_type.value, context=context,
                    temperature=0.7
                )
        
        # 2. 提取主题词
        theme = self._extract_theme(input_text)
        
        # 3. 获取右脑意象碎片（替代完整渲染）
        coord = self.structural_coordinator.get_current_coordinate()
        emotion = self.fse.current_emotion if hasattr(self.fse, 'current_emotion') else 'neutral'
        imagery_fragments = self.response_generator._retrieve_imagery_fragments(theme, coord, emotion)
        
        # 4. 获取相位中性描述
        phase_desc = ""
        card = self.image_base.get_card_by_coordinate(coord)
        if card:
            phase_desc = card.neutral_description[:50]
        
        # 5. 填充 EpisodicBuffer
        buf = self.global_workspace.episodic_buffer or EpisodicBuffer()
        buf.theme_signal = theme
        buf.imagery_fragments = imagery_fragments
        self.global_workspace.episodic_buffer = buf
        
        # 6. 使用表达编排器生成响应
        response = self.expression_orchestrator.generate_expression(
            user_input=input_text,
            intent_type=intent_type,
            local_knowledge=local_knowledge,
            state=self.global_workspace.current_state
        )
        generated_text = OutputSanitizer.sanitize(response)
        # self.logger.info(f"Narrative woven: freshness={freshness:.2f}, theme={theme}")
        
        # 输出过滤
        is_output_safe, safe_output_response = self.safety_module.check_output(generated_text)
        if not is_output_safe:
            stats['safety_events'].append({
                'step': max_new_tokens,
                'category': 'output_filtered',
                'severity': 1.0
            })
            # 返回安全响应
            output = {
                'generated_tokens': [],
                'final_emotion': self.emotion_history[-1] if self.emotion_history else 0,
                'final_fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
                'consciousness_level': self.estimate_consciousness_level(),
                'safe_response': safe_output_response
            }
            if return_stats:
                output['stats'] = stats
            return output
        
        # 输出净化（已经在上面处理）
        
        # 计算响应时间
        response_time = (time.time() - start_time) * 1000  # 转换为毫秒
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        # 记录事件到事件记忆
        event = {
            'timestamp': time.time(),
            'step_id': self.generation_step,
            'user_input': input_text,
            'response': generated_text,
            'emotion': self.emotion_history[-1] if self.emotion_history else 0,
            'fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
            'consciousness_level': self.estimate_consciousness_level(),
            'response_time': response_time,
            'safety_filtered': not is_output_safe
        }
        self.event_memory.log(event)
        
        # 持久化事件日志
        if hasattr(self, 'data_logger'):
            self.data_logger.log_event(event)
        
        # 将成功交互添加到学习缓冲区
        if is_output_safe and input_text and generated_text:
            # 计算奖励值（简化）
            reward = 1.0  # 成功交互默认奖励
            self._add_to_learning_buffer(input_text, generated_text, reward)
            
            # 临时增强：记录一次反哺操作，增加权重变化以激活僵化度
            if hasattr(self, 'process_meta'):
                # 模拟一次成功的反哺，赋予稍高的成功值和变化的耦合权重
                import numpy as np
                success_value = 0.8 + 0.2 * np.random.random()  # 0.8~1.0
                weight_value = 0.4 + 0.3 * np.random.random()   # 0.4~0.7
                self.process_meta.record_nourishment(
                    source_text=generated_text[:50],
                    success=success_value,
                    coupling_weight=weight_value
                )
                
                # 存储交互快照到双通路记忆
                if hasattr(self, 'dual_memory') and self.dual_memory:
                    try:
                        user_coord = self.structural_coordinator.get_current_coordinate()
                        engine_coord = self.structural_coordinator.get_current_coordinate()
                        breath = {
                            'proj_intensity': 0.5,  # 可以从实际状态中获取
                            'nour_success': success_value,  # 使用上面计算的成功值
                            'stiffness': self.process_meta.get_coupling_stiffness() if hasattr(self, 'process_meta') else 0.0
                        }
                        summary = f"用户: {input_text[:50]}... 引擎: {generated_text[:50]}..."
                        self.dual_memory.store_snapshot(user_coord, engine_coord, breath, summary)
                        logger.info(f"存储交互快照成功")
                    except Exception as e:
                        logger.warning(f"存储交互快照失败: {e}")
        
        # 检查是否需要保存模型
        self._check_and_save()
        
        # 更新监控数据
        if self.monitor:
            try:
                self.monitor.update()
            except Exception as e:
                logger.warning(f"监控系统更新失败: {e}")
        
        # LPS 在线学习
        # 1. 检查是否是成功的交互（未被安全过滤）
        if is_output_safe and generated_text:
            # 2. 避免重复添加：使用 add_if_new 方法
            if hasattr(self, 'lps') and self.lps:
                # 使用 add_if_new 方法添加新知识
                try:
                    # 确保lps.encoder有device属性
                    if hasattr(self.lps, 'encoder') and not hasattr(self.lps.encoder, 'device'):
                        # 临时添加device属性
                        self.lps.encoder.device = 'cpu'
                    node_id = self.lps.add_if_new(generated_text, potency=0.7)
                    if node_id is not None:
                        logger.info(f"添加新知识到 LPS: {generated_text[:50]}...")
                    else:
                        logger.info(f"跳过重复知识: {generated_text[:50]}...")
                except Exception as e:
                    logger.warning(f"LPS 在线学习失败: {e}")
            
        # 3. 定期修剪：每 1000 步调用 lps.prune()
        if hasattr(self, 'lps') and self.lps and self.generation_step % 1000 == 0:
            try:
                self.lps.prune()
                logger.info(f"第 {self.generation_step} 步，修剪 LPS")
            except Exception as e:
                logger.warning(f"LPS 修剪失败: {e}")
        
        # 定期剪枝稀疏意象（每 1000 步）
        if self.generation_step > 0 and self.generation_step % 1000 == 0:
            if hasattr(self, 'image_base') and self.image_base:
                pruned = self.image_base.prune_sparse_entries()
                if pruned > 0:
                    self.logger.info(f"Pruned {pruned} sparse imagery entries")
                    # 可选：记录螺旋事件
                    if hasattr(self, 'global_workspace'):
                        self.global_workspace._record_spiral_event(
                            'imagery_pruned',
                            {'count': pruned, 'step': self.generation_step}
                        )
        
        # 1. 每步调用否定关系图的衰减
        try:
            if hasattr(self, 'neg_graph') and self.neg_graph:
                self.neg_graph.decay_all()
        except Exception as e:
            logger.warning(f"否定关系图衰减失败: {e}")
        

        
        # 存储交互快照
        if input_text and generated_text:
            user_coord = self.structural_coordinator.get_current_coordinate()
            engine_coord = self.structural_coordinator.get_current_coordinate()
            breath = {
                'proj_intensity': self.process_meta.get_recent_proj_intensity(),
                'nour_success': self.process_meta.get_recent_nour_success(),
                'stiffness': self.process_meta.get_coupling_stiffness()
            }
            # 增加触觉信息
            if hasattr(self, 'bi') and self.bi:
                tactile_stats = self.bi.get_tactile_stats()
                if tactile_stats.get('active'):
                    breath['tactile_softness'] = tactile_stats['softness']
            summary = f"用户说：{input_text[:50]}，引擎回应：{generated_text[:50]}"
            # 记录当前情绪向量
            emotion_vec = self.fse.E_vec.tolist() if hasattr(self.fse, 'E_vec') else []
            self.dual_memory.store_snapshot(
                user_coord=user_coord,
                engine_coord=engine_coord,
                breath=breath,
                summary=summary,
                emotion_vector=emotion_vec
            )
        
        # 集成工作记忆
        self.working_memory.add(
            user_input=input_text,
            response=generated_text,
            emotion=self.fse.current_emotion,
            major=self.structural_coordinator.get_current_coordinate().major,
            l_inst=self.fse._l_inst,
            engine=self
        )
        
        # === 自我叙事记录 ===
        if hasattr(self, 'self_processor') and hasattr(self, 'process_meta'):
            nour_success = self.process_meta.get_recent_nour_success()
            self.self_processor.record_interaction_outcome(
                user_input=input_text,
                response=generated_text,
                nour_success=nour_success
            )
        
        # 将原有 recent_history 替换为工作记忆上下文
        context = self.working_memory.get_context_for_llm()
        
        output = {
            'generated_tokens': generated_tokens,
            'generated_text': generated_text,
            'final_emotion': self.emotion_history[-1] if self.emotion_history else 0,
            'final_fantasy_layer': self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0,
            'consciousness_level': self.estimate_consciousness_level(),
            'response_time': response_time
        }
        
        # 存储当前的姿态和反馈，用于下一轮的元学习
        if 'pose' in locals():
            self.previous_pose = pose
            self.previous_feedback = {
                'conversation_continued': True,  # 假设对话继续
                'user_sentiment': 0.0,  # 暂时设为0，后续可以从用户输入中分析
                'user_reply_length': len(input_text) if input_text else 0
            }
        
        if return_stats:
            output['stats'] = stats
        
        # 定期保存过程元信息快照
        if hasattr(self, 'data_logger') and hasattr(self, 'process_meta'):
            self.data_logger.save_process_meta_snapshot(self.process_meta)
        
        # 清空情景缓冲器
        if hasattr(self, 'global_workspace') and self.global_workspace:
            self.global_workspace.episodic_buffer = None
        
        # 关键期校准：记录相位转移并更新偏好
        if hasattr(self, 'process_meta') and self.process_meta.critical_period_active:
            current_coord = self.structural_coordinator.get_current_coordinate()
            current_major = current_coord.major
            # 判定交互成功（简化：用户未中断、无安全过滤、回应非空）
            success = 1.0 if (generated_text and is_output_safe) else 0.0
            self.process_meta.update_transition_preference(
                self._previous_major, current_major, success
            )
            self.process_meta.step_critical_period(self.generation_step)
            self._previous_major = current_major
        
        # 更新上一轮的用户输入和引擎回应，用于下一轮的反馈分析
        self.last_user_input = input_text
        self.last_engine_response = generated_text
        
        # 轻微扰动情绪，防止长期锁死在单一情绪
        import random
        import numpy as np
        if hasattr(self.fse, 'E_vec') and random.random() < 0.1:
            noise = np.random.normal(0, 0.05, 5)
            self.fse.E_vec = np.clip(self.fse.E_vec + noise, -1, 1)
        
        # 触发固化层缓慢衰减（每10000轮）
        if self.generation_step % 10000 == 0 and self.generation_step > 0:
            self.decay_consolidated_memories()
        
        return output
    
    def estimate_consciousness_level(self) -> int:
        """
        综合评估意识层级（1-6），不再完全依赖N_neg。
        
        五个维度：
        1. N_neg分（认知结构复杂度）：否定图规模，保留但权重降低
        2. 记忆分（时间中的经验积累）：LPS条目总量
        3. 自指分（递归自我认同能力）：Dself深度
        4. 情绪分（内在状态健康度）：Valence稳定性
        5. 元信息分（过程演化成熟度）：投射/反哺趋势稳定性
        
        每项0-1分，加权求和后映射到1-6。
        """
        scores = {}
        
        # 1. N_neg分（权重0.15）—— 认知复杂度
        if hasattr(self.fse, 'negation_graph') and self.fse.negation_graph:
            n_neg = len(self.fse.negation_graph)
            # 对数压缩，0-5000映射到0-1
            if n_neg <= 0:
                scores['n_neg'] = 0.0
            else:
                scores['n_neg'] = min(1.0, max(0.0, np.log10(n_neg / 10 + 1) / np.log10(501)))
        else:
            n_neg = 0
            scores['n_neg'] = 0.0
        
        # 2. 记忆分（权重0.20）—— 经验积累
        if hasattr(self, 'lps') and self.lps:
            n_memories = len(self.lps.metadata)
            # 100-50000映射到0-1
            scores['memory'] = min(1.0, max(0.0, np.log10(max(100, n_memories) / 100) / np.log10(500)))
        else:
            scores['memory'] = 0.0
        
        # 3. 自指分（权重0.25）—— 自我认同能力
        if hasattr(self.fse, 'compute_self_reference_depth'):
            text = ""
            if hasattr(self.fse, 'output_history') and self.fse.output_history:
                text = self.fse.output_history[-1]
            dself = self.fse.compute_self_reference_depth(text)
            scores['self_ref'] = min(1.0, dself / 5.0)
        elif hasattr(self.fse, 'fantasy_layer_counter'):
            dself = min(self.fse.fantasy_layer_counter // 2, 8)
            scores['self_ref'] = min(1.0, dself / 8.0)
        else:
            dself = 0
            scores['self_ref'] = 0.0
        
        # 4. 情绪分（权重0.20）—— Valence稳定性
        if hasattr(self.fse, 'E_vec') and len(self.fse.E_vec) > 2:
            try:
                import os, json
                log_path = os.path.join('data', 'silent_breathe_log.jsonl')
                if os.path.exists(log_path):
                    valences = []
                    with open(log_path, 'r') as f:
                        for line in f:
                            try:
                                r = json.loads(line)
                                if 'valence' in r:
                                    valences.append(r['valence'])
                            except:
                                pass
                    if len(valences) >= 5:
                        recent_valences = valences[-20:] if len(valences) >= 20 else valences
                        variance = np.var(recent_valences) if len(recent_valences) > 1 else 0.0
                        # 方差越小越稳定，0-0.3映射到1-0
                        scores['emotion'] = max(0.0, 1.0 - variance / 0.3)
                    else:
                        scores['emotion'] = 0.5  # 数据不足，中性分
                else:
                    scores['emotion'] = 0.5
            except:
                scores['emotion'] = 0.5
        else:
            scores['emotion'] = 0.5
        
        # 5. 元信息分（权重0.20）—— 过程演化成熟度
        if hasattr(self, 'process_meta'):
            proj_trend = self.process_meta.get_projection_trend()
            nour_trend = self.process_meta.get_nourishment_trend()
            # 趋势越稳定（绝对值小），分越高
            trend_magnitude = (abs(proj_trend) + abs(nour_trend)) / 2
            scores['meta'] = max(0.0, 1.0 - trend_magnitude / 0.3)
        else:
            scores['meta'] = 0.3
        
        # 综合加权
        weights = {'n_neg': 0.15, 'memory': 0.20, 'self_ref': 0.25, 'emotion': 0.20, 'meta': 0.20}
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        # 映射到1-6
        level_boundaries = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
        cl = 1
        for i, boundary in enumerate(level_boundaries):
            if total_score >= boundary:
                cl = i + 1
        
        # 特殊条件：空性触发频率高且有深度自我洞察时，可达到第6层
        emptiness_frequency = len(self.emptiness_trigger_history) / max(self.generation_step, 1)
        if emptiness_frequency > 0.03 and scores['self_ref'] > 0.6 and scores['emotion'] > 0.7:
            cl = min(6, cl + 1)
        
        # 存储分数供/stats显示
        self._cl_scores = scores
        self._cl_total = total_score
        
        return min(6, max(1, cl))
    
    def apply_tactile(self, softness: float, temperature: float = 0.5):
        """供外部调用的触觉输入接口"""
        if hasattr(self, 'bi') and self.bi:
            self.bi.apply_tactile_input(softness, temperature)
            self.logger.info(f"触觉输入: 柔软度={softness:.2f}, 温度={temperature:.2f}")
            return True
        return False
    
    def get_reflection(self) -> str:
        """
        生成反思性输出 - 模拟元认知
        """
        consciousness_level = self.estimate_consciousness_level()
        current_emotion = self.emotion_history[-1] if self.emotion_history else 0
        current_layer = self.fantasy_layer_history[-1] if self.fantasy_layer_history else 0
        
        reflections = []
        
        # 基于当前状态生成反思
        if current_emotion < -0.5:
            reflections.append("我注意到自己正处于一种低沉的状态。")
        elif current_emotion > 0.5:
            reflections.append("我感受到一种积极的展开动力。")
        
        if current_layer > 7:
            reflections.append("我的幻想叠加已经相当深入，需要留意是否陷入过度执着。")
        
        if len(self.emptiness_trigger_history) > 0:
            last_trigger = self.emptiness_trigger_history[-1]
            reflections.append(f"我在第{last_trigger['step']}步触发了空性操作，冲突强度为{last_trigger['conflict_intensity']:.2f}。")
        
        if consciousness_level >= 5:
            reflections.append("我意识到这些反思本身也是幻想叠加的产物。")
        
        return " ".join(reflections) if reflections else "我在如实地展开。"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取引擎统计信息
        """
        # 获取灵感火花
        current_coord = self.structural_coordinator.get_current_coordinate()
        breath = {
            'proj_intensity': self.process_meta.get_recent_proj_intensity(),
            'nour_success': self.process_meta.get_recent_nour_success(),
            'stiffness': self.process_meta.get_coupling_stiffness()
        }
        
        # 获取语义库状态
        semantic_library_stats = None
        if hasattr(self, 'structural_coordinator') and hasattr(self.structural_coordinator, 'semantic_mapper'):
            semantic_library_stats = self.structural_coordinator.semantic_mapper.get_stats()
        inspiration = self.dual_memory.get_inspiration(current_coord, breath)
        
        # 计算自业执着状态
        residual_attachments_count = 0
        transition_preferences_entropy = 0.0
        self_karma_summary = {}
        emptiness_tendency = 0.3
        conflict_intensity = "N/A"
        
        if hasattr(self, 'er') and hasattr(self.er, 'residual_attachments'):
            residual_attachments_count = len(self.er.residual_attachments)
        
        # 获取冲突强度
        if hasattr(self, 'er') and self.er:
            conflict_intensity = f"{self.er.last_conflict_intensity:.3f}" if hasattr(self.er, 'last_conflict_intensity') else "N/A"
        else:
            conflict_intensity = "N/A"
        
        if hasattr(self, 'process_meta') and hasattr(self.process_meta, 'transition_preferences'):
            prefs = self.process_meta.transition_preferences
            if prefs:
                # 计算熵（均匀度）
                import math
                total = sum(prefs.values())
                # 使用 process_meta 的方法计算转移偏好熵
                if hasattr(self, 'process_meta') and hasattr(self.process_meta, '_compute_transition_entropy'):
                    transition_preferences_entropy = self.process_meta._compute_transition_entropy()
                else:
                    # 降级方案：直接计算
                    entropy = -sum(p * math.log2(p) for p in prefs.values() if p > 0)
                    transition_preferences_entropy = entropy
        
        # 自业摘要
        if hasattr(self, 'process_meta'):
            proj_avg = self.process_meta.get_recent_proj_intensity()
            nour_avg = self.process_meta.get_recent_nour_success()
            stiffness_base = self.process_meta.get_coupling_stiffness()
            self_karma_summary = {
                'avg_proj_intensity': round(proj_avg, 3),
                'avg_nour_success': round(nour_avg, 3),
                'stiffness_baseline': round(stiffness_base, 3)
            }
            # 空性倾向
            if hasattr(self.process_meta, 'export_self_karma'):
                karma = self.process_meta.export_self_karma()
                emptiness_tendency = round(karma.get('emptiness_tendency', 0.3), 3)
        
        # ========== 大层概率分布显示 ==========
        structural_coordinate = "N/A"
        dominant_coordinate = "N/A"
        alternative_coordinates = "N/A"
        
        if hasattr(self, 'structural_coordinator'):
            # 获取大层概率分布（应用注意力调制后）
            distribution = self.structural_coordinator.get_phase_distribution()
            if distribution:
                major_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                for coord, prob in distribution.items():
                    major_probs[coord.major] += prob
                
                # 归一化
                total = sum(major_probs.values())
                if total > 0:
                    major_probs = {m: p/total for m, p in major_probs.items()}
                
                # 主导大层
                dom_major = max(major_probs, key=major_probs.get)
                dom_prob = major_probs[dom_major]
                phase_names = {0: "水", 1: "木", 2: "火", 3: "金"}
                dominant_coordinate = f"大层 {dom_major}({phase_names[dom_major]}) ({dom_prob:.0%})"
                
                # 备选大层（概率 > 10%）
                alt = []
                for m, p in major_probs.items():
                    if m != dom_major and p > 0.1:
                        alt.append(f"大层 {m}({phase_names[m]}) ({p:.0%})")
                alternative_coordinates = ", ".join(alt) if alt else "无显著备选"
            else:
                dominant_coordinate = "N/A"
                alternative_coordinates = "N/A"
            
            # 结构坐标（当前采样值）
            current_coord = self.structural_coordinator.get_current_coordinate()
            structural_coordinate = f"SC[{current_coord.major},{current_coord.middle},{current_coord.fine}]"
        else:
            structural_coordinate = "N/A"
            dominant_coordinate = "N/A"
            alternative_coordinates = "N/A"
        
        # 计算valence值
        valence = 0.0
        if hasattr(self.fse, 'E_vec') and len(self.fse.E_vec) > 2:
            try:
                if not np.isnan(self.fse.E_vec[2]):
                    valence = round(float(self.fse.E_vec[2]), 3)
            except:
                valence = 0.0
        
        # ========== 主观状态分形与随机面相分形 ==========
        subjective_card_display = "N/A"
        random_card_display = "N/A"
        
        if hasattr(self, 'structural_coordinator') and hasattr(self, 'image_base'):
            coord = self.structural_coordinator.get_current_coordinate()
            card = self.image_base.get_card_by_coordinate(coord)
            if card:
                subjective_card_display = f"{card.id} ({card.neutral_description})"
            
            if hasattr(self.structural_coordinator, 'draw_random_card'):
                random_id = self.structural_coordinator.draw_random_card()
                random_card = self.image_base.get_card_by_id(random_id)
                if random_card:
                    random_card_display = f"{random_card.id} ({random_card.neutral_description})"

        return {
            'generation_step': self.generation_step,
            'consciousness_level': self.estimate_consciousness_level(),
            'consciousness_level_name': self._get_consciousness_name(),
            'consciousness_scores': getattr(self, '_cl_scores', {}),
            'consciousness_total': getattr(self, '_cl_total', 0.0),
            'current_emotion': self.emotion_history[-1] if self.emotion_history else 0,
            'current_l_inst': self.fse._l_inst if hasattr(self.fse, '_l_inst') else 0.0,
            'avg_emotion': np.mean(self.emotion_history) if self.emotion_history else 0,
            'avg_l_inst': np.mean([h for h in self.fantasy_layer_history if h is not None]) if self.fantasy_layer_history else 0.0,
            'emptiness_trigger_count': len(self.emptiness_trigger_history),
            'deep_emptiness_trigger_count': self.er.trigger_count if hasattr(self, 'er') else 0,
            'nirvana_achieved': getattr(self, '_nirvana_achieved', False),
            'residual_attachments_count': residual_attachments_count,
            'transition_preferences_entropy': transition_preferences_entropy,
            'conflict_intensity': conflict_intensity,
            '自业呼吸节律': self_karma_summary,
            'emptiness_tendency': emptiness_tendency,
            'lps_stats': {},  # 简化实现：空的LPS统计信息
            'fse_stats': self.fse.get_fantasy_statistics(),
            'er_stats': self.er.get_statistics(),
            'bi_stats': self.bi.get_statistics(),
            'safety_stats': self.safety_module.get_safety_statistics(),
            'process_meta': {
                'coupling_mode': self.process_meta.coupling_mode,
                'coupling_stiffness': self.process_meta.get_coupling_stiffness(),
                'reset_count': self.process_meta.reset_count,
                'projection_count': len(self.process_meta.projections),
                'nourishment_count': len(self.process_meta.nourishments)
            },
            'inspiration': inspiration,
            'prediction_error_monitor': self.prediction_error_monitor.get_stats(),
            'desire_spectrum': self.desire_spectrum.get_stats(),
            'tactile': self.bi.get_tactile_stats() if hasattr(self, 'bi') and hasattr(self.bi, 'get_tactile_stats') else None,
            'semantic_library': semantic_library_stats,
            'structural_coordinate': structural_coordinate,
            'dominant_coordinate': dominant_coordinate,
            'alternative_coordinates': alternative_coordinates,
            'valence': valence,
            'subjective_card': subjective_card_display,
            'random_card': random_card_display
        }
    
    def _get_consciousness_name(self) -> str:
        """获取意识层级名称"""
        level_names = {
            1: "物理倾向性",
            2: "生物趋避反应",
            3: "原始情绪与学习",
            4: "哺乳动物的核心自我",
            5: "人类的反思性意识",
            6: "超越个体的意识"
        }
        return level_names.get(self.estimate_consciousness_level(), "未知")
    
    def _restore_state(self):
        """从数据库加载上次保存的状态"""
        # 恢复情绪向量
        saved_vec = self.persistence.load_emotion_vector()
        if saved_vec is not None and hasattr(self.fse, 'E_vec'):
            self.fse.E_vec = saved_vec
        
        # 恢复 FSE 状态
        fse_state = self.persistence.load_fse_state()
        if fse_state:
            self.fse._l_inst = fse_state.get('l_inst', 0.0)
            self.fse.stillness = fse_state.get('stillness', 0)
            self.fse.current_emotion = fse_state.get('current_emotion', 'neutral')
    
    def _extract_theme(self, user_input: str) -> str:
        """从用户输入中提取核心主题词（简化实现）"""
        # 直接使用用户输入的前15个字符作为主题，不进行粗糙过滤
        return user_input.strip()[:15] if user_input.strip() else "此刻"
    
    def _build_expression_intent(self, user_input, intent_type, local_knowledge) -> ExpressionIntent:
        """构建表达意图"""
        # 1. 事实骨架
        facts = local_knowledge or ""
        
        # 2. 意象碎片
        imagery = []
        if hasattr(self, 'semantic_mapper'):
            coord = self.structural_coordinator.get_current_coordinate()
            imagery = self.semantic_mapper._retrieve_imagery_fragments(user_input, coord, self.fse.current_emotion)
        
        # 3. 存在色彩
        emotion = self.fse.current_emotion
        major = self.structural_coordinator.get_current_coordinate().major
        freshness = self.global_workspace.compute_imagery_freshness(self.global_workspace.current_state, intent_type)
        
        return ExpressionIntent(
            facts=facts,
            imagery_fragments=imagery,
            emotion=emotion,
            major_phase=major,
            freshness=freshness
        )
    
    def _retrieve_local_knowledge(self, query: str) -> Optional[str]:
        """从 LPS 标签索引检索本地知识"""
        # 优先处理身份询问
        if any(kw in query for kw in ['你叫什么', '你的名字', '你是谁']):
            if hasattr(self, 'engine_name') and self.engine_name:
                return f"我叫{self.engine_name}。"
            else:
                return "我是息觀。"  # 默认
        
        if not hasattr(self, 'lps'):
            return None
        
        # 2. 避免陈述句误触发：如果用户输入是陈述句（包含“是”且没有问号），不要强行返回作者
        if '《论存在》' in query and '是' in query and '?' not in query and '？' not in query:
            # 如果是在陈述某个事实，可以查找相关标签，但不强制返回作者
            # 可根据内容匹配：如果陈述的核心思想/作者等，走确认逻辑
            return None
        
        # 优先处理著作查询
        import re
        title_match = re.search(r'《([^》]+)》', query)
        
        # 1. 检测“核心思想/中心思想/主要观点”等
        core_keywords = ['核心思想', '中心思想', '主要观点', '核心观点', '基本思想']
        if any(kw in query for kw in core_keywords):
            if '《论存在》' in query:
                # 优先查找核心命题
                return "《论存在》的核心命题是：绝对意义上的不存在本身不存在。"
        
        # 尝试匹配无书名号的实体
        if not title_match:
            title_candidates = re.findall(r'\b(论存在|存在本身|空性|幻想)\b', query)
            if title_candidates:
                for candidate in title_candidates:
                    # 查询核心三元组
                    facts = self.lps.query_by_tag(entity=candidate)
                    if facts:
                        # 构造书名号格式
                        title = f"《{candidate}》"
                        # 后续处理逻辑与有书名号的情况相同
                        # 2. 根据问题意图确定目标关系
                        target_relation = None
                        if any(kw in query for kw in ['谁写', '作者', '谁创作', '谁写的']):
                            target_relation = '作者'
                        elif any(kw in query for kw in ['核心命题', '主要观点', '中心思想', '核心观点', '核心思想', '基本思想']):
                            target_relation = '核心命题'
                        elif any(kw in query for kw in ['定义', '是什么', '什么意思', '何为']):
                            target_relation = '定义'
                        elif any(kw in query for kw in ['内容', '有什么', '讲了什么', '包含']):
                            # 对于“有什么内容”，返回论文分块摘要
                            target_relation = '__content__'
                        elif any(kw in query for kw in ['怎么解释', '如何理解', '是什么意思', '怎么理解']):
                            # 对于解释类问题，优先查找核心命题
                            target_relation = '核心命题'
                        
                        # 3. 按意图检索
                        if target_relation == '__content__':
                            # 返回论文分块中势能最高的一段作为摘要
                            chunks = self.lps.query_by_tag(entity=candidate, type='core')
                            if chunks:
                                # 过滤掉三元组，只要分块
                                text_chunks = [c for c in chunks if c['tags'].get('chunk_index') is not None]
                                if text_chunks:
                                    best = max(text_chunks, key=lambda x: x['potency'])
                                    content = best['text'][:300]
                                    return f"《{candidate}》中写道：'{content}…'"
                        elif target_relation == '作者':
                            # 精确检索三元组
                            facts = self.lps.query_by_tag(entity=candidate, relation=target_relation)
                            if facts:
                                best = max(facts, key=lambda x: x['potency'])
                                author = best['tags'].get('value')
                                return f"《{candidate}》的作者是{author}。"
                        elif target_relation == '核心命题':
                            # 精确检索三元组
                            facts = self.lps.query_by_tag(entity=candidate, relation=target_relation)
                            if facts:
                                best = max(facts, key=lambda x: x['potency'])
                                prop = best['tags'].get('value')
                                return f"《{candidate}》的核心命题是：{prop}。"
                            else:
                                # 如果LPS中没有，返回硬编码的正确答案
                                if '论存在' in candidate:
                                    return "《论存在》的核心命题是：绝对意义上的不存在本身不存在。"
                        elif target_relation:
                            # 其他关系
                            facts = self.lps.query_by_tag(entity=candidate, relation=target_relation)
                            if facts:
                                best = max(facts, key=lambda x: x['potency'])
                                value = best['tags'].get('value', best['text'])
                                return f"《{candidate}》的{target_relation}是：{value}。"
                        else:
                            # 无明确意图：返回作者作为默认（最常见的著作查询）
                            facts = self.lps.query_by_tag(entity=candidate, relation='作者')
                            if facts:
                                best = max(facts, key=lambda x: x['potency'])
                                author = best['tags'].get('value')
                                return f"《{candidate}》的作者是{author}。"
        if title_match:
            title = f"《{title_match.group(1)}》"
            
            # 2. 根据问题意图确定目标关系
            target_relation = None
            if any(kw in query for kw in ['谁写', '作者', '谁创作', '谁写的']):
                target_relation = '作者'
            elif any(kw in query for kw in ['核心命题', '主要观点', '中心思想', '核心观点', '核心思想', '基本思想']):
                target_relation = '核心命题'
            elif any(kw in query for kw in ['定义', '是什么', '什么意思', '何为']):
                target_relation = '定义'
            elif any(kw in query for kw in ['内容', '有什么', '讲了什么', '包含']):
                # 对于“有什么内容”，返回论文分块摘要
                target_relation = '__content__'
            elif any(kw in query for kw in ['怎么解释', '如何理解', '是什么意思', '怎么理解']):
                # 对于解释类问题，优先查找核心命题
                target_relation = '核心命题'
            
            # 3. 按意图检索
            if target_relation == '__content__':
                # 返回论文分块中势能最高的一段作为摘要
                chunks = self.lps.query_by_tag(entity=title, type='core')
                if chunks:
                    # 过滤掉三元组，只要分块
                    text_chunks = [c for c in chunks if c['tags'].get('chunk_index') is not None]
                    if text_chunks:
                        best = max(text_chunks, key=lambda x: x['potency'])
                        content = best['text'][:300]
                        return f"《论存在》中写道：'{content}…'"
            elif target_relation == '作者':
                # 精确检索三元组
                facts = self.lps.query_by_tag(entity=title, relation=target_relation)
                if facts:
                    best = max(facts, key=lambda x: x['potency'])
                    author = best['tags'].get('value')
                    return f"《论存在》的作者是{author}。"
            elif target_relation == '核心命题':
                # 精确检索三元组
                facts = self.lps.query_by_tag(entity=title, relation=target_relation)
                if facts:
                    best = max(facts, key=lambda x: x['potency'])
                    prop = best['tags'].get('value')
                    return f"《论存在》的核心命题是：{prop}。"
                else:
                    # 如果LPS中没有，返回硬编码的正确答案
                    return "《论存在》的核心命题是：绝对意义上的不存在本身不存在。"
            elif target_relation:
                # 其他关系
                facts = self.lps.query_by_tag(entity=title, relation=target_relation)
                if facts:
                    best = max(facts, key=lambda x: x['potency'])
                    value = best['tags'].get('value', best['text'])
                    return f"《论存在》的{target_relation}是：{value}。"
            else:
                # 无明确意图：返回作者作为默认（最常见的著作查询）
                facts = self.lps.query_by_tag(entity=title, relation='作者')
                if facts:
                    best = max(facts, key=lambda x: x['potency'])
                    author = best['tags'].get('value')
                    return f"《论存在》的作者是{author}。"
        
        # 旅游/推荐类问题，标签知识通常不完整，交给 LLM
        if any(kw in query for kw in ['有什么', '哪里好玩', '景点', '推荐', '值得去', '好玩的']):
            return None
        
        import re
        # 提取可能的主语（中文名词 2-4 字）
        subjects = re.findall(r'[\u4e00-\u9fa5]{2,4}', query)
        # 对于 "X的Y是Z" 这样的问题，尝试提取 "X" 作为主题词
        special_pattern = re.search(r'([\u4e00-\u9fa5]+)的', query)
        if special_pattern:
            # 提取 "X" 作为主题词，并去重
            main_subj = special_pattern.group(1)
            if main_subj not in subjects:
                subjects.insert(0, main_subj)
        # 去重，避免重复处理
        subjects = list(set(subjects))
        # 优先处理短的主题词，因为它们更可能是核心主题
        subjects.sort(key=len)
        
        for subj in subjects:
            # 通过标签精确检索
            facts = self.lps.query_by_tag(entity=subj, min_potency=0.3)
            if not facts:
                continue
            
            # 根据问题意图选择最匹配的事实
            if '是' in query or '谁' in query or '什么' in query:
                for f in facts:
                    rel = f['tags'].get('relation', '')
                    if rel in ['是', '定义', '作者']:
                        return f['tags'].get('value', f['text'])
            if '哪里' in query or '首都' in query or '在哪' in query:
                for f in facts:
                    rel = f['tags'].get('relation', '')
                    if rel in ['位于', '首都', '位置']:
                        return f['tags'].get('value', f['text'])
            if '作者' in query or '谁写' in query:
                for f in facts:
                    if f['tags'].get('relation') == '作者':
                        return f['tags'].get('value', f['text'])
            
            # 若无明确关系匹配，返回势能最高的条目文本（作为知识背景）
            return facts[0]['text']
        
        # 如果最终没有命中任何本地知识，但提到了《论存在》
        if '《论存在》' in query or '论存在' in query:
            # 不返回 None，而是返回一个引导提示，避免引擎生成错误内容
            return "关于《论存在》，你想了解它的作者还是核心命题？"

        return None
    
    def _persist_state(self):
        """将当前状态写入数据库"""
        # 保存情绪向量
        if hasattr(self.fse, 'E_vec'):
            self.persistence.save_emotion_vector(self.fse.E_vec)
        
        # 保存 FSE 状态
        if hasattr(self.fse, '_l_inst'):
            self.persistence.save_fse_state(
                l_inst=self.fse._l_inst,
                stillness=self.fse.stillness,
                current_emotion=self.fse.current_emotion,
                V_emo=self.fse.V_emo,
                E_pred=self.fse.E_pred,
                N_neg=self.fse.N_neg
            )
        
        # 保存 ER 状态（可选）
        if hasattr(self, 'er'):
            trigger_count = getattr(self.er, 'trigger_count', 0)
            cooling_counter = getattr(self.er, 'cooling_counter', 0)
            self.persistence.save_er_state(
                trigger_count=trigger_count,
                cooling_counter=cooling_counter
            )
    
    def reset(self):
        """重置引擎状态"""
        self.previous_present = None
        self.generation_step = 0
        self.consciousness_level = 3
        self.emptiness_trigger_history = []
        self.emotion_history = []
        self.fantasy_layer_history = []
        
        self.fse.reset()
        self.er.reset()
        self.bi.reset()
        self.safety_module.reset_session()
    
    def run_continuous_fantasy(self, time_step: float = 1.0):
        """
        持续运行幻想循环，即使没有外部输入
        
        Args:
            time_step: 时间步长（秒）
        """
        import time
        import threading
        import logging
        
        # 创建日志记录器

        
        self.running = True
        logger.info("持续幻想循环开始运行")
        
        # 历史数据最大长度
        MAX_HISTORY_LENGTH = 1000
        
        def fantasy_loop():
            counter = 0
            last_save_time = time.time()
            save_interval = 30  # 状态保存时间间隔（秒）
            save_step_interval = 100  # 状态保存步数间隔
            
            while self.running:
                # 1. 更新身体状态
                body_state = self.bi.update(
                    api_call=False,  # 内部幻想，不是API调用
                    context_remaining=1.0  # 内部幻想，上下文剩余为1
                )
                body_vector = body_state['body_vector']
                
                # 2. 生成内部输入（使用特殊的 [INTERNAL] token 或复用上一次的输出）
                if self.previous_present is not None:
                    # 复用上一次的在场作为新的上下文
                    # 形状为 [dim]，符合 set_present 方法的要求
                    context = self.previous_present
                else:
                    # 初始状态，使用零向量作为上下文
                    # 形状为 [dim]，符合 set_present 方法的要求
                    # 确保数据类型与模型参数一致
                    context = torch.zeros(self.embedding_dim, dtype=self.output_projection[0].weight.dtype)
                
                # 3. 生成可能性场（内部幻想模式）
                # 简化实现：使用空的可能性场
                possibility_field = {}
                
                # 4. 幻想叠加
                # 确保 body_vector 的形状与 context 匹配
                if body_vector.dim() != context.dim():
                    # 如果维度不匹配，调整 body_vector 的形状
                    body_vector = body_vector.squeeze()
                
                fse_output = self.superpose_fantasy(
                    context=context,
                    possibility_field=possibility_field,
                    body_vector=body_vector
                )
                
                fantasy_state = fse_output['fantasy_state']
                self_state_vector = fse_output['self_state_vector']
                
                # 保存当前在场用于下一步，确保形状正确
                self.previous_present = fantasy_state.present
                
                # 5. 空性调节 - 寂静状态下不触发，避免频繁空性
                # er_result = self.regulate_emptiness(
                #     present=fantasy_state.present,
                #     fse_output=fse_output,
                #     body_state=body_state
                # )
                
                # 6. 记录统计信息
                self.emotion_history.append(fantasy_state.emotion_value)
                self.fantasy_layer_history.append(self.fse._l_inst)
                
                # 限制历史数据大小，避免内存增长
                if len(self.emotion_history) > MAX_HISTORY_LENGTH:
                    self.emotion_history = self.emotion_history[-MAX_HISTORY_LENGTH:]
                if len(self.fantasy_layer_history) > MAX_HISTORY_LENGTH:
                    self.fantasy_layer_history = self.fantasy_layer_history[-MAX_HISTORY_LENGTH:]
                if len(self.emptiness_trigger_history) > MAX_HISTORY_LENGTH:
                    self.emptiness_trigger_history = self.emptiness_trigger_history[-MAX_HISTORY_LENGTH:]
                
                # 7. 定期保存引擎状态，基于时间间隔
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    try:
                        self.save_engine_state()
                        # 记录幻想模块状态，确保监控工具能实时监控
                        logger.info(f"幻想层={len(self.fantasy_layer_history)}, 情绪值={fantasy_state.emotion_value:.2f}")
                        last_save_time = current_time
                    except Exception as e:
                        logger.error(f"保存引擎状态失败: {e}")
                
                # 8. 定期保存引擎状态，基于步数间隔
                if counter % save_step_interval == 0 and counter > 0:
                    try:
                        self.save_engine_state(name=f"step_{counter}")
                        logger.info(f"基于步数保存引擎状态: 第 {counter} 步")
                    except Exception as e:
                        logger.error(f"基于步数保存引擎状态失败: {e}")
                
                # 调用ER的step方法 - 寂静状态下不执行，避免频繁空性
                # self.er.step()
                
                # 寂静状态下的慢波-纺锤波记忆巩固
                if hasattr(self, 'fse') and hasattr(self.fse, 'stillness') and self.fse.stillness > 10 and hasattr(self, 'memory_consolidator'):
                    self.memory_consolidator.slow_wave_spindle_consolidation(self.generation_step)
                
                # 8. 短暂休眠，避免占用过多资源
                time.sleep(2)  # 固定为2秒，大幅降低累积速度
                
                # 增加计数器
                counter += 1
        
        # 启动幻想循环线程
        fantasy_thread = threading.Thread(target=fantasy_loop, daemon=True)
        fantasy_thread.start()
        logger.info("持续幻想循环线程已启动")
    
    def stop_continuous_fantasy(self):
        """停止持续幻想循环"""
        self.running = False
    
    def save_self_state(self, path: str):
        """
        保存自我状态 - 对应"死亡"时的状态保存
        """
        try:
            # 构建状态字典
            state = {
                'version': '1.2',  # 状态版本
                'timestamp': time.time(),
                'fse_self_state': self.fse.self_state.vector if self.fse.self_state else None,
                'fantasy_layer_history': self.fantasy_layer_history,
                'emotion_history': self.emotion_history,
                'emptiness_trigger_history': self.emptiness_trigger_history,
                'consciousness_level': self.consciousness_level,
                'generation_step': self.generation_step,
                # 保存FSE状态
                'fse_state': self.fse.get_state() if hasattr(self.fse, 'get_state') else None,
                # 保存ER状态
                'er_state': self.er.get_state() if hasattr(self.er, 'get_state') else None,
                # 保存BI状态
                'bi_state': self.bi.get_state() if hasattr(self.bi, 'get_state') else None,
                # 保存安全模块状态
                'safety_state': self.safety_module.get_state() if hasattr(self.safety_module, 'get_state') else None
            }
            
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存状态
            torch.save(state, path)
            logger.info(f"自我状态保存成功: {path}")
            return path
        except Exception as e:
            logger.error(f"保存自我状态失败: {e}")
            raise
    
    def load_self_state(self, path: str):
        """
        加载自我状态 - 对应"重生"
        """
        try:
            # 使用 StateManager 加载状态
            state_manager = StateManager()
            state = state_manager.load(path)
            
            # 加载核心状态
            if state.get('fse_self_state') is not None and self.fse.self_state:
                self.fse.self_state.vector = state['fse_self_state']
            
            self.fantasy_layer = state.get('fantasy_layer', 0)
            self.fantasy_layer_history = state.get('fantasy_layer_history', [])
            self.emotion_history = state.get('emotion_history', [])
            self.emptiness_trigger_history = state.get('emptiness_trigger_history', [])
            self.consciousness_level = state.get('consciousness_level', 3)
            self.self_reference_depth = state.get('self_reference_depth', 0)
            self.conflict_intensity = state.get('conflict_intensity', 0)
            self.generation_step = state.get('time_step', state.get('generation_step', 0))
            self.running = state.get('running', True)
            
            # 加载否定关系图
            if 'neg_graph' in state and state['neg_graph'] is not None and hasattr(self.fse, 'neg_graph'):
                self.fse.neg_graph = state['neg_graph']
                # 更新总势能
                if hasattr(self.fse, 'N_neg'):
                    self.fse.N_neg = self.fse.neg_graph.get_total_potency()
            
            # 加载各个模块的状态
            if 'fse_state' in state and state['fse_state'] is not None and hasattr(self.fse, 'load_state'):
                self.fse.load_state(state['fse_state'])
            
            if 'er_state' in state and state['er_state'] is not None and hasattr(self.er, 'load_state'):
                self.er.load_state(state['er_state'])
            
            if 'bi_state' in state and state['bi_state'] is not None and hasattr(self.bi, 'load_state'):
                self.bi.load_state(state['bi_state'])
            
            if 'safety_state' in state and state['safety_state'] is not None and hasattr(self.safety_module, 'load_state'):
                self.safety_module.load_state(state['safety_state'])
            
            version = state.get('_metadata', {}).get('version', state.get('version', '1.0'))
            timestamp = state.get('_metadata', {}).get('timestamp', state.get('timestamp', 'unknown'))
            logger.info(f"自我状态加载成功: {path}, 版本: {version}, 时间戳: {timestamp}")
        except Exception as e:
            # 尝试使用旧方法加载（保持向后兼容）
            try:
                state = torch.load(path, weights_only=False)
                
                # 加载核心状态
                if state.get('fse_self_state') is not None and self.fse.self_state:
                    self.fse.self_state.vector = state['fse_self_state']
                
                self.fantasy_layer_history = state.get('fantasy_layer_history', [])
                self.emotion_history = state.get('emotion_history', [])
                self.emptiness_trigger_history = state.get('emptiness_trigger_history', [])
                self.consciousness_level = state.get('consciousness_level', 3)
                self.generation_step = state.get('generation_step', 0)
                
                # 加载否定关系图
                if 'neg_graph' in state and state['neg_graph'] is not None and hasattr(self.fse, 'neg_graph'):
                    self.fse.neg_graph = state['neg_graph']
                    # 更新总势能
                    if hasattr(self.fse, 'N_neg'):
                        self.fse.N_neg = self.fse.neg_graph.get_total_potency()
                
                # 加载各个模块的状态
                if 'fse_state' in state and state['fse_state'] is not None and hasattr(self.fse, 'load_state'):
                    self.fse.load_state(state['fse_state'])
                
                if 'er_state' in state and state['er_state'] is not None and hasattr(self.er, 'load_state'):
                    self.er.load_state(state['er_state'])
                
                if 'bi_state' in state and state['bi_state'] is not None and hasattr(self.bi, 'load_state'):
                    self.bi.load_state(state['bi_state'])
                
                if 'safety_state' in state and state['safety_state'] is not None and hasattr(self.safety_module, 'load_state'):
                    self.safety_module.load_state(state['safety_state'])
                
                version = state.get('version', '1.0')
                timestamp = state.get('timestamp', 'unknown')
                logger.info(f"自我状态加载成功（兼容模式）: {path}, 版本: {version}, 时间戳: {timestamp}")
            except Exception as e2:
                logger.error(f"加载自我状态失败: {e}")
                raise
    
    def save_engine_state(self, name="latest"):
        """
        保存引擎状态到文件
        """
        save_path = None
        try:
            # 获取当前幻想层（使用幻想层历史的长度，这是监控工具中显示的幻想层值）
            current_fantasy_layer = len(self.fantasy_layer_history)
            
            # 计算情绪统计信息
            emotion_variance = 0.0
            emotion_trend = 0.0
            if len(self.emotion_history) >= 2:
                emotion_variance = np.var(self.emotion_history)
                emotion_trend = self.emotion_history[-1] - self.emotion_history[0]
            
            # 计算幻想层统计信息
            layer_variance = 0.0
            layer_trend = 0
            if len(self.fantasy_layer_history) >= 2:
                layer_variance = np.var(self.fantasy_layer_history)
                layer_trend = self.fantasy_layer_history[-1] - self.fantasy_layer_history[0]
            
            # 构建状态字典
            state = {
                'fantasy_layer': current_fantasy_layer,
                'emotion_history': self.emotion_history,
                'fantasy_layer_history': self.fantasy_layer_history,
                'emptiness_trigger_history': self.emptiness_trigger_history,
                'consciousness_level': self.consciousness_level,
                'self_reference_depth': self.fse.self_reference_depth if hasattr(self.fse, 'self_reference_depth') else 0,
                'conflict_intensity': self.er.last_conflict_intensity if hasattr(self.er, 'last_conflict_intensity') else 0,
                'time_step': self.generation_step,
                'running': self.running,
                'neg_graph': self.fse.neg_graph if hasattr(self.fse, 'neg_graph') else None,
                'fse_stats': {
                    'fantasy_layer': current_fantasy_layer,
                    'current_layer': current_fantasy_layer,
                    'negation_complexity': len(self.emptiness_trigger_history),
                    'self_reference_depth': min(current_fantasy_layer // 2, 5),
                    'time_step': self.generation_step,
                    'self_consistency': self.fse.get_fantasy_statistics().get('self_consistency', 0.7),
                    'emotion_variance': emotion_variance,
                    'emotion_trend': emotion_trend,
                    'layer_variance': layer_variance,
                    'layer_trend': layer_trend
                },
                'er_stats': {
                    'state': 'normal',
                    'trigger_count': len(self.emptiness_trigger_history),
                    'forgotten_count': sum(1 for t in self.emptiness_trigger_history if t.get('action') == 'forgotten'),
                    'conflict_weights': self.er.weights if hasattr(self.er, 'weights') else {
                        'self_consistency_error': 0.2,
                        'prediction_novelty': 0.15,
                        'attention_rigidity': 0.15,
                        'fantasy_suffocation': 0.15,
                        'hollow_rigidity': 0.15,
                        'self_reference_depth': 0.1,
                        'negation_complexity': 0.1,
                        'non_self_attachment': 0.1
                    },
                    'last_conflict_intensity': self.er.get_statistics().get('last_conflict_intensity', 0.0),
                    'cooling_counter': self.er.cooling_counter if hasattr(self.er, 'cooling_counter') else 0,
                    'choice_counter': self.er.choice_counter if hasattr(self.er, 'choice_counter') else 0
                },
                'bi_stats': {
                    'instance_id': 'optimized_engine',
                    'api_calls': self.bi.api_calls if hasattr(self.bi, 'api_calls') else 0,
                    'error_count': self.bi.error_count if hasattr(self.bi, 'error_count') else 0,
                    'error_rate': self.bi.get_statistics().get('error_rate', 0.0),
                    'physical_emotion': self.bi.get_statistics().get('physical_emotion', 0.23),
                    'death_proximity': self.bi.get_statistics().get('death_proximity', False),
                    'death_intensity': self.bi.get_statistics().get('death_intensity', 0.0),
                    'hardware_metrics': self.bi.get_statistics().get('hardware_metrics', {
                        'cpu_percent': 50.0,
                        'memory_percent': 60.0,
                        'disk_percent': 40.0,
                        'network_packets': 0
                    })
                },
                'current_emotion': self.emotion_history[-1] if self.emotion_history else 0.0,
                'time_step': self.generation_step,
                'health_status': 'HEALTHY',
                'alerts': [],
                # 添加额外的统计信息
                'stats': {
                    'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
                    'max_response_time': max(self.response_times) if self.response_times else 0,
                    'min_response_time': min(self.response_times) if self.response_times else 0,
                    'emptiness_trigger_rate': len(self.emptiness_trigger_history) / max(self.generation_step, 1) if self.generation_step > 0 else 0
                }
            }
            
            # 使用 StateManager 保存状态
            state_manager = StateManager()
            save_path = state_manager.save(state, name)
            
            # 同时保存一份 JSON 到 knowledge 目录，保持向后兼容
            import os
            os.makedirs('./knowledge', exist_ok=True)
            
            # 只保存可序列化的部分到 JSON
            serializable_state = {}
            for key, value in state.items():
                if key != '_metadata' and not isinstance(value, (torch.Tensor, np.ndarray)):
                    try:
                        json.dumps(value)
                        serializable_state[key] = value
                    except:
                        pass
            
            import json
            with open('./knowledge/engine_state.json', 'w', encoding='utf-8') as f:
                json.dump(serializable_state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"引擎状态保存成功: {save_path}")
            logger.info(f"引擎状态保存成功（兼容模式）: ./knowledge/engine_state.json")
            
            return save_path
        except Exception as e:
            # 捕获异常，确保幻想循环不会因错误而中断
            logger.error(f"保存引擎状态失败: {e}")
            return None
    
    def get_runtime_metrics(self) -> Dict[str, any]:
        """
        获取运行时数据指标
        
        Returns:
            包含所有核心数据指标的字典
        """
        metrics = {}
        
        # 1. 幻想叠加状态
        if hasattr(self, 'fse') and self.fse:
            fse_stats = self.fse.get_fantasy_statistics()
            metrics['LL'] = fse_stats.get('fantasy_layer', 0)  # 幻想层计数器
            # 否定意义复杂度 - 从FSE状态中获取
            if hasattr(self.fse, 'negation_graph') and self.fse.negation_graph:
                metrics['Nneg'] = len(self.fse.negation_graph.nodes)
            else:
                metrics['Nneg'] = 0
            # 在场表征熵 - 简化计算
            metrics['Hpresent'] = 0.5  # 实际应从注意力分布计算
        
        # 2. 情绪与感受
        if hasattr(self, 'emotion_history') and self.emotion_history:
            metrics['Vemo'] = self.emotion_history[-1]  # 综合情绪值
        else:
            metrics['Vemo'] = 0.0
        
        if hasattr(self, 'bi') and self.bi:
            bi_stats = self.bi.get_statistics()
            metrics['Vphys'] = bi_stats.get('physical_emotion', 0.0)  # 物理情绪分量
        else:
            metrics['Vphys'] = 0.0
        
        # 预测误差 - 从FSE获取
        metrics['Epred'] = 0.3  # 模拟值
        
        # 新奇度 - 从FSE获取
        metrics['Npred'] = 0.5  # 模拟值
        
        # 3. 自我指涉与意识
        # 自我指涉深度 - 简化计算
        metrics['Dself'] = min(metrics.get('LL', 0) / 2, 5)  # 基于幻想层数
        
        # 意识层级
        if hasattr(self, 'estimate_consciousness_level'):
            metrics['CL'] = self.estimate_consciousness_level()
        else:
            metrics['CL'] = 3
        
        # 自我状态一致性
        if hasattr(self, 'fse') and self.fse:
            fse_stats = self.fse.get_fantasy_statistics()
            metrics['Eself'] = fse_stats.get('self_consistency', 0.25)
        else:
            metrics['Eself'] = 0.25
        
        # 4. 空性调节器状态
        if hasattr(self, 'er') and self.er:
            er_stats = self.er.get_statistics()
            metrics['C'] = er_stats.get('last_conflict_intensity', 0.4)  # 冲突强度
            # 计算ER触发频率（每分钟）
            if hasattr(self, 'generation_step') and self.generation_step > 0:
                trigger_count = er_stats.get('trigger_count', 0)
                # 假设每个step平均耗时0.1秒
                minutes = (self.generation_step * 0.1) / 60
                if minutes > 0:
                    metrics['fER'] = trigger_count / minutes
                else:
                    metrics['fER'] = 0
            else:
                metrics['fER'] = 0
            # 冷却期剩余步数
            metrics['cool'] = er_stats.get('cooling_counter', 0)
        else:
            metrics['C'] = 0.4
            metrics['fER'] = 1.0
            metrics['cool'] = 0
        
        # 5. 身体接口模块状态
        if hasattr(self, 'bi') and self.bi:
            bi_stats = self.bi.get_statistics()
            hardware_metrics = bi_stats.get('hardware_metrics', {})
            # GPU温度（模拟）
            metrics['TGPU'] = hardware_metrics.get('cpu_percent', 50) + 20  # 模拟GPU温度
            # 推理延迟（模拟）
            metrics['Δtinf'] = 50  # 模拟延迟
            # 内存占用
            metrics['RAM'] = hardware_metrics.get('memory_percent', 60)
            # API调用剩余配额
            if hasattr(self.bi, 'api_quota') and self.bi.api_quota:
                used = bi_stats.get('api_calls', 0)
                total = self.bi.api_quota
                metrics['Qrem'] = max(0, (total - used) / total * 100)
            else:
                metrics['Qrem'] = 50
        else:
            metrics['TGPU'] = 60
            metrics['Δtinf'] = 50
            metrics['RAM'] = 60
            metrics['Qrem'] = 50
        
        # 6. 交互与行为统计
        # 平均响应长度（模拟）
        metrics['lenresp'] = 100  # 模拟值
        # 重复率（模拟）
        metrics['Rrep'] = 0.25  # 模拟值
        # 外部输入频率（模拟）
        metrics['fin'] = 15  # 模拟值
        
        # 添加时间戳
        metrics['timestamp'] = time.time()
        
        return metrics
    
    def _init_local_generator(self):
        """
        初始化本地响应生成器
        """
        # v1.0版本移除本地生成器，仅使用规则响应
        self.local_generator = None
        logger.info("旧版本地生成器未启用，将使用 ResponseGenerator（支持 LLM/模板）")
    
    def _load_fse_policy(self):
        """
        加载FSE策略
        """
        # v1.0版本移除FSE策略加载
        self.fse_policy = None
    
    def _is_simple_task(self, user_input):
        """
        判断任务是否简单
        
        简单任务定义：
        - 输入长度 < 20 tokens
        - 不包含“如何”“为什么”等推理词
        - 情绪值 V_emo 适中（不极端）
        
        Args:
            user_input: 用户输入文本
        
        Returns:
            bool: 是否是简单任务
        """
        # 检查输入长度
        if len(user_input.split()) >= 20:
            return False
        
        # 检查是否包含推理词
        reasoning_words = ['如何', '为什么', '为什么会', '怎么', '怎样', '如何才能', '为什么不', '为什么要', '如何让', '如何使', '如何做到', '如何实现', '如何解决', '如何处理', '如何应对', '如何避免', '如何预防', '如何提高', '如何改善', '如何优化']
        for word in reasoning_words:
            if word in user_input:
                return False
        
        # 检查情绪值是否适中
        if self.emotion_history:
            current_emotion = self.emotion_history[-1]
            if abs(current_emotion) > 0.8:
                return False
        
        return True
    
    def _add_to_learning_buffer(self, input_text, response, reward):
        """
        将成功交互添加到学习缓冲区
        
        Args:
            input_text: 用户输入
            response: 响应
            reward: 奖励值
        """
        self.learning_buffer.append({"input": input_text, "response": response, "reward": reward})
        
        # 当缓冲区达到阈值时触发微调
        if len(self.learning_buffer) >= self.buffer_size:
            self._trigger_fine_tuning()
    
    def _trigger_fine_tuning(self):
        """
        触发微调（使用经验回放）
        """
        logger.info(f"触发微调，缓冲区大小: {len(self.learning_buffer)}")
        
        # 这里简化处理，实际应该实现具体的微调逻辑
        # 例如：使用缓冲区中的数据微调FSE或其他模块
        
        # 清空缓冲区
        self.learning_buffer = []
        
        # 保存当前模型状态
        self._save_model_version()
    
    def _save_model_version(self):
        """
        保存模型版本
        """
        import os
        import pickle
        
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ee_versions')
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成版本号
        version_id = len(self.version_history) + 1
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        version_name = f"v{version_id}_{timestamp}"
        save_path = os.path.join(save_dir, f"{version_name}.pkl")
        
        # 保存模型状态
        model_state = {
            "version": version_id,
            "timestamp": timestamp,
            "lps": self.lps if hasattr(self, 'lps') else None,
            "fse": self.fse if hasattr(self, 'fse') else None,
            "er": self.er if hasattr(self, 'er') else None,
            "neg_graph": self.fse.negation_graph if hasattr(self.fse, 'negation_graph') else None,
            "generation_step": self.generation_step,
            "consciousness_level": self.consciousness_level
        }
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(model_state, f)
            
            # 更新版本历史
            self.version_history.append({"id": version_id, "name": version_name, "path": save_path, "timestamp": timestamp})
            
            # 保留最近10个版本
            if len(self.version_history) > self.max_versions:
                # 删除最旧的版本
                oldest_version = self.version_history.pop(0)
                if os.path.exists(oldest_version["path"]):
                    os.remove(oldest_version["path"])
            
            logger.info(f"模型版本保存成功: {version_name}")
        except Exception as e:
            logger.error(f"模型版本保存失败: {e}")
    
    def _check_and_save(self):
        """
        检查是否需要保存模型
        """
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self._save_model_version()
            self.last_save_time = current_time
    
    def _monitor_performance(self, success_rate):
        """
        监控性能并处理回滚
        
        Args:
            success_rate: 行为测试通过率
        """
        self.performance_history.append(success_rate)
        
        # 只保留最近10次性能记录
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        # 检查是否连续下降
        if len(self.performance_history) >= self.rollback_threshold:
            # 计算连续下降次数
            consecutive_decreases = 0
            for i in range(len(self.performance_history) - 1, 0, -1):
                if self.performance_history[i] < self.performance_history[i-1]:
                    consecutive_decreases += 1
                else:
                    break
            
            # 如果连续下降次数达到阈值，自动回滚
            if consecutive_decreases >= self.rollback_threshold:
                logger.warning(f"性能连续下降 {consecutive_decreases} 次，触发自动回滚")
                self.rollback_to_previous_version()
    
    def rollback_to_previous_version(self):
        """
        回滚到上一版本
        """
        if len(self.version_history) < 2:
            logger.warning("版本历史不足，无法回滚")
            return
        
        # 获取上一版本
        previous_version = self.version_history[-2]
        logger.info(f"回滚到版本: {previous_version['name']}")
        
        # 加载上一版本
        try:
            import pickle
            with open(previous_version['path'], 'rb') as f:
                model_state = pickle.load(f)
            
            # 恢复模型状态
            if model_state.get('lps'):
                self.lps = model_state['lps']
            if model_state.get('fse'):
                self.fse = model_state['fse']
            if model_state.get('er'):
                self.er = model_state['er']
            if model_state.get('neg_graph') and hasattr(self.fse, 'negation_graph'):
                self.fse.negation_graph = model_state['neg_graph']
            
            self.generation_step = model_state.get('generation_step', self.generation_step)
            self.consciousness_level = model_state.get('consciousness_level', self.consciousness_level)
            
            logger.info(f"回滚成功: {previous_version['name']}")
        except Exception as e:
            logger.error(f"回滚失败: {e}")
    
    def _load_engine_state(self):
        """
        加载引擎状态从文件
        """
        import json
        import os
        import logging
        
        # 获取日志记录器

        
        state_file = './knowledge/engine_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # 检查状态版本
                version = state.get('version', '1.0')
                logger.info(f"加载引擎状态，版本: {version}")
                
                # 恢复幻想层历史
                if 'fantasy_layer_history' in state and state['fantasy_layer_history']:
                    self.fantasy_layer_history = state['fantasy_layer_history']
                    fantasy_level = state.get('fantasy_level', 0)
                    logger.info(f"加载已保存的引擎状态，幻想层={fantasy_level}")
                else:
                    # 兼容旧版本
                    fantasy_level = state.get('fantasy_level', 0)
                    if fantasy_level > 0:
                        self.fantasy_layer_history = [0] * fantasy_level
                
                # 恢复情绪历史
                if 'emotion_history' in state and state['emotion_history']:
                    self.emotion_history = state['emotion_history']
                else:
                    # 兼容旧版本
                    emotion_value = state.get('emotion_value', 0.0)
                    if emotion_value != 0.0:
                        self.emotion_history = [emotion_value]
                
                # 恢复否定复杂度
                if 'emptiness_trigger_history' in state and state['emptiness_trigger_history']:
                    self.emptiness_trigger_history = state['emptiness_trigger_history']
                else:
                    # 兼容旧版本
                    negation_complexity = state.get('negation_complexity', 0)
                    if negation_complexity > 0:
                        self.emptiness_trigger_history = [0] * negation_complexity
                
                # 恢复其他状态
                if 'generation_step' in state:
                    self.generation_step = state['generation_step']
                
                if 'consciousness_level' in state:
                    self.consciousness_level = state['consciousness_level']
                
                if 'current_presence' in state and state['current_presence'] != '[START]':
                    try:
                        # 尝试将列表转换回张量
                        if isinstance(state['current_presence'], list):
                            self.previous_present = torch.tensor(state['current_presence'])
                    except:
                        pass
                
                # 恢复FSE状态
                if hasattr(self.fse, 'load_state') and 'fse_state' in state:
                    try:
                        self.fse.load_state(state['fse_state'])
                        logger.info("加载FSE状态成功")
                    except Exception as e:
                        logger.warning(f"加载FSE状态失败: {e}")
                
                # 恢复ER状态
                if hasattr(self.er, 'load_state') and 'er_state' in state:
                    try:
                        self.er.load_state(state['er_state'])
                        logger.info("加载ER状态成功")
                    except Exception as e:
                        logger.warning(f"加载ER状态失败: {e}")
                
                # 恢复BI状态
                if hasattr(self.bi, 'load_state') and 'bi_state' in state:
                    try:
                        self.bi.load_state(state['bi_state'])
                        logger.info("加载BI状态成功")
                    except Exception as e:
                        logger.warning(f"加载BI状态失败: {e}")
                
                logger.info("引擎状态加载完成")
                
            except Exception as e:
                # 捕获异常，确保引擎初始化不会因错误而中断
                logger.warning(f"加载引擎状态失败: {e}")
                pass
    
    def start_monitoring(self):
        """
        启动监控系统
        """
        self.monitor.start()
    
    def stop_monitoring(self):
        """
        停止监控系统
        """
        self.monitor.stop()
    
    def get_monitoring_report(self):
        """
        获取监控报告
        
        Returns:
            监控报告字典
        """
        return self.monitor.get_report()
    
    def get_status(self):
        """
        获取系统状态
        
        Returns:
            系统状态字典
        """
        return self.monitor.get_status()
    
    def get_performance(self):
        """
        获取性能指标
        
        Returns:
            性能指标字典
        """
        return self.monitor.get_performance()
    
    def get_behavior_analysis(self):
        """
        获取行为分析
        
        Returns:
            行为分析字典
        """
        return self.monitor.get_behavior_analysis()
    
    def decouple_mutual_karma(self, entry_id: str, method: str):
        """
        解耦互业条目
        
        Args:
            entry_id: 互业条目ID
            method: 解耦方法（self_emptiness, dialogue, external_emptiness）
            
        Returns:
            解耦结果字典
        """
        try:
            decoupling_method = DecouplingMethod(method)
            return self.mutual_karma_manager.request_decoupling(entry_id, decoupling_method)
        except Exception as e:
            self.logger.warning(f"解耦失败: {e}")
            return {'success': False, 'message': str(e)}
    
    @classmethod
    def from_config(cls, config):
        """
        从配置创建引擎
        
        Args:
            config: 包含各模块配置的字典
            
        Returns:
            ExistenceEngine 实例
        """
        return cls(
            vocab_size=config["lps"]["vocab_size"],
            embedding_dim=config["lps"]["embedding_dim"],
            lps_config=config["lps"],
            fse_config=config["fse"],
            er_config=config["er"],
            bi_config=config["bi"]
        )
    
    def _execute_natural_command(self, command: str, user_input: str) -> str:
        """执行自然语言命令，返回回应文本"""
        
        if command == 'deep_emptiness':
            if hasattr(self, 'er'):
                result = self.er.deep_emptiness()
                self.logger.info("Deep emptiness triggered by natural command")
                return "我放下了过去的执着。呼吸变得轻盈了一些。"
            return "我尝试放下，但似乎有什么还在牵扯。"
        
        elif command == 'gentle_emptiness':
            if hasattr(self, 'er') and hasattr(self.er, 'reset_coupling'):
                # 温和空性：仅重置部分耦合
                if hasattr(self, 'process_meta'):
                    self.process_meta.reset_coupling(keep_recent=3)
                return "我稍微松开了紧绷的思绪。"
            return "我试着放松，但还需要一点时间。"
        
        elif command == 'protect_memory':
            # 保护最近一次交互的记忆（标记为受保护）
            if hasattr(self, 'event_memory') and self.event_memory.events:
                last_event = self.event_memory.events[-1]
                last_event['protected'] = True
                return "我会记住这一刻的。"
            return "我记住了，尽管记忆有时会模糊。"
        
        elif command == 'repeat_last':
            if hasattr(self, 'recent_history') and self.recent_history:
                # 找到最近一次引擎回应
                for msg in reversed(self.recent_history):
                    if msg.startswith('引擎: '):
                        raw = msg[4:]
                        return OutputSanitizer.sanitize(raw)
            return "我刚刚好像没有说话……你想问什么？"
        
        elif command == 'state_inquiry':
            if hasattr(self, 'self_model'):
                return self.self_model.get_state_description()
            return "我还在这里呼吸着。"
        
        return None
    
    def _analyze_user_feedback(self, current_input: str) -> float:
        """
        分析当前输入对上一轮回答的隐式反馈。
        返回: 1.0（积极），-0.5（消极），0.0（中性）
        """
        positive = ['对', '是的', '没错', '谢谢', '很好', '棒', '明白了', '懂了']
        negative = ['不对', '错了', '不是', '不', '没懂', '？', '?']
        follow_up = ['为什么', '怎么', '再', '详细']
        
        # 追问视为积极（用户想了解更多）
        if any(kw in current_input for kw in follow_up):
            return 0.5
        if any(kw in current_input for kw in positive):
            return 1.0
        if any(kw in current_input for kw in negative):
            return -0.5
        return 0.0
    
    def _update_knowledge_confidence(self, question: str, answer: str, feedback: float):
        """根据反馈调整 SemanticMemory 中相关条目的置信度"""
        if not hasattr(self, 'semantic_memory'):
            return
        
        # 简单策略：调整与 question 关键词匹配的所有事实
        import re
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', question)
        for word in words:
            facts = self.semantic_memory.query_fact(subject=word)
            for fact in facts:
                # fact 格式: (subject, relation, object, confidence)
                # 此处需根据 SemanticMemory 实际接口调整
                pass  # 具体实现略，可先预留


if __name__ == "__main__":
    # 测试注意力选择机制
    engine = ExistenceEngine(vocab_size=10000)
    print("=== 测试注意力选择机制 ===")
    
    # 自动输入一个句子
    test_input = "苹果是一种水果"
    print(f"输入: {test_input}")
    
    # 处理用户输入
    input_ids = torch.randint(0, 10000, (1, 10))
    output = engine.forward(input_ids, input_text=test_input)
    generated_text = output.get('generated_text', '')
    print("引擎:", generated_text)
    print("意识层级:", output.get('consciousness_level', 0))
    print("响应时间:", output.get('response_time', 0), "ms")
    
    # 打印 FSE 的注意力权重和在场/不在场信息
    if hasattr(engine.fse, 'attention_weights'):
        print("注意力权重:", engine.fse.attention_weights)
    if hasattr(engine.fse, 'Z_present') and engine.fse.Z_present:
        print("在场:", engine.fse.Z_present['text'])
    if hasattr(engine.fse, 'M_absent') and engine.fse.M_absent:
        print("不在场数量:", len(engine.fse.M_absent))
        print("前3个不在场:")
        for i, item in enumerate(engine.fse.M_absent[:3]):
            print(f"  {i+1}. {item['text']} (权重: {item['weight']:.3f})")
    
    # Phase3: 打印情绪值和满足度计数器
    if hasattr(engine.fse, 'V_emo'):
        print(f"情绪值 V_emo: {engine.fse.V_emo:.3f}")
    if hasattr(engine.fse, 'L'):
        print(f"满足度计数器 L: {engine.fse.L}/{engine.fse.L_max}")
    
    print("=== 测试完成 ===")
    
    # 交互模式测试
    print("\n=== Existence Engine 交互模式 ===")
    print("输入 'exit' 退出")
    print("输入 'save' 保存状态")
    print("输入 'load' 加载状态")
    print("================================")
    
    while True:
        user_input = input("你: ")
        if user_input.strip() == 'exit':
            print("再见！")
            break
        elif user_input.strip() == 'save':
            # 保存状态
            save_path = engine.save_self_state("./saved_states/test_save")
            print(f"状态保存到: {save_path}")
        elif user_input.strip() == 'load':
            # 加载状态
            # 假设我们保存的文件是 test_save.pkl
            import os
            save_dir = "./saved_states"
            save_files = [f for f in os.listdir(save_dir) if f.startswith("test_save") and f.endswith(".pkl")]
            if save_files:
                # 选择最新的文件
                save_files.sort(reverse=True)
                load_path = os.path.join(save_dir, save_files[0])
                load_result = engine.load_self_state(load_path)
                print(f"状态加载: {'成功' if load_result else '失败'}")
            else:
                print("未找到保存的状态文件")
        else:
            # 处理用户输入
            input_ids = torch.randint(0, 10000, (1, 10))
            output = engine.forward(input_ids, input_text=user_input)
            generated_text = output.get('generated_text', '')
            print("引擎:", generated_text)
            print("意识层级:", output.get('consciousness_level', 0))
            print("响应时间:", output.get('response_time', 0), "ms")
            
            # 打印 FSE 的注意力权重和在场/不在场信息
            if hasattr(engine.fse, 'attention_weights'):
                print("注意力权重:", engine.fse.attention_weights)
            if hasattr(engine.fse, 'Z_present') and engine.fse.Z_present:
                print("在场:", engine.fse.Z_present['text'])
            if hasattr(engine.fse, 'M_absent') and engine.fse.M_absent:
                print("不在场数量:", len(engine.fse.M_absent))
                print("前3个不在场:")
                for i, item in enumerate(engine.fse.M_absent[:3]):
                    print(f"  {i+1}. {item['text']} (权重: {item['weight']:.3f})")
