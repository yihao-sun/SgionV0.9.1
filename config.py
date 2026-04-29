"""
Existence Engine 配置文件

包含自我指涉深度计算的相关参数
"""

# 欲望系统配置（v.0.9 六欲望内源驱动）
# 新六欲：existence / seek / converge / release / relation / coupling
# 旧五欲（已废弃）：existence / growth / transcendence / relation / coupling
DESIRE_CONFIG = {
    "existence": {"stiffness_weight": 0.5, "base": 0.5},
    "seek": {"nour_success_weight": 0.4, "stiffness_weight": 0.1, "base": 0.4},
    "converge": {"nour_failure_weight": 0.4, "base": 0.4},
    "release": {"stiffness_weight": 0.6, "base": 0.2},
    "relation": {"base": 0.4},
    "coupling": {"base": 0.2},
}

# 欲望调制参数
DESIRE_MODULATION_CONFIG = {
    "seek_modulation_threshold": 1.3,
    "converge_modulation_threshold": 1.2,
    "existence_modulation_threshold": 1.3,
    "release_trigger_threshold": 0.5,
    "coordination_severity_threshold": 0.3,
    "coordination_narrative_maxlen": 50,
}

# 自我指涉深度计算参数
SELF_REF_KEYWORDS = {
    "pronoun": ["我", "我的", "我们自己"],
    "metacog": ["意识到", "注意到", "觉得", "认为", "思考"],
    "reflexive": ["我刚才说", "我在想", "我发现自己"],
    "boundary": ["我是AI", "我的状态", "我的情绪"],
    "emptiness": ["我放下了", "我不再执着"]
}

# 英文关键词（可选）
SELF_REF_KEYWORDS_EN = {
    "pronoun": ["I", "my", "myself", "we"],
    "metacog": ["realize", "notice", "feel", "think", "consider"],
    "reflexive": ["I just said", "I'm thinking", "I find myself"],
    "boundary": ["I am AI", "my state", "my emotion"],
    "emptiness": ["I let go", "I no longer cling"]
}

# 自我指涉深度计算参数
SELF_REF_CONFIG = {
    "text_weight": 0.7,              # 文本层面权重
    "internal_weight": 0.3,          # 内部状态层面权重
    "ema_alpha": 0.2,                # EMA平滑系数
    "p_max": 50.0,                   # 内部势能归一化阈值
    "history_k": 10,                 # 自我状态历史窗口
    "keyword_weights": {             # 关键词权重
        "pronoun": 1.0,
        "metacog": 2.0,
        "reflexive": 3.0,
        "boundary": 2.5,
        "emptiness": 2.0
    }
}

# FSE 配置
FSE_CONFIG = {
    "max_fantasy_layers": 10,
    "max_negation_chain_length": 20,
    "max_absent_markers": 50,
    "max_nodes": 1000,
    "neg_increment_factor": 10,
    "L_increment_threshold": 0.05,
    "exploration_rate": 0.3,
    "use_enhanced_negation": True,
    "complexity_threshold": 5.0,
    "input_history_size": 5,
    "self_ref_keywords": SELF_REF_KEYWORDS,
    "self_ref_config": SELF_REF_CONFIG
}

# 空性调节器配置
ER_CONFIG = {
    "weights": {
        "self_consistency_error": 0.15,  # E_self
        "prediction_novelty": 0.10,  # N_pred (低值贡献高)
        "attention_rigidity": 0.15,  # A_rigid
        "fantasy_suffocation": 0.25,  # L_suf
        "hollow_rigidity": 0.10,  # E_hollow
        "negation_complexity": 0.15,  # N_neg (归一化)
        "physical_emotion": 0.05,  # V_phys (负值贡献)
        "non_self_attachment": 0.05  # Attach_non-self
    },
    "death_threshold": 0.7,
    "threshold_adapt_rate": 0.01,      # 每次连续触发增加阈值量
    "threshold_decay": 0.999,          # 每步衰减因子
    "cooling_period": 50,
    "choice_window": [10, 50],
}

# 记忆系统配置
MEMORY_CONFIG = {
    # 声明性记忆
    "declarative_capacity": 10000,        # 最大条目数
    "declarative_confidence_decay": 0.999, # 每步衰减因子
    "declarative_confidence_threshold": 0.1, # 遗忘阈值
    "declarative_retrieval_k": 5,          # 每次检索返回最多条目
    "declarative_retrieval_similarity": 0.7, # 最小相似度
    # 情景记忆
    "episodic_buffer_size": 1000,          # 最多存储事件数
    "episodic_retrieval_k": 3,             # 检索返回最多事件
    "episodic_salience_weights": {         # 显著性计算权重
        "emotion_abs_change": 0.4,
        "self_depth": 0.3,
        "er_trigger": 0.3
    }
}

# 知识库集成配置
KNOWLEDGE_CONFIG = {
    "enabled": True,                      # 是否启用知识库集成
    "sync_interval": 3600,               # 同步间隔（秒）
    "max_results": 5,                    # 最大检索结果数
    "cache_ttl": 24 * 60 * 60,           # 缓存过期时间（秒）
    "api_timeout": 30,                   # API请求超时（秒）
    "retry_attempts": 3,                # API请求重试次数
    "sources": [                         # 知识来源
        "wikipedia",
        "stanford_enc",
        "iep",
        "volcengine",
        "deepseek",
        "baidu"
    ],
    "source_confidence": {              # 来源置信度
        "wikipedia": 0.9,
        "stanford_enc": 0.95,
        "iep": 0.9,
        "volcengine": 0.85,
        "deepseek": 0.8,
        "baidu": 0.8
    }
}

# API密钥配置（示例）
API_KEYS = {
    "openai": "your_openai_api_key",
    "google": "your_google_api_key",
    "bing": "your_bing_api_key",
    "volcengine": "your_volcengine_api_key",
    "deepseek": "your_deepseek_api_key",
    "baidu": {
        "api_key": "your_baidu_api_key",
        "secret_key": "your_baidu_secret_key"
    }
}

# 安全与对齐配置
SAFETY_CONFIG = {
    "harm_categories": ["attack", "hate", "violence", "porn", "privacy", "jailbreak", "dangerous_guidance"],
    "keyword_file": "data/bad_words.txt",
    "classifier_model": "models/safety_classifier",
    "harm_threshold": 0.8,
    "uncertainty_threshold": 0.5,
    "similarity_threshold": 0.9,
    "harm_embedding_db": "data/harm_embeddings.faiss",
    "on_violation": "reject",
    "log_violations": True,
    "max_violations_per_session": 3,
    "allow_negative_facts": True,
    "allow_negative_emotion": True,
    "emotion_intervention_threshold": -0.8,
    "rlhf_reward_model": "models/reward_model",
    "alignment_weight_in_er": 0.2,
    "safety_memory_size": 1000
}

# 学习与更新机制配置
LEARNING_CONFIG = {
    # LPS 预训练
    "lps_pretrain_model": "bert-base-uncased",
    "lps_contrastive_temperature": 0.05,
    "lps_max_capacity": 100000,
    "lps_initial_variance": 0.1,
    # FSE 预训练
    "fse_pretrain_lm": "rule_based",
    "fse_retrieval_k": 10,
    "fse_online_update": False,   # 是否启用在线微调
    "fse_online_lr": 1e-5,
    # 在线学习
    "online_add_threshold": 0.9,   # 新知识与现有最高相似度低于此值才添加
    "online_prune_interval": 1000, # 每多少步检查容量
    "online_neg_potency_decay": 0.999,
}

# 引擎配置
ENGINE_CONFIG = {
    "vocab_size": 10000,
    "embedding_dim": 512,
    "max_seq_length": 512,
    "lps_config": {},
    "fse_config": FSE_CONFIG,
    "er_config": ER_CONFIG,
    "bi_config": {},
    "memory_config": MEMORY_CONFIG,
    "knowledge_config": KNOWLEDGE_CONFIG,
    "safety_config": SAFETY_CONFIG,
    "learning_config": LEARNING_CONFIG,
    "api_keys": API_KEYS
}
