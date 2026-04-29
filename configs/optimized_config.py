"""
Existence Engine 优化配置
"""

# 优化后的模型配置
optimized_config = {
    "lps": {
        "vocab_size": 30522,
        "embedding_dim": 2048,  # 增加到2048
        "num_possibilities": 3000,  # 增加到3000
        "num_heads": 16,  # 增加注意力头数到16
        "num_layers": 10,  # 增加Transformer层数到10
        "dropout": 0.1
    },
    "fse": {
        "vocab_size": 30522,
        "embedding_dim": 2048,  # 与LPS保持一致
        "max_fantasy_layers": 20,  # 增加最大幻想层数到20
        "attention_heads": 16,  # 增加注意力头数到16
        "num_layers": 10,  # 增加Transformer层数到10
        "dropout": 0.1
    },
    "er": {
        "death_threshold": 0.7,
        "cooling_period": 20,
        "choice_window": (10, 50),
        "learning_rate": 0.01
    },
    "bi": {
        "instance_id": "optimized_engine",
        "api_quota": 1000,
        "update_interval": 1.0
    }
}

# 优化后的训练参数
training_config = {
    "lps": {
        "batch_size": 32,
        "epochs": 15,
        "lr": 1e-4,
        "weight_decay": 0.01
    },
    "er": {
        "batch_size": 16,
        "epochs": 8,
        "lr": 1e-5,
        "weight_decay": 0.01
    }
}
