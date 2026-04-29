# 响应生成模块

本模块实现 Existence Engine 的响应生成功能，使用规则基于的响应生成器。

## 目录结构

```
response_generator/
├── __init__.py        # 包初始化文件
├── base.py            # 抽象基类
└── README.md          # 本说明文件
```

## 集成到 Existence Engine

在 engine.py 中集成响应生成模块：

```python
from core.response_generator import ResponseGenerator

class ExistenceEngine:
    def __init__(self, config):
        # ... 初始化 FSE, ER, LPS 等
        self.response_generator = ResponseGenerator()

    def respond(self, user_input: str) -> str:
        # 生成响应
        response = self.response_generator.generate(user_input, self.fse, self.bi)
        return response
```

## 注意事项

- **规则基于的响应**：使用预定义的规则和模板生成响应，无需外部依赖
- **情绪驱动**：响应会根据引擎的情绪状态进行调整
- **无需 API 密钥**：不依赖任何外部 API 服务

## 性能优化

- 规则基于的响应生成速度快，无网络延迟
- 内存占用低，适合在资源受限的环境中运行
