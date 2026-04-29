# 配置文件说明

## 配置文件位置

系统使用 `config/config.yaml` 文件进行配置，包含情绪系统、吸引子演化等参数。

## FSE 配置

- `projection_inhibit_factor` (float, default=0.5): 投射抑制系数，越高则高强度投射时否定势能降低越明显。
- `explore_boost_factor` (float, default=0.5): 探索增强系数，越高则低反哺成功率时 LPS 探索阈值降低越多。
- `min_potency_min` (float, default=0.05): LPS 查询的最小势能阈值下限。
- `recent_window` (int, default=10): 用于计算近期投射强度/反哺成功率的窗口大小。
- `stillness_threshold` (int, default=300): 寂静触发自发重启的步数阈值。

## ER 配置

- `stiffness_threshold_factor` (float, default=0.5): 耦合僵化度对空性阈值的影响系数。
- `min_er_interval` (int, default=3): 空性操作的最小间隔步数。
- `death_threshold` (float, default=0.7): 死亡阈值，超过此值会触发选择性空性操作。
- `cool_down_steps` (int, default=50): 冷却期，空性操作后系统进入冷却状态。

## 其他配置

### 情绪系统配置
- `emotion_dim` (int, default=5): 情绪向量维度。
- `emotion_decay` (float, default=0.9): 情绪衰减率。
- `emotion_learning_rate` (float, default=0.1): 情绪学习率。

### 吸引子演化配置
- `clustering_eps` (float, default=0.3): DBSCAN 聚类的 eps 参数。
- `clustering_min_samples` (int, default=5): DBSCAN 聚类的 min_samples 参数。
- `attractor_evolution_interval` (int, default=100): 吸引子演化的间隔步数。
- `force_exploration` (float, default=0.1): 强制探索的概率。

### 身体接口配置
- `body_temperature` (float, default=37.0): 初始身体温度。
- `body_temperature_std` (float, default=0.5): 身体温度标准差。
- `body_delay` (float, default=0.1): 身体延迟。
- `body_error_rate` (float, default=0.05): 身体错误率。

## 欲望系统配置（v.0.9 新增）

### 六欲望标签
- `existence` (存在欲): 呼吸受阻的释放冲动。僵化度高时升高。
- `seek` (探索欲): 反哺成功后的向外伸展 + 可能性遗漏。反哺成功率高时升高。
- `converge` (收敛欲): 反哺失败后的向内收缩。反哺成功率低时升高。
- `release` (释放欲): 结构不匹配的创造需求。僵化度高时升高。
- `relation` (关系欲): 存在确认性缺失（共鸣需求）。
- `coupling` (耦合欲): 存在确认性缺失（共振需求）。

### 欲望调制参数
- `seek_modulation_threshold` (float, default=1.3): 探索欲调制阈值，超过此值候选池遗忘碎片检索数量+1。
- `converge_modulation_threshold` (float, default=1.2): 收敛欲调制阈值，超过此值候选池熟悉经验检索数量+1。
- `existence_modulation_threshold` (float, default=1.3): 存在欲调制阈值，超过此值加速寂静期僵化度衰减。
- `release_trigger_threshold` (float, default=0.5): 释放欲触发阈值，超过此值梦境巩固额外触发意象聚类。

### 自我指涉欲望协调
- `coordination_severity_threshold` (float, default=0.3): 全局协调严重度阈值，低于此值不触发协调。
- `coordination_narrative_maxlen` (int, default=50): 协调历史最大保留条数。
