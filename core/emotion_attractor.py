"""
情绪吸引子系统 (Emotion Attractor)
哲学对应：《论存在》第7章，意识连续谱的情绪维度。
功能：五维情绪向量识别，预定义原型吸引子，自动发现新吸引子（如 calm_clear），支持吸引子分裂与归档。
主要类：EmotionAttractor, DynamicAttractor
"""
import numpy as np
from utils.logger import get_logger

# 预定义吸引子（五维向量：approach_avoid, arousal, valence, social_connection, self_clarity）
ATTRACTORS = {
    "fear": {
        "center": np.array([-0.9, 0.9, -0.8, -0.6, 0.1]),
        "radius": 0.6,  # 从 0.7 调整
        "action_tendency": "defense, request_resources"
    },
    "joy": {
        "center": np.array([0.7, 0.55, 0.9, 0.8, 0.8]),  # arousal: 0.3→0.55
        "radius": 0.55,  # 从 0.7 调整
        "action_tendency": "maintain, expand"
    },
    "curiosity": {
        "center": np.array([0.3, 0.6, 0.3, 0.0, 0.9]),
        "radius": 0.55,  # 从 0.7 调整
        "action_tendency": "explore, query"
    },
    "sadness": {
        "center": np.array([-0.6, 0.05, -0.9, -0.9, 0.05]),
        "radius": 0.7,  # 保持
        "action_tendency": "withdraw, seek_reconnection"
    },
    "anger": {
        "center": np.array([0.0, 0.8, -0.6, -0.6, 0.3]),  # approach_avoid: -0.5→0.0
        "radius": 0.6,  # 从 0.7 调整
        "action_tendency": "defend_boundary, reject"
    },
    "neutral": {
        "center": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # social: 0.5→0.0
        "radius": 0.5,
        "action_tendency": "neutral, observe"
    }
}

class EmotionAttractor:
    def __init__(self, attractors=ATTRACTORS):
        # 动态吸引子存储: dict[int, DynamicAttractor]
        self.attractors = {}
        # 名称到ID的映射（兼容旧的情绪名称）
        self.name_to_id = {}
        # ID计数器
        self.next_attractor_id = 0
        # 其他属性
        self.current_emotion = None
        self.current_distance = float('inf')
        self.logger = get_logger('emotion_attractor')
        
        # 初始化预定义的原型吸引子
        self._initialize_prototype_attractors(attractors)
    
    def _initialize_prototype_attractors(self, attractors_dict):
        """初始化原型吸引子"""
        for name, attr in attractors_dict.items():
            attractor_id = self.next_attractor_id
            self.next_attractor_id += 1
            
            dynamic_attractor = DynamicAttractor(
                attractor_id=attractor_id,
                center=attr["center"],
                radius=attr["radius"],
                action_tendency=attr["action_tendency"],
                birth_step=0,
                is_prototype=True
            )
            
            self.attractors[attractor_id] = dynamic_attractor
            self.name_to_id[name] = attractor_id
    
    def identify(self, E_vec, weights=None, step=0):
        """
        识别最接近的情绪吸引子
        E_vec: 5维numpy数组
        weights: 5维权重（可选，用于加权距离）
        step: 当前步数
        return: (emotion_name, distance, distances_dict)
        """
        if weights is None:
            weights = np.ones(5)
        min_dist = float('inf')
        best_attractor_id = None
        dist_dict = {}
        
        # 遍历所有动态吸引子
        for attractor_id, attractor in self.attractors.items():
            diff = (E_vec - attractor.center) * weights
            dist = np.sqrt(np.sum(diff**2))
            
            # 找到对应的情绪名称
            emotion_name = self._get_name_from_id(attractor_id)
            dist_dict[emotion_name] = dist
            
            if dist < min_dist:
                min_dist = dist
                best_attractor_id = attractor_id
        
        # 获取最佳吸引子的名称
        best_emotion = self._get_name_from_id(best_attractor_id)
        self.current_emotion = best_emotion
        self.current_distance = min_dist
        
        # 更新吸引子的访问信息
        if best_attractor_id is not None:
            best_attr = self.attractors[best_attractor_id]
            best_attr.access_count += 1
            best_attr.visit_count += 1
            best_attr.last_accessed = step
            # 添加状态历史
            best_attr.state_history.append(E_vec)
            
            # 滑动平均中心更新（仅当距离在半径1.2倍内，表示确实属于该吸引子）
            if best_attr.radius == 0 or min_dist < best_attr.radius * 1.2:
                alpha = best_attr.center_moving_alpha
                best_attr.learned_center = (1 - alpha) * best_attr.learned_center + alpha * E_vec
                best_attr.visit_count_for_center += 1
                
                # 每 N 次访问后同步实际中心
                if best_attr.visit_count_for_center % 100 == 0:
                    old_center = best_attr.center.copy()
                    best_attr.center = best_attr.learned_center.copy()
                    # 记录漂移
                    drift = np.linalg.norm(best_attr.center - old_center)
                    self.logger.debug(f"Attractor {best_attr.id} center drifted by {drift:.4f}")
        
        self.logger.debug(f"Emotion identified: {best_emotion} (dist={min_dist:.3f})")
        return best_emotion, min_dist, dist_dict
    
    def _get_name_from_id(self, attractor_id):
        """从吸引子ID获取情绪名称"""
        for name, id in self.name_to_id.items():
            if id == attractor_id:
                return name
        # 如果找不到，返回ID作为名称
        return f"attractor_{attractor_id}"
    
    def _generate_attractor_name(self, center):
        """根据中心坐标生成描述性名称"""
        # 简易规则：检测哪些维度显著偏离中性
        name_parts = []
        if center[0] < -0.5:
            name_parts.append("avoidant")
        elif center[0] > 0.5:
            name_parts.append("approaching")
        if center[1] > 0.7:
            name_parts.append("tense")
        elif center[1] < 0.3:
            name_parts.append("calm")
        if center[2] < -0.5:
            name_parts.append("unpleasant")
        elif center[2] > 0.5:
            name_parts.append("pleasant")
        if center[3] < -0.5:
            name_parts.append("isolated")
        elif center[3] > 0.5:
            name_parts.append("connected")
        if center[4] < 0.3:
            name_parts.append("diffuse")
        elif center[4] > 0.7:
            name_parts.append("clear")
        
        if not name_parts:
            return "neutral_emergent"
        return "_".join(name_parts[:2])  # 取前两个特征
    
    def is_stuck(self, history, threshold_steps=20):
        """检查最近N步是否陷入同一情绪"""
        if len(history) < threshold_steps:
            return False
        recent = history[-threshold_steps:]
        return len(set(recent)) == 1
    
    def get_action_tendency(self, emotion):
        """获取情绪的行动倾向"""
        # 首先尝试通过名称查找
        if emotion in self.name_to_id:
            attractor_id = self.name_to_id[emotion]
            if attractor_id in self.attractors:
                return self.attractors[attractor_id].action_tendency
        
        # 尝试通过ID查找
        try:
            attractor_id = int(emotion.split('_')[-1])
            if attractor_id in self.attractors:
                return self.attractors[attractor_id].action_tendency
        except (ValueError, IndexError):
            pass
        
        return "neutral"
    
    def _cluster_state_vectors(self, vectors, eps, min_samples):
        """使用 DBSCAN 聚类，返回标签列表和核心样本索引"""
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(vectors)
        return clustering.labels_, clustering.core_sample_indices_
    
    def discover_new_attractors(self, trajectories, current_step, config=None):
        """
        trajectories: list of (E_vec, success, action_tendency)
        返回新添加的吸引子ID列表
        """
        # 配置回退机制
        if config is None:
            from utils.config_loader import Config
            config = Config().get('attractor_evolution', {})
            self.logger.warning("discover_new_attractors called without config, using global defaults")
        
        # 查看配置内容
        print(f"Config received: {config}")
        self.logger.info(f"Config received: {config}")
        
        # 从配置中获取参数
        eps = config.get('clustering_eps', 0.45)
        min_samples = config.get('clustering_min_samples', 8)
        success_threshold = config.get('validity_success_rate', 0.4)
        
        # 验证配置是否被正确读取
        self.logger.info(f"Using attractor evolution config: eps={eps}, min_samples={min_samples}, success_threshold={success_threshold}")
        
        if len(trajectories) < min_samples * 2:
            return []
        
        # 提取状态向量，排除已被现有吸引子覆盖的点
        vectors = []
        successes = []
        tendencies = []
        covered_count = 0
        for vec, succ, tend in trajectories:
            # 检查是否已被已有吸引子覆盖（距离 < 半径）
            covered = False
            for attr in self.attractors.values():
                dist = np.linalg.norm(vec - attr.center)
                if dist < attr.radius:
                    covered = True
                    covered_count += 1
                    break
            if not covered:
                vectors.append(vec)
                successes.append(succ)
                tendencies.append(tend)
        
        self.logger.info(f"Discovery: {len(trajectories)} total points, {covered_count} covered, {len(vectors)} available")
        
        if len(vectors) < config.get('clustering_min_samples', 30):
            return []
        
        # 调用聚类函数
        eps = config.get('clustering_eps', 0.4)
        min_samples = config.get('clustering_min_samples', 30)
        labels, core_idx = self._cluster_state_vectors(np.array(vectors), eps, min_samples)
        
        self.logger.info(f"Clustering: {len(set(labels))-1} clusters found (excluding noise)")
        
        new_ids = []
        for label in set(labels):
            if label == -1:
                continue
            
            # 提取该簇的所有点
            cluster_mask = labels == label
            cluster_vectors = np.array(vectors)[cluster_mask]
            cluster_success = np.array(successes)[cluster_mask]
            cluster_tendencies = np.array(tendencies)[cluster_mask]
            
            self.logger.info(f"Cluster {label}: {len(cluster_vectors)} points")
            
            # 计算平均成功率，过滤掉None值
            cluster_success = cluster_success[cluster_success != None]
            if len(cluster_success) == 0:
                avg_success = 0.0
            else:
                avg_success = np.mean(cluster_success)
            self.logger.info(f"Cluster {label}: avg_success={avg_success:.2f}")
            
            # 从配置中获取成功率阈值
            validity_threshold = config.get('validity_success_rate', 0.4)
            self.logger.info(f"Validity threshold: {validity_threshold:.2f}")
            
            if avg_success < validity_threshold:
                self.logger.info(f"Cluster {label}: rejected due to low success rate")
                continue
            
            # 计算中心（均值）和半径（覆盖90%点）
            center = np.mean(cluster_vectors, axis=0)
            distances = np.linalg.norm(cluster_vectors - center, axis=1)
            radius = np.percentile(distances, 90)
            
            # 检查与现有吸引子是否重叠
            overlap = False
            for attr in self.attractors.values():
                dist_centers = np.linalg.norm(center - attr.center)
                if dist_centers < (radius + attr.radius) * 0.8:
                    overlap = True
                    break
            if overlap:
                self.logger.info(f"Cluster {label}: rejected due to overlap with existing attractor")
                continue
            
            # 确定行动倾向（取出现频率最高的）
            from collections import Counter
            top_tendency = Counter(cluster_tendencies).most_common(1)[0][0]
            
            # 生成描述性名称
            new_name = self._generate_attractor_name(center)
            
            # 创建新吸引子
            new_id = max(self.attractors.keys(), default=0) + 1
            new_attr = DynamicAttractor(
                attractor_id=new_id,
                center=center,
                radius=radius,
                action_tendency=top_tendency,
                birth_step=current_step,
                is_prototype=False
            )
            new_attr.success_rate = avg_success
            self.attractors[new_id] = new_attr
            # 存储名称到ID的映射
            self.name_to_id[new_name] = new_id
            new_ids.append(new_id)
            self.logger.info(f"New attractor discovered: {new_name} (id={new_id}), center={center}, tendency={top_tendency}")
        
        return new_ids
    
    def split_attractor(self, attr_id, current_step, config):
        """
        分裂吸引子
        attr_id: 要分裂的吸引子ID
        current_step: 当前交互步数
        config: 吸引子演化配置
        """
        import numpy as np
        from sklearn.cluster import KMeans
        
        # 检查吸引子是否存在
        if attr_id not in self.attractors:
            self.logger.warning(f"Attractor {attr_id} not found")
            return []
        
        attr = self.attractors[attr_id]
        
        # 检查是否为原型吸引子
        if attr.is_prototype:
            self.logger.info(f"Prototype attractor {attr_id} cannot be split")
            return []
        
        # 从配置读取阈值
        split_visit_threshold = config.get('attractor_evolution', {}).get('split_visit_threshold', 200)
        split_variance_threshold = config.get('attractor_evolution', {}).get('split_variance_threshold', 0.3)
        
        # 检查访问次数和内部方差
        if attr.visit_count <= split_visit_threshold:
            self.logger.info(f"Attractor {attr_id} visit count ({attr.visit_count}) < {split_visit_threshold}, skip split")
            return []
        
        # 计算内部方差
        if not attr.state_history:
            self.logger.info(f"Attractor {attr_id} has no state history, skip split")
            return []
        
        states = np.array(attr.state_history)
        variance = np.var(states, axis=0)
        total_variance = np.sum(variance)
        
        if total_variance <= split_variance_threshold:
            self.logger.info(f"Attractor {attr_id} total variance ({total_variance:.2f}) <= {split_variance_threshold}, skip split")
            return []
        
        self.logger.info(f"Attempting to split attractor {attr_id} with visit count {attr.visit_count} and variance {total_variance:.2f}")
        
        # 使用 K-Means 聚类
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(states)
        
        # 计算两个子簇的中心
        centers = kmeans.cluster_centers_
        distance = np.linalg.norm(centers[0] - centers[1])
        
        if distance <= 0.4:
            self.logger.info(f"Sub-cluster centers distance ({distance:.2f}) <= 0.4, skip split")
            return []
        
        self.logger.info(f"Sub-cluster centers distance: {distance:.2f}, proceeding with split")
        
        # 创建两个新吸引子
        new_ids = []
        for i, center in enumerate(centers):
            # 计算子簇的半径（覆盖90%点）
            cluster_states = states[labels == i]
            distances = np.linalg.norm(cluster_states - center, axis=1)
            radius = np.percentile(distances, 90)
            
            # 计算子簇的平均成功率
            cluster_success = []
            for j, state in enumerate(attr.state_history):
                if labels[j] == i and j < len(attr.success_history):
                    cluster_success.append(attr.success_history[j][1])
            avg_success = np.mean(cluster_success) if cluster_success else 0.5
            
            # 创建新吸引子
            new_id = max(self.attractors.keys(), default=0) + 1
            new_attr = DynamicAttractor(
                attractor_id=new_id,
                center=center,
                radius=radius,
                action_tendency=attr.action_tendency,
                birth_step=current_step,
                is_prototype=False
            )
            new_attr.parent = attr_id
            new_attr.visit_count = len(cluster_states)
            new_attr.success_rate = avg_success
            
            # 添加到吸引子字典
            self.attractors[new_id] = new_attr
            new_ids.append(new_id)
            
            # 添加到父吸引子的子列表
            attr.children.append(new_id)
            
            self.logger.info(f"Created new attractor {new_id} from split, center: {center}, radius: {radius:.2f}, success rate: {avg_success:.2f}")
        
        # 标记原吸引子为已分裂
        attr.split = True
        attr.split_step = current_step
        self.logger.info(f"Marked attractor {attr_id} as split at step {current_step}")
        
        return new_ids
    
    def archive_inactive_attractors(self, current_step, inactive_steps=50000):
        """将长期未访问的非原型吸引子归档（从活跃库移到存档文件）"""
        archived = []
        for attr_id, attr in list(self.attractors.items()):
            if attr.is_prototype:
                continue
            if current_step - attr.last_accessed > inactive_steps:
                # 归档：保存到文件并从活跃字典删除
                self._save_to_archive(attr)
                del self.attractors[attr_id]
                archived.append(attr_id)
                self.logger.info(f"Archived inactive attractor {attr_id}")
        return archived
    
    def _save_to_archive(self, attr):
        """追加到 archive/attractors.json"""
        import json, os
        # 确保目录存在
        os.makedirs("data", exist_ok=True)
        archive_path = "data/attractor_archive.json"
        data = []
        if os.path.exists(archive_path):
            try:
                with open(archive_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
        data.append(attr.to_dict())
        with open(archive_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def force_split(self, attractor_id, current_step):
        """手动触发分裂（用于调试）"""
        if attractor_id in self.attractors:
            return self.split_attractor(attractor_id, current_step, {})
        return []
    
    def merge_attractors(self, config=None):
        """
        检查并合并距离过近的非原型吸引子。
        返回被合并的吸引子ID列表。
        """
        if config is None:
            from utils.config_loader import Config
            config = Config().get('attractor_evolution', {})
        
        merge_distance_threshold = config.get('merge_distance_threshold', 0.3)
        merge_success_diff_threshold = config.get('merge_success_diff_threshold', 0.2)
        
        merged_ids = []
        attr_items = list(self.attractors.items())
        
        for i, (id1, attr1) in enumerate(attr_items):
            if attr1.is_prototype or id1 in merged_ids:
                continue
            for id2, attr2 in attr_items[i+1:]:
                if attr2.is_prototype or id2 in merged_ids:
                    continue
                
                # 计算中心距离
                dist = np.linalg.norm(attr1.center - attr2.center)
                if dist > merge_distance_threshold:
                    continue
                
                # 检查成功率差异
                success_diff = abs(attr1.success_rate - attr2.success_rate)
                if success_diff > merge_success_diff_threshold:
                    continue
                
                # 执行合并：保留访问次数较多的，吸收另一个
                if attr1.visit_count >= attr2.visit_count:
                    keeper, absorbed = attr1, attr2
                    keeper_id, absorbed_id = id1, id2
                else:
                    keeper, absorbed = attr2, attr1
                    keeper_id, absorbed_id = id2, id1
                
                # 加权平均中心
                total_visits = keeper.visit_count + absorbed.visit_count
                w1 = keeper.visit_count / total_visits
                w2 = absorbed.visit_count / total_visits
                keeper.center = w1 * keeper.center + w2 * absorbed.center
                keeper.learned_center = keeper.center.copy()
                
                # 合并半径（取较大者）
                keeper.radius = max(keeper.radius, absorbed.radius)
                
                # 合并成功率
                keeper.success_rate = (keeper.success_rate * keeper.visit_count +
                                       absorbed.success_rate * absorbed.visit_count) / total_visits
                
                # 合并访问计数
                keeper.visit_count = total_visits
                
                # 移除被吸收的吸引子
                del self.attractors[absorbed_id]
                merged_ids.append(absorbed_id)
                
                # 清理名称映射
                for name, nid in list(self.name_to_id.items()):
                    if nid == absorbed_id:
                        del self.name_to_id[name]
                
                self.logger.info(f"Merged attractor {absorbed_id} into {keeper_id} (dist={dist:.3f})")
                break  # keeper 可能继续与其他吸引子比较，但简化起见，每次只处理一对
        
        return merged_ids

# 新增类
class DynamicAttractor:
    def __init__(self, attractor_id, center, radius, action_tendency, birth_step, is_prototype=False):
        self.id = attractor_id
        self.center = np.array(center, dtype=np.float32)
        self.learned_center = self.center.copy()   # 滑动平均中心
        self.radius = radius
        self.action_tendency = action_tendency  # 字符串或函数名
        self.birth_step = birth_step
        self.is_prototype = is_prototype
        self.access_count = 0
        self.visit_count = 0
        self.success_rate = 0.5
        self.children = []      # 子吸引子ID列表
        self.parent = None      # 父吸引子ID
        self.last_accessed = birth_step
        self.state_history = []  # 状态历史
        self.success_history = []  # 成功历史 (state, success)
        self.split = False      # 是否已分裂
        self.split_step = None  # 分裂步数
        self.center_moving_alpha = 0.01 if not is_prototype else 0.001  # 原型漂移慢
        self.visit_count_for_center = 0
        self.split_candidate = False  # 标记是否已进入分裂候选
        self.last_split_check_step = 0

    def to_dict(self):
        return {
            'id': self.id,
            'center': self.center.tolist(),
            'radius': float(self.radius),
            'action_tendency': self.action_tendency,
            'birth_step': self.birth_step,
            'is_prototype': self.is_prototype,
            'access_count': self.access_count,
            'visit_count': self.visit_count,
            'success_rate': float(self.success_rate),
            'children': self.children,
            'parent': self.parent,
            'last_accessed': self.last_accessed,
            'state_history': [state.tolist() for state in self.state_history],
            'success_history': self.success_history,
            'split': self.split,
            'split_step': self.split_step
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data['id'], np.array(data['center']), data['radius'],
            data['action_tendency'], data['birth_step'], data['is_prototype']
        )
        obj.access_count = data['access_count']
        obj.visit_count = data.get('visit_count', 0)
        obj.success_rate = data['success_rate']
        obj.children = data['children']
        obj.parent = data['parent']
        obj.last_accessed = data['last_accessed']
        obj.state_history = [np.array(state) for state in data.get('state_history', [])]
        obj.success_history = data.get('success_history', [])
        obj.split = data.get('split', False)
        obj.split_step = data.get('split_step', None)
        return obj
