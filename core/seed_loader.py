"""
种子文本加载器
功能：扫描 data/seed_texts/ 目录，将文本分块注入 LPS，并提取结构化标签与语义条目。
"""

import os
import re
from pathlib import Path
from typing import Dict, List
from utils.logger import get_logger


class SeedLoader:
    def __init__(self, engine):
        self.engine = engine
        self.logger = get_logger('seed_loader')
        self.seed_dir = Path('data/seed_texts')
    
    def load_all(self) -> Dict:
        """加载所有种子文本，返回统计信息"""
        stats = {
            'files_processed': 0,
            'chunks_added': 0,
            'triplets_extracted': 0,
            'keywords_added': 0
        }
        
        if not self.seed_dir.exists():
            self.logger.info(f"种子目录不存在: {self.seed_dir}，跳过加载")
            return stats
        
        for file_path in self.seed_dir.glob('*.txt'):
            file_stats = self._load_file(file_path)
            stats['files_processed'] += 1
            stats['chunks_added'] += file_stats['chunks']
            stats['triplets_extracted'] += file_stats['triplets']
            stats['keywords_added'] += file_stats['keywords']
        
        self.logger.info(f"种子文本加载完成: {stats}")
        return stats
    
    def _load_file(self, file_path: Path) -> Dict:
        """加载单个文件"""
        stats = {'chunks': 0, 'triplets': 0, 'keywords': 0}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            self.logger.warning(f"读取种子文件失败 {file_path}: {e}")
            return stats
        
        # 分块（chunk_size=512, overlap=50）
        chunk_size = 512
        overlap = 50
        step = chunk_size - overlap
        
        for i, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start:start + chunk_size].strip()
            if not chunk_text:
                continue
            
            # 提取三元组与关键词
            triplets = self._extract_triplets(chunk_text)
            keywords = self._extract_keywords(chunk_text)
            
            # 构建标签
            tags = {
                'type': 'seed_text',
                'source': 'seed',
                'file': file_path.name,
                'chunk_index': i
            }
            if triplets:
                # 取第一个三元组作为主标签（可扩展为多标签）
                subj, rel, obj = triplets[0]
                tags.update({'entity': subj, 'relation': rel, 'value': obj})
                stats['triplets'] += 1
            
            # 存入 LPS，势能 0.6
            if hasattr(self.engine, 'lps') and self.engine.lps:
                node_id = self.engine.lps.add_if_new(chunk_text, potency=0.6, tags=tags)
                if node_id:
                    stats['chunks'] += 1
            
            # 初始化语义条目
            if hasattr(self.engine, 'structural_coordinator') and hasattr(self.engine.structural_coordinator, 'semantic_mapper'):
                self.logger.debug(f"处理关键词: {keywords}")
                for kw in keywords:
                    entry = self.engine.structural_coordinator.semantic_mapper.get_or_create_entry(kw)
                    if entry.get('confidence', 0) == 0.3:  # 新创建
                        stats['keywords'] += 1
                        self.logger.debug(f"创建新语义条目: {kw}")
        
        self.logger.debug(f"加载种子文件 {file_path.name}: {stats}")
        return stats
    
    def _extract_triplets(self, text: str) -> List[tuple]:
        """提取三元组（与 DocumentLearner 保持一致）"""
        triplets = []
        # "X是Y"
        pattern_is = re.compile(r'([^，。！？\s]{2,50})是([^，。！？]{2,50})')
        for match in pattern_is.finditer(text):
            subj, obj = match.groups()
            triplets.append((subj.strip(), "是", obj.strip()))
        # "X位于Y"
        pattern_loc = re.compile(r'([^，。！？\s]{2,50})(?:位于|在)([^，。！？]{2,50})')
        for match in pattern_loc.finditer(text):
            subj, obj = match.groups()
            triplets.append((subj.strip(), "位于", obj.strip()))
        # "X的作者是Y" / "X由Y撰写"
        pattern_author = re.compile(r'《?([^》\s]+)》?(?:的作者是|由)([^，。！？\s]+)')
        for match in pattern_author.finditer(text):
            subj, obj = match.groups()
            triplets.append((f"《{subj}》" if not subj.startswith('《') else subj, "作者", obj.strip()))
        # 识别“核心命题是：X” / “主要观点是：X”
        pattern_core = re.compile(r'(?:核心命题|核心观点|主要观点|中心思想)[是为：]\s*([^。！？\n]+)')
        for match in pattern_core.finditer(text):
            proposition = match.group(1).strip()
            # 尝试关联到前文的书名
            title_match = re.search(r'《([^》]+)》', text)
            if title_match:
                triplets.append((f"《{title_match.group(1)}》", "核心命题", proposition))
        return triplets
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取中文关键词（长度≥2）"""
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        # 去重
        seen = set()
        keywords = []
        for w in words:
            if w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords[:20]  # 每块最多取20个词