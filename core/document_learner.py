"""
文档学习器
功能：从知识源分块学习，内化到语义记忆、LPS 和语义库
"""

import time
import re
from typing import Dict, List
from utils.logger import get_logger
from core.knowledge_source import KnowledgeSource, KnowledgeChunk


class DocumentLearner:
    def __init__(self, engine):
        self.engine = engine
        self.logger = get_logger('document_learner')
    
    def learn(self, source) -> dict:
        """主学习流程：分析 → 归纳 → 生成 → 摄入"""
        stats = {
            'source': source.get_summary(),
            'chunks_processed': 0,
            'triplets_extracted': 0,
            'lps_added': 0,
            'keywords_learned': 0,
            'stiffness_before': 0.0,
            'stiffness_after': 0.0,
            'start_time': time.time(),
            'analysis_summary': '',
            'new_knowledge_count': 0,
            'cross_references_added': 0,
        }

        if hasattr(self.engine, 'process_meta'):
            stats['stiffness_before'] = self.engine.process_meta.get_coupling_stiffness()

        # ====== 第0步：收集全文内容 ======
        full_text = ""
        for chunk in source.fetch_chunks():
            full_text += chunk.content

        # ====== 第1步：分析 ======
        self.logger.info(f"开始分析文档: {source.get_summary()}")
        analysis = self._analyze_document(full_text, source.get_summary())
        stats['analysis_summary'] = analysis.get('summary', '')
        # 提取原始分析文本，用于后续的 Qwen 三元组提取
        analysis_text = analysis.get('_analysis_raw', '')

        # ====== 第2步：分块沉积（基础层） =====
        # 记录当前文档来源，供 _process_chunk 使用
        self._current_source_summary = source.get_summary()

        for chunk in source.fetch_chunks():
            self._process_chunk(chunk, stats)
            stats['chunks_processed'] += 1
            self._record_learning_nourishment(chunk)

        # ====== 第3步：归纳与生成（新增） ======
        # 基于分析报告和分块沉积，生成归纳产物并写入LPS
        induction_result = self._induce_and_generate(analysis, source.get_summary(), full_text)
        stats['new_knowledge_count'] = induction_result.get('new_entries', 0)
        stats['cross_references_added'] = induction_result.get('cross_references', 0)

        # ===== Qwen 三元组提取与写入 =====
        # 将 Step 1 的分析文本用于三元组提取
        qwen_triplets = self._extract_triplets_with_qwen(analysis_text, source.get_summary())
        for subj, rel, obj in qwen_triplets:
            triplet_text = f"{subj} {rel} {obj}"
            tags = {
                'type': 'induced_triplet',
                'source': 'document_ingestion',
                'doc_source': source.get_summary(),
                'entity': subj,
                'relation': rel,
                'value': obj,
                'knowledge_type': 'triplet',
                'ingested_at': time.time(),
            }
            # 补全双记忆标签
            if hasattr(self.engine, 'structural_coordinator'):
                coord = self.engine.structural_coordinator.get_current_coordinate()
                tags['subjective_room'] = coord.as_tarot_code()
                tags['subjective_major'] = coord.major
                if hasattr(self.engine, 'image_base'):
                    card = self.engine.image_base.get_card_by_coordinate(coord)
                    if card:
                        tags['subjective_card_id'] = card.id
            if hasattr(self.engine, 'objective_classifier'):
                tags['objective_room'] = self.engine.objective_classifier.classify(triplet_text)
            if hasattr(self.engine.structural_coordinator, '_infer_major_arcana'):
                reasoned = self.engine.structural_coordinator._infer_major_arcana(triplet_text)
                if reasoned:
                    tags['reasoned_card'] = reasoned
            if hasattr(self.engine.structural_coordinator, 'draw_random_card'):
                tags['random_card'] = self.engine.structural_coordinator.draw_random_card()

            embedding = self.engine.lps.encoder.encode([triplet_text])[0] if self.engine.lps.encoder else None
            node_id = self.engine.lps.add_if_new(triplet_text, embedding, potency=0.8, tags=tags)
            if node_id:
                stats['triplets_extracted'] += 1

        # ====== 第4步：摄入收束 ======
        self._finalize_ingestion(source.get_summary(), stats)

        if hasattr(self.engine, 'process_meta'):
            stats['stiffness_after'] = self.engine.process_meta.get_coupling_stiffness()
        stats['duration'] = time.time() - stats['start_time']

        self._record_learning_completed(source, stats)
        return stats
    
    def _process_chunk(self, chunk: KnowledgeChunk, stats: Dict):
        # 1. 提取三元组，构建标签
        triplets = self._extract_triplets(chunk.content)
        tags = {}
        if triplets:
            # 取第一个三元组作为主标签（可扩展为多标签）
            subj, rel, obj = triplets[0]
            tags = {
                'type': 'document_chunk',
                'source': 'document_ingestion',
                'entity': subj,
                'relation': rel,
                'value': obj,
                'doc_source': self._current_source_summary if hasattr(self, '_current_source_summary') else 'unknown',
                'ingested_at': time.time(),
            }
            # 补全双记忆标签
            self._inject_memory_tags(tags, chunk.content)
            stats['triplets_extracted'] += 1
        
        # 2. 存入 LPS，携带标签
        if hasattr(self.engine, 'lps') and self.engine.lps:
            node_id = self.engine.lps.add_if_new(chunk.content, potency=0.9, tags=tags)
            if node_id:
                stats['lps_added'] += 1
        
        # 3. 提取关键词，初始化语义条目（保持原有逻辑）
        if hasattr(self.engine, 'semantic_mapper'):
            keywords = self.engine.semantic_mapper._extract_keywords(chunk.content)
            for kw in keywords:
                entry = self.engine.semantic_mapper.get_or_create_entry(kw)
                if entry.retrieval_count == 0:  # 新创建
                    stats['keywords_learned'] += 1
    
    def _extract_triplets(self, text: str) -> List[tuple]:
        """简易三元组提取（规则）"""
        triplets = []
        # "X是Y"
        pattern_is = re.compile(r'([^，。！？\s]{2,50})是([^，。！？]{2,50})')
        for match in pattern_is.finditer(text):
            subj, obj = match.groups()
            triplets.append((subj.strip(), "是", obj.strip()))
        # "X位于Y" / "X在Y"
        pattern_loc = re.compile(r'([^，。！？\s]{2,50})(?:位于|在)([^，。！？]{2,50})')
        for match in pattern_loc.finditer(text):
            subj, obj = match.groups()
            triplets.append((subj.strip(), "位于", obj.strip()))
        # "X的作者是Y" / "X由Y撰写"
        pattern_author = re.compile(r'《?([^》\s]+)》?(?:的作者是|由)([^，。！？\s]+)')
        for match in pattern_author.finditer(text):
            subj, obj = match.groups()
            triplets.append((f"《{subj}》" if not subj.startswith('《') else subj, "作者", obj.strip()))
        
        # 识别"《X》是由Y撰写的"或"X由Y共同撰写"
        pattern_author = re.compile(r'《([^》]+)》是由([^撰]+)撰写的')
        for match in pattern_author.finditer(text):
            title, authors = match.groups()
            triplets.append((f"《{title}》", "作者", authors.strip()))
        
        # 同时识别"X和Y共同撰写"变体
        pattern_coauthor = re.compile(r'([^，。]+)与([^，。]+)共同撰写')
        for match in pattern_coauthor.finditer(text):
            authors = f"{match.group(1).strip()}与{match.group(2).strip()}"
            # 需要关联到前面提到的书名，简化处理：若前文有书名号
            title_match = re.search(r'《([^》]+)》', text)
            if title_match:
                triplets.append((f"《{title_match.group(1)}》", "作者", authors))
        # 识别“核心命题是：X” / “主要观点是：X”
        pattern_core = re.compile(r'(?:核心命题|核心观点|主要观点|中心思想)[是为：]\s*([^。！？\n]+)')
        for match in pattern_core.finditer(text):
            proposition = match.group(1).strip()
            # 尝试关联到前文的书名
            title_match = re.search(r'《([^》]+)》', text)
            if title_match:
                triplets.append((f"《{title_match.group(1)}》", "核心命题", proposition))
        return triplets
    
    def _record_learning_nourishment(self, chunk: KnowledgeChunk):
        if not hasattr(self.engine, 'process_meta'):
            return
        self.engine.process_meta.record_nourishment(
            source_text=f"{chunk.source_type}:{chunk.source_identifier}#{chunk.chunk_index}",
            success=1.0,
            coupling_weight=0.5
        )
    
    def _record_learning_completed(self, source: KnowledgeSource, stats: Dict):
        if hasattr(self.engine, 'global_workspace'):
            self.engine.global_workspace._record_spiral_event(
                'learning_completed',
                {
                    'source': source.get_summary(),
                    'chunks': stats['chunks_processed'],
                    'triplets': stats['triplets_extracted'],
                    'keywords': stats['keywords_learned']
                }
            )
    
    def _analyze_document(self, full_text: str, source_summary: str) -> dict:
        """
        三步渐进式分析：
        Step 1: 通读全文，写非结构化摘要
        Step 2: 从摘要中提取实体列表
        Step 3: 逐实体判断新旧关系，生成结构化报告
        """
        sample = full_text[:6000]

        # ====== Step 1: 通读全文，自由分析 ======
        self.logger.info(f"分析 Step 1/3: 通读全文，生成自由分析...")
        step1_prompt = f"""请阅读以下文档，用你自己的话写一份分析报告。

文档来源：{source_summary}
文档内容：
{sample}

请分段写出：
一、核心主题（一句话概括这篇文章在讲什么）
二、关键内容（列出文章中重要的概念、人物、术语、发现，每行一个，尽量完整）
三、核心论断（列出文章提出的主要观点或发现，每句不超过40字）

直接写分析内容，不需要 JSON 或任何特殊格式。"""

        analysis_text = self.engine.response_generator._generate_with_llm(
            step1_prompt,
            self.engine.fse,
            intent='GENERAL_CHAT',
            temperature=0.3
        )

        if not analysis_text:
            self.logger.warning("Step 1 分析文本为空，使用降级方案")
            return self._build_fallback_analysis(sample)

        # ====== Step 2: 从摘要中提取实体列表 ======
        self.logger.info(f"分析 Step 2/3: 提取实体列表...")
        step2_prompt = f"""以下是一份文档的分析报告。请从中提取所有关键实体（概念、人物、术语、发现等）。

分析报告：
{analysis_text[:2000]}

请输出一个 JSON 对象，格式如下：
{{
  "entities": ["实体1", "实体2", ...]
}}

只输出 JSON，不要任何解释。"""

        entities = []
        step2_result = self.engine.response_generator._generate_with_llm(
            step2_prompt,
            self.engine.fse,
            intent='GENERAL_CHAT',
            temperature=0.1
        )

        if step2_result:
            import json, re
            try:
                entities_data = json.loads(step2_result)
                entities = entities_data.get('entities', [])
            except json.JSONDecodeError:
                # 增强容错：用正则提取
                json_match = re.search(r'\{.*\}', step2_result, re.DOTALL)
                if json_match:
                    try:
                        entities_data = json.loads(json_match.group(0))
                        entities = entities_data.get('entities', [])
                    except json.JSONDecodeError:
                        pass

        if not entities:
            # 降级：用规则从分析文本中提取
            entities = self._extract_entities_by_rule(analysis_text)

        self.logger.info(f"Step 2 提取到 {len(entities)} 个实体")

        # ====== Step 3: 逐实体判断新旧关系 ======
        self.logger.info(f"分析 Step 3/3: 判断新旧关系...")
        confirmed = []
        new_knowledge = []
        conflicting = []

        # 对每个实体，查询已有知识库
        for entity in entities[:10]:  # 最多处理10个实体，避免过度消耗
            existing = self._check_existing_knowledge(entity)
            if existing == 'confirmed':
                confirmed.append(entity)
            elif existing == 'conflicting':
                conflicting.append(entity)
            else:
                new_knowledge.append(entity)

        # 从 Step 1 的分析中提取核心命题
        core_propositions = self._extract_propositions_by_rule(analysis_text)

        # 提取摘要
        import re
        summary = ""
        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:
                cleaned = re.sub(r'^[一二三四五六七八九十]+[、.．]\s*', '', line)
                if cleaned and '核心主题' not in cleaned and len(cleaned) > 20:
                    summary = cleaned[:200]
                    break
        if not summary:
            summary = sample[:150] + "..."

        self.logger.info(f"分析完成: 新实体{len(new_knowledge)}个, 已知{len(confirmed)}个, 矛盾{len(conflicting)}个")

        return {
            "summary": summary,
            "key_entities": entities[:10],
            "core_propositions": core_propositions,
            "knowledge_relations": {
                "confirmed": confirmed,
                "new": new_knowledge,
                "conflicting": conflicting
            },
            "_analysis_raw": analysis_text
        }
    
    def _check_existing_knowledge(self, entity: str) -> str:
        """
        检查一个实体与已有知识的关系。
        返回 'confirmed'（已知）、'new'（新知）、'conflicting'（矛盾）之一。
        """
        if not hasattr(self.engine, 'lps') or not self.engine.lps:
            return 'new'

        # 查询 LIFO 中是否有与这个实体相关的知识条目
        existing = self.engine.lps.query_by_tag(entity=entity, min_potency=0.3)
        if not existing:
            # 再尝试向量检索
            embedding = self.engine.lps.encoder.encode([entity])[0] if self.engine.lps.encoder else None
            if embedding is not None:
                similar = self.engine.lps.query(embedding, k=1, min_potency=0.5)
                if similar and similar[0].get('distance', 1) < 0.3:
                    return 'confirmed'
            return 'new'

        # 存在相关条目，判定为已知
        return 'confirmed'


    def _extract_entities_by_rule(self, text: str) -> list:
        """用规则从分析文本中提取实体（降级方案）"""
        import re
        entities = []
        entity_section = False
        for line in text.split('\n'):
            line = line.strip()
            if '关键内容' in line or '关键实体' in line or '二' in line[:3] or '实体' in line:
                entity_section = True
                continue
            if entity_section and line and len(line) > 1:
                if re.match(r'^[三四五六七八九十]+[、.．]', line) or '核心论断' in line or '与已有' in line:
                    break
                cleaned = re.sub(r'^[-·•*▪▸\d+\.\s]+', '', line).strip()
                if cleaned and len(cleaned) < 80 and len(cleaned) > 1:
                    entities.append(cleaned)
            if len(entities) >= 10:
                break
        if not entities:
            # 最后手段：取文本中看起来像实体的词
            words = re.findall(r'[\u4e00-\u9fa5]{2,6}', text)
            entities = list(set(words))[:5]
        return entities


    def _extract_propositions_by_rule(self, text: str) -> list:
        """用规则从分析文本中提取核心命题（降级方案）"""
        import re
        propositions = []
        prop_section = False
        for line in text.split('\n'):
            line = line.strip()
            if '核心论断' in line or '三' in line[:3] or '主要发现' in line or '主要观点' in line:
                prop_section = True
                continue
            if prop_section and line and len(line) > 5:
                if re.match(r'^[四五六七八九十]+[、.．]', line) or '与已有' in line:
                    break
                cleaned = re.sub(r'^[-·•*▪▸\d+\.\s]+', '', line).strip()
                if cleaned and len(cleaned) > 5:
                    propositions.append(cleaned[:100])
            if len(propositions) >= 5:
                break
        if not propositions:
            # 取文本的前几个长句
            sentences = re.split(r'[。！？\n]', text)
            propositions = [s.strip()[:100] for s in sentences if len(s.strip()) > 10][:3]
        return propositions


    def _build_fallback_analysis(self, sample: str) -> dict:
        """当所有分析步骤失败时，构建降级分析结果"""
        return {
            "summary": sample[:150] + "...",
            "key_entities": [],
            "core_propositions": [sample[:100]],
            "knowledge_relations": {"confirmed": [], "new": [sample[:100]], "conflicting": []}
        }
    
    def _extract_triplets_with_qwen(self, analysis_text: str, source_summary: str) -> list:
        """
        基于分析文本，用 Qwen 提取结构化三元组。
        返回三元组列表，每个元素为 (subject, relation, object)。
        """
        prompt = f"""请从以下分析报告中提取所有明确陈述的事实三元组。

分析报告：
{analysis_text[:3000]}

每个三元组包含三个字段：subject（主语）、relation（关系）、object（宾语）。
用 JSON 数组格式输出，例如：
[{{"subject": "J. Tuzo Wilson", "relation": "提出", "object": "地幔热柱假说"}}, ...]

规则：
1. 只提取报告中有明确陈述的事实
2. 关系尽量简洁，如"提出"、"发现"、"包含"、"位于"、"导致"、"支持"
3. 主语和宾语应是核心实体或概念
4. 输出至少 3 个，最多 10 个

只输出 JSON 数组，不要任何解释。"""

        triplets = []
        result_text = self.engine.response_generator._generate_with_llm(
            prompt,
            self.engine.fse,
            intent='GENERAL_CHAT',
            temperature=0.1
        )

        if result_text:
            import json, re
            try:
                triplets = json.loads(result_text)
            except json.JSONDecodeError:
                # 增强容错：用正则提取 JSON 数组
                match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if match:
                    try:
                        triplets = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass

        # 格式转换为 (subject, relation, object) 元组
        extracted = []
        for t in triplets:
            if isinstance(t, dict) and 'subject' in t and 'relation' in t and 'object' in t:
                extracted.append((t['subject'], t['relation'], t['object']))

        self.logger.info(f"Qwen 三元组提取: {len(extracted)} 条")
        return extracted
    
    def _inject_memory_tags(self, tags: dict, text: str):
        """为知识碎片注入完整的双记忆标签"""
        if hasattr(self.engine, 'structural_coordinator'):
            # 主观分形标签
            coord = self.engine.structural_coordinator.get_current_coordinate()
            if hasattr(self.engine, 'image_base'):
                card = self.engine.image_base.get_card_by_coordinate(coord)
                if card:
                    tags['subjective_card_id'] = card.id
                    tags['subjective_room'] = coord.as_tarot_code()
                    tags['subjective_major'] = coord.major

            # 六十四卦客观分类
            if hasattr(self.engine, 'objective_classifier'):
                tags['objective_room'] = self.engine.objective_classifier.classify(text)

            # 推理分形
            if hasattr(self.engine.structural_coordinator, '_infer_major_arcana'):
                reasoned = self.engine.structural_coordinator._infer_major_arcana(text)
                if reasoned:
                    tags['reasoned_card'] = reasoned

            # 随机分形
            if hasattr(self.engine.structural_coordinator, 'draw_random_card'):
                tags['random_card'] = self.engine.structural_coordinator.draw_random_card()
    
    def _induce_and_generate(self, analysis: dict, source_summary: str, full_text: str) -> dict:
        """
        归纳与生成阶段：
        1. 为分析报告中的新知识碎片打上双记忆标签
        2. 建立与已有记忆的交叉引用
        3. 将归纳产物写入LPS
        """
        result = {'new_entries': 0, 'cross_references': 0}

        # 获取当前过程状态用于过程绑定
        process_context = {}
        if hasattr(self.engine, 'structural_coordinator'):
            coord = self.engine.structural_coordinator.get_current_coordinate()
            process_context['subjective_room'] = coord.as_tarot_code()
            process_context['subjective_major'] = coord.major
            # 获取主观分形牌ID
            if hasattr(self.engine, 'image_base'):
                card = self.engine.image_base.get_card_by_coordinate(coord)
                if card:
                    process_context['subjective_card_id'] = card.id

        # 获取客观分类的参考信息
        objective_room = 0
        if hasattr(self.engine, 'objective_classifier'):
            objective_room = self.engine.objective_classifier.classify(source_summary)

        # 处理核心命题：将这些命题作为独立的知识碎片写入LPS
        for prop in analysis.get('core_propositions', []):
            tags = {
                'type': 'induced_knowledge',
                'source': 'document_ingestion',
                'doc_source': source_summary,
                'objective_room': objective_room,
                'knowledge_type': 'core_proposition',
                'ingested_at': time.time(),
            }
            # 添加过程绑定
            tags.update(process_context)

            # 添加三重标签
            if hasattr(self.engine, 'structural_coordinator'):
                # 推理分形
                if hasattr(self.engine.structural_coordinator, '_infer_major_arcana'):
                    reasoned = self.engine.structural_coordinator._infer_major_arcana(prop)
                    if reasoned:
                        tags['reasoned_card'] = reasoned
                # 随机分形
                if hasattr(self.engine.structural_coordinator, 'draw_random_card'):
                    tags['random_card'] = self.engine.structural_coordinator.draw_random_card()

            embedding = self.engine.lps.encoder.encode([prop])[0] if self.engine.lps.encoder else None
            node_id = self.engine.lps.add_if_new(prop, embedding, potency=0.8, tags=tags)
            if node_id:
                result['new_entries'] += 1

        # 处理新发现的实体：为每个新实体创建独立的知识条目
        new_entities = analysis.get('knowledge_relations', {}).get('new', [])
        for entity in new_entities:
            entity_text = f"新知识实体: {entity}"
            tags = {
                'type': 'induced_entity',
                'source': 'document_ingestion',
                'doc_source': source_summary,
                'objective_room': objective_room,
                'knowledge_type': 'new_entity',
                'ingested_at': time.time(),
            }
            tags.update(process_context)

            if hasattr(self.engine, 'structural_coordinator'):
                if hasattr(self.engine.structural_coordinator, '_infer_major_arcana'):
                    reasoned = self.engine.structural_coordinator._infer_major_arcana(entity)
                    if reasoned:
                        tags['reasoned_card'] = reasoned
                if hasattr(self.engine.structural_coordinator, 'draw_random_card'):
                    tags['random_card'] = self.engine.structural_coordinator.draw_random_card()

            embedding = self.engine.lps.encoder.encode([entity_text])[0] if self.engine.lps.encoder else None
            node_id = self.engine.lps.add_if_new(entity_text, embedding, potency=0.7, tags=tags)
            if node_id:
                result['new_entries'] += 1

        # 处理关键实体：为每个实体建立交叉引用索引
        for entity in analysis.get('key_entities', []):
            # 查询已有记忆中与这个实体相关的条目
            existing = self.engine.lps.query_by_tag(entity=entity, min_potency=0.3) if hasattr(self.engine.lps, 'query_by_tag') else []
            if existing:
                # 记录交叉引用（这里简化处理：增加一条关系记录）
                cross_ref_text = f"交叉引用：'{entity}' 在文档 '{source_summary}' 中被提及"
                tags = {
                    'type': 'cross_reference',
                    'source': 'document_ingestion',
                    'doc_source': source_summary,
                    'related_entity': entity,
                    'ingested_at': time.time(),
                }
                tags.update(process_context)
                embedding = self.engine.lps.encoder.encode([cross_ref_text])[0] if self.engine.lps.encoder else None
                self.engine.lps.add_if_new(cross_ref_text, embedding, potency=0.5, tags=tags)
                result['cross_references'] += 1

        return result
    
    def _finalize_ingestion(self, source_summary: str, stats: dict):
        """
        摄入收束阶段：记录摄入元信息，更新文档哈希索引（预留接口）。
        """
        # 记录摄入完成日志
        self.logger.info(f"文档摄入完成: {source_summary}, "
                         f"分块={stats['chunks_processed']}, "
                         f"三元组={stats['triplets_extracted']}, "
                         f"新知识={stats['new_knowledge_count']}, "
                         f"交叉引用={stats['cross_references_added']}")

        # 预留：更新文档哈希索引
        # 当前简化实现：仅记录日志，后续版本可扩展为 SQLite 或 JSON 索引
        # doc_hash = hashlib.md5(source_summary.encode()).hexdigest()
        # self.doc_index[doc_hash] = {'last_ingested': time.time(), 'entry_ids': [...]}
