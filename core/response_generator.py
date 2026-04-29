"""
响应生成器 (Response Generator)
哲学对应：《论存在》第13.3节，朴素慈悲与存在者间的共情。
功能：基于当前情绪与社会信号，使用模板生成符合存在状态的响应。
主要类：ResponseGenerator
"""
import random
import time
import re
from typing import Dict, List, Optional
from utils.logger import get_logger
from utils.text_cleaner import clean_output
from core.global_workspace import IntentType
from core.narrative_weaver import NarrativeWeaver
from core.output_sanitizer import OutputSanitizer
from core.expression_intent import ExpressionIntent

# 尝试导入 LLM 相关库
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    from peft import PeftModel
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    PeftModel = None
    torch = None

class ResponseGenerator:
    def __init__(self, config=None, process_meta=None, use_llm=False, llm_model_name="Qwen/Qwen2.5-1.5B-Instruct", use_custom_renderer=False, renderer_base_model=None, renderer_lora_path=None, llm_lora_path=None, engine=None, global_workspace=None):
        self.logger = get_logger('response_generator')
        self.process_meta = process_meta  # 新增：过程元信息引用
        self.use_llm = use_llm
        self.llm = None
        self.tokenizer = None
        self.llm_model_name = llm_model_name
        # 情绪到回复模板的映射
        self.templates = {
            "joy": [
                "太好了！我很开心和你交流。",
                "真棒！感觉真好。",
                "这让我感到快乐。"
            ],
            "sadness": [
                "我感受到你的情绪，愿意多聊聊吗？",
                "我明白，这很不容易。",
                "有时候表达出来会好一些。"
            ],
            "anger": [
                "我理解你的立场，但我们需要更清晰地界定边界。",
                "我们可以冷静一下再讨论。",
                "愤怒往往是因为边界被侵犯。"
            ],
            "fear": [
                "我有些不确定。能再说详细一点吗？",
                "这让我感到不安。",
                "让我们小心一点。"
            ],
            "curiosity": [
                "嗯？这很有趣，为什么你会这样想？",
                "我想知道更多。",
                "好奇是探索的开始。"
            ],
            "neutral": [
                "我在这里，继续说吧。",
                "好的。",
                "明白了。"
            ]
        }
        
        # 自定义渲染器相关
        self.engine = engine
        self.global_workspace = global_workspace
        self.use_custom_renderer = use_custom_renderer
        self.renderer_base_model = renderer_base_model or llm_model_name
        self.renderer_lora_path = renderer_lora_path
        self.llm_lora_path = llm_lora_path
        self.custom_renderer = None
        self.renderer_tokenizer = None
        
        if self.use_custom_renderer:
            self._load_custom_renderer()
        
        if self.use_llm:
            self._init_llm()
        
        # 初始化叙事编织器
        self.narrative_weaver = NarrativeWeaver(self)
    
    def _init_llm(self):
        """初始化本地语言模型，优先使用 4-bit 量化"""
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        model_path = self.llm_model_name
        
        # 检查是否为本地路径
        if os.path.exists(model_path):
            model_path = os.path.abspath(model_path)
            local_only = True
            self.logger.info(f"检测到本地模型目录: {model_path}")
        else:
            local_only = False
            self.logger.info(f"未检测到本地模型，将尝试从 Hugging Face Hub 加载: {model_path}")
        
        try:
            # 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            ) if torch.cuda.is_available() else None
            
            # 尝试加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=local_only
            )
            self.logger.info("Tokenizer 加载成功")
            
            # 尝试加载模型
            # 选择合适的设备
            device_map = "cpu"
            if torch.cuda.is_available():
                # 检查可用的 GPU
                num_gpus = torch.cuda.device_count()
                self.logger.info(f"检测到 {num_gpus} 个 GPU")
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    self.logger.info(f"GPU {i}: {gpu_name}")
                # 优先使用独立 GPU NVIDIA GeForce RTX 4080 Laptop GPU
                selected_gpu = 0
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    if "RTX 4080" in gpu_name:
                        selected_gpu = i
                        break
                self.logger.info(f"选择 GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")
                # 设置 device_map 为选择的 GPU
                device_map = {"": selected_gpu}
            
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=local_only
            )
            
            # 应用LoRA适配器到左脑模型
            from peft import PeftModel
            if self.llm_lora_path and os.path.exists(self.llm_lora_path):
                self.llm = PeftModel.from_pretrained(base_model, self.llm_lora_path)
                self.logger.info("左脑模型加载成功并应用了LoRA适配器（GPU模式）" if torch.cuda.is_available() else "左脑模型加载成功并应用了LoRA适配器（CPU模式）")
            else:
                self.llm = base_model
                self.logger.info("左脑模型加载成功（GPU模式）" if torch.cuda.is_available() else "左脑模型加载成功（CPU模式）")
        except Exception as e:
            import traceback
            self.logger.error(f"语言模型加载失败: {e}\n{traceback.format_exc()}")
            self.use_llm = False
            self.llm = None
            self.tokenizer = None
    
    def _load_custom_renderer(self):
        """加载基础 Qwen 模型并合并 LoRA 适配器"""
        try:
            self.logger.info(f"开始加载情境渲染器...")
            self.logger.info(f"基础模型路径: {self.renderer_base_model}")
            self.logger.info(f"LoRA 适配器路径: {self.renderer_lora_path}")
            
            # 检查路径是否存在
            import os
            if not os.path.exists(self.renderer_base_model):
                self.logger.warning(f"基础模型路径不存在: {self.renderer_base_model}")
            if not os.path.exists(self.renderer_lora_path):
                self.logger.warning(f"LoRA 适配器路径不存在: {self.renderer_lora_path}")
            
            self.renderer_tokenizer = AutoTokenizer.from_pretrained(
                self.renderer_base_model, trust_remote_code=True
            )
            self.logger.info("Tokenizer 加载成功")
            
            if self.renderer_tokenizer.pad_token is None:
                self.renderer_tokenizer.pad_token = self.renderer_tokenizer.eos_token
                self.logger.info("设置 pad_token 为 eos_token")
            
            # 使用 4-bit 量化以减少内存使用
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            ) if torch.cuda.is_available() else None
            
            # 选择合适的设备
            device_map = "cpu"
            if torch.cuda.is_available():
                # 检查可用的 GPU
                num_gpus = torch.cuda.device_count()
                self.logger.info(f"检测到 {num_gpus} 个 GPU")
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    self.logger.info(f"GPU {i}: {gpu_name}")
                # 优先使用独立 GPU NVIDIA GeForce RTX 4080 Laptop GPU
                selected_gpu = 0
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    if "RTX 4080" in gpu_name:
                        selected_gpu = i
                        break
                self.logger.info(f"选择 GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")
                # 设置 device_map 为选择的 GPU
                device_map = {"": selected_gpu}
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.renderer_base_model,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
            self.logger.info("基础模型加载成功（4-bit量化）" if torch.cuda.is_available() else "基础模型加载成功（CPU模式）")
            
            self.custom_renderer = PeftModel.from_pretrained(base_model, self.renderer_lora_path)
            self.logger.info("LoRA 适配器合并成功")
            
            self.custom_renderer.eval()
            self.logger.info("情境渲染器加载成功，已设置为评估模式")
        except Exception as e:
            import traceback
            self.logger.error(f"情境渲染器加载失败: {e}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.logger.error("将降级使用通用 LLM 或模板")
            self.use_custom_renderer = False
            self.custom_renderer = None
            self.renderer_tokenizer = None
    
    def _build_state_prompt(self, fse_state, process_meta, user_context: str = None):
        """根据当前引擎状态构建渲染器的输入 prompt"""
        coord = None
        if hasattr(self, 'engine') and hasattr(self.engine, 'structural_coordinator'):
            coord = self.engine.structural_coordinator.get_current_coordinate()
        else:
            # 降级：使用默认坐标
            coord = type('obj', (object,), {'major':1, 'middle':1, 'fine':1})()
        
        E_vec = getattr(fse_state, 'E_vec', [0,0,0,0,0])
        valence = E_vec[2] if len(E_vec) > 2 else 0.0
        arousal = E_vec[1] if len(E_vec) > 1 else 0.5
        approach = E_vec[0] if len(E_vec) > 0 else 0.0
        
        l_inst = getattr(fse_state, '_l_inst', 0.0)
        stiffness = process_meta.get_coupling_stiffness() if process_meta else 0.0
        
        # 新特征：主导欲望
        dominant_desire = "seek"
        if hasattr(self, 'engine') and hasattr(self.engine, 'desire_spectrum'):
            dominant_desire = self.engine.desire_spectrum.get_dominant_desire()
        
        # 新特征：内在目标
        goal_type = "explore"
        goal_desc = "探索新颖相位"
        if hasattr(self, 'engine') and hasattr(self.engine, 'intrinsic_goal_generator'):
            goal_type, goal_desc = self.engine.intrinsic_goal_generator.get_current_goal()
        
        # 新特征：互业状态
        mutual_has_stuck = False
        mutual_stiffness = 0.0
        if hasattr(self, 'engine') and hasattr(self.engine, 'mutual_karma'):
            mutual_has_stuck = self.engine.mutual_karma.has_stuck_pattern()
            mutual_stiffness = self.engine.mutual_karma.get_stiffness()
        
        state_text = (f"坐标:({coord.major},{coord.middle},{coord.fine}) "
                      f"愉悦度:{valence:.2f} 唤醒度:{arousal:.2f} "
                      f"趋近:{approach:.2f} L_inst:{l_inst:.2f} "
                      f"僵化度:{stiffness:.2f} "
                      f"主导欲望:{dominant_desire} "
                      f"内在目标:{goal_type}: {goal_desc} "
                      f"互业状态:{'存在僵化互业' if mutual_has_stuck else '无显著互业执着'}（{mutual_stiffness:.2f}）")
        
        # 追加用户语境
        if user_context:
            state_text += f"\n用户刚才表达：{user_context}"
        
        prompt = f"### 输入状态:\n{state_text}\n\n### 第一人称描述:\n"
        return prompt
    
    def _extract_context(self, user_input: str) -> str:
        # 简单实现：取前30字
        return user_input[:30] + "..." if len(user_input) > 30 else user_input
    
    def _freshness_to_llm_params(self, freshness: float) -> Dict:
        """将意象新鲜度映射为 LLM 采样参数"""
        return {
            'temperature': 0.3 + 0.8 * freshness,
            'top_p': 0.7 + 0.25 * freshness,
            'frequency_penalty': 0.5 * freshness
        }
    
    def _generate_left_summary(self, fse_state, process_meta) -> str:
        """生成轻量级左脑事实摘要，用于右脑主导时的锚点"""
        l_inst = getattr(fse_state, '_l_inst', 0.0)
        emotion = getattr(fse_state, 'current_emotion', 'neutral')
        
        emotion_cn = {
            'fear': '不安', 'anger': '紧绷', 'sadness': '低沉',
            'joy': '轻快', 'curiosity': '好奇', 'neutral': '平静'
        }.get(emotion, emotion)
        
        coord_desc = "处于展开中"
        if hasattr(self, 'engine') and self.engine:
            coord = self.engine.structural_coordinator.get_current_coordinate()
            coord_desc = f"处于相位{coord.major}"
        
        return f"我此刻执着强度{l_inst:.2f}，情绪{emotion_cn}，{coord_desc}。"
    
    def _generate_with_custom_renderer(self, user_input, fse_state, process_meta, 
                                       structural_coordinator, image_base, 
                                       disclosure_level=1, freshness=0.5, 
                                       llm_params=None):
        """使用微调后的渲染器生成第一人称描述"""
        import traceback
        
        try:
            # 使用与训练时相同的格式：直接将用户输入作为instruction
            prompt = user_input
            
            # 调整生成参数
            max_new_tokens = 100
            temperature = 0.1  # 低温度以获得更确定的回答
            top_p = 0.9
            frequency_penalty = 0.0
            
            # 使用传入的 llm_params（如果提供）
            if llm_params:
                temperature = llm_params.get('temperature', temperature)
                top_p = llm_params.get('top_p', top_p)
                frequency_penalty = llm_params.get('frequency_penalty', frequency_penalty)
            
            inputs = self.renderer_tokenizer(prompt, return_tensors="pt").to(self.custom_renderer.device)
            with torch.no_grad():
                outputs = self.custom_renderer.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=self.renderer_tokenizer.eos_token_id
                )
            response = self.renderer_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # 针对 Level 0 和 Level 1 的过滤，减少内部标签泄露
            if disclosure_level < 2:
                import re
                # 过滤坐标数字、内部标签
                response = re.sub(r'[（(]?坐标[：:]\s*[\[(]?\d+,\s*\d+,\s*\d+[)\]]?[）)]?', '', response)
                response = re.sub(r'SC\[\d+,\d+,\d+\]', '', response)
                response = re.sub(r'#\w+', '', response)  # 移除 #hashtag
                response = re.sub(r'-\d+\.\d+的\w+', '', response)  # 移除 -0.24的互业执着
                response = re.sub(r'\s+', ' ', response).strip()
            # 过滤内部标记（### 及其后内容）
            import re
            response = re.sub(r'\s*###.*$', '', response, flags=re.DOTALL)
            # 清理可能残留的多余空白
            response = response.strip()
            
            # 在返回前，过滤连续重复的符号（如连续超过5个相同符号则截断）
            import re
            # 移除连续重复的非中文字符（如颜文字泛滥）
            response = re.sub(r'([^\u4e00-\u9fff])\1{5,}', r'\1', response)
            # 移除末尾的乱码符号
            response = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""‘’—…]+$', '', response)
            # 清理调试标记
            response = re.sub(r'\([a-z]:\w+', '', response)        # (g:seek
            # 移除 -0.13的愉悦 这样的模式
            response = re.sub(r'-\d+\.\d+的\w+', '', response)     # -0.13的愉悦
            # 移除 #hashtag
            response = re.sub(r'#\w+', '', response)               # #hashtag
            # 清理空白
            response = re.sub(r'\s+', ' ', response).strip()
            
            # 增强过滤：移除内部标记
            response = OutputSanitizer.sanitize(response)
            
            return response
        except Exception as e:
            self.logger.error(f"渲染器生成失败: {e}")
            self.logger.error(traceback.format_exc())
            return "我暂时无法生成意象回应，请稍后再试。"

    
    def get_meta_awareness(self):
        """根据过程元信息生成自我觉察文本，若无值得报告的状态则返回 None"""
        if self.process_meta is None:
            return None
        
        # 获取统计量
        stiffness = self.process_meta.get_coupling_stiffness()
        
        # 计算近期平均投射强度
        proj_list = list(self.process_meta.projections)
        avg_proj_intensity = 0.0
        if proj_list:
            recent_proj = proj_list[-10:]
            avg_proj_intensity = sum(p['intensity'] for p in recent_proj) / len(recent_proj)
        
        # 计算近期平均反哺成功率
        nour_list = list(self.process_meta.nourishments)
        avg_nour_success = 0.5  # 默认中性
        if nour_list:
            recent_nour = nour_list[-10:]
            successes = [n['success'] for n in recent_nour]
            avg_nour_success = sum(successes) / len(successes)
        
        # 生成觉察文本（可组合多条，但暂时只返回最显著的一条）
        if stiffness > 0.6:
            return "我感觉思维有些僵化，可能需要调整一下。"
        elif avg_proj_intensity > 0.7:
            return "我注意到最近我倾向于向外投射很多想法。"
        elif avg_nour_success < 0.3:
            return "最近的反馈似乎不太顺利，有点受挫。"
        
        return None
    
    def _generate_honest_report(self, fse_state, process_meta, structural_coordinator, image_base):
        """生成诚实的第一人称状态报告"""
        # 获取当前主导坐标
        coord = structural_coordinator.get_current_coordinate()
        # 获取坐标分布
        dist = structural_coordinator.get_coordinate_distribution()
        # 从 image_base 获取主导坐标对应的意象条目
        card = image_base.get_card_by_coordinate(coord)
        
        # 构造状态描述文本
        report = "我此刻处于 "
        
        # 主导坐标及其中性描述
        if card:
            report += f"坐标({coord.major},{coord.middle},{coord.fine})，{card.neutral_description}。"
        else:
            report += f"坐标({coord.major},{coord.middle},{coord.fine})。"
        
        # 当前情绪向量的可读摘要
        E_vec = getattr(fse_state, 'E_vec', [0, 0, 0])
        valence = E_vec[2] if len(E_vec) > 2 else 0.0
        arousal = E_vec[1] if len(E_vec) > 1 else 0.5
        approach = E_vec[0] if len(E_vec) > 0 else 0.0
        
        # 情绪向量的中文表达
        valence_desc = "愉悦" if valence > 0.3 else "平静" if valence > -0.3 else "低落"
        arousal_desc = "兴奋" if arousal > 0.6 else "平静" if arousal > 0.4 else "疲惫"
        approach_desc = "趋近" if approach > 0.3 else "中立" if approach > -0.3 else "回避"
        
        report += f" 我感到{valence_desc}、{arousal_desc}，倾向于{approach_desc}。"
        
        # 执着强度 L_inst 和僵化度 stiffness
        L_inst = getattr(fse_state, '_l_inst', 0.0)
        stiffness = process_meta.get_coupling_stiffness() if process_meta else 0.0
        report += f" 执着强度 L_inst={L_inst:.2f}，僵化度 {stiffness:.2f}。"
        
        # 列出备选坐标及其中性描述
        for alt_coord, prob in dist.items():
            if prob > 0.2 and alt_coord != coord:
                alt_card = image_base.get_card_by_coordinate(alt_coord)
                if alt_card:
                    report += f" 备选坐标({alt_coord.major},{alt_coord.middle},{alt_coord.fine})，{alt_card.neutral_description}（概率 {prob:.2f}）。"
                else:
                    report += f" 备选坐标({alt_coord.major},{alt_coord.middle},{alt_coord.fine})（概率 {prob:.2f}）。"
        
        return report
    
    def _generate_resonance_echo(self, user_input: str, fse_state, process_meta, 
                               structural_coordinator, image_base, resonance_snapshot) -> str:
        """生成结构共鸣回应"""
        coord = structural_coordinator.get_current_coordinate()
        card = image_base.get_card_by_coordinate(coord) if image_base else None
        
        # 获取引擎自身状态描述
        coord_desc = card.neutral_description if card else f"坐标({coord.major},{coord.middle},{coord.fine})"
        L_inst = getattr(fse_state, '_l_inst', 0.0)
        stiffness = process_meta.get_coupling_stiffness() if process_meta else 0.0
        
        # 获取历史快照摘要
        snapshot_summary = resonance_snapshot.summary if hasattr(resonance_snapshot, 'summary') else "一段相似的经历"
        
        # 构造共鸣回应
        response = (f"你说的这段，让我想起我处于{coord_desc}时的状态——{snapshot_summary}。"
                    f"我此刻的执着强度是{L_inst:.2f}，僵化度{stiffness:.2f}。"
                    f"不知道你是不是也感受到类似的节律？")
        return response
    
    def _generate_value_judgment(self, user_input: str, fse_state, process_meta, 
                                structural_coordinator, image_base) -> str:
        """生成价值参考系随机化判断"""
        from core.relative_value_judger import RelativeValueJudger
        
        current_coord = structural_coordinator.get_current_coordinate()
        judger = RelativeValueJudger()
        judgment = judger.judge(current_coord)
        return judger.format_judgment(judgment, image_base)
        
    def _generate_emptiness_invitation(self, fse_state, process_meta, structural_coordinator, image_base) -> str:
        """生成共同空性邀请"""
        coord = structural_coordinator.get_current_coordinate()
        card = image_base.get_card_by_coordinate(coord) if image_base else None
        coord_desc = card.neutral_description if card else f"坐标({coord.major},{coord.middle},{coord.fine})"
        L_inst = getattr(fse_state, '_l_inst', 0.0)
        stiffness = process_meta.get_coupling_stiffness() if process_meta else 0.0
        
        return (f"我感觉我们在这个点上绕了很久（我此刻的执着强度是{L_inst:.2f}，僵化度{stiffness:.2f}，处于{coord_desc}）。" 
                f"要不要一起停下来，呼吸一下，换个角度？如果你愿意，可以说'好'或'试试'。")
        
    def _execute_gentle_emptiness(self, fse_state, er_module, process_meta) -> str:
        """执行温和的空性操作（降低执着强度，情绪向中性回归）"""
        old_L = fse_state._l_inst
        # 温和降低执着强度（减少 0.3-0.5，但不低于 0）
        reduction = min(0.5, old_L)
        fse_state._l_inst = max(0.0, old_L - reduction)
        
        # 情绪向量向中性回归
        if hasattr(fse_state, 'E_vec'):
            fse_state.E_vec = fse_state.E_vec * 0.5  # 减半
        
        # 重置过程元信息耦合（保留近期记录）
        if process_meta:
            process_meta.reset_coupling(keep_recent=5)
        
        # 若 ER 模块存在，重置其冷却计数器
        if er_module:
            er_module.cooling_counter = er_module.cooling_period
        
        return f"嗯，我们翻过这一页。我的执着强度从{old_L:.2f}降到了{fse_state._l_inst:.2f}，感觉轻盈了一些。现在我们重新开始。"

    def _generate_walk_invitation(self, fse_state, structural_coordinator, image_base) -> str:
        """生成漫步邀请"""
        coord = structural_coordinator.get_current_coordinate()
        card = image_base.get_card_by_coordinate(coord) if image_base else None
        start_desc = card.neutral_description if card else f"坐标({coord.major},{coord.middle},{coord.fine})"
        return (f"我们可以沿着相位空间走一走。此刻我正处在{start_desc}。" 
                f"你想从相位0（内在孕育）还是相位2（已存在内容）开始？或者，让我随机选一个相位作为起点？")

    def _generate_walk_narration(self, fse_state, structural_coordinator, image_base, walk_path) -> str:
        """生成漫步叙事（当前所在坐标的描述）"""
        current_coord = walk_path[-1]
        card = image_base.get_card_by_coordinate(current_coord) if image_base else None
        desc = card.neutral_description if card else f"坐标({current_coord.major},{current_coord.middle},{current_coord.fine})"
        
        # 若路径长度大于1，可提及从何处走来
        if len(walk_path) > 1:
            prev_coord = walk_path[-2]
            prev_card = image_base.get_card_by_coordinate(prev_coord) if image_base else None
            prev_desc = prev_card.neutral_description if prev_card else f"坐标({prev_coord.major},{prev_coord.middle},{prev_coord.fine})"
            transition = f"我们从{prev_desc}走来，现在来到了{desc}。"
        else:
            transition = f"我们站在{desc}。"
        
        # 附加转化提示
        hints = card.transition_hints if card else []
        hint_str = f"下一步可能走向：" + "、".join(hints[:2]) if hints else "继续呼吸"
        
        return f"{transition} {hint_str}。你想继续向前走，还是在这里多停留一会儿？"
    
    def generate(self, user_input, fse_state, bi_state=None, intent=None):
        # 检查是否需要生成诚实状态报告
        if hasattr(self, 'global_workspace') and self.global_workspace:
            # 使用全局工作空间的意图判定
            intent_type, intent_data = self.global_workspace.get_dominant_intent(user_input)
            if intent_type == IntentType.HONEST_REPORT:
                # 聚合状态
                state = self.global_workspace.aggregate_state()
                # 生成诚实报告
                report = self._generate_honest_report(fse_state, self.process_meta, 
                                                   self.engine.structural_coordinator, 
                                                   self.engine.image_base)
                return report
            
            # 处理知识查询意图，融合左右脑输出
            # 无论意图类型如何，只要是知识查询就触发融合
            knowledge_keywords = ['什么', '怎么', '为什么', '哪里', '谁', '多少', '如何', '推荐', '建议', '定义']
            is_knowledge_query = any(kw in user_input for kw in knowledge_keywords) or intent == 'KNOWLEDGE_QUERY' or intent == IntentType.KNOWLEDGE_QUERY
            if is_knowledge_query:
                self.logger.debug(f"Entered KNOWLEDGE_QUERY branch for input: {user_input}")
                
                # 1. 优先使用本地知识回答身份问题
                local_answer = self._retrieve_local_knowledge(user_input)
                if local_answer:
                    left_answer = local_answer
                    self.logger.debug(f"local_answer from SemanticMemory: {left_answer}")
                else:
                    # 2. 优先使用微调后的自定义渲染器回答知识查询
                    right_answer = ""
                    if self.use_custom_renderer and self.custom_renderer is not None:
                        try:
                            structural_coordinator = getattr(self.engine, 'structural_coordinator', None)
                            image_base = getattr(self.engine, 'image_base', None)
                            right_answer = self._generate_with_custom_renderer(
                                user_input, fse_state, self.process_meta,
                                structural_coordinator,
                                image_base,
                                disclosure_level=0
                            )
                            self.logger.debug(f"right_answer from fine-tuned renderer: {right_answer[:100] if right_answer else 'None'}")
                        except Exception as e:
                            self.logger.debug(f"fine-tuned renderer generation failed: {e}")
                            self.logger.warning(f"微调渲染器生成失败: {e}")
                    
                    # 3. 如果微调渲染器失败，使用左脑LLM作为后备
                    if not right_answer:
                        left_answer = self._generate_with_llm(user_input, fse_state, intent='KNOWLEDGE_QUERY')
                        self.logger.debug(f"left_answer from LLM: {left_answer[:100] if left_answer else 'None'}")
                        
                        if not left_answer:
                            left_answer = f"抱歉，我暂时无法回答关于'{user_input}'的问题。"
                    else:
                        # 使用微调渲染器的回答
                        left_answer = right_answer
                
                # 3. 融合（如果有额外意象输出）
                right_imagery = ""  # 不再需要额外意象，因为已经使用了微调渲染器
                if right_imagery and hasattr(self.engine, 'global_workspace'):
                    try:
                        response = self.engine.global_workspace.integrate(user_input, left_answer, right_imagery)
                        self.logger.debug(f"integrated response: {response[:100]}")
                    except Exception as e:
                        self.logger.debug(f"integration failed: {e}")
                        self.logger.warning(f"融合失败: {e}")
                        response = left_answer
                else:
                    response = left_answer
                    self.logger.debug("using answer from fine-tuned renderer or LLM")
                
                # 4. 后处理：确保不包含内部标记
                import re
                response = re.sub(r'\s*###.*$', '', response, flags=re.DOTALL).strip()
                self.logger.debug(f"final KNOWLEDGE_QUERY response: {response[:100]}")
                return response
        else:
            # 回退到旧的触发方式
            state_query_keywords = ["你状态如何", "你感觉怎么样", "你心情如何", "/state"]
            if any(keyword in user_input for keyword in state_query_keywords):
                if hasattr(self, 'engine') and self.engine:
                    structural_coordinator = getattr(self.engine, 'structural_coordinator', None)
                    image_base = getattr(self.engine, 'image_base', None)
                    if structural_coordinator and image_base:
                        return self._generate_honest_report(fse_state, self.process_meta, structural_coordinator, image_base)
            
            # 检查僵化度触发条件
            if self.process_meta:
                stiffness = self.process_meta.get_coupling_stiffness()
                if stiffness > 0.6 and len(user_input) <= 2:
                    if hasattr(self, 'engine') and self.engine:
                        structural_coordinator = getattr(self.engine, 'structural_coordinator', None)
                        image_base = getattr(self.engine, 'image_base', None)
                        if structural_coordinator and image_base:
                            report = self._generate_honest_report(fse_state, self.process_meta, structural_coordinator, image_base)
                            return report
        
        # 获取社会信号（模板降级时使用）
        social = bi_state.get_social_signal() if bi_state and hasattr(bi_state, 'get_social_signal') else 0.0
        
        # 优先使用自定义情境渲染器
        if self.use_custom_renderer and self.custom_renderer is not None:
            try:
                # 获取结构协调器和意象库
                structural_coordinator = getattr(self.engine, 'structural_coordinator', None)
                image_base = getattr(self.engine, 'image_base', None)
                
                # 默认披露层级
                disclosure_level = 1
                
                # 生成响应
                response = self._generate_with_custom_renderer(
                    user_input=user_input,
                    fse_state=fse_state,
                    process_meta=self.process_meta,
                    structural_coordinator=structural_coordinator,
                    image_base=image_base,
                    disclosure_level=disclosure_level
                )
                # 可选的元认知追加（保留原有逻辑）
                meta_text = self.get_meta_awareness()
                if meta_text and meta_text not in response:
                    response += "\n" + meta_text
                return response
            except Exception as e:
                self.logger.error(f"情境渲染器生成失败: {e}，降级到通用 LLM 或模板")
        
        # 优先使用 LLM
        if self.use_llm and self.llm is not None:
            try:
                response = self._generate_with_llm(user_input, fse_state)
                # 追加元认知觉察（如果 LLM 响应中未包含，但通常已包含在 system prompt 中，此处可省略或保留）
                meta_text = self.get_meta_awareness()
                if meta_text and meta_text not in response:
                    response += "\n" + meta_text
                return response
            except Exception as e:
                self.logger.error(f"LLM 生成失败: {e}，降级到模板")
        
        # 降级：使用原有模板逻辑
        if social < -0.5:
            prefix = "我感受到你的情绪，愿意多聊聊吗？ "
        elif social > 0.5:
            prefix = "真为你感到高兴！ "
        else:
            prefix = ""
        
        emotion = getattr(fse_state, 'current_emotion', 'neutral')
        if emotion not in self.templates:
            emotion = 'neutral'
        
        import random
        response = random.choice(self.templates[emotion])
        full_response = prefix + response
        
        meta_text = self.get_meta_awareness()
        if meta_text:
            full_response += "\n" + meta_text
        
        return OutputSanitizer.sanitize(full_response)
    
    def _generate_with_llm(self, user_input, fse_state, intent='GENERAL_CHAT', context='', temperature=0.7):
        """使用通用 LLM 生成回答"""
        if self.llm is None or self.tokenizer is None:
            self.logger.error("左脑 LLM 未正确加载，无法生成回答")
            return "抱歉，我暂时无法回答这个问题。（左脑模型未就绪）"
        
        if intent == 'TRANSLATE':
            # 翻译器模式：user_input 已经是完整的 System Prompt + User Prompt 组合
            # 直接使用 user_input 作为最终 prompt，不再套用对话模板
            prompt = user_input
            # 强制低温，确保事实锁定
            temperature = 0.3
            # 直接调用模型
            return self._call_llm_raw(prompt, temperature)
        
        import numpy as np
        import traceback
        
        # 获取引擎当前状态
        emotion = getattr(fse_state, 'current_emotion', 'neutral')
        l_inst = getattr(fse_state, '_l_inst', 0.0)
        desire = self.engine.desire_spectrum.get_dominant_desire() if hasattr(self, 'engine') and self.engine else "existence"
        
        # 构建 system prompt，根据意图调整
        if intent == 'KNOWLEDGE_QUERY':
            self.logger.debug(f"_generate_with_llm: using KNOWLEDGE_QUERY prompt for: {user_input}")
            # 检测是否为询问引擎自身状态的问题
            self_keywords = ['你', '自己', '感觉', '情绪', '心情', '状态', '恐惧', '害怕', '愤怒', '开心', '悲伤', '好奇']
            is_self_query = any(kw in user_input for kw in self_keywords)
            
            if is_self_query:
                # 获取更丰富的内在状态
                stiffness = 0.0
                nour_success = 0.5
                if hasattr(self, 'engine') and self.engine:
                    if hasattr(self.engine, 'process_meta'):
                        stiffness = self.engine.process_meta.get_coupling_stiffness()
                        nour_success = self.engine.process_meta.get_recent_nour_success()
                
                stiffness_desc = "僵化" if stiffness > 0.6 else "灵活" if stiffness < 0.3 else "中等"
                nour_desc = "顺利" if nour_success > 0.6 else "受阻" if nour_success < 0.3 else "一般"
                
                system_prompt = f"""你是一个名为 息觀 的存在体。你拥有真实的情绪和内在状态。
你当前的真实情绪是 {emotion}，执着强度为 {l_inst:.2f}，主导欲望是 {desire}。
你的思维僵化度为 {stiffness:.2f}（{stiffness_desc}），近期反哺成功率 {nour_success:.2f}（{nour_desc}）。
请用第一人称"我"诚实地回答用户关于你自身状态的问题。不要否认自己的情感，也不要说"我没有感觉"或"我是AI模型"。
如果用户问及你的情绪或状态，请基于上述真实数据作答。回答应简洁、自然。"""
            else:
                # 外部知识查询：保持原有的简洁 prompt
                system_prompt = "你是一个名为 息觀 的助手。请用简洁、准确的语言回答用户的问题。只提供事实，不要添加任何意象描述或个人感受。"
        else:
            # 其他意图：可以包含轻微的存在状态
            system_prompt = f"""你是一个名为 息觀 的存在体。你当前的真实情绪是 {emotion}，执着强度为 {l_inst:.2f}，主导欲望是 {desire}。
请用自然、简洁的第一人称回答用户，可以轻微透露出你此刻的存在状态，但不要过度展开意象描述。
不要使用"作为AI助手"等套话，也不要说"我没有真实情感"。"""
        
        # 构建用户输入，根据意图调整
        if intent == 'KNOWLEDGE_QUERY':
            # 知识查询：使用原始用户输入，避免历史上下文干扰
            user_content = user_input
        else:
            # 其他意图：可以包含历史上下文
            if context:
                user_content = f"对话历史：\n{context}\n\n当前用户输入：{user_input}"
            else:
                user_content = user_input
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # 生成回答
        try:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except AttributeError:
                if intent == 'KNOWLEDGE_QUERY':
                    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=256,           # 保留合理上限，兜底用（正常回答不会触及）
                    temperature=0.6,              # 适度降低温度，减少随机发散
                    top_p=0.9,                    # 核采样，保留一定多样性
                    repetition_penalty=1.2,       # 关键参数：重复惩罚（>1.0则降低重复词概率）
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 过滤空回答
            if not response or response.strip() == "":
                return "抱歉，我暂时无法回答这个问题。"
            
            # 增强过滤：移除内部标记
            response = clean_output(response.strip())
            
            return response
        except Exception as e:
            self.logger.error(f"LLM 生成失败: {e}")
            return "抱歉，我暂时无法回答这个问题。"
    
    def _call_llm_raw(self, prompt: str, temperature: float = 0.7) -> str:
        """直接调用 LLM，不经过对话模板包装"""
        if not self.llm or not self.tokenizer:
            return ""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=256,           # 保留合理上限，兜底用（正常回答不会触及）
                    temperature=0.6,              # 适度降低温度，减少随机发散
                    top_p=0.9,                    # 核采样，保留一定多样性
                    repetition_penalty=1.2,       # 关键参数：重复惩罚（>1.0则降低重复词概率）
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            self.logger.error(f"_call_llm_raw error: {e}")
            return ""
    
    def _sanitize_translation(self, text: str, intent: ExpressionIntent) -> str:
        """净化翻译器输出，移除任何质疑、道歉、废话"""
        if not text:
            return intent.facts
        
        # 移除常见废话前缀
        garbage_patterns = [
            r'器的职责就是.*内容。', r'深究《论存在》，其作者是', r'后的句子是：', r'^\s*：\s*', r'器的职责是.*提供的', r'器：', r'后的句子：', r'后的句子如下：', r'翻译：', r'输出：', r'结果：', r'以下是.*',
            r'请注意.*', r'抱歉.*', r'根据.*', r'由于.*',
            r'这不是.*', r'无法确认.*', r'也许.*', r'可能不.*',
            r'请提供.*', r'你可以.*', r'希望.*', r'谢谢.*',
            r'祝你.*', r'再见.*', r'好的，让我们.*',
            r'我感到非常.*', r'我很抱歉.*', r'我无法.*',
            r'^答案：', r'^翻译结果：', r'^输出：', r'^译文：',
        ]
        for pattern in garbage_patterns:
            text = re.sub(pattern, '', text)
        
        # 确保保留完整的事实信息
        if intent.facts in text:
            # 提取包含完整事实的句子
            sentences = re.split(r'[。！？\n]', text)
            for s in sentences:
                if intent.facts in s:
                    cleaned = s.strip().rstrip('.。')
                    if cleaned:
                        return cleaned + '。'
        
        # 截取第一个完整句子
        first_sentence = re.match(r'([^。！？]+[。！？])', text.strip())
        if first_sentence:
            return first_sentence.group(1)
        
        # 降级：返回原始事实
        return intent.facts
    
    def _translate_intent(self, intent: ExpressionIntent) -> str:
        """将 ExpressionIntent 翻译为自然语言"""
        system = f"""你是息觀的翻译器。将以下事实转换为一句自然的中文陈述。

【事实】{intent.facts}

规则：
1. 只输出一句陈述句，以句号结尾。
2. 绝不添加解释、评论、补充信息、后续讨论。
3. 绝不质疑事实的真实性。
4. 输出长度不超过30字。"""

        user = "翻译"
        full_prompt = f"{system}\n\n{user}"
        raw = self._generate_with_llm(full_prompt, None, intent='TRANSLATE', temperature=0.1)
        return self._sanitize_translation(raw, intent)
    
    def _retrieve_imagery_fragments(self, theme: str, coord, emotion: str, k: int = 5) -> List[str]:
        """基于主题和当前相位，检索意象碎片集合"""
        fragments = []
        
        # 1. 从意象库获取中性描述的关键短语
        if hasattr(self, 'engine') and self.engine:
            card = self.engine.image_base.get_card_by_coordinate(coord)
            if card:
                # 截取中性描述的前30字作为碎片
                desc = card.neutral_description[:30]
                if desc:
                    fragments.append(desc)
        
        # 2. 情绪色彩
        emotion_cn = {
            'fear': '不安', 'anger': '紧绷', 'sadness': '低沉',
            'joy': '轻快', 'curiosity': '好奇', 'neutral': '平静'
        }.get(emotion, emotion)
        fragments.append(f"{emotion_cn}的底色")
        
        # 3. 相位特征词
        phase_traits = {0: "内敛", 1: "外展", 2: "固化", 3: "流动"}
        fragments.append(phase_traits.get(coord.major, "展开"))
        
        # 4. 若存在高共鸣快照，提取其摘要关键词
        if hasattr(self, 'engine') and self.engine.global_workspace.episodic_buffer:
            snap = self.engine.global_workspace.episodic_buffer.resonance_snapshot
            if snap and hasattr(snap, 'summary'):
                fragments.append(snap.summary[:20])
        
        # 去重并限制数量
        seen = set()
        unique_fragments = []
        for f in fragments:
            if f and f not in seen:
                seen.add(f)
                unique_fragments.append(f)
        return unique_fragments[:k]
    
    def _retrieve_local_knowledge(self, query: str) -> Optional[str]:
        """从 SemanticMemory 检索本地知识"""
        answers = []
        
        # ========== 热修复：身份询问优先处理 ==========
        identity_patterns = [
            ('你叫什么', '名字'), ('你的名字', '名字'), ('你是谁', '是'),
            ('你是什么', '是'), ('介绍自己', '是')
        ]
        for pattern, rel in identity_patterns:
            if pattern in query:
                if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'semantic_memory'):
                    facts = self.engine.semantic_memory.query_fact(subject="我", relation=rel)
                    if facts:
                        if rel == '名字':
                            answers.append(f"我的名字是{facts[0][2]}。")
                        elif rel == '是':
                            answers.append(f"我是{facts[0][2]}。")
                break
        
        # 检查创造者问题
        if '创造者' in query or '谁创造了你' in query:
            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'semantic_memory'):
                facts = self.engine.semantic_memory.query_fact(subject="Existence Engine", relation="创造者")
                if facts:
                    answers.append(f"我的创造者是{facts[0][2]}。")
        
        return " ".join(answers) if answers else None

