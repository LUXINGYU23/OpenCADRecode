#!/usr/bin/env python3
"""
CAD-Recode 模型定义模块
包含点云编码器和CAD-Recode多模态模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    Qwen3ForCausalLM,  
    Qwen2ForCausalLM,  
    AutoModel,
    AutoTokenizer
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# 导入Qwen2Model用于兼容性
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
except ImportError:
    print("Warning: Could not import Qwen2Model, falling back to AutoModel")
    Qwen2Model = None

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
except ImportError:
    print("Warning: Could not import Qwen3Model, using Qwen3ForCausalLM.model instead")
    Qwen3Model = None

from utils import NUM_POINT_TOKENS


class FourierPointEncoder(nn.Module):
    """傅里叶点云编码器 - 与官方demo.ipynb完全对齐"""
    def __init__(self, hidden_size: int):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (batch_size, num_points, 3)
        Returns:
            encoded_points: (batch_size, num_points, hidden_size)
        """
        if self.projection.weight.dtype != points.dtype:
            points = points.to(self.projection.weight.dtype)
        x = points
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        x = self.projection(x)
        return x


class CADRecode(Qwen3ForCausalLM):
    
    def __init__(self, config):
        # 🚨 关键修复：按照官方demo的初始化方式
        # 首先调用父类初始化
        super().__init__(config)
        
        # 保存配置信息
        self._hidden_size = config.hidden_size
        self.num_point_tokens = NUM_POINT_TOKENS
        
        # 🚨 关键修复：按照官方demo的方式创建点云编码器
        # 官方demo：torch.set_default_dtype(torch.float32) -> 创建编码器 -> torch.set_default_dtype(torch.bfloat16)
        current_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)  # 设置为bfloat16，与官方demo一致

    def enable_input_require_grads(self):
        super().enable_input_require_grads()
        self.point_encoder.requires_grad_(True)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        point_cloud: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_key_values is None or past_key_values.get_seq_length() == 0
        if past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0):
            assert inputs_embeds is None, "不能同时提供inputs_embeds和point_cloud"
            
            # 1. 获取文本token的嵌入
            inputs_embeds = self.model.embed_tokens(input_ids)
            if inputs_embeds.dtype != torch.bfloat16:
                inputs_embeds = inputs_embeds.to(torch.bfloat16)
            # 2. 获取点云嵌入 - 按照官方demo的方式
            if point_cloud is not None:
                point_embeds = self.point_encoder(point_cloud)
            
                point_embeds = point_embeds.bfloat16()        
                # 3. 替换输入嵌入中的点云位置 
                inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            
            # 4. 调整attention_mask（将-1替换为1）
            attention_mask[attention_mask == -1] = 1
            input_ids = None 
            position_ids = None  

        # --- 传递给Qwen模型 ---
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # --- 计算logits和loss ---
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # 🚨 官方demo显式转换为float

        loss = None
        if labels is not None:
            # 标准的因果语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, point_cloud=None, attention_mask=None, **kwargs):
        """为生成准备输入 - 按照官方demo的方式"""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )
        model_inputs['point_cloud'] = kwargs.get('point_cloud', point_cloud)
        
        return model_inputs


def create_model_and_tokenizer(config):
    """创建模型和分词器"""
    from transformers import AutoTokenizer
    
    print(f"Loading model and tokenizer from {config.base_model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        pad_token='<|im_end|>',
        padding_side='left',
        trust_remote_code=True
    )
    
    # 加载模型配置
    model_config = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
    
    # 禁用滑动窗口注意力
    if hasattr(model_config, 'sliding_window'):
        model_config.sliding_window = None
    
    # 创建模型
    model = CADRecode.from_pretrained(
        config.base_model_name,
        config=model_config,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        trust_remote_code=True
    )
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    return model, tokenizer
