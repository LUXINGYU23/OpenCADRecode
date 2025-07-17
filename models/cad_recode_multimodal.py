#!/usr/bin/env python3
"""
CAD-Recode 多模态模型 - 支持BRep和点云输入
基于UV-Net架构的BRep编码器 + 傅里叶点云编码器
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
from .brep_encoder import UVNetGraphEncoder


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


class CADRecodeMultimodal(Qwen3ForCausalLM):
    """
    CAD-Recode多模态模型
    支持BRep和点云两种输入模态
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 保存配置信息
        self._hidden_size = config.hidden_size
        self.num_point_tokens = NUM_POINT_TOKENS
        self.num_brep_tokens = NUM_POINT_TOKENS  # BRep token数量与点云相同
        self.total_modal_tokens = NUM_POINT_TOKENS * 2  # 总共512个模态token
        
        # 创建编码器
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        self.brep_encoder = UVNetGraphEncoder(
            input_dim=7,      # 3 (points) + 3 (normals) + 1 (mask)
            input_edge_dim=6, # 3 (points) + 3 (tangents)
            output_dim=config.hidden_size,
            hidden_dim=128,
            num_layers=3
        )
        
        # 启用梯度检查点
        if hasattr(self, 'gradient_checkpointing_enable'):
            self.gradient_checkpointing_enable()
    
    def enable_input_require_grads(self):
        """启用输入梯度"""
        self.model.embed_tokens.weight.requires_grad = True
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        point_cloud: Optional[torch.Tensor] = None,
        brep_data: Optional[Dict[str, torch.Tensor]] = None,
        brep_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
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
        """
        前向传播 - 支持随机三选一输入策略
        根据brep_mask和point_mask决定哪些样本使用哪些模态
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 只在初始前向时处理模态输入
        if past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0):
            assert inputs_embeds is None, "不能同时提供inputs_embeds和模态输入"
            
            # 获取批次大小
            batch_size = input_ids.shape[0]
            
            # 1. 获取文本token的嵌入
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # 2. 处理BRep嵌入（前256个位置）
            if brep_data is not None and brep_mask is not None:
                # 只处理选择BRep的样本
                brep_indices = torch.where(brep_mask)[0]
                if len(brep_indices) > 0:
                    brep_embeddings = self.brep_encoder(brep_data["graph"])
                    
                    # 将BRep嵌入分配到对应样本位置
                    for idx, sample_idx in enumerate(brep_indices):
                        start_pos = 0
                        end_pos = self.num_brep_tokens
                        if idx < brep_embeddings.shape[0]:
                            inputs_embeds[sample_idx, start_pos:end_pos] = brep_embeddings[idx]
            
            # 3. 处理点云嵌入（后256个位置）
            if point_cloud is not None and point_mask is not None:
                # 只处理选择点云的样本
                point_indices = torch.where(point_mask)[0]
                if len(point_indices) > 0:
                    # 过滤出有效的点云
                    valid_point_clouds = point_cloud[point_indices]
                    if valid_point_clouds.shape[0] > 0:
                        point_embeddings = self.point_encoder(valid_point_clouds)
                        
                        # 将点云嵌入分配到对应样本位置
                        for i, sample_idx in enumerate(point_indices):
                            start_pos = self.num_brep_tokens
                            end_pos = self.num_brep_tokens + self.num_point_tokens
                            inputs_embeds[sample_idx, start_pos:end_pos] = point_embeddings[i]
            
            # 4. 调整attention_mask（将-1/-2替换为1）
            attention_mask = attention_mask.clamp(min=0)
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
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
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
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, 
                                    point_cloud=None, brep_data=None, attention_mask=None,
                                    brep_mask=None, point_mask=None, **kwargs):
        """
        为生成准备输入 - 支持随机三选一策略
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1:] if input_ids is not None else None

        # 如果提供了inputs_embeds，直接使用
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "point_cloud": point_cloud,
            "brep_data": brep_data,
            "brep_mask": brep_mask,
            "point_mask": point_mask,
        })
        return model_inputs


def create_multimodal_model_and_tokenizer(config):
    """创建多模态模型和分词器"""
    from transformers import AutoTokenizer
    
    print(f"Loading multimodal model and tokenizer from {config.base_model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        pad_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    )
    
    # 加载模型配置
    model_config = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
    
    # 禁用滑动窗口注意力
    if hasattr(model_config, 'sliding_window'):
        model_config.sliding_window = None
    
    # 创建模型
    model = CADRecodeMultimodal.from_pretrained(
        config.base_model_name,
        config=model_config,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        trust_remote_code=True
    )
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    return model, tokenizer
