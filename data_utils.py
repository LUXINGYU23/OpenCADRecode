#!/usr/bin/env python3
"""
CAD-Recode 数据处理模块
包含数据集类和数据整理器
"""

import json
import random
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

from utils import (
    NUM_POINT_TOKENS, 
    MAX_SEQ_LENGTH,
    load_data_index,
    load_error_samples,
    get_cached_point_cloud,
    apply_data_augmentation,
    create_default_point_cloud
)


class CADRecodeDataset(torch.utils.data.Dataset):
    """CAD-Recode数据集类"""
    
    def __init__(self, data_root: str, tokenizer: PreTrainedTokenizerBase, config, split: str = "train"):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.num_points = config.n_points
        
        # 加载数据索引
        self.data_index = load_data_index(self.data_root, split)
        
        # 加载错误样本列表（复用现有功能）
        self.error_samples = load_error_samples(self.data_root)
        
        # 过滤错误样本
        self._filter_error_samples()
        
        print(f"Loaded {len(self.data_index)} samples for {split} split")
        print(f"Filtered out {len(self.error_samples)} error samples")

    def _filter_error_samples(self):
        """过滤错误样本"""
        if self.error_samples:
            original_count = len(self.data_index)
            self.data_index = [item for item in self.data_index 
                             if item.get("sample_id", item.get("id")) not in self.error_samples]
            filtered_count = original_count - len(self.data_index)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} known error samples")

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本 - 兼容旧数据格式"""
        try:
            item = self.data_index[idx]
            
            # 支持两种数据格式
            if "sample_id" in item:
                # 新格式 (旧训练代码的格式)
                sample_id = item["sample_id"]
                code_path = item["code_path"]
            else:
                # 简单格式
                sample_id = item.get("id", f"sample_{idx}")
                code_path = item.get("py_file", item.get("code_path"))
            
            if not code_path or not Path(code_path).exists():
                print(f"Code file not found for {sample_id}: {code_path}")
                return self._get_default_sample()
            
            # 读取Python代码
            with open(code_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # 从缓存读取点云
            point_cloud = get_cached_point_cloud(self.data_root, sample_id, self.num_points)
            if point_cloud is None:
                # 如果缓存点云不存在，返回默认样本
                print(f"No point cloud found for {sample_id}, using default")
                return self._get_default_sample()
            
            # 应用数据增强
            point_cloud = apply_data_augmentation(
                point_cloud, 
                self.config.noise_probability, 
                self.config.noise_std,
                is_training=(self.split == "train")
            )
            
            # 按照train_qwen3.py的方式：只在代码前加<|im_start|>
            # 但要确保与官方demo对齐，数据collator会重新处理
            formatted_code = f"<|im_start|>{code_content}"
            
            return {
                "point_cloud": torch.tensor(point_cloud, dtype=torch.float32),
                "target_code": formatted_code
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._get_default_sample()

    def _get_default_sample(self) -> Dict[str, Any]:
        """获取默认样本（出错时使用）"""
        # 创建一个简单的默认样本
        default_code = "import cadquery as cq\nr = cq.Workplane('XY').box(1, 1, 1)"
        
        # 按照train_qwen3.py的方式：只在代码前加<|im_start|>
        # 但要确保与官方demo对齐，数据collator会重新处理
        formatted_code = f"<|im_start|>{default_code}"
        
        # 创建默认点云
        default_points = create_default_point_cloud(self.num_points)
        
        return {
            "point_cloud": torch.tensor(default_points, dtype=torch.float32),
            "target_code": formatted_code
        }


@dataclass
class DataCollatorForCADRecode:
    """CAD-Recode数据整理器 - 处理点云和文本序列"""
    
    tokenizer: PreTrainedTokenizerBase
    max_length: int = MAX_SEQ_LENGTH
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """整理批次数据 - 与官方demo.ipynb的输入构建方式完全对齐"""
        
        batch_point_clouds = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for feature in features:
            try:
                # 获取点云
                point_cloud = feature["point_cloud"]
                target_code = feature["target_code"]
                
                # 分词目标代码（不包括<|im_start|>，因为我们要手动添加）
                # 先移除<|im_start|>前缀
                if target_code.startswith("<|im_start|>"):
                    code_content = target_code[len("<|im_start|>"):]
                else:
                    code_content = target_code
                
                # 按照官方demo方式构建输入，确保训练和推理一致
                # 官方：input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
                # 官方：attention_mask = [-1] * len(point_cloud) + [1]
                # 训练序列：[点云占位符] + <|im_start|> + 代码内容 + <|endoftext|>
                # 推理解码：py_string[begin:end] 其中 end = py_string.find('<|endoftext|>')
                
                # 1. 点云部分：用pad_token_id填充，attention_mask为-1
                point_input_ids = [self.tokenizer.pad_token_id] * NUM_POINT_TOKENS
                point_attention_mask = [-1] * NUM_POINT_TOKENS
                
                # 2. 文本部分：<|im_start|> + 实际代码内容
                start_token_id = self.tokenizer('<|im_start|>')['input_ids'][0]
                
                # 分词代码内容，并添加结束符
                # 为 <|im_start|> 和 <|endoftext|> 预留2个位置
                encoded_code = self.tokenizer(
                    code_content,
                    truncation=True,
                    max_length=self.max_length - NUM_POINT_TOKENS - 2,  # 为点云、start token和end token预留空间
                    padding=False,
                    return_tensors="pt"
                )
                code_input_ids = encoded_code["input_ids"].squeeze(0).tolist()
                
                # 添加结束符
                end_token_id = self.tokenizer.eos_token_id  # <|endoftext|>
                
                # 拼接：点云占位符 + <|im_start|> + 代码内容 + <|endoftext|>
                full_input_ids = point_input_ids + [start_token_id] + code_input_ids + [end_token_id]
                full_attention_mask = point_attention_mask + [1] * (1 + len(code_input_ids) + 1)
                
                # 创建labels：点云部分和start token标记为-100，代码内容和结束符用于计算loss
                point_labels = [-100] * NUM_POINT_TOKENS
                start_label = [-100]  # start token不参与loss计算
                code_labels = code_input_ids  # 代码内容参与loss计算
                end_label = [end_token_id]  # 结束符也参与loss计算，教模型学会停止
                full_labels = point_labels + start_label + code_labels + end_label
                
                batch_point_clouds.append(point_cloud)
                batch_input_ids.append(torch.tensor(full_input_ids, dtype=torch.long))
                batch_attention_mask.append(torch.tensor(full_attention_mask, dtype=torch.long))
                batch_labels.append(torch.tensor(full_labels, dtype=torch.long))
                
            except Exception as e:
                print(f"Error processing feature: {e}")
                # 使用默认样本
                default_point_cloud = torch.randn(NUM_POINT_TOKENS, 3, dtype=torch.float32)
                default_code = "r = cq.Workplane().box(1, 1, 1)"
                
                # 按照相同方式处理默认样本
                point_input_ids = [self.tokenizer.pad_token_id] * NUM_POINT_TOKENS
                point_attention_mask = [-1] * NUM_POINT_TOKENS
                start_token_id = self.tokenizer('<|im_start|>')['input_ids'][0]
                
                encoded_code = self.tokenizer(
                    default_code,
                    truncation=True,
                    max_length=self.max_length - NUM_POINT_TOKENS - 2,  # 为点云、start和end token预留空间
                    padding=False,
                    return_tensors="pt"
                )
                code_input_ids = encoded_code["input_ids"].squeeze(0).tolist()
                
                # 添加结束符
                end_token_id = self.tokenizer.eos_token_id
                
                full_input_ids = point_input_ids + [start_token_id] + code_input_ids + [end_token_id]
                full_attention_mask = point_attention_mask + [1] * (1 + len(code_input_ids) + 1)
                full_labels = [-100] * NUM_POINT_TOKENS + [-100] + code_input_ids + [end_token_id]
                
                batch_point_clouds.append(default_point_cloud)
                batch_input_ids.append(torch.tensor(full_input_ids, dtype=torch.long))
                batch_attention_mask.append(torch.tensor(full_attention_mask, dtype=torch.long))
                batch_labels.append(torch.tensor(full_labels, dtype=torch.long))
        
        # 填充到相同长度
        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            padding_length = max_len - len(input_ids)
            
            if padding_length > 0:
                # 右填充
                padded_input_ids.append(torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                ]))
                padded_attention_mask.append(torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ]))
                padded_labels.append(torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long)
                ]))
            else:
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)
                padded_labels.append(labels)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "point_cloud": torch.stack(batch_point_clouds)
        }
