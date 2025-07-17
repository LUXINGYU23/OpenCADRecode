#!/usr/bin/env python3
"""
CAD-Recode 数据处理模块
包含数据集类和数据整理器
"""

import json
import random
import torch
import dgl
import numpy as np
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


# 新增BRep相关工具函数
def load_brep_graph(data_root: Path, sample_id: str) -> Optional[dgl.DGLGraph]:
    """加载BRep图数据"""
    brep_path = data_root / "brep" / f"{sample_id}.bin"
    if not brep_path.exists():
        return None
    try:
        return dgl.load_graphs(str(brep_path))[0][0]
    except Exception as e:
        print(f"Error loading BRep graph for {sample_id}: {e}")
        return None


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


class CADRecodeMultimodalDataset(torch.utils.data.Dataset):
    """CAD-Recode多模态数据集类 - 支持BRep和点云"""
    
    def __init__(self, data_root: str, tokenizer: PreTrainedTokenizerBase, config, split: str = "train"):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.num_points = config.n_points
        self.num_brep_tokens = config.n_brep_tokens if hasattr(config, 'n_brep_tokens') else NUM_POINT_TOKENS
        self.total_modal_tokens = self.num_brep_tokens + self.num_points
        
        # 加载数据索引
        self.data_index = load_data_index(self.data_root, split)
        
        # 加载错误样本列表
        self.error_samples = load_error_samples(self.data_root)
        
        # 过滤错误样本
        self._filter_error_samples()
        
        # 过滤无效BRep样本
        self._filter_invalid_brep_samples()
        
        print(f"Loaded {len(self.data_index)} samples for {split} split")
        print(f"Filtered out {len(self.error_samples)} error samples")

    def _filter_invalid_brep_samples(self):
        """过滤无效BRep样本"""
        valid_samples = []
        for item in self.data_index:
            sample_id = item.get("sample_id", item.get("id"))
            brep_graph = load_brep_graph(self.data_root, sample_id)
            if brep_graph is not None:
                valid_samples.append(item)
        
        filtered_count = len(self.data_index) - len(valid_samples)
        self.data_index = valid_samples
        print(f"Filtered out {filtered_count} invalid BRep samples")

    def _filter_error_samples(self):
        """过滤错误样本"""
        if self.error_samples:
            original_count = len(self.data_index)
            self.data_index = [item for item in self.data_index 
                             if item.get("sample_id", item.get("id")) not in self.error_samples]
            filtered_count = original_count - len(self.data_index)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} error samples")

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本 - 支持BRep和点云"""
        try:
            item = self.data_index[idx]
            
            # 支持两种数据格式
            if "sample_id" in item:
                sample_id = item["sample_id"]
                code_path = self.data_root / "cad" / f"{sample_id}.py"
            else:
                sample_id = item["id"]
                code_path = self.data_root / "cad" / f"{sample_id}.py"
            
            if not code_path or not Path(code_path).exists():
                return self._get_default_sample()
            
            # 读取Python代码
            with open(code_path, 'r', encoding='utf-8') as f:
                code_content = f.read().strip()
            
            # 加载BRep图数据
            brep_graph = load_brep_graph(self.data_root, sample_id)
            
            # 从缓存读取点云
            point_cloud = get_cached_point_cloud(self.data_root, sample_id, self.num_points)
            if point_cloud is None:
                point_cloud = create_default_point_cloud(self.num_points)
            
            # 应用数据增强
            point_cloud = apply_data_augmentation(
                point_cloud, 
                self.config.noise_probability, 
                self.config.noise_std,
                is_training=(self.split == "train")
            )
            
            # 构建输入格式
            formatted_code = f"<|im_start|>{code_content}"
            
            return {
                "brep_graph": brep_graph,
                "point_cloud": torch.tensor(point_cloud, dtype=torch.float32),
                "target_code": formatted_code
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self._get_default_sample()

    def _get_default_sample(self) -> Dict[str, Any]:
        """获取默认样本（出错时使用）"""
        default_code = "import cadquery as cq\nr = cq.Workplane('XY').box(1, 1, 1)"
        formatted_code = f"<|im_start|>{default_code}"
        
        # 创建默认BRep图（空图）
        default_graph = dgl.graph(([], []), num_nodes=1)
        default_graph.ndata['x'] = torch.zeros(1, 20, 20, 7)
        default_graph.edata['x'] = torch.zeros(0, 20, 6)
        
        # 创建默认点云
        default_points = create_default_point_cloud(self.num_points)
        
        return {
            "brep_graph": default_graph,
            "point_cloud": torch.tensor(default_points, dtype=torch.float32),
            "target_code": formatted_code
        }


@dataclass
class DataCollatorForMultimodalCADRecode:
    """多模态CAD-Recode数据整理器 - 支持随机三选一输入策略"""
    
    tokenizer: PreTrainedTokenizerBase
    max_length: int = MAX_SEQ_LENGTH
    pad_to_multiple_of: Optional[int] = None
    num_brep_tokens: int = NUM_POINT_TOKENS
    num_point_tokens: int = NUM_POINT_TOKENS
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """整理批次数据 - 支持随机三选一输入策略"""
        import random
        
        # 分离不同模态的数据
        brep_graphs = [f["brep_graph"] for f in features]
        point_clouds = torch.stack([f["point_cloud"] for f in features])
        target_codes = [f["target_code"] for f in features]
        
        # 批处理BRep图
        batched_brep_graphs = dgl.batch(brep_graphs)
        
        # 处理文本序列
        batch_size = len(target_codes)
        
        # 为每个样本随机选择输入模态 (0: 仅BRep, 1: 仅点云, 2: 两者都输入)
        modal_choices = [random.choice([0, 1, 2]) for _ in range(batch_size)]
        
        # 为每个样本创建输入序列
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        # 创建模态掩码
        brep_mask = torch.zeros(batch_size, dtype=torch.bool)
        point_mask = torch.zeros(batch_size, dtype=torch.bool)
        
        for i, code in enumerate(target_codes):
            modal_choice = modal_choices[i]
            
            # 设置模态掩码
            if modal_choice == 0:  # 仅BRep
                brep_mask[i] = True
                point_mask[i] = False
            elif modal_choice == 1:  # 仅点云
                brep_mask[i] = False
                point_mask[i] = True
            else:  # 两者都输入
                brep_mask[i] = True
                point_mask[i] = True
            
            # 1. 处理代码内容（移除<|im_start|>前缀）
            if code.startswith("<|im_start|>"):
                code_content = code[len("<|im_start|>"):]
            else:
                code_content = code
            
            # 2. 编码代码内容
            encoded_code = self.tokenizer(
                code_content,
                truncation=True,
                max_length=self.max_length - self.num_brep_tokens - self.num_point_tokens - 2,
                add_special_tokens=False
            )
            
            # 3. 构建完整输入序列
            start_token_id = self.tokenizer('<|im_start|>')['input_ids'][0]
            end_token_id = self.tokenizer.eos_token_id
            
            # 创建input_ids（用padding token填充模态位置）
            input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            
            # 设置模态token位置的attention_mask
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            
            # 文本部分: <|im_start|> + code + <|endoftext|>
            text_start_pos = self.num_brep_tokens + self.num_point_tokens
            text_end_pos = text_start_pos + 1 + len(encoded_code["input_ids"]) + 1
            
            # 填充文本部分
            input_ids[text_start_pos] = start_token_id
            input_ids[text_start_pos + 1:text_start_pos + 1 + len(encoded_code["input_ids"])] = torch.tensor(encoded_code["input_ids"])
            input_ids[text_start_pos + 1 + len(encoded_code["input_ids"])] = end_token_id
            
            # 设置文本部分的attention_mask (1)
            attention_mask[text_start_pos:text_end_pos] = 1
            
            # 根据模态选择设置模态位置的attention_mask
            if brep_mask[i]:
                attention_mask[:self.num_brep_tokens] = -1  # BRep tokens
            if point_mask[i]:
                attention_mask[self.num_brep_tokens:self.num_brep_tokens + self.num_point_tokens] = -2  # Point tokens
            
            # 创建labels
            labels = torch.full((self.max_length,), -100, dtype=torch.long)
            
            # 文本部分参与loss计算（除了<|im_start|>）
            labels[text_start_pos + 1:text_end_pos] = torch.cat([
                torch.tensor(encoded_code["input_ids"]),
                torch.tensor([end_token_id])
            ])
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        # 堆叠批次数据
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        labels = torch.stack(labels_list)
        
        # 根据模态掩码过滤BRep图和点云
        filtered_brep_graphs = []
        filtered_point_clouds = []
        
        for i in range(batch_size):
            if brep_mask[i]:
                filtered_brep_graphs.append(brep_graphs[i])
            if point_mask[i]:
                filtered_point_clouds.append(point_clouds[i])
        
        # 如果没有选择BRep的样本，创建一个空图批次
        if not filtered_brep_graphs:
            empty_graph = dgl.graph(([], []), num_nodes=1)
            empty_graph.ndata['x'] = torch.zeros(1, 20, 20, 7)
            empty_graph.edata['x'] = torch.zeros(0, 20, 6)
            filtered_brep_graphs = [empty_graph]
        
        # 如果没有选择点云的样本，创建空点云
        if not filtered_point_clouds:
            filtered_point_clouds = [torch.zeros(1, self.num_points, 3)]
        
        final_brep_graphs = dgl.batch(filtered_brep_graphs)
        final_point_clouds = torch.stack(filtered_point_clouds) if filtered_point_clouds else torch.zeros(0, self.num_points, 3)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "point_cloud": final_point_clouds,
            "brep_data": {"graph": final_brep_graphs},
            "brep_mask": brep_mask,
            "point_mask": point_mask,
            "modal_choices": torch.tensor(modal_choices)
        }
