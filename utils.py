#!/usr/bin/env python3
"""
CAD-Recode 工具函数模块
包含训练过程中使用的各种工具函数
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

# 配置常量
NUM_POINT_TOKENS = 256  # 论文中提到的点云token数量
MAX_CODE_TOKENS = 768   # 代码token最大数量
SPECIAL_TOKENS = 2      # <|im_start|> 和 <|endoftext|>
MAX_SEQ_LENGTH = NUM_POINT_TOKENS + MAX_CODE_TOKENS + SPECIAL_TOKENS  # 总序列长度


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 模型配置
    base_model_name: str = "Qwen/Qwen2-1.5B"
    model_save_path: str = "checkpoints_v2"
    
    # 数据配置
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    max_seq_length: int = MAX_SEQ_LENGTH
    n_points: int = NUM_POINT_TOKENS
    
    # 训练配置
    max_steps: int = 100000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # 评估和保存配置
    eval_steps: int = 2000
    save_steps: int = 2000
    logging_steps: int = 50
    seed: int = 42
    
    # 硬件配置
    device: str = "auto"
    mixed_precision: str = "bf16"
    num_workers: int = 0
    
    # 数据增强配置
    noise_probability: float = 0.5
    noise_std: float = 0.01
    
    # 实验跟踪
    experiment_name: str = "cad-recode-v2"
    use_swanlab: bool = True


def setup_environment():
    """设置环境变量"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def set_random_seeds(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_error_samples(data_root: Path) -> set:
    """加载错误样本列表"""
    error_file = data_root / "error_samples.json"
    if error_file.exists():
        with open(error_file, 'r') as f:
            return set(json.load(f))
    return set()


def load_data_index(data_root: Path, split: str) -> List[Dict[str, Any]]:
    """加载数据索引 - 兼容旧格式"""
    # 首先尝试加载split特定的索引文件（旧格式）
    index_file = data_root / f"{split}_index.json"
    if index_file.exists():
        print(f"Loading index from {index_file}")
        with open(index_file, 'r') as f:
            return json.load(f)
    
    # 如果没有split特定的索引文件，尝试通用索引文件
    index_file = data_root / "index.json"
    if index_file.exists():
        print(f"Loading index from {index_file}")
        with open(index_file, 'r') as f:
            return json.load(f)
    
    # 如果没有索引文件，扫描目录结构
    print(f"No index file found, scanning directory: {data_root}")
    data_list = []
    for batch_dir in data_root.glob("batch_*"):
        if batch_dir.is_dir():
            for py_file in batch_dir.glob("*.py"):
                sample_id = f"{batch_dir.name}_{py_file.stem}"
                data_list.append({
                    "sample_id": sample_id,
                    "code_path": str(py_file),
                    "relative_path": f"{batch_dir.name}/{py_file.name}",
                    "split": split
                })
    return data_list


def get_cached_point_cloud(data_root: Path, sample_id: str, num_points: int) -> Optional[np.ndarray]:
    """从缓存读取点云 - 支持全局缓存目录"""
    try:
        # 尝试本地缓存
        cache_file = data_root / "point_cloud_cache" / f"{sample_id}.npy"
        
        if not cache_file.exists():
            # 尝试全局缓存目录
            global_cache = Path("/root/cad-recode/point_cloud_cache") 
            cache_file = global_cache / f"{sample_id}.npy"
        
        if cache_file.exists():
            point_cloud = np.load(cache_file)
            # 确保点云形状正确
            if point_cloud.shape[0] != num_points:
                print(f"Warning: cached point cloud for {sample_id} has wrong shape: {point_cloud.shape}")
                return None
            return point_cloud.astype(np.float32)
        else:
            print(f"Warning: no cached point cloud found for {sample_id}")
            return None
            
    except Exception as e:
        print(f"Error loading cached point cloud for {sample_id}: {e}")
        return None


def apply_data_augmentation(point_cloud: np.ndarray, noise_probability: float, noise_std: float, is_training: bool = True) -> np.ndarray:
    """应用数据增强"""
    if is_training and random.random() < noise_probability:
        # 添加高斯噪声
        noise = np.random.normal(0, noise_std, point_cloud.shape)
        point_cloud = point_cloud + noise
    return point_cloud


def create_default_point_cloud(num_points: int) -> np.ndarray:
    """创建默认点云（单位立方体的8个顶点）"""
    default_points = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float32)
    
    # 扩展到所需的点数
    if len(default_points) < num_points:
        # 重复点以达到所需数量
        repeat_times = num_points // len(default_points) + 1
        default_points = np.tile(default_points, (repeat_times, 1))[:num_points]
    
    return default_points


def inference_single_sample(model, tokenizer, point_cloud: np.ndarray, max_new_tokens: int = 768) -> str:
    """单样本推理 - 与官方demo.ipynb完全对齐"""
    device = next(model.parameters()).device
    
    # input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
    # attention_mask = [-1] * len(point_cloud) + [1]
    
    input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
    attention_mask = [-1] * len(point_cloud) + [1]
    
    # 转移到设备
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
    point_cloud = torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).to(device)
    
    # 生成 - 与demo一致
    with torch.no_grad():
        batch_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_cloud=point_cloud,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码 - 与demo一致
    py_string = tokenizer.batch_decode(batch_ids)[0]
    begin = py_string.find('<|im_start|>') + 12
    end = py_string.find('<|endoftext|>')
    py_string = py_string[begin:end] if end != -1 else py_string[begin:]
    
    return py_string


def merge_lora_weights_to_model(peft_model, base_model_name: str = None, save_path: str = None, 
                               safe_serialization: bool = True, max_shard_size: str = "5GB"):
    """
    合并LoRA权重到基础模型
    
    Args:
        peft_model: PEFT模型实例
        base_model_name: 基础模型名称（用于保存tokenizer和config）
        save_path: 保存路径，如果为None则返回合并后的模型
        safe_serialization: 是否使用safe_serialization
        max_shard_size: 最大分片大小
    
    Returns:
        merged_model: 合并后的模型（如果save_path为None）
    """
    try:
        print("🔄 Starting LoRA weight merging...")
        
        # 合并权重
        merged_model = peft_model.merge_and_unload()
        print("✅ LoRA weights merged successfully")
        
        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存合并后的模型
            print(f"💾 Saving merged model to: {save_path}")
            merged_model.save_pretrained(
                save_path,
                safe_serialization=safe_serialization,
                max_shard_size=max_shard_size
            )
            
            # 如果提供了基础模型名称，保存tokenizer和config
            if base_model_name:
                try:
                    from transformers import AutoTokenizer, AutoConfig
                    
                    print("💾 Saving tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_name,
                        pad_token='<|im_end|>',
                        padding_side='left',
                        trust_remote_code=True
                    )
                    tokenizer.save_pretrained(save_path)
                    
                    print("💾 Saving config...")
                    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
                    if hasattr(config, 'sliding_window'):
                        config.sliding_window = None
                    config.save_pretrained(save_path)
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not save tokenizer/config: {e}")
            
            # 显示保存信息
            try:
                model_files = list(save_path.glob("*.safetensors")) + list(save_path.glob("*.bin"))
                total_size = sum(f.stat().st_size for f in model_files)
                print(f"📊 Merged model size: {total_size / (1024**3):.2f} GB")
                print(f"📁 Files saved: {len(model_files)} model files")
            except Exception as e:
                print(f"⚠️  Could not calculate model size: {e}")
                
            print(f"✅ Merged model saved successfully to: {save_path}")
            
        return merged_model
        
    except Exception as e:
        print(f"❌ Error merging LoRA weights: {e}")
        raise e


def save_lora_checkpoint_with_merge(trainer, output_dir: str, base_model_name: str = None, 
                                   auto_merge: bool = True, keep_lora_only: bool = True):
    """
    保存LoRA检查点并可选择性地自动合并权重
    
    Args:
        trainer: Hugging Face Trainer实例
        output_dir: 输出目录
        base_model_name: 基础模型名称
        auto_merge: 是否自动合并权重
        keep_lora_only: 是否保留LoRA权重文件
    """
    try:
        output_dir = Path(output_dir)
        
        # 保存LoRA权重 - 直接调用原始的保存方法
        print(f"💾 Saving LoRA checkpoint to: {output_dir}")
        
        # 确保目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态字典和配置
        if hasattr(trainer.model, 'save_pretrained'):
            trainer.model.save_pretrained(output_dir)
        
        # 保存tokenizer
        if hasattr(trainer, 'tokenizer') and trainer.tokenizer:
            trainer.tokenizer.save_pretrained(output_dir)
        
        if auto_merge and hasattr(trainer.model, 'merge_and_unload'):
            # 创建合并模型的子目录
            merged_dir = output_dir / "merged_model"
            
            print(f"🔄 Auto-merging LoRA weights...")
            merge_lora_weights_to_model(
                peft_model=trainer.model,
                base_model_name=base_model_name,
                save_path=str(merged_dir)
            )
            
            # 如果不保留LoRA权重，移除相关文件
            if not keep_lora_only:
                lora_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
                for lora_file in lora_files:
                    lora_path = output_dir / lora_file
                    if lora_path.exists():
                        lora_path.unlink()
                        print(f"🗑️  Removed LoRA file: {lora_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving LoRA checkpoint: {e}")
        return False


class LoRACheckpointCallback:
    """LoRA训练过程中的自动合并回调"""
    
    def __init__(self, base_model_name: str = None, auto_merge: bool = True, 
                 keep_lora_only: bool = True, merge_final_only: bool = False):
        self.base_model_name = base_model_name
        self.auto_merge = auto_merge
        self.keep_lora_only = keep_lora_only
        self.merge_final_only = merge_final_only
    
    def on_save(self, trainer, output_dir: str, is_final: bool = False):
        """在保存时调用"""
        # 如果设置了只在最终保存时合并，且当前不是最终保存，则跳过
        if self.merge_final_only and not is_final:
            return
            
        if self.auto_merge and hasattr(trainer.model, 'merge_and_unload'):
            save_lora_checkpoint_with_merge(
                trainer=trainer,
                output_dir=output_dir,
                base_model_name=self.base_model_name,
                auto_merge=self.auto_merge,
                keep_lora_only=self.keep_lora_only or not is_final  # 最终保存时可以选择不保留LoRA
            )
