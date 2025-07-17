#!/usr/bin/env python3
"""
CAD-Recode LoRA推理脚本
支持加载LoRA权重进行点云到代码的推理


使用方法:
python inference_lora.py --base_model Qwen/Qwen3-1.7B-Base --lora_path checkpoints_qwen3_lora --point_cloud_file data/val/point_cloud_cache/0.npy"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import numpy as np
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel
from models import CADRecode
from utils import inference_single_sample


def load_lora_model(base_model_path: str, lora_path: str, device: str = "auto"):
    """
    加载LoRA模型，自动检测并优先使用合并后的模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径
        device: 设备
    
    Returns:
        model, tokenizer: 加载的模型和分词器
    """
    lora_path = Path(lora_path)
    merged_model_path = lora_path / "merged_model"
    
    # 检查是否存在合并后的模型
    if merged_model_path.exists() and (merged_model_path / "config.json").exists():
        print(f"🎯 Found merged model, loading from: {merged_model_path}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            merged_model_path,
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
        
        # 加载合并后的模型
        model_config = AutoConfig.from_pretrained(merged_model_path, trust_remote_code=True)
        if hasattr(model_config, 'sliding_window'):
            model_config.sliding_window = None
        
        model = CADRecode.from_pretrained(
            merged_model_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print(f"✅ Merged model loaded successfully")
        
    else:
        print(f"🔧 No merged model found, loading LoRA adapter from: {lora_path}")
        print(f"Base model: {base_model_path}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
        
        # 加载模型配置
        model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 禁用滑动窗口注意力
        if hasattr(model_config, 'sliding_window'):
            model_config.sliding_window = None
        
        # 加载基础模型
        base_model = CADRecode.from_pretrained(
            base_model_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        # 合并权重以提高推理速度
        print("🔄 Merging LoRA weights for faster inference...")
        model = model.merge_and_unload()
        
        print(f"✅ LoRA adapter loaded and merged successfully")
    
    # 移动到指定设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_point_cloud(file_path: str, num_points: int = 256) -> np.ndarray:
    """加载点云文件"""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    # 支持多种格式
    if file_path.endswith('.npy'):
        point_cloud = np.load(file_path)
    elif file_path.endswith('.txt'):
        point_cloud = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # 确保形状正确
    if point_cloud.shape[1] != 3:
        raise ValueError(f"Point cloud should have 3 coordinates, got shape: {point_cloud.shape}")
    
    # 调整点数
    if point_cloud.shape[0] != num_points:
        if point_cloud.shape[0] > num_points:
            # 随机采样
            indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
            point_cloud = point_cloud[indices]
        else:
            # 重复采样
            repeat_times = num_points // point_cloud.shape[0] + 1
            point_cloud = np.tile(point_cloud, (repeat_times, 1))[:num_points]
    
    return point_cloud.astype(np.float32)


def create_sample_point_cloud(shape: str = "cube", num_points: int = 256) -> np.ndarray:
    """创建示例点云"""
    if shape == "cube":
        # 创建立方体点云
        points = []
        for i in range(num_points):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1) 
            z = np.random.uniform(-1, 1)
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)
    
    elif shape == "sphere":
        # 创建球体点云
        points = []
        for i in range(num_points):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, 1)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)
    
    elif shape == "cylinder":
        # 创建圆柱体点云
        points = []
        for i in range(num_points):
            theta = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 1)
            z = np.random.uniform(-1, 1)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)
    
    else:
        raise ValueError(f"Unknown shape: {shape}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CAD-Recode LoRA Inference")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA weights path")
    parser.add_argument("--point_cloud_file", type=str, help="Point cloud file (.npy or .txt)")
    parser.add_argument("--shape", type=str, choices=["cube", "sphere", "cylinder"], 
                        default="cube", help="Generate sample point cloud shape")
    parser.add_argument("--num_points", type=int, default=256, help="Number of points")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output", type=str, help="Output file to save generated code")
    
    args = parser.parse_args()
    
    # 加载模型
    print("Loading LoRA model...")
    model, tokenizer = load_lora_model(args.base_model, args.lora_path, args.device)
    
    # 加载或创建点云
    if args.point_cloud_file:
        print(f"Loading point cloud from: {args.point_cloud_file}")
        point_cloud = load_point_cloud(args.point_cloud_file, args.num_points)
    else:
        print(f"Creating sample {args.shape} point cloud...")
        point_cloud = create_sample_point_cloud(args.shape, args.num_points)
    
    print(f"Point cloud shape: {point_cloud.shape}")
    
    # 进行推理
    print("Generating CadQuery code...")
    generated_code = inference_single_sample(
        model, tokenizer, point_cloud, args.max_new_tokens
    )
    
    # 输出结果
    print("\n" + "="*50)
    print("Generated CadQuery Code:")
    print("="*50)
    print(generated_code)
    print("="*50)
    
    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        print(f"\nCode saved to: {args.output}")
    
    # 简单的代码验证
    try:
        # 检查是否包含基本的CadQuery语法
        if "cadquery" in generated_code.lower() or "cq." in generated_code:
            print("\n✅ Generated code contains CadQuery syntax")
        else:
            print("\n⚠️  Generated code may not contain valid CadQuery syntax")
            
        # 检查语法错误
        compile(generated_code, '<string>', 'exec')
        print("✅ Generated code is syntactically valid Python")
        
    except SyntaxError as e:
        print(f"\n❌ Syntax error in generated code: {e}")
    except Exception as e:
        print(f"\n⚠️  Could not validate code: {e}")


if __name__ == "__main__":
    main()
