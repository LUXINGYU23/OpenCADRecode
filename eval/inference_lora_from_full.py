#!/usr/bin/env python3
"""
CAD-Recode LoRA微调模型推理脚本
用于测试从full模型微调后的LoRA模型效果
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
import torch
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 导入自定义模块
from utils import inference_single_sample, get_cached_point_cloud, load_data_index
from models import CADRecode
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_lora_model(lora_model_path: str, use_merged: bool = True):
    """加载LoRA微调模型"""
    
    if use_merged:
        # 尝试加载合并后的模型
        merged_path = Path(lora_model_path) / "merged_model"
        if merged_path.exists():
            print(f"🔄 Loading merged model from: {merged_path}")
            
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                str(merged_path),
                pad_token='<|im_end|>',
                padding_side='left',
                trust_remote_code=True
            )
            
            # 加载模型
            model = CADRecode.from_pretrained(
                str(merged_path),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            print(f"✅ Successfully loaded merged LoRA model")
            return model, tokenizer
    
    # 加载LoRA权重
    print(f"🔄 Loading LoRA model from: {lora_model_path}")
    
    # 首先确定基础模型路径
    config_file = Path(lora_model_path) / "config.yaml"
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        base_model_name = config.get('base_model_name', 'checkpoints_qwen3_sft')
    else:
        base_model_name = 'checkpoints_qwen3_sft'  # 默认路径
    
    print(f"🔄 Base model: {base_model_name}")
    
    # 加载分词器
    if os.path.exists(base_model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
    else:
        # 从LoRA目录加载
        tokenizer = AutoTokenizer.from_pretrained(
            lora_model_path,
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
    
    # 加载基础模型
    if os.path.exists(base_model_name):
        model = CADRecode.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Base model not found: {base_model_name}")
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    print(f"✅ Successfully loaded LoRA model")
    return model, tokenizer


def batch_inference(model, tokenizer, data_root: str, output_dir: str, 
                   split: str = "val", max_samples: int = 10,
                   max_new_tokens: int = 768):
    """批量推理"""
    
    print(f"Starting batch inference on {split} split...")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    code_dir = output_dir / "generated_code"
    stl_dir = output_dir / "generated_stl"
    results_dir = output_dir / "results"
    
    code_dir.mkdir(exist_ok=True)
    stl_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # 加载数据索引
    data_index = load_data_index(Path(data_root), split)
    
    if max_samples > 0:
        data_index = data_index[:max_samples]
    
    print(f"Processing {len(data_index)} samples...")
    
    results = []
    device = next(model.parameters()).device
    
    for i, item in enumerate(data_index):
        try:
            # 获取样本ID
            sample_id = item.get("sample_id", item.get("id", f"sample_{i}"))
            print(f"Processing sample {i+1}/{len(data_index)}: {sample_id}")
            
            # 加载点云
            point_cloud = get_cached_point_cloud(Path(data_root), sample_id, 256)
            if point_cloud is None:
                print(f"⚠️  No point cloud found for {sample_id}, skipping...")
                continue
            
            # 推理生成代码
            generated_code = inference_single_sample(
                model, tokenizer, point_cloud, max_new_tokens=max_new_tokens
            )
            
            # 保存生成的代码
            code_file = code_dir / f"{sample_id}.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            # 尝试执行代码生成STL
            stl_generated = False
            stl_file = None
            stl_error = None
            
            try:
                # 执行CadQuery代码
                exec_globals = {}
                exec(generated_code, exec_globals)
                
                # 寻找结果变量（通常是'r'）
                if 'r' in exec_globals:
                    result_obj = exec_globals['r']
                    
                    # 导出STL
                    stl_file = stl_dir / f"{sample_id}.stl"
                    result_obj.val().exportStl(str(stl_file))
                    stl_generated = True
                    stl_file = str(stl_file)
                    print(f"✅ STL generated: {stl_file}")
                else:
                    stl_error = "No result variable 'r' found in generated code"
                    
            except Exception as e:
                stl_error = f"Error executing CadQuery code: {str(e)}"
                print(f"❌ STL generation failed: {stl_error}")
            
            # 加载真实代码（如果存在）
            ground_truth_code = ""
            code_path = item.get("code_path")
            if code_path and Path(code_path).exists():
                with open(code_path, 'r', encoding='utf-8') as f:
                    ground_truth_code = f.read()
            
            # 记录结果
            result = {
                "sample_id": sample_id,
                "generated_code": generated_code,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model.peft_config['default'].base_model_name_or_path if hasattr(model, 'peft_config') else 'merged_model'),
                "code_file": str(code_file),
                "stl_generated": stl_generated,
                "stl_file": stl_file,
                "stl_error": stl_error,
                "ground_truth_code": ground_truth_code
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing sample {sample_id}: {e}")
            continue
    
    # 保存结果
    results_file = results_dir / "inference_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 统计结果
    total_samples = len(results)
    successful_stl = sum(1 for r in results if r["stl_generated"])
    success_rate = successful_stl / total_samples if total_samples > 0 else 0
    
    print(f"\n📊 Inference Results:")
    print(f"Total samples: {total_samples}")
    print(f"Successful STL generation: {successful_stl}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Results saved to: {results_file}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CAD-Recode LoRA Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="LoRA model path")
    parser.add_argument("--data_root", type=str, default="data", help="Data root directory")
    parser.add_argument("--output_dir", type=str, default="inference_results_lora", help="Output directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Data split")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum samples to process (0 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Maximum new tokens to generate")
    parser.add_argument("--use_merged", action="store_true", help="Use merged model if available")
    
    args = parser.parse_args()
    
    print("="*50)
    print("CAD-Recode LoRA Model Inference")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples}")
    print(f"Use merged: {args.use_merged}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载模型
    model, tokenizer = load_lora_model(args.model_path, use_merged=args.use_merged)
    model = model.to(device)
    model.eval()
    
    # 批量推理
    results = batch_inference(
        model=model,
        tokenizer=tokenizer,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    print("🎉 Inference completed!")


if __name__ == "__main__":
    main()
