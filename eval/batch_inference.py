#!/usr/bin/env python3
"""
CAD-Recode 批量推理脚本
支持加载训练好的模型，对点云数据进行批量推理，生成CadQuery代码并保存为.stl文件

主要功能：
1. 加载训练好的CAD-Recode模型
2. 批量读取点云数据
3. 生成CadQuery代码
4. 执行代码生成.stl文件
5. 支持多种输出格式和评估指标

使用方法:
python batch_inference.py --model_path checkpoints_qwen3_sft --data_path data/val --output_dir inference_results
"""

import os
import sys
import json
import yaml
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from utils import TrainingConfig, setup_environment, get_cached_point_cloud, inference_single_sample
from models import CADRecode, create_model_and_tokenizer
from data_utils import load_data_index, load_error_samples

# CadQuery相关导入
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    print("Warning: CadQuery not installed. STL generation will be disabled.")
    print("Install with: pip install cadquery")


class BatchInferenceConfig:
    """批量推理配置类"""
    def __init__(self, **kwargs):
        # 模型和数据路径
        self.model_path = kwargs.get('model_path', 'checkpoints_qwen3_sft')
        self.data_path = kwargs.get('data_path', 'data/val')
        self.output_dir = kwargs.get('output_dir', 'inference_results')
        
        # 推理参数
        self.max_new_tokens = kwargs.get('max_new_tokens', 768)
        self.batch_size = kwargs.get('batch_size', 1)  # 推理时通常使用小batch
        self.device = kwargs.get('device', 'auto')
        
        # 输出控制
        self.save_code = kwargs.get('save_code', True)
        self.save_stl = kwargs.get('save_stl', True)
        self.save_step = kwargs.get('save_step', True)  # 新增STEP格式支持
        self.save_predictions = kwargs.get('save_predictions', True)
        
        # 数据处理
        self.num_samples = kwargs.get('num_samples', None)  # None表示处理所有样本
        self.skip_errors = kwargs.get('skip_errors', True)
        
        # CadQuery执行参数
        self.cadquery_timeout = kwargs.get('cadquery_timeout', 30)  # 秒
        self.stl_quality = kwargs.get('stl_quality', 0.1)  # STL网格质量
        self.step_mode = kwargs.get('step_mode', 'default')  # STEP导出模式: 'default', 'fused'


class CadQueryExecutor:
    """CadQuery代码执行器"""
    
    def __init__(self, config: BatchInferenceConfig):
        self.config = config
        self.success_count = 0
        self.error_count = 0
        self.errors = []
        self.stl_success_count = 0
        self.step_success_count = 0
    
    def execute_cadquery_code(self, code: str, stl_output_path: str = None, step_output_path: str = None) -> Tuple[bool, str, Dict[str, bool]]:
        """执行CadQuery代码并生成STL和STEP文件 - 参考demo实现"""
        if not CADQUERY_AVAILABLE:
            return False, "CadQuery not available", {"stl": False, "step": False}
        
        try:
            # 清理代码
            code = self._clean_code(code)
            
            # 创建安全的执行环境
            safe_globals = {
                'cq': cq,
                'cadquery': cq,
                'math': __import__('math'),
                'numpy': np,
                'np': np,
                '__builtins__': {
                    '__import__': __import__,
                    'abs': abs, 'min': min, 'max': max, 'range': range, 'len': len,
                    'float': float, 'int': int, 'str': str, 'print': print,
                }
            }
            
            # 执行代码 - 参考demo中的exec(py_string, globals())
            exec(code, safe_globals)
            
            # 查找结果对象 - 参考demo中的globals()['r'].val()
            compound = None
            
            # 首先尝试查找名为'r'的变量（demo中的标准做法）
            if 'r' in safe_globals:
                try:
                    compound = safe_globals['r'].val()
                except:
                    # 如果r不是Workplane对象，直接使用
                    compound = safe_globals['r']
            
            # 如果没有找到'r'，查找其他可能的Workplane对象
            if compound is None:
                for name, obj in safe_globals.items():
                    if isinstance(obj, cq.Workplane) and not name.startswith('_'):
                        try:
                            compound = obj.val()
                            break
                        except:
                            compound = obj
                            break
            
            # 查找其他可能的变量名
            if compound is None:
                for var_name in ['result', 'shape', 'model', 'part']:
                    if var_name in safe_globals:
                        obj = safe_globals[var_name]
                        if isinstance(obj, cq.Workplane):
                            try:
                                compound = obj.val()
                                break
                            except:
                                compound = obj
                                break
            
            if compound is None:
                return False, "No valid CadQuery result object found in code", {"stl": False, "step": False}
            
            export_results = {"stl": False, "step": False}
            
            # 导出STL文件 - 参考demo实现
            if stl_output_path:
                stl_success = self._export_stl_demo_style(compound, stl_output_path)
                export_results["stl"] = stl_success
                if stl_success:
                    self.stl_success_count += 1
            
            # 导出STEP文件 - 参考demo实现
            if step_output_path:
                step_success = self._export_step_demo_style(compound, step_output_path)
                export_results["step"] = step_success
                if step_success:
                    self.step_success_count += 1
            
            # 如果至少有一个格式导出成功，则认为整体成功
            overall_success = any(export_results.values())
            if overall_success:
                self.success_count += 1
            else:
                self.error_count += 1
            
            return overall_success, "Success", export_results
            
        except Exception as e:
            error_msg = f"Error executing CadQuery code: {str(e)}"
            self.errors.append(error_msg)
            self.error_count += 1
            return False, error_msg, {"stl": False, "step": False}
    
    def _export_stl_demo_style(self, compound, output_path: str) -> bool:
        """使用demo风格导出STL文件"""
        try:
            # 确保输出路径有.stl扩展名
            if not output_path.lower().endswith('.stl'):
                output_path = output_path + '.stl'
            
            # 参考demo: compound.tessellate(0.001, 0.1)
            vertices, faces = compound.tessellate(
                tolerance=self.config.stl_quality, 
                angularTolerance=0.1
            )
            
            # 参考demo: trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
            try:
                import trimesh
                vertex_coords = [(v.x, v.y, v.z) for v in vertices]
                mesh = trimesh.Trimesh(vertex_coords, faces)
                # 参考demo: mesh.export('/tmp/1.stl')
                mesh.export(output_path)
                print(f"STL exported to {output_path}")
                return True
            except ImportError:
                print("Trimesh not available, falling back to CadQuery export")
                # 如果trimesh不可用，回退到CadQuery的导出方法
                # 创建临时Workplane包装compound
                temp_wp = cq.Workplane().add(compound)
                temp_wp.export(output_path, 
                              tolerance=self.config.stl_quality,
                              angularTolerance=0.1)
                print(f"STL exported to {output_path}")
                return True
                
        except Exception as e:
            print(f"STL export error: {str(e)}")
            return False
    
    def _export_step_demo_style(self, compound, output_path: str) -> bool:
        """使用demo风格导出STEP文件"""
        try:
            # 确保输出路径有.step或.stp扩展名
            if not (output_path.lower().endswith('.step') or output_path.lower().endswith('.stp')):
                output_path = output_path + '.step'
            
            # 参考demo: cq.exporters.export(compound, '/tmp/1.step')
            cq.exporters.export(compound, output_path)
            print(f"STEP exported to {output_path}")
            return True
                
        except Exception as e:
            print(f"STEP export error: {str(e)}")
            return False
    
    def _clean_code(self, code: str) -> str:
        """清理和预处理CadQuery代码"""
        # 移除可能的markdown标记
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1].split('```')[0]
        
        # 移除可能的特殊token
        for token in ['<|im_start|>', '<|im_end|>', '<|endoftext|>']:
            code = code.replace(token, '')
        
        # 确保代码以适当的导入开始
        if 'import cadquery as cq' not in code and 'cadquery' in code:
            code = 'import cadquery as cq\n' + code
        
        return code.strip()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        total = self.success_count + self.error_count
        return {
            'total_executed': total,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / total if total > 0 else 0.0,
            'stl_success_count': self.stl_success_count,
            'step_success_count': self.step_success_count,
            'stl_success_rate': self.stl_success_count / total if total > 0 else 0.0,
            'step_success_rate': self.step_success_count / total if total > 0 else 0.0,
            'errors': self.errors
        }


def load_trained_model(model_path: str, device: str = 'auto') -> Tuple[CADRecode, Any]:
    """加载训练好的模型"""
    model_path = Path(model_path)
    
    # 检查模型文件是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # 尝试加载配置
    config_file = model_path / "config.yaml"
    if config_file.exists():
        print(f"Loading config from {config_file}")
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        # 使用默认配置
        print("No config.yaml found, using default configuration")
        config = TrainingConfig()
        # 尝试从模型目录推断基础模型
        if 'qwen3' in str(model_path).lower():
            config.base_model_name = "Qwen/Qwen3-1.7B-Base"
        elif 'qwen2' in str(model_path).lower():
            config.base_model_name = "Qwen/Qwen2-1.5B"
    
    print(f"Loading model from {model_path}")
    print(f"Base model: {config.base_model_name}")
    
    # 加载分词器
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        pad_token='<|im_end|>',
        padding_side='left',
        trust_remote_code=True
    )
    
    # 加载模型
    model = CADRecode.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer


def load_inference_data(data_path: str, num_samples: Optional[int] = None, skip_errors: bool = True) -> List[Dict[str, Any]]:
    """加载推理数据"""
    data_path = Path(data_path)
    
    # 加载数据索引
    data_index = load_data_index(data_path, "val")
    
    # 加载错误样本列表
    if skip_errors:
        error_samples = load_error_samples(data_path)
        if error_samples:
            original_count = len(data_index)
            data_index = [item for item in data_index 
                         if item.get("sample_id", item.get("id")) not in error_samples]
            filtered_count = original_count - len(data_index)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} known error samples")
    
    # 限制样本数量
    if num_samples is not None and num_samples < len(data_index):
        data_index = data_index[:num_samples]
        print(f"Limited to {num_samples} samples")
    
    print(f"Loaded {len(data_index)} samples for inference")
    return data_index


def run_batch_inference(config: BatchInferenceConfig):
    """执行批量推理"""
    # 设置环境
    setup_environment()
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    code_dir = output_dir / "generated_code"
    stl_dir = output_dir / "generated_stl"
    step_dir = output_dir / "generated_step"  # 新增STEP目录
    results_dir = output_dir / "results"
    
    if config.save_code:
        code_dir.mkdir(exist_ok=True)
    if config.save_stl:
        stl_dir.mkdir(exist_ok=True)
    if config.save_step:
        step_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # 加载模型
    print("Loading trained model...")
    model, tokenizer = load_trained_model(config.model_path, config.device)
    print(model)
    # 加载数据
    print("Loading inference data...")
    data_index = load_inference_data(config.data_path, config.num_samples, config.skip_errors)
    
    # 初始化CadQuery执行器
    executor = CadQueryExecutor(config)
    
    # 存储推理结果
    inference_results = []
    
    # 开始推理
    print("Starting batch inference...")
    data_root = Path(config.data_path)
    
    for idx, item in enumerate(tqdm(data_index, desc="Inference")):
        try:
            # 获取样本信息
            if "sample_id" in item:
                sample_id = item["sample_id"]
                code_path = item.get("code_path")
            else:
                sample_id = item.get("id", f"sample_{idx}")
                code_path = item.get("py_file", item.get("code_path"))
            
            # 加载点云
            point_cloud = get_cached_point_cloud(data_root, sample_id, 256)  # 使用固定的256个点
            if point_cloud is None:
                print(f"Warning: No point cloud found for {sample_id}, skipping")
                continue
            
            # 执行推理
            generated_code = inference_single_sample(
                model, tokenizer, point_cloud, config.max_new_tokens
            )
            
            # 准备结果
            result = {
                'sample_id': sample_id,
                'generated_code': generated_code,
                'timestamp': datetime.now().isoformat(),
                'model_path': str(config.model_path)
            }
            
            # 保存生成的代码
            if config.save_code:
                code_file = code_dir / f"{sample_id}.py"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                result['code_file'] = str(code_file)
            
            # 执行CadQuery代码生成STL和STEP文件
            stl_file = None
            step_file = None
            
            if config.save_stl:
                stl_file = stl_dir / f"{sample_id}.stl"
            
            if config.save_step:
                step_file = step_dir / f"{sample_id}.step"
            
            if stl_file or step_file:
                success, error_msg, export_results = executor.execute_cadquery_code(
                    generated_code, 
                    str(stl_file) if stl_file else None,
                    str(step_file) if step_file else None
                )
                
                result['overall_success'] = success
                result['overall_error'] = error_msg if not success else None
                
                # STL结果
                if config.save_stl:
                    result['stl_generated'] = export_results.get('stl', False)
                    result['stl_file'] = str(stl_file) if export_results.get('stl', False) else None
                
                # STEP结果
                if config.save_step:
                    result['step_generated'] = export_results.get('step', False)
                    result['step_file'] = str(step_file) if export_results.get('step', False) else None
            
            # 加载真实代码（如果存在）用于比较
            if code_path and Path(code_path).exists():
                with open(code_path, 'r', encoding='utf-8') as f:
                    result['ground_truth_code'] = f.read()
            
            inference_results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            if not config.skip_errors:
                raise
            continue
    
    # 保存推理结果
    if config.save_predictions:
        results_file = results_dir / "inference_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, indent=2, ensure_ascii=False)
        print(f"Inference results saved to {results_file}")
    
    # 生成统计报告
    statistics = executor.get_statistics()
    statistics['total_samples'] = len(inference_results)
    statistics['config'] = config.__dict__
    
    stats_file = results_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Total samples processed: {len(inference_results)}")
    print(f"Overall success rate: {statistics['success_rate']:.2%}")
    if config.save_stl:
        print(f"STL generation success rate: {statistics['stl_success_rate']:.2%}")
    if config.save_step:
        print(f"STEP generation success rate: {statistics['step_success_rate']:.2%}")
    print(f"Output directory: {output_dir}")
    print(f"Statistics saved to: {stats_file}")
    
    if statistics['error_count'] > 0:
        print(f"\nTop errors:")
        for error in statistics['errors'][:5]:
            print(f"  - {error}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CAD-Recode Batch Inference Script")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model directory")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to inference data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    
    # 可选参数
    parser.add_argument("--max_new_tokens", type=int, default=768,
                       help="Maximum new tokens to generate")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu)")
    
    # 输出控制
    parser.add_argument("--no_code", action="store_true",
                       help="Don't save generated code files")
    parser.add_argument("--no_stl", action="store_true",
                       help="Don't generate STL files")
    parser.add_argument("--no_step", action="store_true",
                       help="Don't generate STEP files")
    parser.add_argument("--no_predictions", action="store_true",
                       help="Don't save prediction results")
    
    # 其他选项
    parser.add_argument("--include_errors", action="store_true",
                       help="Include known error samples in inference")
    parser.add_argument("--cadquery_timeout", type=int, default=30,
                       help="Timeout for CadQuery execution (seconds)")
    parser.add_argument("--step_mode", type=str, default="default", 
                       choices=["default", "fused"],
                       help="STEP export mode: default or fused")
    
    args = parser.parse_args()
    
    # 创建配置
    config = BatchInferenceConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        device=args.device,
        save_code=not args.no_code,
        save_stl=not args.no_stl,
        save_step=not args.no_step,
        save_predictions=not args.no_predictions,
        skip_errors=not args.include_errors,
        cadquery_timeout=args.cadquery_timeout,
        step_mode=args.step_mode
    )
    
    print("Batch Inference Configuration:")
    print(f"  Model Path: {config.model_path}")
    print(f"  Data Path: {config.data_path}")
    print(f"  Output Dir: {config.output_dir}")
    print(f"  Max New Tokens: {config.max_new_tokens}")
    print(f"  Device: {config.device}")
    print(f"  Save Code: {config.save_code}")
    print(f"  Save STL: {config.save_stl}")
    print(f"  Save STEP: {config.save_step}")
    if config.save_step:
        print(f"  STEP Mode: {config.step_mode}")
    if config.num_samples:
        print(f"  Sample Limit: {config.num_samples}")
    
    # 运行推理
    run_batch_inference(config)


if __name__ == "__main__":
    main()
