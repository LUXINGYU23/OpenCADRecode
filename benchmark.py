#!/usr/bin/env python3
"""
CAD-Recode Benchmark Script
对训练的模型和官方模型在验证数据集上进行评估和比较

主要功能：
1. 加载多个模型（官方模型 + 训练的模型）
2. 在data/val数据集上进行批量推理
3. 生成CadQuery代码和STL/STEP文件
4. 计算几何指标（IoU、Chamfer距离等）
5. 生成详细的评估报告

使用方法:
python benchmark.py --models official checkpoints_qwen3_sft --num_samples 100
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import argparse
import traceback
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from utils import (
    setup_environment, 
    get_cached_point_cloud, 
    inference_single_sample,
    NUM_POINT_TOKENS
)
from models import CADRecode, create_model_and_tokenizer
from data_utils import load_data_index, load_error_samples
from metric import MetricsCalculator, SamplingConfig, MetricsResult
from batch_inference import CadQueryExecutor, BatchInferenceConfig

# 导入官方模型定义（如果可用）
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    print("Warning: CadQuery not installed. STL generation will be disabled.")


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    path: str
    model_type: str  # 'official', 'sft', 'lora', 'lora_from_full'
    base_model: Optional[str] = None  # LoRA模型需要指定base model
    

@dataclass
class BenchmarkConfig:
    """Benchmark配置"""
    models: List[ModelConfig]
    data_path: str = "data/val"
    output_dir: str = "benchmark_results"
    num_samples: Optional[int] = None  # None表示全部样本
    batch_size: int = 1
    max_new_tokens: int = 768
    device: str = "auto"
    
    # 输出控制
    save_code: bool = True
    save_stl: bool = True
    save_step: bool = True
    
    # 评估参数
    n_points_metric: int = 2000  # 用于计算指标的点数
    cadquery_timeout: int = 30
    skip_failed_generations: bool = True
    
    # 并行参数
    use_multiprocessing: bool = True


class OfficialCADRecode:
    """官方CAD-Recode模型包装类"""
    
    def __init__(self):
        from transformers import AutoTokenizer
        import torch.nn as nn
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers import Qwen2ForCausalLM, Qwen2Model, PreTrainedModel
        
        class FourierPointEncoder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
                self.register_buffer('frequencies', frequencies, persistent=False)
                self.projection = nn.Linear(51, hidden_size)

            def forward(self, points):
                x = points
                x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
                x = torch.cat((points, x.sin(), x.cos()), dim=-1)
                x = self.projection(x)
                return x
        
        class OfficialCADRecode(Qwen2ForCausalLM):
            def __init__(self, config):
                PreTrainedModel.__init__(self, config)
                self.model = Qwen2Model(config)
                self.vocab_size = config.vocab_size
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                
                torch.set_default_dtype(torch.float32)
                self.point_encoder = FourierPointEncoder(config.hidden_size)
                torch.set_default_dtype(torch.bfloat16)

            def forward(self, input_ids=None, attention_mask=None, point_cloud=None, 
                       position_ids=None, past_key_values=None, inputs_embeds=None, 
                       labels=None, use_cache=None, output_attentions=None, 
                       output_hidden_states=None, return_dict=None, cache_position=None):
                
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                # concatenate point and text embeddings
                if past_key_values is None or past_key_values.get_seq_length() == 0:
                    assert inputs_embeds is None
                    inputs_embeds = self.model.embed_tokens(input_ids)
                    point_embeds = self.point_encoder(point_cloud).bfloat16()
                    inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
                    attention_mask[attention_mask == -1] = 1
                    input_ids = None
                    position_ids = None

                # decoder outputs
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
                    cache_position=cache_position)

                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states)
                logits = logits.float()

                loss = None
                if labels is not None:
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
                    attentions=outputs.attentions)

            def prepare_inputs_for_generation(self, *args, **kwargs):
                model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
                model_inputs['point_cloud'] = kwargs['point_cloud']
                return model_inputs
        
        self.OfficialCADRecode = OfficialCADRecode


class BenchmarkRunner:
    """Benchmark运行器"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化CadQuery执行器
        batch_config = BatchInferenceConfig(
            output_dir=str(self.output_dir),
            cadquery_timeout=config.cadquery_timeout,
            save_code=config.save_code,
            save_stl=config.save_stl,
            save_step=config.save_step
        )
        self.cadquery_executor = CadQueryExecutor(batch_config)
        
        # 初始化指标计算器
        sampling_config = SamplingConfig(n_points=config.n_points_metric)
        self.metrics_calculator = MetricsCalculator(sampling_config=sampling_config)
        
        # 加载数据索引
        self.data_index = load_data_index(Path(config.data_path), "val")
        self.error_samples = load_error_samples(Path(config.data_path))
        
        # 处理简单格式的数据索引（val目录直接包含.py文件的情况）
        if not self.data_index:
            print("No index found, scanning val directory for .py files...")
            data_path = Path(config.data_path)
            self.data_index = []
            for py_file in sorted(data_path.glob("*.py")):
                sample_id = py_file.stem  # 文件名去掉.py后缀
                self.data_index.append({
                    "id": sample_id,
                    "py_file": str(py_file),
                    "code_path": str(py_file)  # 兼容两种格式
                })
        
        # 过滤错误样本
        if self.error_samples:
            original_count = len(self.data_index)
            self.data_index = [item for item in self.data_index 
                             if item.get("sample_id", item.get("id")) not in self.error_samples]
            filtered_count = original_count - len(self.data_index)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} known error samples")
        
        # 限制样本数量
        if config.num_samples:
            self.data_index = self.data_index[:config.num_samples]
        
        print(f"Loaded {len(self.data_index)} samples for benchmarking")
        
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, model_config: ModelConfig) -> Tuple[Any, Any]:
        """加载模型和分词器"""
        print(f"\\n🔄 Loading model: {model_config.name} ({model_config.model_type})")
        
        if model_config.model_type == "official":
            return self._load_official_model(model_config)
        elif model_config.model_type == "sft":
            return self._load_sft_model(model_config)
        elif model_config.model_type in ["lora", "lora_from_full"]:
            return self._load_lora_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
    
    def _load_official_model(self, model_config: ModelConfig) -> Tuple[Any, Any]:
        """加载官方模型"""
        from transformers import AutoTokenizer
        
        # 检查官方模型是否存在
        model_path = Path(model_config.path)
        if not model_path.exists():
            # 尝试直接从HuggingFace加载
            print(f"Local model not found, trying to load from HuggingFace: filapro/cad-recode-v1.5")
            model_config.path = "filapro/cad-recode-v1.5"
        
        # 使用官方推荐的加载方式
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attn_implementation = None  # 不使用 flash_attention_2，使用普通注意力机制
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2-1.5B',
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
        
        # 动态创建官方模型类
        official_model_cls = OfficialCADRecode().OfficialCADRecode
        
        model = official_model_cls.from_pretrained(
            model_config.path,
            torch_dtype='auto',
            attn_implementation=attn_implementation,
            trust_remote_code=True
        ).eval().to(device)
        
        print(f"✅ Official model loaded from {model_config.path}")
        return model, tokenizer
    
    def _load_sft_model(self, model_config: ModelConfig) -> Tuple[Any, Any]:
        """加载SFT模型"""
        from transformers import AutoTokenizer, AutoConfig
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.path,
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
        
        # 加载模型
        model_config_obj = AutoConfig.from_pretrained(model_config.path, trust_remote_code=True)
        if hasattr(model_config_obj, 'sliding_window'):
            model_config_obj.sliding_window = None
        
        model = CADRecode.from_pretrained(
            model_config.path,
            config=model_config_obj,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        
        # 移动到设备
        device = torch.device("cuda" if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")
        model = model.to(device)
        
        print(f"✅ SFT model loaded from {model_config.path}")
        return model, tokenizer
    
    def _load_lora_model(self, model_config: ModelConfig) -> Tuple[Any, Any]:
        """加载LoRA模型"""
        from inference_lora import load_lora_model
        
        if not model_config.base_model:
            # 根据模型类型推断base model
            if model_config.model_type == "lora":
                model_config.base_model = "Qwen/Qwen3-1.7B-Base"
            else:  # lora_from_full
                # 需要用户指定或从配置推断
                model_config.base_model = "checkpoints_qwen3_sft"
        
        model, tokenizer = load_lora_model(
            model_config.base_model,
            model_config.path,
            device=self.config.device
        )
        
        print(f"✅ LoRA model loaded from {model_config.path}")
        return model, tokenizer
    
    def inference_single_sample_official(self, model, tokenizer, point_cloud: np.ndarray, sample_id: str) -> str:
        """使用官方模型进行单样本推理 - 完全按照demo.ipynb的方式"""
        try:
            # 按照官方demo的方式构建输入
            input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
            attention_mask = [-1] * len(point_cloud) + [1]
            
            with torch.no_grad():
                batch_ids = model.generate(
                    input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
                    attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(model.device),
                    point_cloud=torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).to(model.device),
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            py_string = tokenizer.batch_decode(batch_ids)[0]
            begin = py_string.find('<|im_start|>') + 12
            end = py_string.find('<|endoftext|>')
            if end == -1:
                end = len(py_string)
            py_string = py_string[begin:end]
            
            return py_string.strip()
            
        except Exception as e:
            print(f"Error in official inference for {sample_id}: {e}")
            return f"# Error during inference: {e}\\nr = cq.Workplane().box(1, 1, 1)"
    
    def inference_single_sample_custom(self, model, tokenizer, point_cloud: np.ndarray, sample_id: str) -> str:
        """使用自定义模型进行单样本推理"""
        try:
            return inference_single_sample(
                model=model,
                tokenizer=tokenizer,
                point_cloud=point_cloud,
                max_new_tokens=self.config.max_new_tokens
            )
        except Exception as e:
            print(f"Error in custom inference for {sample_id}: {e}")
            return f"# Error during inference: {e}\\nr = cq.Workplane().box(1, 1, 1)"
    
    def run_benchmark(self):
        """运行benchmark"""
        print(f"\\n🚀 Starting benchmark with {len(self.config.models)} models on {len(self.data_index)} samples")
        
        # 加载所有模型
        for model_config in self.config.models:
            model, tokenizer = self.load_model(model_config)
            self.models[model_config.name] = model
            self.tokenizers[model_config.name] = tokenizer
        
        # 创建输出目录结构
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        for model_config in self.config.models:
            model_dir = results_dir / model_config.name
            model_dir.mkdir(exist_ok=True)
            (model_dir / "generated_code").mkdir(exist_ok=True)
            (model_dir / "generated_stl").mkdir(exist_ok=True)
            (model_dir / "generated_step").mkdir(exist_ok=True)
        
        # 运行推理
        all_results = {}
        for model_config in self.config.models:
            print(f"\\n📊 Running inference with {model_config.name}...")
            model_results = self._run_model_inference(model_config)
            all_results[model_config.name] = model_results
        
        # 计算几何指标
        print(f"\\n📏 Computing geometric metrics...")
        metrics_results = self._compute_metrics(all_results)
        
        # 生成报告
        print(f"\\n📋 Generating benchmark report...")
        self._generate_report(all_results, metrics_results)
        
        print(f"\\n✅ Benchmark completed! Results saved to {self.output_dir}")
    
    def _run_model_inference(self, model_config: ModelConfig) -> Dict[str, Any]:
        """运行单个模型的推理"""
        model = self.models[model_config.name]
        tokenizer = self.tokenizers[model_config.name]
        model_dir = self.output_dir / "results" / model_config.name
        
        results = {
            "model_config": asdict(model_config),
            "samples": {},
            "generation_stats": {
                "total_samples": len(self.data_index),
                "successful_generations": 0,
                "failed_generations": 0,
                "successful_stl": 0,
                "failed_stl": 0
            }
        }
        
        for i, sample_info in enumerate(tqdm(self.data_index, desc=f"Inference {model_config.name}")):
            sample_id = sample_info.get("sample_id", sample_info.get("id"))
            
            # 加载点云
            point_cloud = get_cached_point_cloud(
                Path(self.config.data_path), 
                sample_id, 
                num_points=NUM_POINT_TOKENS
            )
            
            if point_cloud is None:
                print(f"Warning: No point cloud found for sample {sample_id}")
                continue
            
            # 推理生成代码
            if model_config.model_type == "official":
                generated_code = self.inference_single_sample_official(model, tokenizer, point_cloud, sample_id)
            else:
                generated_code = self.inference_single_sample_custom(model, tokenizer, point_cloud, sample_id)
            
            # 保存生成的代码
            if self.config.save_code:
                code_path = model_dir / "generated_code" / f"{sample_id}.py"
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
            
            # 执行CadQuery代码生成几何体
            stl_path = model_dir / "generated_stl" / f"{sample_id}.stl"
            step_path = model_dir / "generated_step" / f"{sample_id}.step"
            
            success, error_msg, file_status = self.cadquery_executor.execute_cadquery_code(
                generated_code,
                str(stl_path) if self.config.save_stl else None,
                str(step_path) if self.config.save_step else None
            )
            
            # 记录结果
            sample_result = {
                "sample_id": sample_id,
                "generated_code": generated_code,
                "execution_success": success,
                "error_message": error_msg if not success else None,
                "stl_generated": file_status.get("stl", False),
                "step_generated": file_status.get("step", False)
            }
            
            results["samples"][sample_id] = sample_result
            
            # 更新统计
            if success:
                results["generation_stats"]["successful_generations"] += 1
                if file_status.get("stl", False):
                    results["generation_stats"]["successful_stl"] += 1
                else:
                    results["generation_stats"]["failed_stl"] += 1
            else:
                results["generation_stats"]["failed_generations"] += 1
                results["generation_stats"]["failed_stl"] += 1
        
        # 保存模型结果
        results_file = model_dir / "inference_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _prepare_gt_step_files(self) -> Path:
        """为GT数据生成STEP文件"""
        gt_step_dir = self.output_dir / "gt_step_files"
        gt_step_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Preparing GT STEP files...")
        
        # 创建临时的CadQuery执行器用于生成GT STEP文件
        temp_config = BatchInferenceConfig(
            output_dir=str(self.output_dir),
            cadquery_timeout=self.config.cadquery_timeout,
            save_code=False,
            save_stl=False,
            save_step=True
        )
        gt_executor = CadQueryExecutor(temp_config)
        
        generated_count = 0
        failed_count = 0
        
        for sample_info in tqdm(self.data_index, desc="Generating GT STEP files"):
            sample_id = sample_info.get("sample_id", sample_info.get("id"))
            
            # 处理不同的数据格式
            if "code_path" in sample_info:
                gt_code_path = Path(sample_info["code_path"])
            else:
                # 简单格式，直接使用sample_id作为文件名
                gt_code_path = Path(self.config.data_path) / f"{sample_id}.py"
            
            gt_step_path = gt_step_dir / f"{sample_id}.step"
            
            # 如果STEP文件已存在，跳过
            if gt_step_path.exists():
                generated_count += 1
                continue
            
            # 读取GT代码
            if not gt_code_path.exists():
                print(f"Warning: GT code file not found: {gt_code_path}")
                failed_count += 1
                continue
            
            try:
                with open(gt_code_path, 'r', encoding='utf-8') as f:
                    gt_code = f.read()
                
                # 执行GT代码生成STEP文件
                success, error_msg, file_status = gt_executor.execute_cadquery_code(
                    gt_code,
                    stl_output_path=None,  # 不需要STL
                    step_output_path=str(gt_step_path)
                )
                
                if success and file_status.get("step", False):
                    generated_count += 1
                else:
                    print(f"Failed to generate GT STEP for {sample_id}: {error_msg}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"Error processing GT code for {sample_id}: {e}")
                failed_count += 1
        
        print(f"GT STEP generation completed: {generated_count} success, {failed_count} failed")
        return gt_step_dir
    
    def _compute_metrics(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算几何指标 - 使用STEP格式进行BREP计算"""
        metrics_results = {}
        
        # 为GT数据生成STEP文件
        gt_step_dir = self._prepare_gt_step_files()
        
        for model_name, model_results in all_results.items():
            print(f"  Computing metrics for {model_name}...")
            model_dir = self.output_dir / "results" / model_name
            pred_step_dir = model_dir / "generated_step"
            
            # 收集需要计算指标的文件对
            file_pairs = []
            for sample_id, sample_result in model_results["samples"].items():
                if sample_result["step_generated"]:
                    gt_step = gt_step_dir / f"{sample_id}.step"
                    pred_step = pred_step_dir / f"{sample_id}.step"
                    
                    if gt_step.exists() and pred_step.exists():
                        file_pairs.append((str(gt_step), str(pred_step)))
            
            if not file_pairs:
                print(f"    No valid file pairs found for {model_name}")
                metrics_results[model_name] = {"error": "No valid file pairs"}
                continue
            
            print(f"    Found {len(file_pairs)} valid GT-Prediction pairs for metric calculation")
            
            # 直接计算指标 - 使用简化的方法
            try:
                batch_results = []
                print(f"    Computing metrics for each pair...")
                
                for i, (gt_file, pred_file) in enumerate(file_pairs):
                    try:
                        result = self.metrics_calculator.compute_files_metrics(
                            ext="step",
                            file1=gt_file,
                            file2=pred_file,
                            matching_threshold=None,
                            include_normals=True,
                            use_icp=True
                        )
                        
                        if result is not None:
                            result_dict = result.to_dict()
                            result_dict['file1'] = gt_file.replace("\\", "/")
                            result_dict['file2'] = pred_file.replace("\\", "/")
                            result_dict['pair_index'] = i
                            batch_results.append(result_dict)
                            print(f"      ✅ Pair {i+1}/{len(file_pairs)}: Success")
                        else:
                            print(f"      ❌ Pair {i+1}/{len(file_pairs)}: Failed")
                    except Exception as e:
                        print(f"      ❌ Pair {i+1}/{len(file_pairs)}: Error - {e}")
                
                print(f"    Computed {len(batch_results)} successful metric results for {model_name}")
                # 计算平均指标
                if batch_results:
                    metrics = self._compute_average_metrics(batch_results)
                    metrics["total_pairs"] = len(file_pairs)
                    metrics["successful_pairs"] = len(batch_results)
                    
                    print(f"    Metrics computed for {metrics['successful_pairs']}/{metrics['total_pairs']} pairs")
                else:
                    metrics = {"error": "No successful metric computations"}
                
                metrics_results[model_name] = metrics
                
                # 保存详细指标
                metrics_file = model_dir / "detailed_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump({
                        "summary": metrics,
                        "batch_results": batch_results,
                        "file_pairs": [(str(p[0]), str(p[1])) for p in file_pairs]
                    }, f, indent=2)
                
            except Exception as e:
                print(f"    Error computing metrics for {model_name}: {e}")
                traceback.print_exc()
                metrics_results[model_name] = {"error": str(e)}
        
        return metrics_results
    
    def _compute_average_metrics(self, batch_results: List[Dict]) -> Dict[str, float]:
        """计算平均指标"""
        if not batch_results:
            return {}
        
        # 根据metric.py定义的指标
        metrics = [
            'chamfer_distance', 'hausdorff_distance', 'earth_mover_distance', 'rms_error',
            'iou_voxel', 'iou_brep', 'matching_rate', 'coverage_rate', 
            'normal_consistency', 'computation_time'
        ]
        
        def get_values(key):
            values = []
            for res in batch_results:
                value = res.get(key)
                # 只处理数值类型的值
                if value is not None and isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(value)
            return values
        
        result = {}
        for metric in metrics:
            vals = get_values(metric)
            if len(vals) > 0:
                result[f"{metric}_mean"] = float(np.mean(vals))
                result[f"{metric}_std"] = float(np.std(vals))
                result[f"{metric}_median"] = float(np.median(vals))
                result[f"{metric}_min"] = float(np.min(vals))
                result[f"{metric}_max"] = float(np.max(vals))
        
        return result
    
    def _generate_report(self, all_results: Dict[str, Any], metrics_results: Dict[str, Any]):
        """生成benchmark报告"""
        report = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "total_samples": len(self.data_index)
            },
            "models": {}
        }
        
        # 汇总每个模型的结果
        for model_name in all_results.keys():
            model_data = all_results[model_name]
            model_metrics = metrics_results.get(model_name, {})
            
            # 生成统计
            stats = model_data["generation_stats"]
            metrics_summary = {}
            
            if "error" not in model_metrics:
                # 主要指标
                main_metrics = [
                    "chamfer_distance", "hausdorff_distance", "earth_mover_distance",
                    "iou_voxel", "iou_brep", "matching_rate", "coverage_rate", "normal_consistency"
                ]
                
                for metric in main_metrics:
                    mean_key = f"{metric}_mean"
                    std_key = f"{metric}_std"
                    if mean_key in model_metrics:
                        metrics_summary[metric] = {
                            "mean": model_metrics[mean_key],
                            "std": model_metrics[std_key]
                        }
            
            report["models"][model_name] = {
                "config": model_data["model_config"],
                "generation_stats": stats,
                "metrics": metrics_summary if metrics_summary else model_metrics
            }
        
        # 保存报告
        report_file = self.output_dir / "benchmark_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        # 打印总结
        self._print_summary(report)
    
    def _generate_markdown_report(self, report: Dict):
        """生成Markdown格式的报告"""
        md_content = f"""# CAD-Recode Benchmark Report

**Generated:** {report['benchmark_info']['timestamp']}  
**Total Samples:** {report['benchmark_info']['total_samples']}

## Model Comparison

| Model | Type | Success Rate | STL Success | Chamfer↓ | IoU↑ | Matching Rate↑ |
|-------|------|-------------|-------------|----------|------|----------------|
"""
        
        for model_name, model_data in report["models"].items():
            config = model_data["config"]
            stats = model_data["generation_stats"]
            metrics = model_data["metrics"]
            
            # 计算成功率
            total = stats["total_samples"]
            success_rate = stats["successful_generations"] / total * 100 if total > 0 else 0
            stl_success_rate = stats["successful_stl"] / total * 100 if total > 0 else 0
            
            # 提取关键指标
            chamfer = metrics.get("chamfer_distance", {})
            iou = metrics.get("iou_brep", {}) or metrics.get("iou_voxel", {})
            matching = metrics.get("matching_rate", {})
            
            chamfer_str = f"{chamfer.get('mean', 0):.4f}" if isinstance(chamfer, dict) else "N/A"
            iou_str = f"{iou.get('mean', 0):.4f}" if isinstance(iou, dict) else "N/A"
            matching_str = f"{matching.get('mean', 0):.4f}" if isinstance(matching, dict) else "N/A"
            
            md_content += f"| {model_name} | {config['model_type']} | {success_rate:.1f}% | {stl_success_rate:.1f}% | {chamfer_str} | {iou_str} | {matching_str} |\\n"
        
        md_content += f"""
## Detailed Results

"""
        
        for model_name, model_data in report["models"].items():
            metrics = model_data["metrics"]
            md_content += f"""### {model_name}

**Generation Statistics:**
- Total Samples: {model_data['generation_stats']['total_samples']}
- Successful Generations: {model_data['generation_stats']['successful_generations']}
- Failed Generations: {model_data['generation_stats']['failed_generations']}
- Successful STL: {model_data['generation_stats']['successful_stl']}

"""
            if isinstance(metrics, dict) and "error" not in metrics:
                md_content += """**Geometric Metrics:**
"""
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        md_content += f"- {metric_name}: {metric_data['mean']:.6f} ± {metric_data['std']:.6f}\\n"
            else:
                md_content += f"**Metrics Error:** {metrics.get('error', 'Unknown error')}\\n"
            
            md_content += "\\n"
        
        # 保存Markdown报告
        md_file = self.output_dir / "benchmark_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _print_summary(self, report: Dict):
        """打印结果总结"""
        print(f"\\n{'='*60}")
        print(f"🎯 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        for model_name, model_data in report["models"].items():
            stats = model_data["generation_stats"]
            metrics = model_data["metrics"]
            
            print(f"\\n📊 {model_name} ({model_data['config']['model_type']})")
            print(f"   Generation Success: {stats['successful_generations']}/{stats['total_samples']} ({stats['successful_generations']/stats['total_samples']*100:.1f}%)")
            print(f"   STL Success: {stats['successful_stl']}/{stats['total_samples']} ({stats['successful_stl']/stats['total_samples']*100:.1f}%)")
            
            if isinstance(metrics, dict) and "error" not in metrics:
                # 打印关键指标
                key_metrics = ["chamfer_distance", "iou_brep", "matching_rate"]
                for metric in key_metrics:
                    if metric in metrics:
                        data = metrics[metric]
                        if isinstance(data, dict) and "mean" in data:
                            print(f"   {metric}: {data['mean']:.6f} ± {data['std']:.6f}")
            else:
                print(f"   Metrics: {metrics.get('error', 'Error occurred')}")
        
        print(f"\\n✅ Complete results saved to: {self.output_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CAD-Recode Benchmark Script")
    
    parser.add_argument("--models", nargs="+", required=True,
                       help="List of models to benchmark. Use 'official' for the official model, or provide paths to your trained models.")
    parser.add_argument("--model_types", nargs="+", 
                       help="Model types corresponding to --models. Options: official, sft, lora, lora_from_full")
    parser.add_argument("--base_models", nargs="+",
                       help="Base models for LoRA models (required for lora type)")
    
    parser.add_argument("--data_path", default="data/val",
                       help="Path to validation data")
    parser.add_argument("--output_dir", default="benchmark_results",
                       help="Output directory for results")
    
    parser.add_argument("--num_samples", type=int,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=768,
                       help="Maximum new tokens for generation")
    
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--no_stl", action="store_true",
                       help="Skip STL generation")
    parser.add_argument("--no_step", action="store_true",
                       help="Skip STEP generation")
    parser.add_argument("--no_code", action="store_true",
                       help="Skip code saving")
    
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout for CadQuery execution")
    parser.add_argument("--metric_points", type=int, default=2000,
                       help="Number of points for metric calculation")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置环境
    setup_environment()
    
    # 构建模型配置
    model_configs = []
    for i, model_path in enumerate(args.models):
        if model_path == "official":
            model_config = ModelConfig(
                name="official",
                path="cad-recode-v1.5",  # 官方模型路径
                model_type="official"
            )
        else:
            # 推断模型类型
            if args.model_types and i < len(args.model_types):
                model_type = args.model_types[i]
            else:
                # 根据路径推断
                if "lora" in model_path.lower():
                    if "from_full" in model_path.lower():
                        model_type = "lora_from_full"
                    else:
                        model_type = "lora"
                else:
                    model_type = "sft"
            
            # 基础模型
            base_model = None
            if model_type in ["lora", "lora_from_full"]:
                if args.base_models and i < len(args.base_models):
                    base_model = args.base_models[i]
                elif model_type == "lora":
                    base_model = "Qwen/Qwen3-1.7B-Base"
                elif model_type == "lora_from_full":
                    base_model = "checkpoints_qwen3_sft"  # 默认值，可能需要调整
            
            model_config = ModelConfig(
                name=Path(model_path).name,
                path=model_path,
                model_type=model_type,
                base_model=base_model
            )
        
        model_configs.append(model_config)
    
    # 创建benchmark配置
    config = BenchmarkConfig(
        models=model_configs,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        save_code=not args.no_code,
        save_stl=not args.no_stl,
        save_step=not args.no_step,
        n_points_metric=args.metric_points,
        cadquery_timeout=args.timeout
    )
    
    # 运行benchmark
    runner = BenchmarkRunner(config)
    runner.run_benchmark()


if __name__ == "__main__":
    main()
