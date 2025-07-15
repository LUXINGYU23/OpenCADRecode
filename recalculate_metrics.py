#!/usr/bin/env python3
"""
CAD-Recode Benchmark Metrics Recalculation Script
重新计算已完成推理的模型的几何指标，增加内存监控和样本跳过功能
"""

import os
import sys
import json
import argparse
import traceback
import psutil
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

# 导入必要的自定义模块
from utils import setup_environment, NUM_POINT_TOKENS
from data_utils import load_error_samples
from metric import MetricsCalculator, SamplingConfig, MetricsResult
from batch_inference import CadQueryExecutor, BatchInferenceConfig

# 尝试导入CadQuery
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    print("Warning: CadQuery not installed. STL/STEP processing will be limited.")

# 尝试导入OCC
try:
    import OCC
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False
    print("Warning: OCC not installed. Some STEP processing features will be disabled.")


@dataclass
class RecalculateConfig:
    """重新计算配置"""
    results_dir: str  # 原有benchmark结果目录
    output_dir: Optional[str] = None  # 新结果输出目录，默认在原有目录下
    num_samples: Optional[int] = None  # 限制样本数量，None表示全部
    n_points_metric: int = 2000  # 用于计算指标的点数
    cadquery_timeout: int = 30
    device: str = "auto"
    memory_threshold: float = 0.8  # 内存使用阈值 (0.0-1.0)
    skip_failed_samples: bool = True  # 跳过失败的样本


class MemoryMonitor:
    """内存监控工具"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.stop_event = None
    
    def check_memory(self) -> float:
        """检查当前内存使用率"""
        memory_info = psutil.virtual_memory()
        return memory_info.percent / 100.0
    
    def is_over_threshold(self) -> bool:
        """检查内存是否超过阈值"""
        return self.check_memory() >= self.threshold
    
    def run_monitor(self, check_interval: float = 1.0) -> None:
        """运行内存监控线程"""
        import threading
        
        self.monitoring = True
        self.stop_event = threading.Event()
        
        def monitor_loop():
            while not self.stop_event.is_set():
                if self.is_over_threshold():
                    print(f"Memory usage ({self.check_memory()*100:.1f}%) exceeded threshold ({self.threshold*100:.1f}%)!")
                time.sleep(check_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitor(self) -> None:
        """停止内存监控线程"""
        if self.monitoring and self.stop_event:
            self.stop_event.set()
            self.monitor_thread.join()
            self.monitoring = False


class MetricsRecalculator:
    """指标重新计算工具"""
    
    def __init__(self, config: RecalculateConfig):
        self.config = config
        
        # 设置结果目录
        self.base_results_dir = Path(config.results_dir)
        if not self.base_results_dir.exists():
            raise ValueError(f"Results directory not found: {self.base_results_dir}")
        
        # 设置输出目录
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
        else:
            self.output_dir = self.base_results_dir / "recalculated_metrics"
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建日志文件
        self.log_file = self.output_dir / "recalculation.log"
        self._log("Starting metrics recalculation...")
        
        # 初始化指标计算器
        sampling_config = SamplingConfig(n_points=config.n_points_metric)
        self.metrics_calculator = MetricsCalculator(sampling_config=sampling_config)
        
        # 加载原有配置
        self.benchmark_config = self._load_benchmark_config()
        
        # 加载数据索引
        self.data_index = self._load_data_index()
        
        # 过滤错误样本
        self.error_samples = self._load_error_samples()
        if self.error_samples:
            original_count = len(self.data_index)
            self.data_index = [item for item in self.data_index 
                             if item.get("sample_id", item.get("id")) not in self.error_samples]
            filtered_count = original_count - len(self.data_index)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} known error samples")
                self._log(f"Filtered out {filtered_count} known error samples")
        
        # 限制样本数量
        if config.num_samples:
            self.data_index = self.data_index[:config.num_samples]
        
        print(f"Loaded {len(self.data_index)} samples for metrics recalculation")
        self._log(f"Loaded {len(self.data_index)} samples for metrics recalculation")
        
        # 获取所有模型名称
        self.model_names = self._get_model_names()
        print(f"Found {len(self.model_names)} models to process: {', '.join(self.model_names)}")
        self._log(f"Found {len(self.model_names)} models to process: {', '.join(self.model_names)}")
        
        # 准备GT STEP文件
        self.gt_step_dir = self._prepare_gt_step_files()
        
        # 初始化内存监控器
        self.memory_monitor = MemoryMonitor(threshold=config.memory_threshold)
        
        # 记录失败的样本
        self.failed_samples = defaultdict(list)
    
    def _log(self, message: str) -> None:
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # 打印到控制台
        print(log_message)
        
        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def _load_benchmark_config(self) -> Dict:
        """加载原有的benchmark配置"""
        config_file = self.base_results_dir / "benchmark_report.json"
        if not config_file.exists():
            self._log(f"Warning: Benchmark report not found: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            return report.get('benchmark_info', {}).get('config', {})
        except Exception as e:
            self._log(f"Error loading benchmark config: {e}")
            return {}
    
    def _load_data_index(self) -> List[Dict]:
        """加载数据索引"""
        # 尝试从原脚本的方式加载数据索引
        dataset_type = self.benchmark_config.get('dataset_type', 'legacy')
        data_path = self.benchmark_config.get('data_path', 'data/val')
        split = self.benchmark_config.get('split', 'val')
        train_test_json = self.benchmark_config.get('train_test_json')
        
        if dataset_type == "fusion360":
            return self._load_fusion360_data_index(data_path, split, train_test_json)
        else:
            # 加载legacy格式数据索引
            from data_utils import load_data_index
            return load_data_index(Path(data_path), split)
    
    def _load_fusion360_data_index(self, data_path: str, split: str, train_test_json: str) -> List[Dict]:
        """加载fusion360数据集索引"""
        # 确定train_test.json文件路径
        if train_test_json and Path(train_test_json).exists():
            train_test_file = Path(train_test_json)
        else:
            # 默认在fusion360dataset目录下查找
            train_test_file = Path(data_path).parent / "fusion360dataset" / "train_test.json"
        
        if not train_test_file.exists():
            self._log(f"Warning: Train/test split file not found: {train_test_file}")
            return []
        
        self._log(f"Loading fusion360 {split} split from {train_test_file}")
        
        # 加载train/test拆分
        try:
            with open(train_test_file, 'r', encoding='utf-8') as f:
                train_test_data = json.load(f)
        except Exception as e:
            self._log(f"Error loading train/test split: {e}")
            return []
        
        if split not in train_test_data:
            self._log(f"Split '{split}' not found in {train_test_file}")
            return []
        
        sample_ids = train_test_data[split]
        
        # 构建数据索引 - 基于fusion360数据集结构
        data_index = []
        
        # fusion360数据集的reconstruction目录路径
        reconstruction_dir = Path("cad-recode/fusion360dataset/reconstruction")
        if not reconstruction_dir.exists():
            # 尝试其他可能的路径
            reconstruction_dir = Path("fusion360dataset/reconstruction")
            if not reconstruction_dir.exists():
                reconstruction_dir = Path(data_path).parent / "fusion360dataset" / "reconstruction"
        
        self._log(f"Using fusion360 reconstruction directory: {reconstruction_dir}")
        
        found_samples = 0
        missing_samples = 0
        
        for sample_id in sample_ids:
            # 查找对应的.step文件
            step_file = None
            
            # 首先尝试精确匹配
            exact_match = reconstruction_dir / f"{sample_id}.step"
            if exact_match.exists():
                step_file = exact_match
            else:
                # 尝试模糊匹配
                for step_candidate in reconstruction_dir.glob(f"*{sample_id}*.step"):
                    step_file = step_candidate
                    break
            
            if step_file and step_file.exists():
                data_index.append({
                    "id": sample_id,
                    "sample_id": sample_id,
                    "step_file": str(step_file),
                    "split": split,
                    "data_source": "fusion360_step"
                })
                found_samples += 1
            else:
                self._log(f"Warning: STEP file not found for sample {sample_id}")
                missing_samples += 1
        
        self._log(f"Loaded {found_samples} fusion360 samples, {missing_samples} missing")
        return data_index
    
    def _load_error_samples(self) -> set:
        """加载错误样本列表"""
        data_path = self.benchmark_config.get('data_path', 'data/val')
        from data_utils import load_error_samples
        return load_error_samples(Path(data_path))
    
    def _get_model_names(self) -> List[str]:
        """获取所有模型名称"""
        results_dir = self.base_results_dir / "results"
        if not results_dir.exists():
            self._log(f"Warning: Results directory not found: {results_dir}")
            return []
        
        model_names = []
        for item in results_dir.iterdir():
            if item.is_dir() and (item / "inference_results.json").exists():
                model_names.append(item.name)
        
        return sorted(model_names)
    
    def _prepare_gt_step_files(self) -> Path:
        """准备GT STEP文件"""
        # 检查是否已有GT STEP文件
        gt_step_dir = self.base_results_dir / "gt_step_files"
        if gt_step_dir.exists() and len(list(gt_step_dir.glob("*.step"))) > 0:
            self._log(f"Using existing GT STEP files from {gt_step_dir}")
            return gt_step_dir
        
        # 否则创建新的GT STEP目录
        gt_step_dir = self.output_dir / "gt_step_files"
        gt_step_dir.mkdir(exist_ok=True)
        gt_stl_dir = self.output_dir / "gt_stl_files"
        gt_stl_dir.mkdir(exist_ok=True)

        self._log(f"Preparing GT STEP files in {gt_step_dir}")

        # 创建临时的CadQuery执行器
        batch_config = BatchInferenceConfig(
            output_dir=str(self.output_dir),
            cadquery_timeout=self.config.cadquery_timeout,
            save_code=False,
            save_stl=True,
            save_step=True
        )
        executor = CadQueryExecutor(batch_config)

        generated_count = 0
        failed_count = 0

        for sample_info in tqdm(self.data_index, desc="Preparing GT STEP files"):
            sample_id = sample_info.get("sample_id", sample_info.get("id"))
            gt_step_path = gt_step_dir / f"{sample_id}.step"
            gt_stl_path = gt_stl_dir / f"{sample_id}.stl"

            # 如果已存在则跳过
            if gt_step_path.exists() and gt_stl_path.exists():
                generated_count += 1
                continue

            # 处理fusion360的情况
            if sample_info.get("data_source") == "fusion360_step":
                step_file = sample_info.get("step_file")
                if step_file and Path(step_file).exists():
                    # 复制STEP文件
                    import shutil
                    shutil.copy(step_file, gt_step_path)
                    
                    # 生成STL
                    try:
                        stl_file = self._step_to_stl(str(gt_step_path), sample_id)
                        if stl_file and Path(stl_file).exists():
                            shutil.copy(stl_file, gt_stl_path)
                        generated_count += 1
                    except Exception as e:
                        self._log(f"Error generating STL for {sample_id}: {e}")
                        failed_count += 1
                else:
                    self._log(f"STEP file not found for {sample_id}")
                    failed_count += 1
                continue

            # 处理legacy情况
            if "code_path" in sample_info:
                gt_code_path = Path(sample_info["code_path"])
            else:
                data_path = self.benchmark_config.get('data_path', 'data/val')
                gt_code_path = Path(data_path) / f"{sample_id}.py"

            if not gt_code_path.exists():
                self._log(f"GT code file not found: {gt_code_path}")
                failed_count += 1
                continue

            try:
                with open(gt_code_path, 'r', encoding='utf-8') as f:
                    gt_code = f.read()

                # 执行代码生成STEP和STL
                success, error_msg, file_status = executor.execute_cadquery_code(
                    gt_code,
                    stl_output_path=str(gt_stl_path),
                    step_output_path=str(gt_step_path)
                )

                if success and file_status.get("step", False):
                    generated_count += 1
                else:
                    self._log(f"Failed to generate GT STEP for {sample_id}: {error_msg}")
                    failed_count += 1

            except Exception as e:
                self._log(f"Error processing {sample_id}: {e}")
                failed_count += 1

        self._log(f"GT STEP preparation complete: {generated_count} success, {failed_count} failed")
        return gt_step_dir
    
    def _step_to_stl(self, step_file: str, sample_id: str) -> Optional[str]:
        """将STEP文件转换为STL文件"""
        try:
            if not OCC_AVAILABLE:
                self._log("OCC not available, cannot convert STEP to STL")
                return None

            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Extend.DataExchange import write_stl_file
            
            # 读取STEP文件
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(step_file)
            if status != IFSelect_RetDone:
                return None
            
            step_reader.TransferRoots()
            shape = step_reader.OneShape()
            
            # 创建临时STL文件
            stl_dir = Path("temp_stl_recalc")
            stl_dir.mkdir(exist_ok=True)
            stl_file = stl_dir / f"{sample_id}_temp.stl"
            
            # 写入STL文件
            write_stl_file(shape, str(stl_file))
            
            if stl_file.exists():
                return str(stl_file)
            
            return None
            
        except Exception as e:
            self._log(f"Error converting STEP to STL: {e}")
            return None
    
    def _get_model_inference_results(self, model_name: str) -> Dict:
        """获取模型的推理结果"""
        results_file = self.base_results_dir / "results" / model_name / "inference_results.json"
        if not results_file.exists():
            self._log(f"Warning: Inference results not found for {model_name}: {results_file}")
            return {}
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._log(f"Error loading inference results for {model_name}: {e}")
            return {}
    
    def _find_common_success_samples(self) -> set:
        """找到所有模型都成功生成STEP的样本ID"""
        common_samples = None
        
        for model_name in self.model_names:
            try:
                results = self._get_model_inference_results(model_name)
                success_ids = set()
                
                for sample_id, sample_result in results.get("samples", {}).items():
                    if sample_result.get("step_generated", False):
                        # 检查文件是否实际存在
                        step_file = self.base_results_dir / "results" / model_name / "generated_step" / f"{sample_id}.step"
                        gt_step_file = self.gt_step_dir / f"{sample_id}.step"
                        
                        if step_file.exists() and gt_step_file.exists():
                            success_ids.add(sample_id)
                
                if common_samples is None:
                    common_samples = success_ids
                else:
                    common_samples &= success_ids
                    
            except Exception as e:
                self._log(f"Error processing {model_name}: {e}")
                continue
        
        return common_samples if common_samples else set()
    
    def _run_with_memory_check(self, func: Callable, *args, **kwargs) -> Any:
        """运行函数并检查内存使用情况"""
        # 检查初始内存使用情况
        initial_memory = self.memory_monitor.check_memory()
        self._log(f"Initial memory usage: {initial_memory*100:.1f}%")
        
        # 如果初始内存已经超过阈值，返回None
        if initial_memory >= self.config.memory_threshold:
            self._log(f"Memory already above threshold ({self.config.memory_threshold*100:.1f}%) before starting computation. Skipping...")
            return None
        
        # 开始内存监控
        self.memory_monitor.run_monitor()
        
        try:
            # 运行函数
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            self._log(f"Exception during computation: {e}")
            return None
        finally:
            # 停止内存监控
            self.memory_monitor.stop_monitor()
            
            # 检查最终内存使用情况
            final_memory = self.memory_monitor.check_memory()
            self._log(f"Final memory usage: {final_memory*100:.1f}%")
            
            # 如果内存使用过高，尝试释放内存
            if final_memory >= self.config.memory_threshold:
                self._log(f"Memory usage after computation is high ({final_memory*100:.1f}%). Attempting to free memory...")
                self._free_memory()
    
    def _free_memory(self) -> None:
        """尝试释放内存"""
        # 清除GPU缓存（如果使用GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 执行垃圾回收
        import gc
        gc.collect()
        
        # 检查内存使用情况
        memory_after = self.memory_monitor.check_memory()
        self._log(f"Memory usage after cleanup: {memory_after*100:.1f}%")
    
    def recalculate_metrics(self) -> Dict[str, Any]:
        """重新计算所有模型的指标"""
        self._log("\nStarting metrics recalculation...")
        print("\nStarting metrics recalculation...")
        
        # 找到所有模型都成功的样本
        common_samples = self._find_common_success_samples()
        self._log(f"Found {len(common_samples)} common successful samples across all models")
        print(f"Found {len(common_samples)} common successful samples across all models")
        
        if not common_samples:
            self._log("No common successful samples found. Cannot calculate metrics.")
            print("No common successful samples found. Cannot calculate metrics.")
            return {}
        
        # 为每个模型计算指标
        all_metrics = {}
        
        for model_name in tqdm(self.model_names, desc="Processing models"):
            self._log(f"\nProcessing model: {model_name}")
            print(f"\nProcessing model: {model_name}")
            
            # 获取模型的生成结果目录
            model_step_dir = self.base_results_dir / "results" / model_name / "generated_step"
            if not model_step_dir.exists():
                self._log(f"STEP directory not found for {model_name}")
                print(f"STEP directory not found for {model_name}")
                continue
            
            # 收集有效的文件对
            file_pairs = []
            for sample_id in common_samples:
                gt_step = self.gt_step_dir / f"{sample_id}.step"
                pred_step = model_step_dir / f"{sample_id}.step"
                
                if gt_step.exists() and pred_step.exists():
                    file_pairs.append((str(gt_step), str(pred_step), sample_id))
            
            self._log(f"Found {len(file_pairs)} valid GT-Prediction pairs for {model_name}")
            print(f"Found {len(file_pairs)} valid GT-Prediction pairs for {model_name}")
            
            # 计算每个文件对的指标
            batch_results = []
            failed_pairs = []
            
            for i, (gt_file, pred_file, sample_id) in enumerate(tqdm(file_pairs, desc="Calculating metrics")):
                self._log(f"Processing pair {i+1}/{len(file_pairs)}: {sample_id}")
                
                # 检查内存状态
                current_memory = self.memory_monitor.check_memory()
                if current_memory >= self.config.memory_threshold:
                    self._log(f"Memory threshold ({self.config.memory_threshold*100:.1f}%) reached before processing sample {sample_id}. Skipping...")
                    failed_pairs.append((sample_id, f"Memory threshold reached: {current_memory*100:.1f}%"))
                    self.failed_samples[model_name].append(sample_id)
                    continue
                
                # 包装计算函数，添加内存监控
                def compute_metrics_wrapper():
                    return self.metrics_calculator.compute_files_metrics(
                        ext="step",
                        file1=gt_file,
                        file2=pred_file,
                        matching_threshold=None,
                        include_normals=True,
                        use_icp=True
                    )
                
                # 运行计算，带有内存监控
                result = self._run_with_memory_check(compute_metrics_wrapper)
                
                if result is not None:
                    result_dict = result.to_dict()
                    result_dict['sample_id'] = sample_id
                    result_dict['gt_file'] = gt_file
                    result_dict['pred_file'] = pred_file
                    result_dict['pair_index'] = i
                    batch_results.append(result_dict)
                    
                    # 每10个样本打印一次进度
                    if (i + 1) % 10 == 0:
                        self._log(f"Completed {i + 1}/{len(file_pairs)} pairs")
                        print(f"  Completed {i + 1}/{len(file_pairs)} pairs")
                else:
                    error_msg = f"Failed to compute metrics for {sample_id}"
                    self._log(error_msg)
                    print(f"  {error_msg}")
                    failed_pairs.append((sample_id, error_msg))
                    self.failed_samples[model_name].append(sample_id)
                    
                    # 如果配置了跳过失败样本，则继续处理下一个
                    if self.config.skip_failed_samples:
                        continue
                    else:
                        # 否则停止处理此模型
                        break
            
            # 保存失败的样本
            if failed_pairs:
                failed_dir = self.output_dir / "failed_samples" / model_name
                failed_dir.mkdir(exist_ok=True, parents=True)
                failed_file = failed_dir / "failed_pairs.json"
                
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_pairs, f, indent=2)
                
                self._log(f"Saved {len(failed_pairs)} failed pairs to {failed_file}")
            
            # 计算平均指标
            if batch_results:
                metrics = self._compute_average_metrics(batch_results)
                metrics["total_pairs"] = len(file_pairs)
                metrics["successful_pairs"] = len(batch_results)
                metrics["failed_pairs"] = len(failed_pairs)
                
                # 保存详细结果
                detailed_dir = self.output_dir / "detailed_metrics" / model_name
                detailed_dir.mkdir(exist_ok=True, parents=True)
                
                with open(detailed_dir / "detailed_metrics.json", 'w', encoding='utf-8') as f:
                    json.dump(batch_results, f, indent=2)
                
                all_metrics[model_name] = metrics
                self._log(f"Successfully computed metrics for {len(batch_results)}/{len(file_pairs)} pairs")
                print(f"Successfully computed metrics for {len(batch_results)}/{len(file_pairs)} pairs")
            else:
                all_metrics[model_name] = {"error": "No successful metric computations"}
                self._log("No successful metric computations for this model")
        
        # 保存失败样本汇总
        if self.failed_samples:
            with open(self.output_dir / "all_failed_samples.json", 'w', encoding='utf-8') as f:
                json.dump(self.failed_samples, f, indent=2)
            self._log(f"Saved all failed samples summary to {self.output_dir / 'all_failed_samples.json'}")
        
        # 生成报告
        self._generate_report(all_metrics)
        return all_metrics
    
    def _compute_average_metrics(self, batch_results: List[Dict]) -> Dict[str, float]:
        """计算平均指标"""
        if not batch_results:
            return {}
        
        # 定义需要计算的指标
        metrics = [
            'chamfer_distance', 'hausdorff_distance', 'earth_mover_distance', 'rms_error',
            'iou_voxel', 'iou_brep', 'matching_rate', 'coverage_rate', 
            'normal_consistency', 'computation_time'
        ]
        
        def get_values(key):
            values = []
            for res in batch_results:
                value = res.get(key)
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
    
    def _generate_report(self, metrics_results: Dict):
        """生成报告"""
        # 加载原报告
        report_file = self.base_results_dir / "benchmark_report.json"
        if report_file.exists():
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
            except Exception as e:
                self._log(f"Error loading original report: {e}")
                report = {"benchmark_info": {}, "models": {}}
        else:
            report = {"benchmark_info": {}, "models": {}}
        
        # 更新指标
        for model_name, metrics in metrics_results.items():
            if model_name not in report["models"]:
                report["models"][model_name] = {"metrics": {}}
            
            # 提取主要指标用于摘要
            main_metrics = [
                "chamfer_distance", "hausdorff_distance", "earth_mover_distance",
                "iou_voxel", "iou_brep", "matching_rate", "coverage_rate", "normal_consistency"
            ]
            
            metrics_summary = {}
            for metric in main_metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                if mean_key in metrics:
                    metrics_summary[metric] = {
                        "mean": metrics[mean_key],
                        "std": metrics[std_key]
                    }
            
            report["models"][model_name]["metrics"] = metrics_summary if metrics_summary else metrics
        
        # 添加失败样本信息
        report["recalculation_info"] = {
            "timestamp": datetime.now().isoformat(),
            "memory_threshold": self.config.memory_threshold,
            "failed_samples": self.failed_samples
        }
        
        # 保存更新后的报告
        new_report_file = self.output_dir / "updated_benchmark_report.json"
        with open(new_report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        self._log(f"\nReport saved to {new_report_file}")
        print(f"\nReport saved to {new_report_file}")
        return report
    
    def _generate_markdown_report(self, report: Dict):
        """生成Markdown报告"""
        md_content = f"""# CAD-Recode Benchmark Metrics (Recalculated)

**Generated:** {datetime.now().isoformat()}  
**Total Samples:** {report.get('benchmark_info', {}).get('total_samples', 'N/A')}  
**Memory Threshold:** {self.config.memory_threshold*100:.1f}%  

## Model Comparison

| Model | Success Rate | STL Success | Chamfer↓ | IoU↑ | Matching Rate↑ | Failed Samples |
|-------|-------------|-------------|----------|------|----------------|----------------|
"""
        
        for model_name, model_data in report["models"].items():
            stats = model_data.get("generation_stats", {})
            metrics = model_data.get("metrics", {})
            
            # 计算成功率
            total = stats.get("total_samples", 0)
            if total > 0:
                success_rate = stats.get("successful_generations", 0) / total * 100
                stl_success_rate = stats.get("successful_stl", 0) / total * 100
            else:
                success_rate = 0
                stl_success_rate = 0
            
            # 提取关键指标
            chamfer = metrics.get("chamfer_distance", {})
            iou = metrics.get("iou_brep", {}) or metrics.get("iou_voxel", {})
            matching = metrics.get("matching_rate", {})
            
            chamfer_str = f"{chamfer.get('mean', 0):.4f}" if isinstance(chamfer, dict) else "N/A"
            iou_str = f"{iou.get('mean', 0):.4f}" if isinstance(iou, dict) else "N/A"
            matching_str = f"{matching.get('mean', 0):.4f}" if isinstance(matching, dict) else "N/A"
            
            # 失败样本数量
            failed_count = len(self.failed_samples.get(model_name, []))
            
            md_content += f"| {model_name} | {success_rate:.1f}% | {stl_success_rate:.1f}% | {chamfer_str} | {iou_str} | {matching_str} | {failed_count} |\\n"
        
        md_content += "\n## Detailed Metrics\n"
        
        for model_name, model_data in report["models"].items():
            md_content += f"### {model_name}\n\n"
            
            # 生成统计
            stats = model_data.get("generation_stats", {})
            md_content += f"- Generation Success: {stats.get('successful_generations', 0)}/{stats.get('total_samples', 0)}\n"
            md_content += f"- STL Success: {stats.get('successful_stl', 0)}/{stats.get('total_samples', 0)}\n"
            md_content += f"- Metrics Computed: {metrics_results.get(model_name, {}).get('successful_pairs', 0)}/{metrics_results.get(model_name, {}).get('total_pairs', 0)}\n"
            md_content += f"- Failed Samples: {len(self.failed_samples.get(model_name, []))}\n\n"
            
            # 指标
            metrics = model_data.get("metrics", {})
            if "error" in metrics:
                md_content += f"**Metrics Error:** {metrics['error']}\n\n"
                continue
            
            md_content += "| Metric | Mean | Std | Min | Max |\n"
            md_content += "|--------|------|-----|-----|-----|\n"
            
            for metric, values in metrics.items():
                if isinstance(values, dict) and "mean" in values and "std" in values:
                    md_content += f"| {metric} | {values['mean']:.6f} | {values['std']:.6f} | {values.get('min', 0):.6f} | {values.get('max', 0):.6f} |\n"
            
            md_content += "\n"
        
        # 添加失败样本部分
        if self.failed_samples:
            md_content += "## Failed Samples\n\n"
            
            for model_name, samples in self.failed_samples.items():
                if samples:
                    md_content += f"### {model_name} ({len(samples)} failed)\n\n"
                    md_content += ", ".join(samples) + "\n\n"
        
        # 保存Markdown报告
        md_file = self.output_dir / "updated_benchmark_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CAD-Recode Benchmark Metrics Recalculation")
    
    parser.add_argument("--results_dir", required=True,
                       help="Path to the original benchmark results directory")
    parser.add_argument("--output_dir",
                       help="Directory to save updated results (default: results_dir/recalculated_metrics)")
    parser.add_argument("--num_samples", type=int,
                       help="Number of samples to process (default: all)")
    parser.add_argument("--metric_points", type=int, default=2000,
                       help="Number of points for metric calculation")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout for CadQuery operations")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置环境
    setup_environment()
    
    # 创建配置
    config = RecalculateConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        n_points_metric=args.metric_points,
        cadquery_timeout=args.timeout,
        device=args.device
    )
    
    # 初始化并运行重新计算
    try:
        recalculator = MetricsRecalculator(config)
        recalculator.recalculate_metrics()
        print("\nMetrics recalculation completed successfully!")
    except Exception as e:
        print(f"Error during recalculation: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()