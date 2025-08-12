#!/usr/bin/env python3
"""
高效批量预缓存点云数据脚本
采用多进程并行处理，通过随机化和迭代重启机制筛选错误样本

主要功能：
1. 多进程并行处理，无单样本进程隔离
2. 随机化样本顺序，每次运行重新打乱
3. 迭代重启机制，崩溃时记录第一个失败样本
4. 自动跳过已缓存和已知错误的样本
5. 支持处理验证集的批量子文件夹结构

使用方法:
python precache_point_clouds_efficient.py --config configs/train_config.yaml
python precache_point_clouds_efficient.py --train_data_path data/train --val_data_path data/val
"""

import os
import sys
import json
import yaml
import argparse
import logging
import warnings
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np
from tqdm import tqdm

# 导入训练配置和工具函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_cad_recode import TrainingConfig, setup_logger

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class PrecacheConfig:
    """预缓存配置"""
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    n_points: int = 256
    max_workers: int = 8  # 进程数量
    timeout: int = 15  # 单个样本的超时时间
    force_recache: bool = False  # 是否强制重新生成缓存
    log_level: str = "INFO"
    max_retries: int = 10  # 最大重试次数


def execute_cadquery_code(code_file: str, n_points: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """执行CadQuery代码生成点云（高效版本，无进程隔离）"""
    try:
        import cadquery as cq
        import trimesh
        import torch
        from pytorch3d.ops import sample_farthest_points
        
        # 读取代码文件
        if not os.path.exists(code_file):
            return None, f"code_file_not_found|{code_file}"
        
        with open(code_file, 'r', encoding='utf-8') as f:
            code_content = f.read().strip()
        
        if not code_content:
            return None, "empty_code_file"
        
        # 执行代码
        exec_globals = {
            'cadquery': cq,
            'cq': cq,
            '__builtins__': __builtins__
        }
        
        exec(code_content, exec_globals)
        
        if 'r' not in exec_globals or exec_globals['r'] is None:
            return None, "no_result_variable"
            
        compound = exec_globals['r'].val()
        
        # 生成临时STEP文件
        temp_step_path = f"/tmp/temp_model_{os.getpid()}_{hash(code_content) % 10000}.step"
        
        try:
            # 导出STEP文件
            cq.exporters.export(compound, temp_step_path)
            
            # 转换为点云
            if temp_step_path.lower().endswith('.stl'):
                mesh = trimesh.load_mesh(temp_step_path)
            else:
                try:
                    mesh = trimesh.load_mesh(temp_step_path, force='mesh')
                except:
                    # tessellation fallback
                    mesh_data = compound.tessellate(0.1)
                    if len(mesh_data[0]) > 0:
                        vertices = np.array([(v.x, v.y, v.z) for v in mesh_data[0]])
                        faces = np.array(mesh_data[1])
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    else:
                        return None, "tessellation_failed"
            
            # 标准化网格
            if not mesh.is_empty and mesh.vertices is not None and len(mesh.vertices) > 0:
                # 移动到原点
                mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
                # 缩放到单位立方体
                if max(mesh.extents) > 0:
                    mesh.apply_scale(2.0 / max(mesh.extents))
                
                # 采样点云
                vertices, _ = trimesh.sample.sample_surface(mesh, 8192)
                
                # 使用最远点采样
                if len(vertices) >= n_points:
                    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
                    _, ids = sample_farthest_points(vertices_tensor, K=n_points)
                    ids = ids[0].numpy()
                    selected_points = vertices[ids]
                else:
                    indices = np.random.choice(len(vertices), n_points, replace=True)
                    selected_points = vertices[indices]
                
                return selected_points.astype(np.float32), None
            else:
                return None, "empty_mesh"
                
        finally:
            # 清理临时文件
            if os.path.exists(temp_step_path):
                os.remove(temp_step_path)
                
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def process_single_sample(args: Tuple[str, str, str, int, bool]) -> Tuple[str, bool, str]:
    """处理单个样本（高效版本）"""
    sample_id, code_file, cache_file, n_points, force_recache = args
    
    try:
        # 检查是否需要跳过已有缓存
        if not force_recache and os.path.exists(cache_file):
            try:
                # 验证缓存文件的有效性
                cached_pc = np.load(cache_file)
                if cached_pc.shape == (n_points, 3):
                    return sample_id, True, "cached"
                else:
                    # 缓存文件格式不正确，删除重新生成
                    os.remove(cache_file)
            except Exception as e:
                # 缓存文件损坏，删除重新生成
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                return sample_id, False, f"corrupted_cache|{type(e).__name__}: {str(e)[:100]}"
        
        # 执行CadQuery代码
        point_cloud, error_msg = execute_cadquery_code(code_file, n_points)
        
        if point_cloud is not None:
            # 保存点云
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, point_cloud)
            return sample_id, True, "generated"
        else:
            return sample_id, False, error_msg
            
    except Exception as e:
        return sample_id, False, f"unexpected_error|{type(e).__name__}: {str(e)[:100]}"


def load_data_index(data_path: str) -> List[Dict[str, Any]]:
    """加载数据索引 - 支持验证集的批量子文件夹"""
    data_path = Path(data_path)
    
    # 支持多种索引文件名
    possible_index_files = [
        data_path / "train_index.json",
        data_path / "val_index.json", 
        data_path / "test_index.json",
        data_path / "index.json"
    ]
    
    for index_file in possible_index_files:
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {index_file}: {e}")
                continue
    
    # 如果没有索引文件，尝试自动扫描
    print(f"No index file found, scanning directory: {data_path}")
    data_items = []
    
    if data_path.exists():
        # 检查是否是验证集结构（有batch_xx子文件夹）
        batch_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('batch_')]
        
        if batch_dirs:
            # 验证集结构：扫描所有batch_xx子文件夹
            print(f"Found {len(batch_dirs)} batch directories")
            for batch_dir in batch_dirs:
                for code_file in batch_dir.rglob("*.py"):
                    if code_file.is_file():
                        # 使用相对路径作为sample_id
                        relative_path = code_file.relative_to(data_path)
                        sample_id = str(relative_path.with_suffix(''))
                        data_items.append({
                            "id": sample_id,
                            "code_file": str(code_file)
                        })
        else:
            # 常规结构：直接扫描目录
            for code_file in data_path.rglob("*.py"):
                if code_file.is_file():
                    sample_id = code_file.stem
                    data_items.append({
                        "id": sample_id,
                        "code_file": str(code_file)
                    })
    
    return data_items


class EfficientPrecacheManager:
    """高效预缓存管理器"""
    
    def __init__(self, config: PrecacheConfig):
        self.config = config
        self.logger = setup_logger()
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # 统计信息
        self.error_samples = set()
        self.retry_count = 0
    
    def _get_cache_dir_and_error_file(self, data_path: str) -> Tuple[Path, Path]:
        """获取缓存目录和错误文件路径"""
        data_path = Path(data_path)
        cache_dir = data_path / "point_cloud_cache"
        cache_dir.mkdir(exist_ok=True)
        error_file = cache_dir / "error_samples.json"
        return cache_dir, error_file
    
    def _load_error_samples(self, error_file: Path):
        """加载已知错误样本列表"""
        if error_file.exists():
            try:
                with open(error_file, 'r') as f:
                    error_list = json.load(f)
                    self.error_samples = set(error_list)
                self.logger.info(f"Loaded {len(self.error_samples)} known error samples")
            except Exception as e:
                self.logger.warning(f"Failed to load error samples: {e}")
                self.error_samples = set()
    
    def _save_error_samples(self, error_file: Path):
        """保存错误样本列表"""
        try:
            with open(error_file, 'w') as f:
                json.dump(list(self.error_samples), f, indent=2)
            self.logger.info(f"Saved {len(self.error_samples)} error samples to {error_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save error samples: {e}")
    
    def _get_processing_stats(self, data_items: List[Dict[str, Any]], cache_dir: Path) -> Dict[str, int]:
        """获取处理统计信息"""
        total_samples = len(data_items)
        cached_samples = 0
        error_samples = len(self.error_samples)
        
        # 计算已缓存的样本数
        for item in data_items:
            sample_id = item.get('sample_id', item.get('id', 'unknown'))
            
            # 缓存文件路径 - 将路径分隔符替换为下划线，直接放在缓存目录下
            cache_filename = sample_id.replace('/', '_').replace('\\', '_') + '.npy'
            cache_file = cache_dir / cache_filename
            
            if cache_file.exists():
                try:
                    cached_pc = np.load(cache_file)
                    if cached_pc.shape == (self.config.n_points, 3):
                        cached_samples += 1
                except:
                    pass
        
        remaining_samples = total_samples - cached_samples - error_samples
        
        return {
            "total": total_samples,
            "cached": cached_samples,
            "error": error_samples,
            "remaining": remaining_samples,
            "completion_rate": (cached_samples / total_samples) * 100 if total_samples > 0 else 0
        }
    
    def _prepare_tasks(self, data_items: List[Dict[str, Any]], data_path: str, cache_dir: Path) -> List[Tuple[str, str, str, int, bool]]:
        """准备处理任务列表"""
        tasks = []
        
        for item in data_items:
            sample_id = item.get('sample_id', item.get('id', 'unknown'))
            code_file = item.get('code_path', item.get('code_file', ''))
            
            # 如果路径不存在，尝试构建相对路径
            if not os.path.exists(code_file):
                relative_path = item.get('relative_path', '')
                if relative_path:
                    code_file = os.path.join(data_path, relative_path)
            
            # 跳过已知错误样本
            if sample_id in self.error_samples:
                self.logger.debug(f"Skipping known error sample: {sample_id}")
                continue
            
            # 缓存文件路径 - 将路径分隔符替换为下划线，直接放在缓存目录下
            cache_filename = sample_id.replace('/', '_').replace('\\', '_') + '.npy'
            cache_file = cache_dir / cache_filename
            
            tasks.append((
                sample_id,
                code_file,
                str(cache_file),
                self.config.n_points,
                self.config.force_recache
            ))
        
        return tasks
    
    def _process_tasks_batch(self, tasks: List[Tuple[str, str, str, int, bool]], split_name: str) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
        """处理一批任务"""
        results = {"total": len(tasks), "successful": 0, "cached": 0, "failed": 0}
        failed_samples = []
        first_error_recorded = False  # 标记是否已记录第一个错误
        
        if not tasks:
            return results, failed_samples
        
        # 随机打乱任务顺序
        random.shuffle(tasks)
        
        self.logger.info(f"Processing {len(tasks)} samples (retry {self.retry_count + 1}/{self.config.max_retries})")
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交任务
            future_to_task = {executor.submit(process_single_sample, task): task for task in tasks}
            
            # 处理结果
            with tqdm(total=len(tasks), desc=f"Precaching {split_name}") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    sample_id = task[0]
                    
                    try:
                        sample_id, success, message = future.result()
                        
                        if success:
                            if message == "cached":
                                results["cached"] += 1
                            else:
                                results["successful"] += 1
                            pbar.set_postfix({
                                "Success": results["successful"], 
                                "Cached": results["cached"],
                                "Failed": results["failed"]
                            })
                        else:
                            results["failed"] += 1
                            # 只记录第一个错误样本
                            if not first_error_recorded:
                                failed_samples.append((sample_id, message))
                                first_error_recorded = True
                                self.logger.info(f"Recording first error sample: {sample_id}")
                            pbar.set_postfix({
                                "Success": results["successful"], 
                                "Cached": results["cached"],
                                "Failed": results["failed"]
                            })
                            
                    except Exception as e:
                        results["failed"] += 1
                        # 只记录第一个错误样本
                        if not first_error_recorded:
                            failed_samples.append((sample_id, f"process_error: {e}"))
                            first_error_recorded = True
                            self.logger.info(f"Recording first error sample: {sample_id}")
                    
                    pbar.update(1)
        
        return results, failed_samples
    
    def precache_dataset(self, data_path: str, split_name: str) -> Dict[str, int]:
        """预缓存单个数据集（支持迭代重启）"""
        self.logger.info(f"Starting precache for {split_name} dataset: {data_path}")
        
        # 获取缓存目录和错误文件
        cache_dir, error_file = self._get_cache_dir_and_error_file(data_path)
        self._load_error_samples(error_file)
        
        # 加载数据索引
        data_items = load_data_index(data_path)
        if not data_items:
            self.logger.warning(f"No data items found in {data_path}")
            return {"total": 0, "successful": 0, "cached": 0, "failed": 0}
        
        self.logger.info(f"Found {len(data_items)} samples in {split_name} dataset")
        
        # 获取初始统计信息
        initial_stats = self._get_processing_stats(data_items, cache_dir)
        self.logger.info(f"Initial status: {initial_stats['cached']} cached, {initial_stats['error']} errors, {initial_stats['remaining']} remaining")
        self.logger.info(f"Current completion rate: {initial_stats['completion_rate']:.1f}%")
        
        # 迭代重启处理
        overall_results = {"total": len(data_items), "successful": 0, "cached": 0, "failed": 0}
        
        for retry in range(self.config.max_retries):
            self.retry_count = retry
            
            # 准备任务列表（跳过已知错误样本）
            tasks = self._prepare_tasks(data_items, data_path, cache_dir)
            
            if not tasks:
                self.logger.info(f"No more tasks to process for {split_name}")
                break
            
            self.logger.info(f"Retry {retry + 1}/{self.config.max_retries}: Processing {len(tasks)} remaining samples")
            
            try:
                # 处理当前批次
                batch_results, failed_samples = self._process_tasks_batch(tasks, split_name)
                
                # 更新总体结果
                overall_results["successful"] += batch_results["successful"]
                overall_results["cached"] += batch_results["cached"]
                overall_results["failed"] += batch_results["failed"]
                
                # 如果没有失败样本，完成处理
                if not failed_samples:
                    self.logger.info(f"All samples processed successfully for {split_name}")
                    break
                
                # 记录失败样本（每次重启只记录第一个）
                for sample_id, error_msg in failed_samples:
                    self.error_samples.add(sample_id)
                    self.logger.info(f"Added error sample to blacklist: {sample_id} - {error_msg}")
                
                # 保存错误样本列表
                self._save_error_samples(error_file)
                
                self.logger.info(f"Retry {retry + 1} completed: {batch_results['successful']} successful, {batch_results['cached']} cached, {batch_results['failed']} failed")
                self.logger.info(f"Added {len(failed_samples)} error sample(s) to blacklist (first error only per retry)")
                
                # 如果失败样本过多，可能需要调整策略
                if batch_results['failed'] > batch_results['successful'] + batch_results['cached']:
                    self.logger.warning(f"High failure rate in retry {retry + 1}, consider adjusting timeout or worker count")
                
            except Exception as e:
                self.logger.error(f"Retry {retry + 1} crashed: {e}")
                # 如果进程池崩溃，记录当前处理的第一个样本为错误样本
                if tasks:
                    first_sample_id = tasks[0][0]
                    self.error_samples.add(first_sample_id)
                    self._save_error_samples(error_file)
                    self.logger.info(f"Added potentially problematic sample to error list: {first_sample_id}")
                
                # 等待一段时间后重试
                time.sleep(2)
                continue
        
        # 保存最终失败样本日志
        if self.error_samples:
            fail_log_file = cache_dir / f"failed_samples_final.json"
            try:
                with open(fail_log_file, 'w') as f:
                    json.dump(list(self.error_samples), f, indent=2)
                self.logger.info(f"Saved final failed samples to: {fail_log_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save final failed samples: {e}")
        
        # 统计报告
        total = overall_results["total"]
        successful = overall_results["successful"]
        cached = overall_results["cached"]
        failed = overall_results["failed"]
        
        self.logger.info(f"\n{split_name.upper()} Dataset Precaching Results:")
        self.logger.info(f"  Total samples: {total}")
        self.logger.info(f"  Successfully generated: {successful} ({successful/total*100:.1f}%)")
        self.logger.info(f"  Already cached: {cached} ({cached/total*100:.1f}%)")
        self.logger.info(f"  Failed: {failed} ({failed/total*100:.1f}%)")
        self.logger.info(f"  Error samples blacklisted: {len(self.error_samples)}")
        self.logger.info(f"  Cache directory: {cache_dir}")
        self.logger.info(f"  Retries completed: {self.retry_count + 1}/{self.config.max_retries}")
        
        # 提供下一步建议
        if failed > 0:
            remaining_failed = failed - len(self.error_samples)
            if remaining_failed > 0:
                self.logger.info(f"  Note: {remaining_failed} samples failed but were not blacklisted (only first error per retry is recorded)")
                self.logger.info(f"  Recommendation: Run the script again to continue processing remaining samples")
        
        return overall_results
    
    def precache_all(self) -> Dict[str, Dict[str, int]]:
        """预缓存所有数据集"""
        self.logger.info("Starting efficient precache process for all datasets...")
        self.logger.info(f"Configuration: {self.config}")
        
        all_results = {}
        
        # 处理训练数据
        if os.path.exists(self.config.train_data_path):
            train_results = self.precache_dataset(self.config.train_data_path, "train")
            all_results["train"] = train_results
        else:
            self.logger.warning(f"Train data path not found: {self.config.train_data_path}")
        
        # 处理验证数据
        if os.path.exists(self.config.val_data_path):
            val_results = self.precache_dataset(self.config.val_data_path, "val")
            all_results["val"] = val_results
        else:
            self.logger.warning(f"Val data path not found: {self.config.val_data_path}")
        
        # 总体统计
        total_samples = sum(results["total"] for results in all_results.values())
        total_successful = sum(results["successful"] for results in all_results.values())
        total_cached = sum(results["cached"] for results in all_results.values())
        total_failed = sum(results["failed"] for results in all_results.values())
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("OVERALL EFFICIENT PRECACHING RESULTS:")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total samples processed: {total_samples}")
        self.logger.info(f"Successfully generated: {total_successful} ({total_successful/total_samples*100:.1f}%)")
        self.logger.info(f"Already cached: {total_cached} ({total_cached/total_samples*100:.1f}%)")
        self.logger.info(f"Failed: {total_failed} ({total_failed/total_samples*100:.1f}%)")
        self.logger.info(f"Error samples: {len(self.error_samples)}")
        self.logger.info(f"{'='*60}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="高效预缓存CAD-Recode训练数据的点云")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--train_data_path", type=str, default="./data/train", help="训练数据路径")
    parser.add_argument("--val_data_path", type=str, default="./data/val", help="验证数据路径")
    parser.add_argument("--n_points", type=int, default=256, help="点云点数")
    parser.add_argument("--max_workers", type=int, default=8, help="最大并行进程数")
    parser.add_argument("--timeout", type=int, default=15, help="单个样本超时时间(秒)")
    parser.add_argument("--force_recache", action="store_true", help="强制重新生成所有缓存")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--max_retries", type=int, default=10, help="最大重试次数")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config and os.path.exists(args.config):
        # 从训练配置文件加载
        with open(args.config, 'r') as f:
            train_config_dict = yaml.safe_load(f)
        
        # 创建预缓存配置
        config = PrecacheConfig(
            train_data_path=train_config_dict.get('train_data_path', args.train_data_path),
            val_data_path=train_config_dict.get('val_data_path', args.val_data_path),
            n_points=train_config_dict.get('n_points', args.n_points),
            max_workers=args.max_workers,
            timeout=args.timeout,
            force_recache=args.force_recache,
            log_level=args.log_level,
            max_retries=args.max_retries
        )
    else:
        # 从命令行参数创建
        config = PrecacheConfig(
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            n_points=args.n_points,
            max_workers=args.max_workers,
            timeout=args.timeout,
            force_recache=args.force_recache,
            log_level=args.log_level,
            max_retries=args.max_retries
        )
    
    # 设置随机种子
    random.seed(int(time.time()))
    
    # 执行预缓存
    try:
        manager = EfficientPrecacheManager(config)
        results = manager.precache_all()
        
        print("\n🎉 高效预缓存完成！")
        print("现在可以开始训练，所有有效的点云数据已预先生成。")
        print("缓存保存在各数据集目录的 point_cloud_cache 子目录中")
        print("错误样本已被记录并将在未来的训练中跳过。")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ 预缓存被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 预缓存过程出错: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("🚀 CAD-Recode 高效点云预缓存工具")
    print("="*60)
    exit(main())
