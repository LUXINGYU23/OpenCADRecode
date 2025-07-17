#!/usr/bin/env python3
"""
CAD-Recode 训练脚本 - 基于论文优化版
支持从点云生成CadQuery代码的多模态模型训练

基于CAD-Recode论文的关键优化：
1. 动态点云生成与缓存
2. 数据增强（高斯噪声，50%概率，std=0.01）
3. Test-time采样推理（生成10组代码，选择最佳）
4. 代码提示支持（import cadquery as cq）
5. 优化训练参数（10万步，batch_size=18，lr=2e-4等）

使用方法:
python train_cad_recode.py --config configs/train_config.yaml
"""

import os
import sys
import json
import yaml
import argparse
import logging
import warnings
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import transformers
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    Qwen2Config,
    Qwen2ForCausalLM, 
    Qwen2Model, 
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import get_last_checkpoint

import tempfile
import cadquery as cq

# 导入SwanLab用于实验跟踪和可视化
try:
    import swanlab
    from swanlab.integration.huggingface import SwanLabCallback
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not installed. Install with: pip install swanlab")

# 导入OpenCascade相关模块用于STEP文件处理
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Extend.DataExchange import write_stl_file
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False

# 忽略一些警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置随机种子
def set_seed(seed: int = 42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 日志设置
def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger('CADRecode_Training')
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 模型组件定义 (来自demo.ipynb)
class FourierPointEncoder(nn.Module):
    """傅里叶点云编码器"""
    def __init__(self, hidden_size: int):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)  # 3 + 8*3*2 = 51

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [batch_size, n_points, 3]
        x = points
        # 计算傅里叶特征
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        x = self.projection(x)
        return x


class CADRecode(Qwen2ForCausalLM):
    """CAD-Recode多模态模型"""
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 临时切换到float32来初始化点云编码器
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(original_dtype)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                point_cloud: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.Tensor] = None) -> CausalLMOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 在没有past_key_values或者是新的序列时，融合点云和文本嵌入
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            if point_cloud is not None:
                # 编码点云
                point_embeds = self.point_encoder(point_cloud)
                if point_embeds.dtype != inputs_embeds.dtype:
                    point_embeds = point_embeds.to(inputs_embeds.dtype)
                
                # 将点云嵌入插入到指定位置 (attention_mask == -1的位置)
                point_mask = (attention_mask == -1)
                if point_mask.any():
                    inputs_embeds[point_mask] = point_embeds.reshape(-1, point_embeds.shape[-1])
                    attention_mask[point_mask] = 1
            
            input_ids = None
            position_ids = None

        # 通过Qwen2模型
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
            # 计算语言建模损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
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
        if 'point_cloud' in kwargs:
            model_inputs['point_cloud'] = kwargs['point_cloud']
        return model_inputs


@dataclass
class TrainingConfig:
    """训练配置 - 基于CAD-Recode论文优化"""
    # 模型配置
    base_model_name: str = "Qwen/Qwen2-1.5B"
    model_save_path: str = "./checkpoints"
    
    # 数据配置
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    max_seq_length: int = 1024
    n_points: int = 256
    
    # 训练配置 - 基于论文优化
    max_training_steps: int = 100000  # 论文中的10万次迭代
    batch_size: int = 4              # 论文推荐batch size
    gradient_accumulation_steps: int = 4  # 调整以适应GPU内存
    learning_rate: float = 2e-4       # 论文中的0.0002
    weight_decay: float = 0.01        # 论文设置
    warmup_steps: int = 1000          # 论文中的1千次warmup
    
    # 数据增强配置 - 基于论文
    noise_std: float = 0.01           # 高斯噪声标准差
    noise_probability: float = 0.5    # 添加噪声的概率
    
    # Test-time策略配置
    num_inference_samples: int = 10   # 推理时生成样本数
    use_test_time_sampling: bool = True
    
    # 优化器配置
    optimizer: str = "adamw"
    scheduler: str = "linear_warmup"  # 线性warmup + 余弦衰减
    
    # 系统配置
    num_workers: int = 0  # 避免多进程嵌套问题
    save_steps: int = 4000           # 保存间隔，必须是eval_steps的倍数
    eval_steps: int = 2000           # 评估间隔
    logging_steps: int = 100
    seed: int = 42
    
    # 硬件配置
    device: str = "auto"
    mixed_precision: str = "bf16"     # 论文使用的精度
    
    # 训练策略配置
    use_code_prompt: bool = True      # 使用代码提示
    code_prompt: str = "import cadquery as cq\n"  # 初始代码token
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 计算总epoch数（基于步数）
        if hasattr(self, 'train_dataset_size') and self.train_dataset_size > 0:
            effective_batch_size = self.batch_size * self.gradient_accumulation_steps
            self.num_epochs = max(1, self.max_training_steps * effective_batch_size // self.train_dataset_size)
        else:
            # 如果没有数据集大小信息，使用默认值
            self.num_epochs = 10


class CADDataset(Dataset):
    """CAD数据集类 - 动态生成点云版本"""
    
    def __init__(self, data_path: str, tokenizer, config: TrainingConfig, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.logger = logging.getLogger('CADRecode_Training')
        
        # 点云缓存目录
        self.cache_dir = self.data_path / "point_cloud_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 错误样本跟踪
        self.error_samples_file = self.data_path / "error_samples.json"
        self.error_samples = self._load_error_samples()
        
        # 加载数据索引
        self.data_list = self._load_data_index()
        
        # 过滤错误样本
        self._filter_error_samples()
        
        self.logger.info(f"Loaded {len(self.data_list)} samples for {split} split")
        if len(self.error_samples) > 0:
            self.logger.info(f"Skipping {len(self.error_samples)} known error samples")
    
    def _load_error_samples(self) -> set:
        """加载已知的错误样本列表"""
        if self.error_samples_file.exists():
            try:
                with open(self.error_samples_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()
    
    def _save_error_samples(self):
        """保存错误样本列表"""
        try:
            with open(self.error_samples_file, 'w') as f:
                json.dump(list(self.error_samples), f)
        except Exception as e:
            self.logger.warning(f"Failed to save error samples: {e}")
    
    def _filter_error_samples(self):
        """过滤掉已知的错误样本"""
        if self.error_samples:
            original_count = len(self.data_list)
            self.data_list = [item for item in self.data_list 
                            if item["sample_id"] not in self.error_samples]
            filtered_count = original_count - len(self.data_list)
            if filtered_count > 0:
                self.logger.info(f"Filtered out {filtered_count} known error samples")
    
    def _mark_sample_as_error(self, sample_id: str):
        """将样本标记为错误样本"""
        if sample_id not in self.error_samples:
            self.error_samples.add(sample_id)
            self._save_error_samples()
            self.logger.warning(f"Marked sample {sample_id} as error sample")
    
    def _load_data_index(self) -> List[Dict[str, Any]]:
        """加载数据索引文件"""
        index_file = self.data_path / f"{self.split}_index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            self.logger.error(f"Index file not found: {index_file}")
            return []
    
    def _execute_cadquery_code_safe(self, py_code: str, timeout: int = 10) -> Optional[np.ndarray]:
        """安全执行CadQuery代码并生成点云 - 强化进程隔离版本"""
        import subprocess
        import tempfile
        import signal
        import pickle
        import base64
        
        def write_execution_script(code: str, result_file: str, sample_id: str) -> str:
            """创建独立的执行脚本"""
            script_content = f'''#!/usr/bin/env python3
import sys
import os
import contextlib
import io
import tempfile
import signal
import pickle
import traceback

# 设置信号处理，防止无限挂起
def timeout_handler(signum, frame):
    sys.exit(124)  # 超时退出码

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout - 1})  # 留1秒给cleanup

try:
    # 导入所需模块
    import cadquery as cq
    import trimesh
    import numpy as np
    import torch
    from pytorch3d.ops import sample_farthest_points
    
    # 重定向所有输出
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            # 代码执行
            code = {repr(code)}
            exec_globals = {{
                'cadquery': cq,
                'cq': cq,
                '__builtins__': __builtins__
            }}
            
            exec(code, exec_globals)
            
            if 'r' in exec_globals and exec_globals['r'] is not None:
                compound = exec_globals['r'].val()
                
                # 生成临时STEP文件
                temp_step_path = f"/tmp/temp_model_{{os.getpid()}}_{sample_id}_{{hash(code) % 10000}}.step"
                
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
                                sys.exit(1)
                    
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
                        if len(vertices) >= 256:
                            vertices_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
                            _, ids = sample_farthest_points(vertices_tensor, K=256)
                            ids = ids[0].numpy()
                            selected_points = vertices[ids]
                        else:
                            indices = np.random.choice(len(vertices), 256, replace=True)
                            selected_points = vertices[indices]
                        
                        # 保存结果
                        with open("{result_file}", "wb") as f:
                            pickle.dump(selected_points.astype(np.float32), f)
                        
                        sys.exit(0)  # 成功
                    else:
                        sys.exit(2)  # 空网格
                        
                except Exception as e:
                    sys.exit(3)  # 网格处理错误
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_step_path):
                        os.remove(temp_step_path)
            else:
                sys.exit(4)  # 没有结果
                
        except Exception as e:
            sys.exit(5)  # 代码执行错误
            
except Exception as e:
    sys.exit(6)  # 导入或其他错误
finally:
    signal.alarm(0)  # 取消alarm
'''
            return script_content
        
        try:
            sample_id = abs(hash(py_code)) % 100000
            
            # 创建临时文件用于结果传递
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_path = script_file.name
                script_content = write_execution_script(py_code, script_path + '.result', sample_id)
                script_file.write(script_content)
            
            result_file = script_path + '.result'
            
            try:
                # 使用subprocess执行脚本，完全隔离
                result = subprocess.run(
                    [sys.executable, script_path],
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                    cwd='/tmp',  # 在tmp目录执行
                    env={**os.environ, 'PYTHONPATH': '/tmp'}  # 限制环境
                )
                
                # 检查执行结果
                if result.returncode == 0 and os.path.exists(result_file):
                    try:
                        with open(result_file, 'rb') as f:
                            point_cloud = pickle.load(f)
                        return point_cloud
                    except Exception as e:
                        self.logger.debug(f"Failed to load result: {e}")
                        return None
                else:
                    # 记录错误代码
                    error_messages = {
                        1: "Empty tessellation",
                        2: "Empty mesh",
                        3: "Mesh processing error",
                        4: "No result variable",
                        5: "Code execution error",
                        6: "Import/system error",
                        124: "Timeout",
                        -9: "Killed",
                        -11: "Segmentation fault"
                    }
                    error_msg = error_messages.get(result.returncode, f"Unknown error {result.returncode}")
                    self.logger.debug(f"CadQuery execution failed: {error_msg}")
                    return None
                    
            except subprocess.TimeoutExpired:
                self.logger.debug(f"CadQuery execution timeout after {timeout}s")
                return None
                
            finally:
                # 清理临时文件
                for temp_file in [script_path, result_file]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                            
        except Exception as e:
            self.logger.debug(f"Error in CadQuery execution setup: {e}")
            return None
                
        except Exception as e:
            self.logger.debug(f"Process execution failed: {e}")
            return None
    
    def _step_to_point_cloud(self, step_file_path: str) -> Optional[np.ndarray]:
        """将STEP文件转换为点云"""
        try:
            import trimesh
            from pytorch3d.ops import sample_farthest_points
            import torch
            
            if not OCC_AVAILABLE:
                # 如果没有OpenCascade，尝试直接加载STL格式
                if step_file_path.lower().endswith('.stl'):
                    mesh = trimesh.load_mesh(step_file_path)
                else:
                    return None
            else:
                # 使用OpenCascade处理STEP文件
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_file:
                    temp_stl_path = tmp_file.name
                
                try:
                    # 读取STEP文件
                    step_reader = STEPControl_Reader()
                    status = step_reader.ReadFile(step_file_path)
                    
                    if status == IFSelect_RetDone:
                        step_reader.TransferRoots()
                        shape = step_reader.Shape()
                        
                        # 导出为STL
                        write_stl_file(shape, temp_stl_path)
                        
                        # 使用trimesh加载STL
                        mesh = trimesh.load_mesh(temp_stl_path)
                    else:
                        return None
                        
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_stl_path):
                        os.remove(temp_stl_path)
            
            # 标准化网格
            if not mesh.is_empty and mesh.vertices is not None:
                # 移动到原点
                mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
                # 缩放到单位立方体
                if max(mesh.extents) > 0:
                    mesh.apply_scale(2.0 / max(mesh.extents))
                
                # 采样点云
                vertices, _ = trimesh.sample.sample_surface(mesh, 8192)
                
                # 使用最远点采样
                if len(vertices) >= self.config.n_points:
                    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
                    _, ids = sample_farthest_points(vertices_tensor, K=self.config.n_points)
                    ids = ids[0].numpy()
                    selected_points = vertices[ids]
                else:
                    # 如果点数不够，进行重复采样
                    indices = np.random.choice(len(vertices), self.config.n_points, replace=True)
                    selected_points = vertices[indices]
                
                return selected_points.astype(np.float32)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error converting to point cloud: {e}")
            return None
    
    def _get_point_cloud_from_code(self, code_content: str, sample_id: str) -> Optional[np.ndarray]:
        """从CadQuery代码生成点云，使用缓存机制，包含错误样本跟踪"""
        # 检查是否是已知错误样本
        if sample_id in self.error_samples:
            return None
            
        # 检查缓存
        cache_file = self.cache_dir / f"{sample_id}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except:
                # 缓存文件损坏，删除并重新生成
                cache_file.unlink(missing_ok=True)
        
        # 生成点云（使用进程隔离）
        try:
            point_cloud = self._execute_cadquery_code_safe(code_content, timeout=10)
            
            if point_cloud is not None:
                # 保存到缓存
                try:
                    np.save(cache_file, point_cloud)
                except Exception as e:
                    self.logger.warning(f"Failed to cache point cloud for {sample_id}: {e}")
                return point_cloud
            else:
                # 执行失败，标记为错误样本
                self._mark_sample_as_error(sample_id)
                return None
                
        except Exception as e:
            # 进程执行出现严重错误（如段错误），标记为错误样本
            self.logger.error(f"Serious error processing sample {sample_id}: {e}")
            self._mark_sample_as_error(sample_id)
            return None
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def _apply_data_augmentation(self, point_cloud: np.ndarray) -> np.ndarray:
        """应用数据增强 - 基于CAD-Recode论文策略"""
        if self.split == "train" and np.random.random() < self.config.noise_probability:
            # 添加高斯噪声（标准差0.01，50%概率）
            noise = np.random.normal(0, self.config.noise_std, point_cloud.shape)
            point_cloud = point_cloud + noise.astype(np.float32)
            
            self.logger.debug(f"Applied noise augmentation with std={self.config.noise_std}")
        
        return point_cloud
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据样本 - 优化版"""
        item = self.data_list[idx]
        sample_id = item["sample_id"]
        
        # 加载CadQuery代码
        try:
            with open(item["code_path"], 'r', encoding='utf-8') as f:
                cadquery_code = f.read().strip()
        except Exception as e:
            self.logger.error(f"Failed to load code for {sample_id}: {e}")
            return self._get_default_sample()
        
        # 生成点云
        point_cloud = self._get_point_cloud_from_code(cadquery_code, sample_id)
        if point_cloud is None:
            self.logger.warning(f"Failed to generate point cloud for {sample_id}, using random points")
            point_cloud = np.random.randn(self.config.n_points, 3).astype(np.float32)
        
        # 应用数据增强
        point_cloud = self._apply_data_augmentation(point_cloud)
        
        # 确保点云形状正确
        if point_cloud.shape[0] != self.config.n_points:
            if point_cloud.shape[0] > self.config.n_points:
                indices = np.random.choice(point_cloud.shape[0], self.config.n_points, replace=False)
                point_cloud = point_cloud[indices]
            else:
                indices = np.random.choice(point_cloud.shape[0], self.config.n_points, replace=True)
                point_cloud = point_cloud[indices]
        
        # 构建提示模板 - 添加代码提示
        if self.config.use_code_prompt:
            prompt = "<|im_start|>" + self.config.code_prompt
            full_text = prompt + cadquery_code + "<|endoftext|>"
        else:
            prompt = "<|im_start|>"
            full_text = prompt + cadquery_code + "<|endoftext|>"
        
        # 分词
        try:
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].squeeze(0)
        except Exception as e:
            self.logger.error(f"Tokenization failed for {sample_id}: {e}")
            return self._get_default_sample()
        
        # 创建attention_mask，点云位置标记为-1
        attention_mask = torch.ones_like(input_ids)
        
        # 为点云预留位置 (在prompt之前)
        point_mask = torch.full((self.config.n_points,), -1, dtype=torch.long)
        prompt_start_token = self.tokenizer("<|im_start|>")["input_ids"][0]
        
        # 找到prompt开始位置
        prompt_start_indices = (input_ids == prompt_start_token).nonzero(as_tuple=True)[0]
        if len(prompt_start_indices) > 0:
            prompt_start_idx = prompt_start_indices[0].item()
            has_prompt = True
        else:
            prompt_start_idx = -1  # 表示没有找到prompt
            has_prompt = False
            
        # 插入点云掩码
        input_ids_with_points = torch.cat([
            torch.full((self.config.n_points,), self.tokenizer.pad_token_id, dtype=torch.long),
            input_ids
        ])
        attention_mask_with_points = torch.cat([
            point_mask,
            attention_mask
        ])
        
        # 创建labels (只对代码部分计算损失)
        labels = input_ids_with_points.clone()
        labels[:self.config.n_points] = -100  # 点云部分不计算损失
        
        # 找到实际代码开始位置（跳过prompt）
        if self.config.use_code_prompt:
            # 如果使用代码提示，跳过import语句
            code_prompt_tokens = self.tokenizer(self.config.code_prompt)["input_ids"]
            code_start_offset = len(code_prompt_tokens)
        else:
            code_start_offset = 1  # 只跳过<|im_start|>
        
        if has_prompt:
            code_start_idx = self.config.n_points + prompt_start_idx + code_start_offset
            labels[:code_start_idx] = -100
        
        return {
            "input_ids": input_ids_with_points,
            "attention_mask": attention_mask_with_points,
            "labels": labels,
            "point_cloud": torch.tensor(point_cloud, dtype=torch.float32)
        }
    
    def _get_default_sample(self) -> Dict[str, torch.Tensor]:
        """获取默认样本（用于错误情况）"""
        # 生成最小化的默认数据
        default_text = "<|im_start|>r = cq.Workplane().box(1, 1, 1)<|endoftext|>"
        tokenized = self.tokenizer(
            default_text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"].squeeze(0)
        
        # 添加点云位置
        point_mask = torch.full((self.config.n_points,), -1, dtype=torch.long)
        input_ids_with_points = torch.cat([
            torch.full((self.config.n_points,), self.tokenizer.pad_token_id, dtype=torch.long),
            input_ids
        ])
        attention_mask_with_points = torch.cat([
            point_mask,
            torch.ones_like(input_ids)
        ])
        
        labels = input_ids_with_points.clone()
        labels[:self.config.n_points + 1] = -100  # 点云和prompt不计算损失
        
        return {
            "input_ids": input_ids_with_points,
            "attention_mask": attention_mask_with_points,
            "labels": labels,
            "point_cloud": torch.randn(self.config.n_points, 3, dtype=torch.float32)
        }


class CADDataCollator:
    """CAD数据整理器"""
    
    def __init__(self, tokenizer, pad_token_id: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """安全的数据整理，带错误处理"""
        try:
            # 过滤无效的features
            valid_features = []
            for feature in features:
                if (feature is not None and 
                    "input_ids" in feature and 
                    "attention_mask" in feature and 
                    "labels" in feature and 
                    "point_cloud" in feature):
                    valid_features.append(feature)
            
            if not valid_features:
                # 如果没有有效features，返回默认batch
                return self._get_default_batch()
            
            # 获取最大序列长度
            max_length = max(f["input_ids"].size(0) for f in valid_features)
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            batch_point_clouds = []
            
            for feature in valid_features:
                try:
                    input_ids = feature["input_ids"]
                    attention_mask = feature["attention_mask"]
                    labels = feature["labels"]
                    point_cloud = feature["point_cloud"]
                    
                    # 填充序列
                    padding_length = max_length - input_ids.size(0)
                    if padding_length > 0:
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((padding_length,), self.pad_token_id, dtype=input_ids.dtype)
                        ])
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(padding_length, dtype=attention_mask.dtype)
                        ])
                        labels = torch.cat([
                            labels,
                            torch.full((padding_length,), -100, dtype=labels.dtype)
                        ])
                    
                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_labels.append(labels)
                    batch_point_clouds.append(point_cloud)
                    
                except Exception as e:
                    # 如果单个feature处理失败，跳过它
                    continue
            
            # 检查是否有有效的批次数据
            if not batch_input_ids:
                return self._get_default_batch()
            
            return {
                "input_ids": torch.stack(batch_input_ids),
                "attention_mask": torch.stack(batch_attention_mask),
                "labels": torch.stack(batch_labels),
                "point_cloud": torch.stack(batch_point_clouds)
            }
            
        except Exception as e:
            # 如果整个批次处理失败，返回默认批次
            import logging
            logger = logging.getLogger('CADRecode_Training')
            logger.warning(f"Error in data collator: {e}, returning default batch")
            return self._get_default_batch()
    
    def _get_default_batch(self) -> Dict[str, torch.Tensor]:
        """返回默认的安全批次"""
        batch_size = 1
        seq_len = 128
        n_points = 256
        
        return {
            "input_ids": torch.full((batch_size, seq_len), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "labels": torch.full((batch_size, seq_len), -100, dtype=torch.long),
            "point_cloud": torch.randn(batch_size, n_points, 3, dtype=torch.float32)
        }


class CADTrainer(Trainer):
    """自定义Trainer类 - 增强日志记录和错误处理"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'steps': []
        }
        self.error_count = 0
        self.max_consecutive_errors = 5
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算损失 - 兼容transformers新版本并增强错误处理"""
        try:
            labels = inputs.get("labels")
            if labels is not None:
                # 移除-100标记的位置（不参与损失计算）
                valid_mask = (labels != -100)
                if not valid_mask.any():
                    # 如果没有有效标签，返回零损失
                    loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
                    if return_outputs:
                        # 创建虚拟输出
                        outputs = type('MockOutputs', (), {
                            'loss': loss,
                            'logits': torch.zeros(labels.shape + (model.config.vocab_size,), device=labels.device)
                        })()
                        return (loss, outputs)
                    return loss
            
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs.get('loss')
            
            if loss is None:
                # 如果模型没有返回损失，手动计算
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs.get('logits')
                if logits is not None and labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss = torch.tensor(0.0, device=inputs.get('input_ids', torch.tensor([])).device, requires_grad=True)
            
            # 检查损失值是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                self.error_count += 1
                self.logger.warning(f"Invalid loss detected (NaN/Inf), error count: {self.error_count}")
                loss = torch.tensor(0.1, device=loss.device, requires_grad=True)  # 使用小的默认损失
                
                if self.error_count >= self.max_consecutive_errors:
                    self.logger.error(f"Too many consecutive errors ({self.error_count}), may need to check data quality")
            else:
                self.error_count = 0  # 重置错误计数
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error in compute_loss: {e}, error count: {self.error_count}")
            
            # 返回安全的默认损失
            device = inputs.get('input_ids', torch.tensor([])).device
            loss = torch.tensor(0.1, device=device, requires_grad=True)
            
            if return_outputs:
                # 创建虚拟输出以避免后续错误
                outputs = type('MockOutputs', (), {
                    'loss': loss,
                    'logits': torch.zeros((1, 1, getattr(model.config, 'vocab_size', 32000)), device=device)
                })()
                return (loss, outputs)
            
            return loss
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """增强的日志记录功能 - 兼容新版transformers"""
        # 调用父类的log方法，传递所有参数
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # 记录到历史
        if 'train_loss' in logs:
            self.loss_history['train_loss'].append(logs['train_loss'])
            self.loss_history['steps'].append(self.state.global_step)
        
        if 'eval_loss' in logs:
            self.loss_history['eval_loss'].append(logs['eval_loss'])
        
        if 'learning_rate' in logs:
            self.loss_history['learning_rate'].append(logs['learning_rate'])
        
        # 详细日志格式
        step = self.state.global_step
        epoch = self.state.epoch
        
        log_msg = f"Step {step:6d} | Epoch {epoch:.2f}"
        
        if 'train_loss' in logs:
            log_msg += f" | Train Loss: {logs['train_loss']:.6f}"
        
        if 'eval_loss' in logs:
            log_msg += f" | Eval Loss: {logs['eval_loss']:.6f}"
        
        if 'learning_rate' in logs:
            log_msg += f" | LR: {logs['learning_rate']:.2e}"
        
        # GPU内存使用情况
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            log_msg += f" | GPU Mem: {memory_used:.1f}GB"
        
        # 记录到logger
        logger = logging.getLogger('CADRecode_Training')
        logger.info(log_msg)
        
        # 每1000步保存损失历史
        if step % 1000 == 0:
            self.save_loss_history()
    
    def save_loss_history(self):
        """保存损失历史到文件"""
        import json
        history_file = Path(self.args.output_dir) / "loss_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
        except Exception as e:
            logger = logging.getLogger('CADRecode_Training')
            logger.warning(f"Failed to save loss history: {e}")
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """评估时的回调"""
        super().on_evaluate(args, state, control, model, **kwargs)
        
        # 创建即时可视化
        if state.global_step % 5000 == 0:  # 每5000步生成一次图表
            try:
                self.create_loss_plot()
            except Exception as e:
                logger = logging.getLogger('CADRecode_Training')
                logger.warning(f"Failed to create loss plot: {e}")
    
    def create_loss_plot(self):
        """Create loss curve plot"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Training loss
            if self.loss_history['train_loss']:
                steps = self.loss_history['steps'][-len(self.loss_history['train_loss']):]
                ax1.plot(steps, self.loss_history['train_loss'], 'b-', alpha=0.7)
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Loss')
                ax1.grid(True, alpha=0.3)
            
            # Learning rate
            if self.loss_history['learning_rate']:
                steps = self.loss_history['steps'][-len(self.loss_history['learning_rate']):]
                ax2.plot(steps, self.loss_history['learning_rate'], 'orange')
                ax2.set_title('Learning Rate')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Learning Rate')
                ax2.set_yscale('log')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = Path(self.args.output_dir) / f"loss_plot_step_{self.state.global_step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger = logging.getLogger('CADRecode_Training')
            logger.info(f"📊 Loss plot saved: {plot_path}")
            
        except ImportError:
            pass  # Skip if matplotlib is not available


def create_model_and_tokenizer(config: TrainingConfig) -> Tuple[CADRecode, AutoTokenizer]:
    """创建模型和分词器（修复注意力机制警告）"""
    logger = logging.getLogger('CADRecode_Training')
    
    # 加载分词器
    logger.info(f"Loading tokenizer from {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        pad_token='<|im_end|>',
        padding_side='left',
        trust_remote_code=True
    )
    
    # 加载模型配置并修复滑动窗口注意力问题
    model_config = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
    
    # 禁用滑动窗口注意力以避免实现警告
    if hasattr(model_config, 'sliding_window'):
        model_config.sliding_window = None
        logger.info("Disabled sliding window attention to avoid implementation issues")
    
    # 强制使用 'eager' (默认) 实现以避免 Qwen2+FlashAttention+右填充的bug
    # attn_implementation = "eager"
    # logger.warning("Flash Attention has been forcibly disabled. Using 'eager' attention implementation.")
    
    # 根据用户请求，重新启用Flash Attention 2以节省内存
    attn_implementation = "flash_attention_2"
    logger.info("Using 'flash_attention_2' attention implementation as requested to save memory.")

    try:
        model = CADRecode.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
            trust_remote_code=True
        )
        
        # 复制预训练权重
        model.model.load_state_dict(base_model.model.state_dict(), strict=False)
        model.lm_head.load_state_dict(base_model.lm_head.state_dict(), strict=False)
        
        logger.info("Successfully loaded pretrained weights")
        del base_model  # 释放内存
    except Exception as e:
        logger.warning(f"Could not load pretrained weights: {e}")
    
    return model, tokenizer


def train_model(config: TrainingConfig):
    """主训练函数 - 基于CAD-Recode论文优化"""
    # 设置随机种子
    set_seed(config.seed)
    
    # 创建输出目录
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(config.model_save_path, "training.log")
    logger = setup_logger(log_file)
    
    logger.info("Starting CAD-Recode training with optimized strategy")
    logger.info(f"Training strategy based on CAD-Recode paper:")
    logger.info(f"  - Max training steps: {config.max_training_steps}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Warmup steps: {config.warmup_steps}")
    logger.info(f"  - Data augmentation: noise_std={config.noise_std}, prob={config.noise_probability}")
    logger.info(f"Configuration: {config}")
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer(config)
    
    # 创建数据集
    logger.info("Loading datasets...")
    train_dataset = CADDataset(config.train_data_path, tokenizer, config, "train")
    val_dataset = CADDataset(config.val_data_path, tokenizer, config, "val")
    
    # 更新配置中的数据集大小（用于计算epoch数）
    config.train_dataset_size = len(train_dataset)
    config.__post_init__()
    
    logger.info(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 创建数据整理器
    data_collator = CADDataCollator(tokenizer)
    
    # 计算训练参数
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    total_steps = config.max_training_steps
    estimated_epochs = max(1, total_steps * effective_batch_size // len(train_dataset))
    
    logger.info(f"Training parameters:")
    logger.info(f"  - Effective batch size: {effective_batch_size}")
    logger.info(f"  - Total training steps: {total_steps}")
    logger.info(f"  - Estimated epochs: {estimated_epochs}")
    
    # 设置训练参数 - 基于论文优化
    training_args = TrainingArguments(
        output_dir=config.model_save_path,
        num_train_epochs=1,  # Set epochs to 1, control training with max_steps
        max_steps=config.max_training_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        bf16=config.mixed_precision == 'bf16',
        fp16=config.mixed_precision == 'fp16',
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True if config.val_data_path else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        # gradient_checkpointing=True,  # 启用梯度检查点以节省内存
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=config.num_workers,
        save_total_limit=5,
        report_to=["none"],  # 使用SwanLab回调而不是内置报告工具
        logging_dir=str(Path(config.model_save_path) / "logs"),
    )

    # 准备回调函数
    callbacks = []
    
    # 添加SwanLab回调用于实验跟踪
    if SWANLAB_AVAILABLE:
        try:
            swanlab_callback = SwanLabCallback(
                project="CAD-Recode",
                experiment_name=f"cad-recode-training-{config.base_model_name.replace('/', '-')}",
                description="CAD-Recode training with optimized strategy based on paper",
                config={
                    "model_name": config.base_model_name,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "max_training_steps": config.max_training_steps,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "n_points": config.n_points,
                    "max_seq_length": config.max_seq_length,
                    "noise_std": config.noise_std,
                    "noise_probability": config.noise_probability,
                    "mixed_precision": config.mixed_precision,
                }
            )
            callbacks.append(swanlab_callback)
            logger.info("SwanLab callback added for experiment tracking")
        except Exception as e:
            logger.warning(f"Failed to initialize SwanLab callback: {e}")
    
    # 创建 Trainer
    trainer = CADTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,  # 添加回调函数
    )
    
    # 检查是否有检查点可以恢复
    checkpoint = get_last_checkpoint(config.model_save_path)
    if checkpoint:
        logger.info(f"Resuming training from checkpoint: {checkpoint}")
    
    # 开始训练
    logger.info("Starting training with CAD-Recode optimized strategy...")
    logger.info(f"Expected training time: ~12 hours (based on paper, single H100)")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存最终模型
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.model_save_path)
    
    # 保存训练配置
    config_path = os.path.join(config.model_save_path, "training_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    
    logger.info("Training completed!")
    logger.info(f"Final training loss: {train_result.training_loss}")
    logger.info(f"Total training steps: {train_result.global_step}")
    
    return trainer, train_result


def test_time_sampling_inference(model, tokenizer, point_cloud: np.ndarray, 
                                config: TrainingConfig, 
                                max_new_tokens: int = 768) -> str:
    """Test-time采样策略推理 - 基于CAD-Recode论文"""
    if not config.use_test_time_sampling:
        # 标准推理
        return single_inference(model, tokenizer, point_cloud, max_new_tokens)
    
    logger = logging.getLogger('CADRecode_Training')
    logger.info(f"Using test-time sampling with {config.num_inference_samples} samples")
    
    candidates = []
    
    # 生成多个候选代码
    for i in range(config.num_inference_samples):
        try:
            code = single_inference(model, tokenizer, point_cloud, max_new_tokens)
            if code and code.strip():
                candidates.append(code)
        except Exception as e:
            logger.warning(f"Failed to generate candidate {i}: {e}")
            continue
    
    if not candidates:
        logger.error("No valid candidates generated")
        return "r = cq.Workplane().box(1, 1, 1)"  # 默认代码
    
    if len(candidates) == 1:
        return candidates[0]
    
    # 评估候选代码，选择几何距离最小的
    best_code = candidates[0]
    best_score = float('inf')
    
    for code in candidates:
        try:
            # 计算Chamfer距离
            score = evaluate_code_geometry(code, point_cloud)
            if score < best_score:
                best_score = score
                best_code = code
        except Exception as e:
            logger.warning(f"Failed to evaluate candidate: {e}")
            continue
    
    logger.info(f"Selected best candidate with score: {best_score}")
    return best_code


def single_inference(model, tokenizer, point_cloud: np.ndarray, 
                    max_new_tokens: int = 768) -> str:
    """单次推理"""
    # 准备输入 (基于demo.ipynb)
    input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
    attention_mask = [-1] * len(point_cloud) + [1]
    
    with torch.no_grad():
        batch_ids = model.generate(
            input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
            attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(model.device),
            point_cloud=torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).to(model.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,  # 启用采样
            temperature=0.8,  # 控制随机性
            top_p=0.9,       # nucleus采样
        )
    
    # 解码生成的代码
    py_string = tokenizer.batch_decode(batch_ids)[0]
    begin = py_string.find('<|im_start|>') + 12
    end = py_string.find('<|endoftext|>')
    py_string = py_string[begin: end]
    
    return py_string


def evaluate_code_geometry(code: str, target_point_cloud: np.ndarray, 
                          timeout: int = 5) -> float:
    """评估生成代码的几何质量（Chamfer距离）"""
    try:
        import trimesh
        from scipy.spatial import cKDTree
        
        def target_func(code, result_queue):
            try:
                # 执行代码生成CAD模型
                exec_globals = {'cq': cq}
                exec(code, exec_globals)
                
                if 'r' in exec_globals:
                    compound = exec_globals['r'].val()
                    
                    # 导出为临时STEP文件
                    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp_file:
                        temp_step_path = tmp_file.name
                    
                    try:
                        cq.exporters.export(compound, temp_step_path)
                        
                        # 转换为点云
                        generated_pc = step_to_point_cloud_for_eval(temp_step_path, len(target_point_cloud))
                        if generated_pc is not None:
                            # 计算Chamfer距离
                            cd = compute_chamfer_distance(target_point_cloud, generated_pc)
                            result_queue.put((True, cd))
                        else:
                            result_queue.put((False, float('inf')))
                    finally:
                        if os.path.exists(temp_step_path):
                            os.remove(temp_step_path)
                else:
                    result_queue.put((False, float('inf')))
                    
            except Exception as e:
                result_queue.put((False, float('inf')))
        
        # 使用进程执行（避免CadQuery内存泄漏）
        result_queue = Queue()
        process = Process(target=target_func, args=(code, result_queue))
        process.start()
        process.join(timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return float('inf')
        
        if not result_queue.empty():
            success, score = result_queue.get()
            return score if success else float('inf')
        
        return float('inf')
        
    except Exception as e:
        return float('inf')


def step_to_point_cloud_for_eval(step_file_path: str, n_points: int = 256) -> Optional[np.ndarray]:
    """用于评估的STEP转点云函数"""
    try:
        import trimesh
        from pytorch3d.ops import sample_farthest_points
        import torch
        
        if not OCC_AVAILABLE:
            return None
        
        # 使用OpenCascade处理STEP文件
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_file:
            temp_stl_path = tmp_file.name
        
        try:
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(step_file_path)
            
            if status == IFSelect_RetDone:
                step_reader.TransferRoots()
                shape = step_reader.Shape()
                write_stl_file(shape, temp_stl_path)
                mesh = trimesh.load_mesh(temp_stl_path)
                
                # 标准化网格
                if not mesh.is_empty and mesh.vertices is not None:
                    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
                    if max(mesh.extents) > 0:
                        mesh.apply_scale(2.0 / max(mesh.extents))
                    
                    # 采样点云
                    vertices, _ = trimesh.sample.sample_surface(mesh, 8192)
                    if len(vertices) >= n_points:
                        vertices_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
                        _, ids = sample_farthest_points(vertices_tensor, K=n_points)
                        ids = ids[0].numpy()
                        return vertices[ids].astype(np.float32)
            
            return None
            
        finally:
            if os.path.exists(temp_stl_path):
                os.remove(temp_stl_path)
                
    except Exception as e:
        return None


def compute_chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """计算两个点云之间的Chamfer距离"""
    try:
        from scipy.spatial import cKDTree
        
        # 确保点云形状一致
        if pc1.shape[0] != pc2.shape[0]:
            min_points = min(pc1.shape[0], pc2.shape[0])
            if pc1.shape[0] > min_points:
                indices = np.random.choice(pc1.shape[0], min_points, replace=False)
                pc1 = pc1[indices]
            if pc2.shape[0] > min_points:
                indices = np.random.choice(pc2.shape[0], min_points, replace=False)
                pc2 = pc2[indices]
        
        # 计算Chamfer距离
        tree1 = cKDTree(pc1)
        tree2 = cKDTree(pc2)
        
        dist1, _ = tree1.query(pc2, k=1)
        dist2, _ = tree2.query(pc1, k=1)
        
        chamfer_dist = np.mean(np.square(dist1)) + np.mean(np.square(dist2))
        return float(chamfer_dist)
        
    except Exception as e:
        return float('inf')


def create_optimized_model_with_inference(config: TrainingConfig):
    """创建支持优化推理的模型"""
    model, tokenizer = create_model_and_tokenizer(config)
    
    # 添加推理方法到模型
    def inference_method(point_cloud: np.ndarray, max_new_tokens: int = 768) -> str:
        return test_time_sampling_inference(model, tokenizer, point_cloud, config, max_new_tokens)
    
    model.optimized_inference = inference_method
    return model, tokenizer


def setup_signal_handlers():
    """设置信号处理器以优雅处理崩溃"""
    def signal_handler(signum, frame):
        if signum == signal.SIGSEGV:
            print(f"\n[CRITICAL] Segmentation fault detected! Attempting graceful shutdown...")
            logger = logging.getLogger('CADRecode_Training')
            logger.error(f"Segmentation fault detected at frame: {frame}")
            
            # 尝试保存当前状态
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
            except:
                pass
            
            # 记录错误信息
            logger.error("Training terminated due to segmentation fault")
            sys.exit(1)
        elif signum == signal.SIGINT:
            print(f"\n[INFO] Received interrupt signal, shutting down gracefully...")
            sys.exit(0)
        elif signum == signal.SIGTERM:
            print(f"\n[INFO] Received termination signal, shutting down gracefully...")
            sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    # 设置信号处理器
    setup_signal_handlers()
    
    parser = argparse.ArgumentParser(description="CAD-Recode训练脚本")
    parser.add_argument("--config", type=str, default=None,
                       help="训练配置文件路径 (YAML格式)")
    parser.add_argument("--train_data_path", type=str, default="./data/train",
                       help="训练数据路径")
    parser.add_argument("--val_data_path", type=str, default="./data/val",
                       help="验证数据路径")
    parser.add_argument("--model_save_path", type=str, default="./checkpoints",
                       help="模型保存路径")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2-1.5B",
                       help="基础模型名称")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="学习率")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备")
    
    args = parser.parse_args()
    
    # 初始化日志
    logger = setup_logger()
    
    # 创建训练配置
    if args.config and os.path.exists(args.config):
        # 从配置文件加载
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 过滤掉不属于TrainingConfig的字段
        from dataclasses import fields
        valid_fields = {field.name for field in fields(TrainingConfig)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        logger.info(f"Loaded config with fields: {list(filtered_config.keys())}")
        config = TrainingConfig(**filtered_config)
    else:
        # 从命令行参数创建
        config = TrainingConfig(
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            model_save_path=args.model_save_path,
            base_model_name=args.base_model_name,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device
        )
    
    # 训练模型
    try:
        # 设置随机种子
        set_seed(config.seed)
        
        # 设置设备信息
        if torch.cuda.is_available():
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            torch.cuda.empty_cache()
        
        # 开始训练
        logger.info("Starting CAD-Recode training with robust error handling...")
        
        trainer, train_result = train_model(config)
        
        # 显示训练完成信息
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final model saved to: {config.model_save_path}")
        print(f"Total training steps: {train_result.global_step}")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print("\nTo run inference, use:")
        print("python demo_optimized_strategy.py --model_path <path_to_model>")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\n[INFO] Training interrupted by user. Cleaning up...")
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        return 1
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory error: {e}")
            print("\n[ERROR] GPU out of memory! Try reducing batch_size or max_seq_length")
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            
        else:
            logger.error(f"Runtime error during training: {e}")
            print(f"\n[ERROR] Runtime error: {e}")
            import traceback
            traceback.print_exc()
        
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
