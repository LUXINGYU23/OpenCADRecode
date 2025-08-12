#!/usr/bin/env python3
"""
CAD-Recode 训练脚本 v2.0 - 精简版
支持从点云生成CadQuery代码的多模态模型训练

主要特性：
1. 基于train_qwen3的架构设计
2. 支持切换基座模型（Qwen2/Qwen3等）
3. 使用标准Trainer进行监督微调
4. 集成SwanLab进行实验跟踪
5. 模块化设计，分离模型、数据和工具函数

使用方法:
python train/train_cad_recode_full.py --config configs/train_config_sft.yaml
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
import argparse
import yaml
from typing import Tuple

from transformers import TrainingArguments, Trainer

# 导入自定义模块
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils',)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models',)))
from utils import TrainingConfig, setup_environment, set_random_seeds
from models import create_model_and_tokenizer
from data_utils import CADRecodeDataset, DataCollatorForCADRecode

# 导入SwanLab用于实验跟踪
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not installed. Install with: pip install swanlab")

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(config: TrainingConfig):
    """主训练函数"""
    # 设置环境和随机种子
    setup_environment()
    set_random_seeds(config.seed)
    
    # 创建输出目录
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 初始化SwanLab
    if config.use_swanlab and SWANLAB_AVAILABLE:
        swanlab.init(
            project="cad-recode-qwen3-full",
            experiment_name=config.experiment_name,
            config=config.__dict__
        )
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer(config)
    #打印模型结构
    print(model)
    # 创建数据集
    print("Loading datasets...")
    train_dataset = CADRecodeDataset(config.train_data_path, tokenizer, config, "train")
    val_dataset = CADRecodeDataset(config.val_data_path, tokenizer, config, "val") if config.val_data_path else None
    
    # 创建数据collator
    data_collator = DataCollatorForCADRecode(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        pad_to_multiple_of=8  # 可选：提高效率
    )   
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=config.model_save_path,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        bf16=config.mixed_precision == "bf16",
        fp16=config.mixed_precision == "fp16",
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,  # 保留point_cloud列
        gradient_checkpointing=True,
        dataloader_num_workers=config.num_workers,
        save_total_limit=3,
        report_to=["none"],  # 我们使用SwanLab
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 添加SwanLab回调
    if config.use_swanlab and SWANLAB_AVAILABLE:
        from transformers import TrainerCallback
        
        class SwanLabCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    swanlab.log(logs)
        
        trainer.add_callback(SwanLabCallback())
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.model_save_path)
    
    # 保存配置
    with open(os.path.join(config.model_save_path, "config.yaml"), 'w') as f:
        yaml.dump(config.__dict__, f)
    
    print("Training completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CAD-Recode Training Script v2.0 - Modularized")
    parser.add_argument("--config", type=str, required=True, help="Training configuration file")
    parser.add_argument("--base_model", type=str, help="Override base model name")
    parser.add_argument("--experiment_name", type=str, help="Override experiment name")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = TrainingConfig(**config_dict)
    
    # 覆盖配置
    if args.base_model:
        config.base_model_name = args.base_model
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    print(f"Training Configuration:")
    print(f"  Base Model: {config.base_model_name}")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    
    # 开始训练
    train_model(config)


if __name__ == "__main__":
    main()
