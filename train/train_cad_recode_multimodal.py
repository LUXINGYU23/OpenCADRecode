#!/usr/bin/env python3
"""
CAD-Recode 多模态训练脚本 v2.0
支持BRep和点云两种输入模态
"""
import os
import argparse
import warnings
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer

# 导入自定义模块
from utils import TrainingConfig, setup_environment, set_random_seeds
from models.cad_recode_multimodal import create_multimodal_model_and_tokenizer
from utils.data_utils import (
    CADRecodeMultimodalDataset, 
    DataCollatorForMultimodalCADRecode,
    NUM_POINT_TOKENS
)

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def train_multimodal_model(config: TrainingConfig):
    """主训练函数"""
    # 设置环境和随机种子
    setup_environment()
    set_random_seeds(config.seed)
    
    # 创建输出目录
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 创建模型和分词器
    model, tokenizer = create_multimodal_model_and_tokenizer(config)
    
    # 打印模型结构
    print(model)
    
    # 创建数据集
    print("Loading datasets...")
    train_dataset = CADRecodeMultimodalDataset(
        config.train_data_path, 
        tokenizer, 
        config, 
        "train"
    )
    val_dataset = CADRecodeMultimodalDataset(
        config.val_data_path, 
        tokenizer, 
        config, 
        "val"
    ) if config.val_data_path else None
    
    # 创建数据collator
    data_collator = DataCollatorForMultimodalCADRecode(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        pad_to_multiple_of=8,
        num_brep_tokens=NUM_POINT_TOKENS,
        num_point_tokens=NUM_POINT_TOKENS
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
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=config.num_workers,
        save_total_limit=3,
        report_to=["none"],
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
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.model_save_path)
    
    print("Training completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CAD-Recode Multimodal Training Script v2.0")
    parser.add_argument("--config", type=str, required=True, help="Training configuration file")
    parser.add_argument("--base_model", type=str, help="Override base model name")
    parser.add_argument("--experiment_name", type=str, help="Override experiment name")
    
    args = parser.parse_args()
    
    # 加载配置
    config = TrainingConfig.from_yaml(args.config)
    
    # 覆盖配置
    if args.base_model:
        config.base_model_name = args.base_model
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # 开始训练
    train_multimodal_model(config)


if __name__ == "__main__":
    main()
