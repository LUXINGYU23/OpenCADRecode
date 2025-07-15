#!/usr/bin/env python3
"""
CAD-Recode LoRA微调训练脚本 v2.0
支持从点云生成CadQuery代码的多模态模型LoRA微调

主要特性：
1. 基于PEFT库的LoRA微调
2. 支持切换基座模型（Qwen2/Qwen3等）
3. 使用标准Trainer进行监督微调
4. 集成SwanLab进行实验跟踪
5. 内存友好的LoRA训练方案
6. 支持QLoRA（量化LoRA）

使用方法:
python train_cad_recode_lora.py --config configs/train_config_lora.yaml
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
import argparse
import yaml
from typing import Tuple
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 导入自定义模块
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


class LoRATrainingConfig(TrainingConfig):
    """LoRA训练配置类，继承基础配置并添加LoRA特定配置"""
    
    def __init__(self, **kwargs):
        # LoRA特定参数列表
        lora_specific_params = {
            'use_lora', 'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules',
            'use_qlora', 'load_in_4bit', 'load_in_8bit', 'bnb_4bit_compute_dtype',
            'bnb_4bit_use_double_quant', 'bnb_4bit_quant_type',
            'auto_merge_lora', 'auto_merge_final', 'keep_lora_only', 
            'keep_lora_final', 'merge_final_only', 'load_from_full_model', 'freeze_base_model'
        }
        
        # 分离基础配置和LoRA配置
        base_kwargs = {k: v for k, v in kwargs.items() if k not in lora_specific_params}
        
        # 调用父类初始化
        super().__init__(**base_kwargs)
        
        # LoRA配置
        self.use_lora: bool = kwargs.get('use_lora', True)
        self.lora_r: int = kwargs.get('lora_r', 16)
        self.lora_alpha: int = kwargs.get('lora_alpha', 32)
        self.lora_dropout: float = kwargs.get('lora_dropout', 0.1)
        self.lora_target_modules: list = kwargs.get('lora_target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
        
        # 量化配置 (QLoRA)
        self.use_qlora: bool = kwargs.get('use_qlora', False)
        self.load_in_4bit: bool = kwargs.get('load_in_4bit', False)
        self.load_in_8bit: bool = kwargs.get('load_in_8bit', False)
        self.bnb_4bit_compute_dtype: str = kwargs.get('bnb_4bit_compute_dtype', "bfloat16")
        self.bnb_4bit_use_double_quant: bool = kwargs.get('bnb_4bit_use_double_quant', True)
        self.bnb_4bit_quant_type: str = kwargs.get('bnb_4bit_quant_type', "nf4")
        
        # 自动合并配置
        self.auto_merge_lora: bool = kwargs.get('auto_merge_lora', True)
        self.auto_merge_final: bool = kwargs.get('auto_merge_final', True)
        self.keep_lora_only: bool = kwargs.get('keep_lora_only', True)
        self.keep_lora_final: bool = kwargs.get('keep_lora_final', True)
        self.merge_final_only: bool = kwargs.get('merge_final_only', False)
        
        # 从full模型训练的特殊配置
        self.load_from_full_model: bool = kwargs.get('load_from_full_model', False)
        self.freeze_base_model: bool = kwargs.get('freeze_base_model', False)


def create_quantization_config(config: LoRATrainingConfig):
    """创建量化配置"""
    if not (config.use_qlora or config.load_in_4bit or config.load_in_8bit):
        return None
    
    if config.load_in_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        )
    elif config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    
    return None


def create_lora_model(model, tokenizer, config: LoRATrainingConfig):
    """创建LoRA模型"""
    if not config.use_lora:
        return model
    
    # 如果使用量化，准备模型
    if config.use_qlora or config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # 确保点云编码器需要梯度
    if hasattr(model, 'point_encoder'):
        for param in model.point_encoder.parameters():
            param.requires_grad = True
        print("✅ Point encoder gradients enabled")
        
        # 确保点云编码器数据类型与模型一致
        model_dtype = next(model.parameters()).dtype
        if model.point_encoder.projection.weight.dtype != model_dtype:
            model.point_encoder = model.point_encoder.to(model_dtype)
            print(f"✅ Point encoder dtype set to {model_dtype}")
    
    # 配置LoRA - 添加点云编码器到目标模块
    target_modules = config.lora_target_modules.copy()
    
    # 检查是否需要添加点云编码器模块
    if hasattr(model, 'point_encoder'):
        print("🎯 Adding point encoder to LoRA target modules")
        # 添加点云编码器的线性层
        if hasattr(model.point_encoder, 'projection'):
            target_modules.append("point_encoder.projection")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["point_encoder"],  # 保存点云编码器的所有参数
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 再次确保点云编码器需要梯度（PEFT可能会影响）
    if hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'point_encoder'):
        for param in model.base_model.model.point_encoder.parameters():
            param.requires_grad = True
        print("✅ Point encoder gradients re-enabled after LoRA")
    elif hasattr(model, 'point_encoder'):
        for param in model.point_encoder.parameters():
            param.requires_grad = True
        print("✅ Point encoder gradients re-enabled after LoRA (direct access)")
    
    # 启用输入梯度（对于CADRecode模型）
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
        print("✅ Input gradients enabled")
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    return model


def create_lora_model_and_tokenizer(config: LoRATrainingConfig):
    """创建LoRA模型和分词器"""
    from transformers import AutoTokenizer, AutoConfig
    from models import CADRecode
    import os
    
    print(f"Loading model and tokenizer from {config.base_model_name}")
    
    # 检查是否从full模型加载
    is_local_model = os.path.exists(config.base_model_name) and os.path.isdir(config.base_model_name)
    load_from_full = getattr(config, 'load_from_full_model', False) or is_local_model
    
    if load_from_full:
        print(f"🔄 Loading from trained full model: {config.base_model_name}")
        
        # 从本地已训练模型加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
        
        # 从本地加载模型配置
        model_config = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
        
        # 禁用滑动窗口注意力
        if hasattr(model_config, 'sliding_window'):
            model_config.sliding_window = None
        
        # 创建量化配置
        quantization_config = create_quantization_config(config)
        
        # 确定数据类型
        if quantization_config:
            torch_dtype = torch.float16  # 量化时使用fp16
        else:
            torch_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        
        # 从本地已训练模型加载
        model = CADRecode.from_pretrained(
            config.base_model_name,
            config=model_config,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        print(f"✅ Successfully loaded trained CADRecode model from {config.base_model_name}")
        
    else:
        print(f"🔄 Loading base model: {config.base_model_name}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            pad_token='<|im_end|>',
            padding_side='left',
            trust_remote_code=True
        )
        
        # 加载模型配置
        model_config = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
        
        # 禁用滑动窗口注意力
        if hasattr(model_config, 'sliding_window'):
            model_config.sliding_window = None
        
        # 创建量化配置
        quantization_config = create_quantization_config(config)
        
        # 确定数据类型
        if quantization_config:
            torch_dtype = torch.float16  # 量化时使用float16
        else:
            torch_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        
        # 创建模型
        model = CADRecode.from_pretrained(
            config.base_model_name,
            config=model_config,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    
    # 应用LoRA
    model = create_lora_model(model, tokenizer, config)
    
    # 不启用梯度检查点以避免兼容性问题
    # if not quantization_config:
    #     model.gradient_checkpointing_enable()
    
    return model, tokenizer


def train_model(config: LoRATrainingConfig):
    """主训练函数"""
    # 设置环境和随机种子
    setup_environment()
    set_random_seeds(config.seed)
    
    # 创建输出目录
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 初始化SwanLab
    if config.use_swanlab and SWANLAB_AVAILABLE:
        swanlab.init(
            project="cad-recode-lora",
            experiment_name=config.experiment_name,
            config=config.__dict__
        )
    
    # 创建模型和分词器
    model, tokenizer = create_lora_model_and_tokenizer(config)
    
    # 调试模型梯度状态
    debug_model_gradients(model, "LoRA Model")
    
    # 打印模型信息
    print("Model structure:")
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
        bf16=config.mixed_precision == "bf16" and not (config.load_in_4bit or config.load_in_8bit),
        fp16=config.mixed_precision == "fp16" or (config.load_in_4bit or config.load_in_8bit),
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,  # 保留point_cloud列
        gradient_checkpointing=False,  # 禁用梯度检查点以避免兼容性问题
        dataloader_num_workers=config.num_workers,
        save_total_limit=3,
        report_to=["none"],  # 我们使用SwanLab
        ddp_find_unused_parameters=False,  # LoRA训练优化
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
    
    # 添加LoRA自动合并回调（如果启用）
    if config.use_lora and hasattr(config, 'auto_merge_lora') and config.auto_merge_lora:
        from utils import LoRACheckpointCallback
        from transformers import TrainerCallback
        
        class LoRAMergeCallback(TrainerCallback):
            def __init__(self, base_model_name, auto_merge=True, keep_lora_only=True, merge_final_only=False):
                self.base_model_name = base_model_name
                self.auto_merge = auto_merge
                self.keep_lora_only = keep_lora_only
                self.merge_final_only = merge_final_only
            
            def on_save(self, args, state, control, **kwargs):
                # 如果设置了只在最终保存时合并，且当前不是最终保存，则跳过
                is_final = (state.global_step >= args.max_steps)
                if self.merge_final_only and not is_final:
                    return
                
                if self.auto_merge:
                    from utils import merge_lora_weights_to_model
                    from pathlib import Path
                    
                    model = kwargs.get('model')
                    if model and hasattr(model, 'merge_and_unload'):
                        output_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                        merged_dir = output_dir / "merged_model"
                        
                        try:
                            print(f"🔄 Auto-merging LoRA weights for checkpoint-{state.global_step}...")
                            merge_lora_weights_to_model(
                                peft_model=model,
                                base_model_name=self.base_model_name,
                                save_path=str(merged_dir)
                            )
                            print(f"✅ Merged model saved to: {merged_dir}")
                        except Exception as e:
                            print(f"❌ Error merging LoRA weights: {e}")
        
        lora_callback = LoRAMergeCallback(
            base_model_name=config.base_model_name,
            auto_merge=config.auto_merge_lora,
            keep_lora_only=getattr(config, 'keep_lora_only', True),
            merge_final_only=getattr(config, 'merge_final_only', False)
        )
        
        trainer.add_callback(lora_callback)
    
    # 开始训练
    print("Starting LoRA training...")
    trainer.train()
    
    # 保存最终模型
    print("Saving final LoRA model...")
    if config.use_lora:
        # 直接保存LoRA权重
        trainer.save_model()
        
        # 如果配置了自动合并，进行最终合并
        auto_merge = getattr(config, 'auto_merge_final', True)
        if auto_merge:
            from utils import merge_lora_weights_to_model
            from pathlib import Path
            
            merged_dir = Path(config.model_save_path) / "merged_model"
            try:
                print(f"🔄 Creating final merged model...")
                merge_lora_weights_to_model(
                    peft_model=model,
                    base_model_name=config.base_model_name,
                    save_path=str(merged_dir)
                )
                print(f"✅ Final merged model saved to: {merged_dir}")
            except Exception as e:
                print(f"❌ Error creating final merged model: {e}")
        
        # 额外保存tokenizer（确保在主目录）
        tokenizer.save_pretrained(config.model_save_path)
        
        print(f"✅ LoRA weights saved to: {config.model_save_path}")
    else:
        # 如果不使用LoRA，保存完整模型
        trainer.save_model()
        tokenizer.save_pretrained(config.model_save_path)
    
    # 保存配置
    with open(os.path.join(config.model_save_path, "config.yaml"), 'w') as f:
        yaml.dump(config.__dict__, f)
    
    print("LoRA training completed!")
    
    # 打印训练后的参数统计
    if config.use_lora:
        print("\nTrainable parameters after training:")
        model.print_trainable_parameters()
        
        # 打印保存的文件信息
        save_path = Path(config.model_save_path)
        lora_files = list(save_path.glob("adapter_*"))
        merged_path = save_path / "merged_model"
        
        print(f"\n📁 Saved files:")
        print(f"  LoRA weights: {len(lora_files)} files in {save_path}")
        if merged_path.exists():
            merged_files = list(merged_path.glob("*.safetensors")) + list(merged_path.glob("*.bin"))
            print(f"  Merged model: {len(merged_files)} files in {merged_path}")
            
            # 计算模型大小
            try:
                total_size = sum(f.stat().st_size for f in merged_files)
                print(f"  Merged model size: {total_size / (1024**3):.2f} GB")
            except:
                pass


def debug_model_gradients(model, name="Model"):
    """调试模型梯度状态"""
    print(f"\n🔍 {name} gradient status:")
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "point_encoder" in name:
                print(f"  ✅ {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        else:
            if "point_encoder" in name:
                print(f"  ❌ {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    print(f"  📊 Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    return trainable_params, total_params


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CAD-Recode LoRA Training Script v2.0")
    parser.add_argument("--config", type=str, required=True, help="Training configuration file")
    parser.add_argument("--base_model", type=str, help="Override base model name")
    parser.add_argument("--experiment_name", type=str, help="Override experiment name")
    parser.add_argument("--lora_r", type=int, help="Override LoRA rank")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit quantization)")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = LoRATrainingConfig(**config_dict)
    
    # 覆盖配置
    if args.base_model:
        config.base_model_name = args.base_model
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.use_qlora:
        config.use_qlora = True
        config.load_in_4bit = True
    
    print(f"LoRA Training Configuration:")
    print(f"  Base Model: {config.base_model_name}")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Use LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"  LoRA Rank: {config.lora_r}")
        print(f"  LoRA Alpha: {config.lora_alpha}")
        print(f"  LoRA Dropout: {config.lora_dropout}")
        print(f"  Target Modules: {config.lora_target_modules}")
        print(f"  Auto Merge: {config.auto_merge_lora}")
        print(f"  Auto Merge Final: {config.auto_merge_final}")
        print(f"  Merge Final Only: {config.merge_final_only}")
    print(f"  Use QLoRA: {config.use_qlora}")
    if config.use_qlora or config.load_in_4bit:
        print(f"  4-bit Quantization: {config.load_in_4bit}")
        print(f"  Compute Dtype: {config.bnb_4bit_compute_dtype}")
    
    # 开始训练
    train_model(config)


if __name__ == "__main__":
    main()
