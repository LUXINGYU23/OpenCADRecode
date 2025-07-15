# CAD-Recode LoRA 训练使用指南

## 概述

CAD-Recode LoRA版本提供了内存友好的模型微调方案，支持在较小显存环境中训练大型多模态模型。

## 主要特性

- 🚀 **内存效率**: LoRA只训练少量参数(<1%)，显著降低显存需求
- ⚡ **训练速度**: 更快的训练和收敛速度
- 💾 **存储优化**: 只需保存LoRA权重(几MB vs 几GB)
- 🔄 **灵活性**: 可以轻松切换和合并不同的适配器
- 🎯 **QLoRA支持**: 支持4位量化，进一步降低显存需求

## 文件结构

```
├── train_cad_recode_lora.py      # LoRA训练脚本
├── inference_lora.py             # LoRA推理脚本  
├── merge_lora_weights.py         # LoRA权重合并脚本
├── configs/
│   ├── train_config_lora.yaml    # 标准LoRA配置
│   └── train_config_qlora.yaml   # QLoRA配置(低显存)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

确保安装了以下关键依赖：
- `peft>=0.7.0` (LoRA支持)
- `bitsandbytes>=0.41.0` (量化支持)
- `accelerate>=0.24.0` (训练加速)

### 2. 标准LoRA训练

适用于16GB+显存环境：

```bash
python train_cad_recode_lora.py --config configs/train_config_lora.yaml
```

### 3. QLoRA训练(低显存)

适用于12GB及以下显存环境：

```bash
python train_cad_recode_lora.py --config configs/train_config_qlora.yaml
```

或者使用命令行参数：

```bash
python train_cad_recode_lora.py --config configs/train_config_lora.yaml --use_qlora
```

### 4. LoRA推理

使用训练好的LoRA权重进行推理：

```bash
# 使用点云文件推理
python inference_lora.py \
    --base_model Qwen/Qwen3-1.7B-Base \
    --lora_path checkpoints_qwen3_lora \
    --point_cloud_file data/test/sample.npy

# 使用示例点云推理
python inference_lora.py \
    --base_model Qwen/Qwen3-1.7B-Base \
    --lora_path checkpoints_qwen3_lora \
    --shape cube \
    --output generated_code.py
```

### 5. 合并LoRA权重

训练完成后，会自动生成两种形式的模型：

#### 自动合并（推荐）

训练脚本会自动合并LoRA权重：

```bash
# 训练完成后会生成：
checkpoints_qwen3_lora/
├── adapter_config.json          # LoRA配置
├── adapter_model.safetensors    # LoRA权重
├── tokenizer_config.json        # 分词器配置
├── config.yaml                  # 训练配置
└── merged_model/                # 🎯 自动合并的完整模型
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── ...
```

#### 手动合并

如果需要手动合并LoRA权重：

```bash
python merge_lora_weights.py \
    --base_model Qwen/Qwen3-1.7B-Base \
    --lora_path checkpoints_qwen3_lora \
    --output_path merged_model
```

#### 推理时自动检测

推理脚本会自动检测并优先使用合并后的模型：

```bash
# 会自动使用 checkpoints_qwen3_lora/merged_model（如果存在）
python inference_lora.py \
    --base_model Qwen/Qwen3-1.7B-Base \
    --lora_path checkpoints_qwen3_lora \
    --shape cube
```

### 6. 模型结构

训练后的目录结构：

```
checkpoints_qwen3_lora/
├── 📁 LoRA权重文件
│   ├── adapter_config.json      # LoRA适配器配置
│   ├── adapter_model.safetensors # LoRA权重(通常几MB)
│   └── tokenizer_config.json    # 分词器配置
├── 📁 合并后的完整模型 (自动生成)
│   └── merged_model/
│       ├── config.json          # 模型配置
│       ├── model.safetensors    # 完整模型权重(几GB)
│       ├── tokenizer_config.json
│       └── generation_config.json
└── 📄 训练配置
    └── config.yaml              # 训练时的配置参数
```

**优势：**
- 🔄 **灵活性**: 保留LoRA权重便于后续微调
- ⚡ **推理速度**: 合并模型无需动态加载适配器
- 💾 **存储优化**: LoRA权重文件小，便于分发
- 🎯 **自动化**: 无需手动操作，训练完成即可使用

## 配置说明

### LoRA参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_r` | 16 | LoRA秩，控制参数量和性能平衡 |
| `lora_alpha` | 32 | LoRA缩放因子，通常设为2*lora_r |
| `lora_dropout` | 0.1 | LoRA dropout率 |
| `lora_target_modules` | [q_proj, k_proj, v_proj, ...] | 目标模块列表 |

### 自动合并配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `auto_merge_lora` | true | 是否在训练过程中自动合并权重 |
| `auto_merge_final` | true | 是否在最终保存时合并权重 |
| `keep_lora_only` | true | 训练过程中是否保留LoRA权重文件 |
| `keep_lora_final` | true | 最终保存是否保留LoRA权重文件 |
| `merge_final_only` | false | 是否只在最终保存时合并（节省训练时间） |

### QLoRA参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_qlora` | false | 是否启用QLoRA |
| `load_in_4bit` | false | 4位量化 |
| `bnb_4bit_compute_dtype` | "bfloat16" | 计算数据类型 |
| `bnb_4bit_quant_type` | "nf4" | 量化类型 |

## 显存需求估算

| 配置 | 模型大小 | 显存需求 | 训练参数量 |
|------|----------|----------|------------|
| Qwen3-1.7B + LoRA | 1.7B | ~8GB | ~0.8M (0.05%) |
| Qwen3-1.7B + QLoRA | 1.7B | ~6GB | ~0.8M (0.05%) |
| Qwen2-7B + LoRA | 7B | ~16GB | ~3.1M (0.04%) |
| Qwen2-7B + QLoRA | 7B | ~12GB | ~3.1M (0.04%) |

## 最佳实践

### 1. 选择合适的配置

- **显存 ≥ 16GB**: 使用标准LoRA配置
- **显存 < 16GB**: 使用QLoRA配置
- **显存 < 12GB**: 考虑使用更小的模型或降低batch size

### 2. LoRA参数调优

- **高性能**: `r=64, alpha=128`
- **平衡**: `r=16, alpha=32` (推荐)
- **低参数**: `r=8, alpha=16`

### 3. 训练策略

- **学习率**: LoRA通常使用稍高的学习率(2e-4 ~ 5e-4)
- **Batch Size**: LoRA可以使用更大的batch size
- **训练步数**: LoRA收敛更快，可以减少训练步数

### 4. 模型选择

- **快速实验**: Qwen3-1.7B
- **更好性能**: Qwen2-7B (需要更多显存)
- **平衡方案**: Qwen2.5-3B

## 故障排除

### 1. 显存不足

```bash
# 降低batch size
per_device_train_batch_size: 4

# 增加梯度累积
gradient_accumulation_steps: 8

# 启用QLoRA
use_qlora: true
```

### 2. 训练不收敛

```bash
# 增加LoRA秩
lora_r: 32
lora_alpha: 64

# 调整学习率
learning_rate: 0.0001

# 增加warmup步数
warmup_steps: 2000
```

### 3. 推理速度慢

```bash
# 合并LoRA权重
python merge_lora_weights.py --base_model ... --lora_path ... --output_path merged_model

# 使用合并后的模型进行推理
```

## 高级用法

### 1. 多LoRA适配器

可以为不同任务训练多个LoRA适配器：

```python
# 加载不同的LoRA适配器
model.load_adapter("task1_lora", adapter_name="task1")
model.load_adapter("task2_lora", adapter_name="task2")

# 切换适配器
model.set_adapter("task1")
```

### 2. 增量训练

在已有LoRA基础上继续训练：

```bash
python train_cad_recode_lora.py \
    --config configs/train_config_lora.yaml \
    --resume_from_checkpoint checkpoints_qwen3_lora/checkpoint-10000
```

### 3. 模型评估

```bash
# 在验证集上评估
python train_cad_recode_lora.py \
    --config configs/train_config_lora.yaml \
    --do_eval \
    --eval_steps 1000
```

## 参考资源

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [PEFT库文档](https://huggingface.co/docs/peft)
- [BitsAndBytes文档](https://github.com/TimDettmers/bitsandbytes)
