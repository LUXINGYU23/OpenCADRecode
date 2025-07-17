# CAD-Recode 重构版本说明

## 代码结构

经过重构，原来的单文件训练脚本已经被分解为以下模块化组件：

### 1. 核心模块

- **`utils.py`**: 工具函数模块
  - `TrainingConfig`: 训练配置类
  - `setup_environment()`: 环境设置
  - `set_random_seeds()`: 随机种子设置
  - `load_data_index()`: 数据索引加载
  - `get_cached_point_cloud()`: 点云缓存读取
  - `apply_data_augmentation()`: 数据增强
  - `inference_single_sample()`: 单样本推理

- **`models.py`**: 模型定义模块
  - `FourierPointEncoder`: 傅里叶点云编码器
  - `CADRecode`: CAD-Recode多模态模型
  - `create_model_and_tokenizer()`: 模型和分词器创建

- **`datasets.py`**: 数据处理模块
  - `CADRecodeDataset`: CAD-Recode数据集类
  - `DataCollatorForCADRecode`: 数据整理器

### 2. 主要脚本

- **`train_cad_recode_full.py`**: 精简训练脚本
  - 仅包含训练流程逻辑
  - 导入和使用模块化组件
  - SwanLab集成和训练监控

- **`inference.py`**: 推理脚本
  - 加载训练好的模型
  - 单样本点云到代码生成
  - 支持自定义输出路径

## 使用方法

### 训练
```bash
python train_cad_recode_full.py --config configs/train_config_sft.yaml
```

### 推理
```bash
python batch_inference.py --model_path checkpoints_qwen3_sft --data_path data/val --output_dir inference_results
```

## 主要改进

1. **模块化设计**: 将原来的单文件拆分为功能明确的模块
2. **代码复用**: 工具函数可以在不同脚本间共享
3. **易于维护**: 每个模块职责单一，便于调试和修改
4. **扩展性**: 新功能可以轻松添加到对应模块
5. **清晰的依赖关系**: 模块间的依赖关系明确且合理

## 配置文件

训练配置仍然通过YAML文件管理，所有参数都在`TrainingConfig`类中定义，包括：
- 模型配置（基座模型、保存路径等）
- 数据配置（数据路径、序列长度等）
- 训练配置（学习率、批次大小等）
- 硬件配置（精度、设备等）
- 实验跟踪配置
