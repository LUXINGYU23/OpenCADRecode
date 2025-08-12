# OpenCADRecode 

[English Version](README_en.md)

> 说明：本中英文 README 文档由 GPT5 自动生成 / 翻译，可能存在遗漏或偏差，请以实际代码与配置文件为准。

> 数据来源说明：原始 CAD-Recode (`.py`) 代码数据集不随本仓库分发，请前往官方发布地址下载：  
> Hugging Face 数据集: https://huggingface.co/datasets/filapro/cad-recode-v1.5  
> 官方上游项目仓库: https://github.com/filaPro/cad-recode  
> 本仓库为对 CAD-Recode 的开源复现与扩展项目（训练 / LoRA & QLoRA / 多模态 / 评测流水线），非官方仓库；如有冲突请以官方为准。

> 从点云/STEP 逆向生成可执行 CadQuery Python CAD 程序的多模态代码生成与重建基准工程。包含：数据准备 → 点云缓存 → （SFT / LoRA / QLoRA / 多模态）训练 → 批量推理 → 几何指标评估 (STEP / STL) → Benchmark 报告全流程。

<h4 align="center">支持官方模型 (filapro/cad-recode / cad-recode-v1.5) 与自训练/LoRA 适配器模型基准对比</h4>

---
## 目录
- [1. 环境安装](#1-环境安装)
- [2. 数据准备](#2-数据准备)
  - [2.1 CAD-Recode 原始代码数据集结构](#21-cad-recode-原始代码数据集结构)
  - [2.2 Fusion360 数据集 (STEP)](#22-fusion360-数据集-step)
  - [2.3 点云生成与缓存](#23-点云生成与缓存)
- [3. 训练](#3-训练)
  - [3.1 SFT 全参数训练](#31-sft-全参数训练)
  - [3.2 LoRA / QLoRA 微调](#32-lora--qlora-微调)
  - [3.3 多模态 (BRep + Point Cloud) 训练](#33-多模态-brep--point-cloud-训练)
  - [3.4 常见配置说明](#34-常见配置说明)
- [4. 推理 & 批量生成](#4-推理--批量生成)
- [5. Benchmark 评估流程](#5-benchmark-评估流程)
  - [5.1 指标说明](#51-指标说明)
  - [5.2 报告结构](#52-报告结构)
- [6. 当前 Benchmark 示例结果](#6-当前-benchmark-示例结果)
- [7. 模型/权重组织](#7-模型权重组织)

---
## 1. 环境安装
推荐使用 Python 3.10+，确保 GPU (A100 / 4090 / V100) 或至少 16GB 显存；QLoRA 可在更低显存运行。

```bash
# 克隆仓库
git clone https://github.com/LUXINGYU23/OpenCADRecode.git
cd OpenCADRecode
bash set_up.sh
```
> 国内可设置 HF_ENDPOINT / 镜像源：脚本内部已设置 `HF_ENDPOINT=https://hf-mirror.com`。

---
## 2. 数据准备
### 2.1 CAD-Recode 原始代码数据集结构
数据以 CadQuery Python 代码形式提供，每个样本一个 `.py`；点云不直接存储，训练 & 推理阶段从缓存读取。

> 数据来源说明：原始 CAD-Recode (`.py`) 代码数据集不随本仓库分发，请前往官方发布地址下载：
> Hugging Face 数据集: https://huggingface.co/datasets/filapro/cad-recode-v1.5  
> 官方上游项目仓库: https://github.com/filaPro/cad-recode  
> 本仓库是对 CAD-Recode 的开源复现与扩展（提供训练脚本 / LoRA & QLoRA / 多模态基线 / 评测流水线等），并非官方仓库；若本文档与官方存在差异，请以官方数据与模型为准。

```
data/
  train/
    batch_0000/*.py
    batch_0001/*.py
    ...
  val/
    *.py  或 batch_*/*.py
```
可选索引文件：`val_index.json / train_index.json / index.json`。若缺失会自动扫描目录。

**索引文件生成**  
```bash
python generate_indices.py --data-root data --splits train val
```
生成结果：`data/train_index.json`、`data/val_index.json` 与合并的 `data/index.json`。若需要重新生成（覆盖已有文件）添加 `--overwrite`：
```bash
python generate_indices.py --data-root data --splits train val --overwrite
```
包含 test 集时：
```bash
python generate_indices.py --data-root data --splits train val test
```
脚本逻辑：
- 扫描 `batch_*` 目录与顶层 `.py` 文件
- 自动跳过各 split 目录下的 `error_samples.json` 中列出的样本
- `sample_id` 规则：`batch_XXXX_filename` 或顶层 `filename`
- 生成的合并 `index.json` 可被部分脚本统一读取


### 2.2 Fusion360 数据集 (STEP)
将 Fusion360 Gallery reconstruction STEP 文件放入：
```
fusion360dataset/
  reconstruction/*.step
  train_test.json  # {"train": [...], "test": [...]} 用于 split 划分
```
Benchmark 时通过 `--dataset_type fusion360 --split test --data_path fusion360dataset` 调用；脚本会：
1. 读取 STEP → 采样点云 (优先缓存)  
2. 生成预测代码 → CadQuery 执行输出 STL / STEP  
3. 与 GT STEP 做几何指标对比。

### 2.3 点云生成与缓存
训练 / 推理统一读取缓存点云：`get_cached_point_cloud(data_root, sample_id, num_points)`：
检索顺序：
1. `data/<split>/point_cloud_cache/{sample_id}.npy`
2. `/root/opencadrecode/point_cloud_cache/{sample_id}.npy` （兼容旧：`/root/cad-recode/point_cloud_cache/`）

格式：`(NUM_POINT_TOKENS=256, 3)`；需归一化到 [-1,1]。

生成策略：
- CAD-Recode 原始代码：离线脚本（你可编写批处理，执行每个 `.py` 用 CadQuery 生成 STL → 采样点云 → 保存 .npy）。
- Fusion360：`eval/benchmark.py` 在 `_get_point_cloud_from_step` 中：STEP → (OCC) → STL → FPS 采样 256 点 → 归一化 → 缓存到 `point_cloud_cache_fusion360/`。

> 若无缓存会警告并跳过；建议预先批量生成减少 I/O。

示例（自建离线点云缓存伪代码）：
```python
import cadquery as cq, numpy as np, pathlib
from utils.utils import NUM_POINT_TOKENS
# 遍历代码 -> eval 生成 shape -> 采样点 -> 归一化 -> 保存 .npy
```

---
## 3. 训练
配置文件位于 `configs/`。

### 3.1 SFT 全参数训练
脚本：`train/train_cad_recode.py` (或 `train_cad_recode_full.py`)。示例：
```bash
python train/train_cad_recode.py \
  --config configs/train_config_sft.yaml
```

### 3.2 LoRA / QLoRA 微调
脚本：`train/train_cad_recode_lora.py`
```bash
python train/train_cad_recode_lora.py \
  --config configs/train_config_lora.yaml \
  --experiment_name lora-qwen3-1p7b-exp1

# 启用 QLoRA (4bit)
python train/train_cad_recode_lora.py \
  --config configs/train_config_lora.yaml --use_qlora
```
特性：
- 自动合并权重：保存目录 `checkpoints_qwen3_lora/merged_model/` 输出合并后完整模型 (`model.safetensors` 等)。
- 保留适配器：`adapter_model.safetensors` 与配置文件。
- 支持从已全参训练模型继续 LoRA (`load_from_full_model=true`)。


### 3.4 常见配置说明
核心参数 (以 LoRA 配置为例)：
- `base_model_name`: Qwen/Qwen3-1.7B-Base (可换 Qwen2/Qwen3 其它尺寸)
- `max_seq_length`: 1024 = 256 点 token + 768 代码 token + 特殊 token
- `lora_r / lora_alpha / lora_dropout`: 结构秩、缩放、正则
- `use_qlora + load_in_4bit`: 低显存微调
- `noise_probability / noise_std`: 点云数据增强
- `auto_merge_lora / auto_merge_final`: 训练中 & 结束后自动合并

---
## 4. 推理 & 批量生成
单样本推理核心函数：`utils.utils.inference_single_sample`，遵循官方 prompt 结构：
```
[PAD]*256 + <|im_start|> + 生成CadQuery代码
```
批量推理：
```bash
python eval/batch_inference.py \
  --model_path checkpoints_qwen3_lora/merged_model \
  --data_path data/val --output_dir inference_results
```
LoRA 适配器推理示例：
```bash
python eval/inference_lora.py \
  --base_model Qwen/Qwen3-1.7B-Base \
  --lora_path checkpoints_qwen3_lora
```
输出包含：`generated_code/*.py`、可选 `generated_stl/*.stl`、`generated_step/*.step`。

---
## 5. Benchmark 评估流程
脚本：`eval/benchmark.py`

### 5.1 基本用法
Legacy 验证集：
```bash
python eval/benchmark.py \
  --models official checkpoints_qwen3_sft \
  --data_path data/val --num_samples 200
```
Fusion360 测试集：
```bash
python eval/benchmark.py \
  --models official checkpoints_qwen3_sft \
  --dataset_type fusion360 --split test \
  --data_path fusion360dataset --train_test_json fusion360dataset/train_test.json \
  --num_samples 100
```
参数说明：
- `--models`: 传入多个模型目录或关键字 `official`
- 自动检测类型：`official | sft | lora | lora_from_full`
- LoRA 模型需提供 base model (`model_config.base_model` 自动推断或修改源码)
- 生成结构：
```
benchmark_results/
  results/<model_name>/generated_code
  results/<model_name>/generated_stl
  results/<model_name>/generated_step
  gt_step_files/  # GT 生成或复制
  benchmark_report.json / .md
```

### 5.2 指标说明
在共同成功生成 STEP 的样本交集上计算：
- 几何距离：Chamfer, Hausdorff, Earth Mover (EMD), RMS
- 拓扑/覆盖：IoU (brep / voxel), Matching Rate, Coverage Rate
- 法向一致：Normal Consistency
- 性能：Computation Time

指标计算流程：GT STEP ↔ Pred STEP → 采样点集 (含 ICP 对齐可选) → 批量统计 → 平均 / 方差 / 中位数。

---
## 6. 当前 Benchmark 示例结果
示例（Fusion360 测试集）：
| Model | Type | Success Rate | STL Success | Chamfer↓ | IoU↑ | Matching Rate↑ |
|-------|------|-------------|-------------|----------|------|----------------|
| official | official | 80.0% | 80.0% | 0.0181 | 0.8793 | 0.8220 |
| Ours | sft | 70.0% | 70.0% | 0.0207 | 0.8700 | 0.8234 |

---
## 7. 模型/权重组织
```
checkpoints_qwen3_sft/          # 全参数SFT
checkpoints_qwen3_sft/checkpoint-100000/
checkpoints_qwen3_lora/         # LoRA 适配器
checkpoints_qwen3_lora/merged_model/  # 合并后权重
cad-recode-v1.5/                # 官方v1.5 (外部发布模型保留原名)
```
推理时：
- 使用合并模型：直接传 `.../merged_model`
- 使用 LoRA 适配器：加载 base + adapter (`eval/inference_lora.py`)

---
欢迎 Issue / PR 反馈改进。若本项目对你有帮助，请⭐支持。
