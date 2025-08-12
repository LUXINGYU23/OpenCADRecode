<!-- Auto-generated English version of the Chinese README. Keep sections aligned. -->
# OpenCADRecode (formerly CAD-Recode v2.0)

> Note: The Chinese and English README documents were auto-generated / translated by GPT5 and may contain mistakes or omissions. In case of any discrepancy, the source code and configuration files are the source of truth.

> Dataset origin: The original CAD-Recode (`.py`) code dataset is NOT bundled here. Please download from the official release:  
> Hugging Face dataset: https://huggingface.co/datasets/filapro/cad-recode-v1.5  
> Upstream repository: https://github.com/filaPro/cad-recode  
> This repo is an open-source reproduction & extension (training, LoRA / QLoRA, multi-modal, benchmarking) and is NOT the official one; defer to upstream if inconsistent.

> End-to-end open pipeline for reverse engineering parametric CAD (CadQuery Python code) from point clouds / STEP: data prep → point cloud caching → (SFT / LoRA / QLoRA / Multi-modal) training → batch inference → geometric benchmarking (STEP / STL) → report.

<p align="center"><b>Supports official models (filapro/cad-recode, cad-recode-v1.5) and your own SFT / LoRA adapters side-by-side.</b></p>

---
## Contents
1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
   - [2.1 Original CAD-Recode Code Dataset](#21-original-cad-recode-code-dataset)
   - [2.2 Fusion360 Dataset (STEP)](#22-fusion360-dataset-step)
   - [2.3 Point Cloud Generation & Cache](#23-point-cloud-generation--cache)
3. [Training](#3-training)
   - [3.1 Full SFT](#31-full-sft)
   - [3.2 LoRA / QLoRA](#32-lora--qlora)
   - [3.3 Multi-modal (BRep + Point Cloud)](#33-multi-modal-brep--point-cloud)
   - [3.4 Key Config Parameters](#34-key-config-parameters)
4. [Inference & Batch Generation](#4-inference--batch-generation)
5. [Benchmark Pipeline](#5-benchmark-pipeline)
   - [5.1 Metrics](#51-metrics)
   - [5.2 Report Layout](#52-report-layout)
6. [Example Benchmark Result](#6-example-benchmark-result)
7. [Model / Checkpoint Layout](#7-model--checkpoint-layout)
8. [Quick Reproduction Guide](#8-quick-reproduction-guide)
9. [Citation](#9-citation)

---
## 1. Environment Setup
Python 3.10+. GPU recommended (A100 / 4090 / V100). QLoRA allows low VRAM.
```bash
git clone https://github.com/LUXINGYU23/OpenCADRecode.git
cd OpenCADRecode
bash set_up.sh
```
We set `HF_ENDPOINT=https://hf-mirror.com` internally for faster downloads (China friendly).

---
## 2. Data Preparation
### 2.1 Original CAD-Recode Code Dataset
Each sample = one CadQuery Python file. Point clouds are not stored in dataset; they are looked up in cache on-the-fly.

> Dataset origin: The original CAD-Recode (`.py`) code dataset is NOT distributed with this repo. Please download from the official release:  
> Hugging Face dataset: https://huggingface.co/datasets/filapro/cad-recode-v1.5  
> Upstream official repository: https://github.com/filaPro/cad-recode  
> This project is an open-source reproduction & extension (training scripts, LoRA / QLoRA, multi-modal baseline, benchmarking pipeline) and is NOT the official repo; if any discrepancy arises, defer to the upstream data & models.
```
data/
  train/batch_0000/*.py
  train/batch_0001/*.py
  val/*.py or val/batch_*/.py
  dataset_info.json
  error_samples.json (optional exclusion list)
```
Optional indices: `train_index.json`, `val_index.json`, or `index.json`. Fallback: directory scan.

**Index file generation**  
```bash
python generate_indices.py --data-root data --splits train val
```
Outputs: `data/train_index.json`, `data/val_index.json`, plus merged `data/index.json`. Overwrite with:
```bash
python generate_indices.py --data-root data --splits train val --overwrite
```
Include a test split:
```bash
python generate_indices.py --data-root data --splits train val test
```
Script behavior:
- Scans both `batch_*` subfolders and top-level `.py` files
- Skips samples listed in each split's `error_samples.json`
- `sample_id` rule: `batch_XXXX_filename` or just `filename` (top-level)
- Merged `index.json` provides a unified list for consumers


### 2.2 Fusion360 Dataset (STEP)
```
fusion360dataset/
  reconstruction/*.step
  train_test.json   # {"train": [...], "test": [...]} for splits
```
Benchmark example:
```bash
python eval/benchmark.py \
  --models official checkpoints_qwen3_sft \
  --dataset_type fusion360 --split test \
  --data_path fusion360dataset --train_test_json fusion360dataset/train_test.json \
  --num_samples 100
```
Pipeline: load STEP → sample point cloud (cache first) → generate code → CadQuery execution → STL/STEP → metric vs GT.

### 2.3 Point Cloud Generation & Cache
Order in `get_cached_point_cloud`:
1. `data/<split>/point_cloud_cache/{id}.npy`
2. `$OPENCADRECODE_CACHE_ROOT/point_cloud_cache/{id}.npy`
3. `/root/opencadrecode/point_cloud_cache/{id}.npy`
4. Legacy: `/root/cad-recode/point_cloud_cache/{id}.npy`
Shape: `(256, 3)` normalized [-1,1].
Generation:
- From code: run CadQuery → mesh → sample (FPS) → normalize → save .npy
- From Fusion360 STEP: automatic (STEP → STL → FPS) in `eval/benchmark.py`.

---
## 3. Training
### 3.1 Full SFT
```bash
python train/train_cad_recode.py --config configs/train_config_sft.yaml
```
### 3.2 LoRA / QLoRA
```bash
python train/train_cad_recode_lora.py \
  --config configs/train_config_lora.yaml \
  --experiment_name lora-qwen3-1p7b-exp1

python train/train_cad_recode_lora.py \
  --config configs/train_config_lora.yaml --use_qlora
```
Features: auto merge (merged_model), adapter retention, resume from full model.
### 3.3 Multi-modal
```bash
python train/train_cad_recode_multimodal.py --config configs/train_config_sft.yaml
```
### 3.4 Key Config
`base_model_name`, `max_seq_length`, LoRA parameters, quantization flags, augmentation, auto merge flags.

---
## 4. Inference & Batch Generation
```bash
python eval/batch_inference.py \
  --model_path checkpoints_qwen3_lora/merged_model \
  --data_path data/val --output_dir inference_results
```
LoRA adapter:
```bash
python eval/inference_lora.py \
  --base_model Qwen/Qwen3-1.7B-Base \
  --lora_path checkpoints_qwen3_lora
```
Outputs: generated_code / generated_stl / generated_step.

---
## 5. Benchmark Pipeline
Script: `eval/benchmark.py`
### 5.1 Metrics
Chamfer, Hausdorff, EMD, RMS, IoU (brep / voxel), Matching Rate, Coverage Rate, Normal Consistency, computation_time.
### 5.2 Layout
```
benchmark_results/
  results/<model_name>/
  gt_step_files/
  benchmark_report.json / .md
```

---
## 6. Example Benchmark Result (Fusion360 testset)
| Model | Type | Success Rate | STL Success | Chamfer↓ | IoU↑ | Matching Rate↑ |
|-------|------|-------------|-------------|----------|------|----------------|
| official | official | 80.0% | 80.0% | 0.0181 | 0.8793 | 0.8220 |
| Ours | sft | 70.0% | 70.0% | 0.0207 | 0.8700 | 0.8234 |

---
## 7. Model / Checkpoint Layout
```
checkpoints_qwen3_sft/
checkpoints_qwen3_sft/checkpoint-100000/
checkpoints_qwen3_lora/
checkpoints_qwen3_lora/merged_model/
cad-recode-v1.5/
```

---
## 8. Quick Reproduction Guide
```bash
pip install -r requirements.txt
python train/train_cad_recode_lora.py --config configs/train_config_lora.yaml
python eval/batch_inference.py --model_path checkpoints_qwen3_lora/merged_model --data_path data/val
python eval/benchmark.py --models official checkpoints_qwen3_lora --data_path data/val --num_samples 100
```
---
Issues / PRs welcome. If useful, please ⭐.
