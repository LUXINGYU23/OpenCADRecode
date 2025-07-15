#!/bin/bash

# CAD-Recode LoRA从Full模型微调启动脚本
# 使用已训练的full模型作为起点进行LoRA微调

echo "==========================================="
echo "CAD-Recode LoRA从Full模型微调"
echo "==========================================="

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
fi

# 检查必要的文件和目录
echo "🔍 检查训练环境..."

# 检查已训练的full模型
FULL_MODEL_PATH="checkpoints_qwen3_sft"
if [ ! -d "$FULL_MODEL_PATH" ]; then
    echo "❌ 未找到已训练的full模型: $FULL_MODEL_PATH"
    echo "请先完成full模型训练"
    exit 1
fi

# 检查数据目录
if [ ! -d "data/train" ] || [ ! -d "data/val" ]; then
    echo "❌ 数据目录不完整，请检查 data/train 和 data/val"
    exit 1
fi

# 检查配置文件
CONFIG_FILE="configs/train_config_lora_from_full.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "✅ 环境检查完成"
echo ""

# 显示配置信息
echo "🔧 训练配置:"
echo "  Full模型路径: $FULL_MODEL_PATH"
echo "  配置文件: $CONFIG_FILE"
echo "  输出目录: checkpoints_qwen3_lora_from_full"
echo ""

# 询问用户确认
read -p "是否开始LoRA微调? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

# 设置Python路径和环境变量
export PYTHONPATH="$PWD:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# 启动训练
echo "🚀 开始LoRA微调训练..."
echo "详细日志将保存到 lora_from_full_training.log"
echo ""

# 运行训练脚本并保存日志
python train_cad_recode_lora.py \
    --config "$CONFIG_FILE" \
    --experiment_name "cad-recode-lora-from-full-$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee lora_from_full_training.log

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 LoRA微调训练完成!"
    echo ""
    
    # 检查输出目录
    OUTPUT_DIR="checkpoints_qwen3_lora_from_full"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 训练输出目录: $OUTPUT_DIR"
        echo "   文件结构:"
        ls -la "$OUTPUT_DIR"
        
        # 检查合并模型
        if [ -d "$OUTPUT_DIR/merged_model" ]; then
            echo ""
            echo "✅ 找到合并后的模型: $OUTPUT_DIR/merged_model"
            echo "   可以直接使用此模型进行推理"
        fi
        
        echo ""
        echo "🧪 运行推理测试:"
        echo "python inference_lora_from_full.py \\"
        echo "    --model_path $OUTPUT_DIR \\"
        echo "    --data_root data \\"
        echo "    --output_dir inference_results_lora_from_full \\"
        echo "    --max_samples 5 \\"
        echo "    --use_merged"
    fi
else
    echo ""
    echo "❌ 训练过程中出现错误"
    echo "请检查日志文件: lora_from_full_training.log"
fi
