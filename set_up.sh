#!/bin/bash
# CAD-Recode Environment Setup Script
# 使用conda创建虚拟环境并安装所有依赖

set -e  # 遇到错误时立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
ENV_NAME="cad-recode"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.0.0"
CUDA_VERSION="11.8"  # 根据您的CUDA版本调整

echo -e "${BLUE}===== CAD-Recode Environment Setup =====${NC}"
echo -e "${YELLOW}Environment Name: ${ENV_NAME}${NC}"
echo -e "${YELLOW}Python Version: ${PYTHON_VERSION}${NC}"
echo -e "${YELLOW}PyTorch Version: ${PYTORCH_VERSION}${NC}"
echo -e "${YELLOW}CUDA Version: ${CUDA_VERSION}${NC}"
echo ""

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

# 检查是否已存在同名环境
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment ${ENV_NAME} already exists.${NC}"
    read -p "Do you want to remove it and create a new one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME}
    else
        echo -e "${BLUE}Using existing environment.${NC}"
        conda activate ${ENV_NAME}
        echo -e "${GREEN}Environment ${ENV_NAME} activated.${NC}"
        exit 0
    fi
fi

# 创建conda环境
echo -e "${BLUE}Creating conda environment: ${ENV_NAME}${NC}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# 激活环境
echo -e "${BLUE}Activating environment: ${ENV_NAME}${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 更新conda和pip
echo -e "${BLUE}Updating conda and pip...${NC}"
conda update -n base -c defaults conda -y
pip install --upgrade pip

# 安装PyTorch相关 (根据CUDA版本选择)
echo -e "${BLUE}Installing PyTorch and related packages...${NC}"
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # CPU版本
    pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装PyTorch3D (需要特殊处理)
echo -e "${BLUE}Installing PyTorch3D...${NC}"
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html

# 安装Transformers和相关ML包
echo -e "${BLUE}Installing Transformers and ML packages...${NC}"
pip install transformers>=4.40.0
pip install datasets
pip install trl  # 用于SFT训练
pip install accelerate  # 用于分布式训练
pip install bitsandbytes  # 用于量化

# 安装点云和3D处理包
echo -e "${BLUE}Installing 3D processing packages...${NC}"
pip install trimesh
pip install open3d
pip install scipy
pip install scikit-image

# 安装CAD相关包
echo -e "${BLUE}Installing CAD-related packages...${NC}"
pip install cadquery

# 尝试安装OpenCascade (可选)
echo -e "${BLUE}Installing OpenCascade (optional)...${NC}"
conda install -c conda-forge pythonocc-core -y || echo -e "${YELLOW}Warning: Failed to install pythonocc-core${NC}"

# 安装配置和工具包
echo -e "${BLUE}Installing utility packages...${NC}"
pip install pyyaml
pip install tqdm
pip install numpy
pip install matplotlib

# 安装开发和调试工具
echo -e "${BLUE}Installing development tools...${NC}"
pip install jupyter
pip install ipython
pip install swanlab  # 用于实验跟踪

# 安装其他有用的包
echo -e "${BLUE}Installing additional useful packages...${NC}"
pip install wandb  # 另一个实验跟踪工具
pip install tensorboard  # 可视化工具
pip install rich  # 美化终端输出

# 验证安装
echo -e "${BLUE}Verifying installation...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import trimesh; print('Trimesh: OK')"
python -c "import open3d; print('Open3D: OK')"
python -c "import cadquery; print('CadQuery: OK')"

# 创建快速激活脚本
cat > activate_cad_recode.sh << EOF
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
echo -e "${GREEN}CAD-Recode environment activated!${NC}"
echo "You can now run:"
echo "  python train_cad_recode_full.py --config configs/train_config_sft.yaml"
EOF

chmod +x activate_cad_recode.sh

echo -e "${GREEN}===== Setup Complete! =====${NC}"
echo ""
echo -e "${YELLOW}To activate the environment in the future, run:${NC}"
echo -e "${BLUE}  conda activate ${ENV_NAME}${NC}"
echo -e "${YELLOW}Or use the convenience script:${NC}"
echo -e "${BLUE}  ./activate_cad_recode.sh${NC}"
echo ""
echo -e "${YELLOW}To test the installation:${NC}"
echo -e "${BLUE}  python -c \"import torch, transformers, cadquery; print('All imports successful!')\"${NC}"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
