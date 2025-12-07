#!/bin/bash
set -e

# COLORS
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
NC="\033[0m"

log() { echo -e "${GREEN}✔ $1${NC}"; }
info() { echo -e "${CYAN}→ $1${NC}"; }
warn() { echo -e "${YELLOW}! $1${NC}"; }

info "STARTING S2C SETUP..."

# 0. Disable auto tmux (optional)
touch ~/.no_auto_tmux

# 1. System dependencies
info "Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential cmake pkg-config \
    libeigen3-dev libcrypt-dev graphviz \
    aria2 git wget unzip curl

log "System dependencies installed."

# 2. Conda environment (Python 3.8)
info "Setting up conda environment..."

eval "$(conda shell.bash hook)"

if conda env list | grep -w s2c >/dev/null; then
    warn "Conda env 's2c' already exists — reusing it."
else
    conda create -n s2c python=3.8 -y
fi

conda activate s2c
log "Activated conda env: s2c"

# 3. REQUIRED: PyTorch 1.8.2 + CUDA 11.1 (compatible w/ S2C)
info "Installing PyTorch 1.8.2 + CUDA 11.1..."

pip install --upgrade pip setuptools wheel

pip install \
  torch==1.8.2+cu111 \
  torchvision==0.9.2+cu111 \
  torchaudio==0.8.2 \
  -f https://download.pytorch.org/whl/torch_stable.html

log "PyTorch installed."

# 4. torch-scatter (correct wheel!)
info "Installing torch-scatter wheel..."

pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-1.8.0+cu111.html

log "torch-scatter installed."

# 5. Python dependencies
info "Installing Python dependencies..."

pip install \
    opencv-python pycocotools matplotlib \
    onnxruntime onnx gdown \
    ftfy regex tqdm pillow imageio \
    scikit-image scikit-learn pandas \
    protobuf timm graphviz

# PydenseCRF requires eigen + cython (already installed)
pip install pydensecrf

# Install SAM
# pip install "git+https://github.com/facebookresearch/segment-anything.git"

# Optional: install your own forked SAM patch
pip install "git+https://github.com/danganhdat/segment-anything.git"

log "Python packages installed."

# 6. libtiff fix
info "Checking libtiff..."

cd /usr/lib/x86_64-linux-gnu/
if [ ! -f "libtiff.so.5" ]; then
    sudo ln -s libtiff.so.6 libtiff.so.5
    log "libtiff.so.5 symlink created."
else
    warn "libtiff.so.5 already exists."
fi
cd -

# 7. Project folders
mkdir -p pretrained data
log "Folders prepared."

# 8. Download VOC dataset
VOC_URL="https://web.archive.org/web/20250604190242/http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_TAR="VOC2012.tar"

info "Downloading PASCAL VOC 2012..."

if [ ! -f "$VOC_TAR" ]; then
    aria2c -x 16 -s 16 "$VOC_URL" -o "$VOC_TAR"
else
    warn "$VOC_TAR already exists — skipping download."
fi

if [ ! -d "data/VOC2012" ]; then
    tar -xf "$VOC_TAR" -C data/
    if [ -d "data/VOCdevkit/VOC2012" ]; then
        mv data/VOCdevkit/VOC2012 data/VOC2012
        rm -rf data/VOCdevkit
    fi
    log "VOC2012 extracted."
else
    warn "VOC2012 already exists."
fi

# 9. Download pretrained weights
info "Downloading pretrained weights..."

# ResNet-38
if [ ! -f "pretrained/resnet_38d.params" ]; then
    gdown --fuzzy "https://drive.google.com/file/d/1fpb4vah3e-Ynx4cv5upUcqnpJFY_FTja/view" \
        -O pretrained/resnet_38d.params
else
    warn "resnet_38d.params already exists."
fi

# SAM ViT-H
if [ ! -f "pretrained/sam_vit_h.pth" ]; then
    aria2c -x 16 -s 16 \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
        -o pretrained/sam_vit_h.pth
else
    warn "sam_vit_h.pth already exists."
fi

log "All pretrained weights downloaded."

# ============================================================
log "🎉 FULL S2C SETUP COMPLETE!"
echo -e "${GREEN}You can now run training or scripts inside the 's2c' conda env.${NC}"