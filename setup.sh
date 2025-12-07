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

# 0. Disable auto tmux (optional)
info "Disabling auto tmux..."
touch ~/.no_auto_tmux

# 1. Install system packages
info "Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential cmake libeigen3-dev libcrypt-dev graphviz \
    pkg-config aria2 git wget unzip curl

log "System dependencies installed."

# 2. Install Python (optional)
if [ ! -d "$HOME/anaconda3" ]; then
    info "Installing Anaconda Python..."
    wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh -O anaconda.sh
    bash anaconda.sh -b -p $HOME/anaconda3
    eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
    echo "source ~/anaconda3/bin/activate" >> ~/.bashrc
    log "Anaconda installed."
else
    warn "Anaconda already installed — skipping."
    eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
fi

# 3. Create Python venv
info "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
log "Virtual environment activated."

# 4. Install Python dependencies
info "Installing PyTorch + dependencies..."

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio torch_scatter

pip install opencv-python pycocotools matplotlib onnxruntime onnx gdown
pip install git+https://github.com/danganhdat/segment-anything.git

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    warn "requirements.txt not found — skipping."
fi

log "Python dependencies installed."

# 5. Fix libtiff issue (sometimes required)
info "Fixing libtiff.so.5 if missing..."
cd /usr/lib/x86_64-linux-gnu/
if [ ! -f "libtiff.so.5" ]; then
    sudo ln -s libtiff.so.6 libtiff.so.5
    log "libtiff.so.5 symlink created."
else
    warn "libtiff.so.5 already exists — skipping."
fi
cd -

# 6. Setup project folders
info "Preparing folders..."
mkdir -p pretrained data
log "Folders ready."

# 7. Download PASCAL VOC 2012
VOC_URL="https://web.archive.org/web/20250604190242/http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_TAR="VOC2012.tar"

info "Downloading PASCAL VOC 2012 dataset..."

if [ ! -f "$VOC_TAR" ]; then
    aria2c -x 16 -s 16 "$VOC_URL" -o "$VOC_TAR"
else
    warn "$VOC_TAR already exists — skipping download."
fi

if [ ! -d "data/VOC2012" ]; then
    info "Extracting VOC..."
    mkdir -p data
    tar -xf "$VOC_TAR" -C data/

    if [ -d "data/VOCdevkit/VOC2012" ]; then
        mv data/VOCdevkit/VOC2012 data/VOC2012
        rm -rf data/VOCdevkit
    fi
    log "VOC2012 ready."
else
    warn "data/VOC2012 already exists — skipping extract."
fi

# 8. Download pretrained weights
info "Downloading pretrained weights..."

# ResNet-38 (Google Drive)
RESNET_URL="https://drive.google.com/file/d/1fpb4vah3e-Ynx4cv5upUcqnpJFY_FTja/view"
if [ ! -f "pretrained/resnet_38d.params" ]; then
    gdown --fuzzy "$RESNET_URL" -O pretrained/resnet_38d.params
else
    warn "resnet_38d.params already exists."
fi

# SAM ViT-H
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
if [ ! -f "pretrained/sam_vit_h.pth" ]; then
    aria2c -x 16 -s 16 "$SAM_URL" -o pretrained/sam_vit_h.pth
else
    warn "sam_vit_h.pth already exists."
fi

log "All pretrained weights downloaded."

# DONE
log "FULL SETUP COMPLETE 🎉"
echo -e "${GREEN}Run your training or scripts now inside the venv.${NC}"
