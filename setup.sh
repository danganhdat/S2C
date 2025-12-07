#!/bin/bash
set -e

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
NC="\033[0m"

log()  { echo -e "${GREEN}✔ $1${NC}"; }
info() { echo -e "${CYAN}→ $1${NC}"; }
warn() { echo -e "${YELLOW}! $1${NC}"; }

info "STARTING S2C SETUP (CUDA 12.8.1, PyTorch 2.8.0)..."

# 1. System deps
info "Installing system packages..."
sudo apt update -y
sudo apt install -y \
    build-essential cmake pkg-config \
    libeigen3-dev libcrypt-dev graphviz \
    aria2 git wget unzip curl

log "System deps installed."

# ------------------------------------------------------------
# NOTE: You will activate env manually, so we DO NOT run:
# eval "$(conda ...)"
# conda activate ...
# ------------------------------------------------------------

warn "Make sure your conda env is already ACTIVATED before running this!"
sleep 1

# 2. pip upgrade
info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch 2.8.0 (CUDA 12.8)
info "Installing PyTorch 2.8.0 + CUDA 12.8..."
pip install torch==2.8.0+cu128 torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cu128

log "PyTorch 2.8.0 installed."

# 4. torch-scatter (MUST build from source)
info "Building torch-scatter from source for torch 2.8.0 + cu128..."

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

log "torch-scatter installed (compiled)."

# 5. Python deps
info "Installing Python dependencies..."

pip install \
    opencv-python pycocotools matplotlib \
    onnxruntime onnx gdown \
    ftfy regex tqdm pillow imageio \
    scikit-image scikit-learn pandas \
    protobuf timm graphviz

# pydensecrf requires eigen + cython + compiler
pip install pydensecrf

# Install SAM fork
pip install "git+https://github.com/danganhdat/segment-anything.git"

log "Python deps installed."

# 6. Fix libtiff bug (PIL)
info "Checking libtiff..."
cd /usr/lib/x86_64-linux-gnu/
if [ ! -f "libtiff.so.5" ]; then
    sudo ln -s libtiff.so.6 libtiff.so.5
    log "Created libtiff.so.5 symlink."
else
    warn "libtiff.so.5 already exists."
fi
cd -

# 7. Project dirs
mkdir -p pretrained data
log "Folders ready."

# 8. VOC dataset
VOC_URL="https://web.archive.org/web/20250604190242/http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_TAR="VOC2012.tar"

info "Downloading VOC2012..."

if [ ! -f "$VOC_TAR" ]; then
    aria2c -x 16 -s 16 "$VOC_URL" -o "$VOC_TAR"
else
    warn "VOC2012.tar exists."
fi

if [ ! -d "data/VOC2012" ]; then
    tar -xf "$VOC_TAR" -C data/
    if [ -d "data/VOCdevkit/VOC2012" ]; then
        mv data/VOCdevkit/VOC2012 data/VOC2012
        rm -rf data/VOCdevkit
    fi
    log "Extracted VOC2012."
else
    warn "VOC2012 folder exists."
fi

# 9. pretrained models
info "Downloading pretrained weights..."

if [ ! -f "pretrained/resnet_38d.params" ]; then
    gdown --fuzzy \
      "https://drive.google.com/file/d/1fpb4vah3e-Ynx4cv5upUcqnpJFY_FTja/view" \
      -O pretrained/resnet_38d.params
else
    warn "resnet_38d.params exists."
fi

if [ ! -f "pretrained/sam_vit_h.pth" ]; then
    aria2c -x 16 -s 16 \
      "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
      -o pretrained/sam_vit_h.pth
else
    warn "sam_vit_h.pth exists."
fi

log "Weights downloaded."

echo -e "${GREEN}🎉 S2C SETUP COMPLETE (CUDA 12.8 + Torch 2.8)!${NC}"
echo -e "${CYAN}Remember: YOU must manually activate your conda env.${NC}"
