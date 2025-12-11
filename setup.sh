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

# System deps
# System deps
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
# pip upgrade
# pip upgrade
info "Upgrading pip..."
pip install --upgrade pip
pip install gdown kagglehub
# pip install -r requirements.txt
# Install SAM fork
# pip install "git+https://github.com/danganhdat/segment-anything.git"

log "Python deps installed."

# Fix libtiff bug (PIL)
# Fix libtiff bug (PIL)
info "Checking libtiff..."
cd /usr/lib/x86_64-linux-gnu/
if [ ! -f "libtiff.so.5" ]; then
    sudo ln -s libtiff.so.6 libtiff.so.5
    log "Created libtiff.so.5 symlink."
else
    warn "libtiff.so.5 already exists."
fi
cd -

# Project dirs
mkdir -p pretrained data se/default
# Project dirs
mkdir -p pretrained data se/default
log "Folders ready."

# seg map
info "Downloading SE maps from KaggleHub..."
python3 download_se_maps.py
log "SE maps downloaded into se/default."

# VOC dataset
VOC_URL="https://datasets.cms.waikato.ac.nz/ufdl/data/pascalvoc2012/VOCtrainval_11-May-2012.tar"
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

# pretrained models
# pretrained models
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