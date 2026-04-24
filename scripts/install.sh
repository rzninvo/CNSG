#!/usr/bin/env bash
set -euo pipefail

# Get the script directory and root directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo -e "${GREEN}"
echo "=========================================="
echo "  CNSG Project Installation Script"
echo "=========================================="
echo -e "${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    log_error "conda is not installed. Please install miniconda or anaconda first."
    exit 1
fi

# Step 1: Initialize git submodules
echo -e "\n${YELLOW} Step 1: Initializing git submodules ${NC}"
cd "$ROOT_DIR"
log_info "Running git submodule update --init --recursive..."
git submodule update --init --recursive
log_success "Submodules initialized"

# Step 2: Install Habitat-Sim
echo -e "\n${YELLOW} Step 2: Installing Habitat-Sim ${NC}"
cd "$ROOT_DIR/habitat-sim"

# Source conda
eval "$(conda shell.bash hook)"

# Check if habitat-source conda env exists
if conda env list | grep -q "^habitat-source "; then
    log_warning "Conda environment 'habitat-source' already exists. Skipping creation."
else
    log_info "Creating conda environment 'habitat-source' with Python 3.9 and cmake 3.14.0..."
    conda create -n habitat-source python=3.9 cmake=3.14.0 -y
    log_success "Conda environment 'habitat-source' created"
fi

# Activate habitat-source environment
log_info "Activating habitat-source environment..."
conda activate habitat-source

# Install Python requirements
log_info "Installing Python requirements for habitat-sim..."
pip install -r requirements.txt

# # Install system dependencies (Linux only)
# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#     log_info "Installing system dependencies (requires sudo)..."
#     sudo apt-get update || true
#     sudo apt-get install -y --no-install-recommends \
#         libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
#     log_success "System dependencies installed"
# else
#     log_warning "Not on Linux, skipping apt-get dependencies. Please install them manually if needed."
# fi

# Build habitat-sim with bullet physics
log_info "Building habitat-sim with bullet physics (this may take a while)..."
python setup.py install --bullet
log_success "Habitat-sim built successfully"

# Step 3: Install Habitat-Lab
echo -e "\n${YELLOW} Step 3: Installing Habitat-Lab ${NC}"
cd "$ROOT_DIR/habitat-lab"

# Should still be in habitat-source env
log_info "Installing habitat-lab (should be in habitat-source env)..."
pip install -e habitat-lab
log_success "Habitat-lab installed"

# Install PyTorch with CUDA 12.8 (RTX 5090 / Blackwell) in habitat-source env
log_info "Installing PyTorch cu128 for habitat-source env..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
log_info "Pinning huggingface_hub<1.0 for transformers compatibility..."
pip install "huggingface_hub<1.0"
log_success "PyTorch (cu128) installed in habitat-source"

# Install GUI/audio extras (match install_hm3d.sh so the viewer runs)
log_info "Installing GUI and audio Python packages..."
pip install PySide6 SpeechRecognition gTTS playsound3
log_info "Installing audio packages via conda..."
conda install -c conda-forge pyaudio alsa-plugins jack speex -y
if command -v espeak &> /dev/null; then
    log_success "espeak already installed"
else
    log_warning "espeak not installed. Run manually: sudo apt-get install -y espeak"
fi

# Deactivate habitat-source env
log_info "Deactivating habitat-source environment..."
conda deactivate

# Step 4: Install mesh pipeline
echo -e "\n${YELLOW} Step 4: Installing Mesh Pipeline ${NC}"
cd "$ROOT_DIR/mesh_pipeline"

# Check if CNSG-meshing conda env exists
if conda env list | grep -q "^CNSG-meshing "; then
    log_warning "Conda environment 'CNSG-meshing' already exists. Skipping creation."
else
    log_info "Creating conda environment 'CNSG-meshing' with Python 3.10.6..."
    conda create -n CNSG-meshing python=3.10.6 -y
    log_success "Conda environment 'CNSG-meshing' created"
fi

# Activate CNSG-meshing environment
log_info "Activating CNSG-meshing environment..."
conda activate CNSG-meshing

# Install Python requirements
log_info "Installing Python requirements for mesh pipeline..."
pip install -r requirements.txt
log_info "Upgrading torch to cu128 (for RTX 5090 / Blackwell)..."
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
log_success "Mesh pipeline requirements installed"


# Download data
log_info "Downloading mesh pipeline data..."
bash scripts/download_data.sh
log_success "Data downloaded"

# Step 5: Wire HGE assets into habitat-sim's scene_datasets layout so the
# README's `python examples/mr_viewer.py --scene ./data/scene_datasets/HGE/
# HGE.basis.glb ...` invocation works out of the box. We symlink rather than
# copy — the source bytes in mesh_pipeline/data/ (basis) and data/maps/hge/
# (semantics) are the authoritative location; symlinks keep both paths in
# sync when we rebuild semantics.
echo -e "\n${YELLOW} Step 5: Wiring HGE into habitat-sim/data/scene_datasets/ ${NC}"

SCENE_DATASETS_DIR="$ROOT_DIR/habitat-sim/data/scene_datasets"
HGE_DIR="$SCENE_DATASETS_DIR/HGE"
MESH_PIPELINE_DATA="$ROOT_DIR/mesh_pipeline/data"
BUILD_DIR="$ROOT_DIR/data/maps/hge"

mkdir -p "$HGE_DIR"

# Shipped from the Google Drive bundle (required for any render of HGE).
# Scene config + basis stage live relative to scene_datasets/ per the README.
log_info "Linking HGE.scene_dataset_config.json ..."
ln -sfn "$MESH_PIPELINE_DATA/HGE.scene_dataset_config.json" \
       "$SCENE_DATASETS_DIR/HGE.scene_dataset_config.json"

log_info "Linking HGE/HGE.basis.glb ..."
ln -sfn "$MESH_PIPELINE_DATA/HGE.basis.glb" "$HGE_DIR/HGE.basis.glb"

# Room names + semantic assets: prefer the freshly-built versions under
# data/maps/hge/ when they exist (cnsg/segmentation/build_hge output), fall
# back to whatever the Google Drive bundle shipped. Loud [WARN] if neither —
# the viewer still runs but SemanticSensor returns zero regions/objects.
link_first_existing() {
    local target="$1"
    shift
    for src in "$@"; do
        if [[ -f "$src" ]]; then
            ln -sfn "$src" "$target"
            log_info "   linked $(basename "$target") -> $src"
            return 0
        fi
    done
    log_warning "[WARN] $(basename "$target"): expected=readable one of [$*], got=none-exist, fallback=not-linked"
    return 1
}

link_first_existing "$SCENE_DATASETS_DIR/HGE.semantic.glb" \
    "$BUILD_DIR/HGE.semantic.glb" "$MESH_PIPELINE_DATA/HGE.semantic.glb"
link_first_existing "$SCENE_DATASETS_DIR/HGE.semantic.txt" \
    "$BUILD_DIR/HGE.semantic.txt" "$MESH_PIPELINE_DATA/HGE.semantic.txt"
link_first_existing "$SCENE_DATASETS_DIR/room_id_to_name_map.json" \
    "$BUILD_DIR/room_id_to_name_map.json" "$MESH_PIPELINE_DATA/room_id_to_name_map.json"

log_success "scene_datasets/HGE wired up"

# Step 6: Ask about (re)building semantics. The legacy run_segmentation_
# pipeline.sh is gone (replaced by our Phase-3 cnsg/segmentation pipeline
# in `scripts/build_hge_semantics.sh`).
echo -e "\n${YELLOW} Step 6: HGE Semantic Mesh (Optional Rebuild) ${NC}"
echo -e "${BLUE}The Google Drive bundle ships a pre-built semantic mesh.${NC}"
echo -e "Rebuild it from scratch (Mask2Former + SAM 3, ~30 min on RTX 5090)?"
echo -e "Skip unless you changed prompts / confidence in HgeBuildConfig."
echo ""
read -p "Run the semantic build pipeline? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Running cnsg.segmentation.build_hge (needs the cnsg-seg env) ..."
    bash "$ROOT_DIR/scripts/build_hge_semantics.sh"
    log_success "Semantic build done — Step 5 symlinks already point at the fresh output"

    # Re-run symlink step so any new files land correctly
    link_first_existing "$SCENE_DATASETS_DIR/HGE.semantic.glb" \
        "$BUILD_DIR/HGE.semantic.glb" "$MESH_PIPELINE_DATA/HGE.semantic.glb"
    link_first_existing "$SCENE_DATASETS_DIR/HGE.semantic.txt" \
        "$BUILD_DIR/HGE.semantic.txt" "$MESH_PIPELINE_DATA/HGE.semantic.txt"
    link_first_existing "$SCENE_DATASETS_DIR/room_id_to_name_map.json" \
        "$BUILD_DIR/room_id_to_name_map.json" "$MESH_PIPELINE_DATA/room_id_to_name_map.json"
else
    log_warning "Skipping semantic rebuild; using whatever's on disk."
    echo -e "You can run it later with: ${YELLOW}bash scripts/build_hge_semantics.sh${NC}"
fi
# install some usefull libraries

# Deactivate conda env
conda deactivate

# Final summary
echo -e "\n${GREEN}"
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo -e "${NC}"
echo -e "Environment usage:"
echo -e "  " For Habitat-Sim:    ${GREEN}conda activate habitat-source${NC}"
echo -e "  " For Mesh Pipeline:  ${GREEN}conda activate CNSG-meshing${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Activate the habitat-source environment: ${YELLOW}conda activate habitat-source${NC}"
echo -e "  2. Navigate to habitat-sim: ${YELLOW}cd habitat-sim${NC}"
echo -e "  3. Run the ETH HG E floor viewer: ${YELLOW}python examples/mr_viewer.py${NC}"
echo ""
log_success "Happy coding!"
