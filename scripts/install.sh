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

# Check if habitat conda env exists
if conda env list | grep -q "^habitat "; then
    log_warning "Conda environment 'habitat' already exists. Skipping creation."
else
    log_info "Creating conda environment 'habitat' with Python 3.9 and cmake 3.14.0..."
    conda create -n habitat python=3.9 cmake=3.14.0 -y
    log_success "Conda environment 'habitat' created"
fi

# Activate habitat environment
log_info "Activating habitat environment..."
conda activate habitat

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

# Should still be in habitat env
log_info "Installing habitat-lab (should be in habitat env)..."
pip install -e habitat-lab
log_success "Habitat-lab installed"

# Deactivate habitat env
log_info "Deactivating habitat environment..."
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
log_success "Mesh pipeline requirements installed"

log_info "Installing additional Python packages..."
read -r -p "Install Python/conda audio & TTS packages? [Y/n]: " _install_tts
if [[ -z "${_install_tts}" || "${_install_tts,,}" == "y" || "${_install_tts,,}" == "yes" ]]; then
    pip install PySide6 SpeechRecognition gTTS pyttsx3 || log_warning "pip install of TTS/audio packages failed"
    conda install -c conda-forge pyaudio alsa-plugins jack speex -y || log_warning "conda install of audio packages failed"

    log_info "Installing additional system dependencies (requires sudo)..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update || true
        sudo apt-get install -y espeak || log_warning "apt-get install espeak failed"
        log_success "espeak installed (or attempted)"
    else
        log_warning "apt-get not found — please install 'espeak' manually if needed."
    fi

    log_success "Additional packages installed (or attempted)"
else
    log_info "Skipping audio/TTS package installation"
fi

# Download data
log_info "Downloading mesh pipeline data..."
bash scripts/download_data.sh
log_success "Data downloaded"

# Step 5: Ask about segmentation
echo -e "\n${YELLOW} Step 5: Semantic Segmentation (Optional) ${NC}"
echo -e "${BLUE}Do you want to run the segmentation pipeline?${NC}"
echo -e "This will generate semantic meshes for Habitat-Sim to use with ETH HG E floor."
echo -e "Note: This can take several hours depending on your hardware."
echo ""
read -p "Run segmentation pipeline? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Running segmentation pipeline..."
    bash scripts/run_segmentation_pipeline.sh
    log_success "Segmentation pipeline completed"

    # Copy files to habitat-sim
    echo -e "\n${YELLOW} Step 6: Copying Semantic Data to Habitat-Sim ${NC}"

    # Create scene_datasets directory if it doesn't exist
    SCENE_DATASETS_DIR="$ROOT_DIR/habitat-sim/data/scene_datasets"
    log_info "Creating scene_datasets directory at: $SCENE_DATASETS_DIR"
    mkdir -p "$SCENE_DATASETS_DIR"

    # Copy scene dataset config to scene_datasets
    log_info "Copying HGE.scene_dataset_config.json..."
    cp "$ROOT_DIR/mesh_pipeline/data/HGE.scene_dataset_config.json" "$SCENE_DATASETS_DIR/"

    # Create HGE directory
    HGE_DIR="$SCENE_DATASETS_DIR/HGE"
    log_info "Creating HGE directory at: $HGE_DIR"
    mkdir -p "$HGE_DIR"

    # Copy files to HGE directory
    log_info "Copying semantic files to HGE directory..."

    FILES_TO_COPY=(
        "room_id_to_name_map.json"
        "HGE.semantic.txt"
        "HGE.semantic.scn"
        "HGE.semantic.glb"
        "HGE.basis.glb"
    )

    for file in "${FILES_TO_COPY[@]}"; do
        SOURCE_FILE="$ROOT_DIR/mesh_pipeline/data/$file"
        if [[ -f "$SOURCE_FILE" ]]; then
            cp "$SOURCE_FILE" "$HGE_DIR/"
            log_info "   Copied $file"
        else
            log_warning "   File not found: $file (may be generated by segmentation pipeline)"
        fi
    done

    log_success "Semantic data copied to habitat-sim!"
    log_info "Scene dataset location: $SCENE_DATASETS_DIR/HGE"
else
    log_warning "Skipping segmentation pipeline."
    echo -e "You can run it later with: ${YELLOW}bash mesh_pipeline/scripts/run_segmentation_pipeline.sh${NC}"
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
echo -e "  " For Habitat-Sim:    ${GREEN}conda activate habitat${NC}"
echo -e "  " For Mesh Pipeline:  ${GREEN}conda activate CNSG-meshing${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Activate the habitat environment: ${YELLOW}conda activate habitat${NC}"
echo -e "  2. Test Habitat-Sim by running: ${YELLOW}python examples/example.py${NC}"
echo ""
log_success "Happy coding!"
