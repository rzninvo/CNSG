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
echo "  CNSG HM3D Quick Setup"
echo "=========================================="
echo -e "${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    log_error "conda is not installed. Please install miniconda or anaconda first."
    exit 1
fi

# Initialize git submodules
echo -e "\n${YELLOW}▶▶▶ Step 1: Initializing git submodules ◀◀◀${NC}"
cd "$ROOT_DIR"
log_info "Running git submodule update --init --recursive..."
git submodule update --init --recursive
log_success "Submodules initialized"

# Setup conda environment
echo -e "\n${YELLOW}▶▶▶ Step 2: Setting up Conda Environment ◀◀◀${NC}"

# Source conda
eval "$(conda shell.bash hook)"

# Check if habitat-default conda env exists
if conda env list | grep -q "^habitat-default "; then
    log_warning "Conda environment 'habitat-default' already exists. Skipping creation."
else
    log_info "Creating conda environment 'habitat-default' with Python 3.9 and cmake 3.14.0..."
    conda create -n habitat-default python=3.9 cmake=3.14.0 -y
    log_success "Conda environment 'habitat-default' created"
fi

# Activate habitat-default environment
log_info "Activating habitat-default environment..."
conda activate habitat-default

# Install Habitat-Sim from conda-forge
echo -e "\n${YELLOW}▶▶▶ Step 3: Installing Habitat-Sim ◀◀◀${NC}"

# Detect if we should use headless mode
if [[ -z "${DISPLAY:-}" ]]; then
    log_info "No display detected, installing habitat-sim in headless mode..."
    conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
else
    log_info "Display detected, installing habitat-sim with display support..."
    conda install habitat-sim withbullet -c conda-forge -c aihabitat -y
fi
log_success "Habitat-sim installed"

# Install Habitat-Lab
echo -e "\n${YELLOW}▶▶▶ Step 4: Installing Habitat-Lab ◀◀◀${NC}"
cd "$ROOT_DIR/habitat-lab"
log_info "Installing habitat-lab..."
pip install -e habitat-lab
log_success "Habitat-lab installed"

# Install Habitat-Sim requirements
echo -e "\n${YELLOW}▶▶▶ Step 5: Installing Habitat-Sim Requirements ◀◀◀${NC}"
cd "$ROOT_DIR/habitat-sim"
log_info "Installing Python packages from habitat-sim requirements.txt..."
pip install -r requirements.txt
log_success "Habitat-sim requirements installed"

# Install GUI and Audio dependencies
echo -e "\n${YELLOW}▶▶▶ Step 6: Installing GUI and Audio Dependencies ◀◀◀${NC}"
log_info "Installing Python packages for GUI, speech recognition, and audio..."
pip install PySide6 SpeechRecognition gTTS playsound3

log_info "Installing audio packages via conda..."
conda install -c conda-forge pyaudio alsa-plugins jack speex -y

# Install espeak (Linux only) — SKIPPED in automated run (requires sudo, user will run manually)
if command -v espeak &> /dev/null; then
    log_success "espeak already installed"
else
    log_warning "espeak not installed. After this script finishes, run manually:"
    echo -e "    ${YELLOW}sudo apt-get install -y espeak${NC}"
fi

# Install PyTorch with CUDA 12.8 support (required for RTX 5090 / Blackwell / sm_120)
echo -e "\n${YELLOW}▶▶▶ Step 6b: Installing PyTorch with CUDA 12.8 (for RTX 5090) ◀◀◀${NC}"
log_info "Installing torch/torchvision/torchaudio from PyTorch cu128 index..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
log_success "PyTorch (cu128) installed"

# Download LoRA adapter weights
echo -e "\n${YELLOW}▶▶▶ Step 7: Setting up LoRA Adapter Weights ◀◀◀${NC}"
cd "$ROOT_DIR"
LORA_DIR="$ROOT_DIR/finetuning/phi3-mr-lora-fixed-v3"

log_info "Creating LoRA adapter directory at: $LORA_DIR"
mkdir -p "$LORA_DIR"

log_info "Checking for LoRA adapter weights..."
if [[ -f "$LORA_DIR/adapter_config.json" ]] && [[ -f "$LORA_DIR/adapter_model.safetensors" ]]; then
    log_success "LoRA adapter weights already exist"
else
    log_warning "LoRA adapter weights not found"
    echo -e "\n${BLUE}Please download the LoRA adapter weights from:${NC}"
    echo -e "${YELLOW}https://huggingface.co/FBondi/phi3-mr-lora-weights${NC}"
    echo -e "\nPlace the following files in: ${GREEN}$LORA_DIR${NC}"
    echo -e "  - adapter_config.json"
    echo -e "  - adapter_model.safetensors"
fi

# Ask about HM3D dataset download
echo -e "\n${YELLOW}▶▶▶ Step 8: Download HM3D Dataset (Optional) ◀◀◀${NC}"
echo -e "${BLUE}Do you want to download the HM3D dataset now?${NC}"
echo -e "This requires a Matterport API token from:"
echo -e "${YELLOW}https://my.matterport.com/settings/account/devtools${NC}"
echo ""
read -p "Download HM3D dataset? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Enter your Matterport API Token ID (username): " API_TOKEN_ID
    read -sp "Enter your Matterport API Token Secret (password): " API_TOKEN_SECRET
    echo ""

    log_info "Downloading HM3D minival dataset..."
    python -m habitat_sim.utils.datasets_download \
        --username "$API_TOKEN_ID" \
        --password "$API_TOKEN_SECRET" \
        --uids hm3d_minival_v0.2

    log_success "HM3D dataset downloaded"
else
    log_warning "Skipping HM3D dataset download."
    echo -e "You can download it later with:"
    echo -e "${YELLOW}python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2${NC}"
fi

# Deactivate conda env
conda deactivate

# Final summary
echo -e "\n${GREEN}"
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo -e "${NC}"
echo -e "To run the HM3D house environment with the finetuned model:"
echo -e "  1. Activate habitat-default environment: ${YELLOW}conda activate habitat-default${NC}"
echo -e "  2. Navigate to habitat-sim: ${YELLOW}cd habitat-sim${NC}"
echo -e "  3. Run the viewer: ${YELLOW}python examples/mr_viewer.py --backend=local --finetuned-model=True${NC}"
echo ""
echo -e "${BLUE}Note:${NC} The base model will be downloaded automatically from Hugging Face on first run."
echo ""
log_success "Happy exploring!"
