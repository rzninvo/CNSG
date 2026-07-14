# CNSG Installation and Setup Guide

Choose your installation path based on what you want to explore:

---

# Option 1: HM3D House with Finetuned Model (Recommended)

This installation is for exploring **HM3D house environments** with our finetuned language model. This is a simpler setup that installs Habitat-Sim from conda-forge (no building from source) and doesn't require the mesh pipeline or segmentation.

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Anaconda)
- Git

## Installation

Run the quick setup script:

```bash
git clone https://github.com/rzninvo/CNSG.git
cd CNSG
bash scripts/install_hm3d.sh
```

This script will:
1. Initialize git submodules
2. Create the `habitat-default` conda environment
3. Install Habitat-Sim from conda-forge (auto-detects headless mode)
4. Install Habitat-Lab
5. Install GUI and audio dependencies
6. Setup LoRA adapter weights directory
7. Optionally download the HM3D dataset

### Download HM3D Dataset

During installation, you'll be prompted to download the HM3D dataset. You'll need a Matterport API Token from [https://my.matterport.com/settings/account/devtools](https://my.matterport.com/settings/account/devtools).

If you skip this during installation, you can download it later:

```bash
conda activate habitat-default
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```

## Running the HM3D House Environment

Activate the habitat-default environment and run the viewer:

```bash
conda activate habitat-default
cd habitat-sim
python examples/mr_viewer.py --backend=local --finetuned-model=True
```

The base model (`microsoft/Phi-3-mini-4k-instruct`) will be downloaded automatically from Hugging Face on first run.

Use **W/A/S/D** keys to move and arrow keys or mouse to look around.

### Backend Options

**Local Model (finetuned) - Recommended:**
```bash
# Requires LoRA adapter weights (see LoRA section below)
python examples/mr_viewer.py --backend=local --finetuned-model=True
```

**Local Model (base):**
```bash
python examples/mr_viewer.py --backend=local
```

**OpenAI Backend:**
```bash
# Create .env file with your API key in the project root
echo "OPENAI_API_KEY=your_api_key_here" > .env
python examples/mr_viewer.py --backend=openai
```

## Web Demo — Conversational Navigation Assistant

Instead of the desktop viewer window, you can run the whole experience as a **web app**: the Habitat-Sim scene is streamed to your browser, you move with the same keys, and the LLM chat sits next to it. Everything (web app + live video + chat) is served by a **single command on one port**.

The prerequisites (Node.js, `xvfb`, `flask-sock`) are already installed by `scripts/install_hm3d.sh`. For public sharing you also need `cloudflared` (see below).

Run it from the repo root (the first run builds the web app automatically):

```bash
conda activate habitat-default
cd CNSG
```

**1. Default (works everywhere) — headless, no local window, software rendering:**
```bash
./run_demo.sh
```

**2. GPU-accelerated, still no local window (WSL / NVIDIA):**
```bash
./run_demo.sh --gpu
```

**3. GPU + public URL to share with any device (any network):**
```bash
./run_demo.sh --gpu --public
```

When it finishes loading it prints a **READY** banner with the URL(s) to open, e.g.:

```
      Local:   http://localhost:5001/assistant
      Public:  https://<random>.trycloudflare.com/assistant
```

Open the printed URL (ending in **/assistant**). In the browser:
- **Hover the viewer** and use **W/A/S/D** to move, **arrow keys** to look, **Z/X** for up/down, and other viewer shortcuts like **B** (bounding boxes).
- **Drag** inside the viewer with the mouse to look around.
- Type in the **chat panel** on the right to ask for directions (or use the mic / speaker buttons).

Useful options: `--port <n>`, `--width <w> --height <h>`, `--base-model` (non‑finetuned LLM), `--no-xvfb` (GPU with a visible window).

**Public sharing prerequisite (`--public`):** install `cloudflared` once (no account needed):
```bash
mkdir -p ~/.local/bin
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared
chmod +x ~/.local/bin/cloudflared
```
On WSL2 the local `172.x` address is **not** reachable from other devices — always share the `https://…trycloudflare.com/assistant` URL from the READY banner.

> **Note (native Linux, no WSL):** `--gpu` uses WSL's Mesa `d3d12` driver and falls back to software elsewhere. On a native Linux box with a display, use `./run_demo.sh --no-xvfb` for GPU rendering; on a headless GPU server you need a GPU-backed X server (or the default software mode `./run_demo.sh`).

> **Important:** The `habitat-default` environment uses Habitat-Sim from conda-forge and is specifically for HM3D datasets. Do not use this environment with the ETH HG E floor scene.

---

# Option 2: ETH HG E Floor with Semantic Segmentation (Optional)

This installation is for exploring the **ETH HG E floor academic building** with full semantic segmentation capabilities. This requires building Habitat-Sim from source and running a mesh processing pipeline.

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Anaconda)
- Git
- Docker (for the localization pipeline)

## Installation

Run the automated installation script:

```bash
git clone https://github.com/rzninvo/CNSG.git
cd CNSG
bash scripts/install.sh
```

This script will:

1. Initialize git submodules
2. Create and configure the `habitat-source` conda environment
3. Build and install habitat-sim with Bullet physics from source
4. Install habitat-lab
5. Create and configure the `CNSG-meshing` conda environment for the mesh pipeline
6. Download required mesh data
7. Optionally run the segmentation pipeline for ETH HG E floor

### Environment Usage

After installation, you'll have two conda environments:

```bash
# For Habitat-Sim built from source (ETH HG E floor)
conda activate habitat-source

# For Mesh Pipeline (3D reconstruction and segmentation)
conda activate CNSG-meshing
```

### Docker Container for Localization

The localization pipeline requires a Docker container:

```bash
cd mesh_pipeline/third_party/lamar-benchmark
docker build --target lamar -t lamar:lamar -f Dockerfile ./
```

## Running the ETH HG E Floor Environment

Activate the habitat-source environment and run the viewer with the HGE scene:

```bash
conda activate habitat-source
cd habitat-sim
python examples/mr_viewer.py --scene ./data/scene_datasets/HGE/HGE.basis.glb --dataset data/scene_datasets/HGE.scene_dataset_config.json
```

Use **W/A/S/D** keys to move and arrow keys or mouse to look around. Press **K** to toggle semantic visualization.

### Backend Options

By default, the viewer uses OpenAI as the backend. You can choose different backends:

**OpenAI Backend (default):**
```bash
# Create .env file with your API key in the project root
echo "OPENAI_API_KEY=your_api_key_here" > .env
python examples/mr_viewer.py --scene ./data/scene_datasets/HGE/HGE.basis.glb --dataset data/scene_datasets/HGE.scene_dataset_config.json --backend=openai
```

**Local Model (base):**
```bash
python examples/mr_viewer.py --scene ./data/scene_datasets/HGE/HGE.basis.glb --dataset data/scene_datasets/HGE.scene_dataset_config.json --backend=local
```

**Local Model (finetuned):**
```bash
# Requires LoRA adapter weights (see LoRA section below)
python examples/mr_viewer.py --scene ./data/scene_datasets/HGE/HGE.basis.glb --dataset data/scene_datasets/HGE.scene_dataset_config.json --backend=local --finetuned-model=True
```

> **Important:** The `habitat-source` environment is specifically built for the ETH HG E floor with semantic segmentation. Do not use this environment with HM3D datasets.

---

# LoRA Adapter Weights (for Finetuned Model)

To use the finetuned model with either installation option, download the LoRA adapter weights:

1. Download from [https://huggingface.co/FBondi/phi3-mr-lora-weights](https://huggingface.co/FBondi/phi3-mr-lora-weights)
2. Create the target directory:
   ```bash
   mkdir -p finetuning/phi3-mr-lora-fixed-v3
   ```
3. Place the downloaded files in `finetuning/phi3-mr-lora-fixed-v3/`

The directory must contain:
```
adapter_config.json
adapter_model.safetensors
```

The application will load the adapter at runtime from `finetuning/phi3-mr-lora-fixed-v3/`.
