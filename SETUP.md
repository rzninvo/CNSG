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
