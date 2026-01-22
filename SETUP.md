# CNSG Installation and Setup Guide

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Anaconda)
- Git
- Docker (for the localization pipeline)

---

## Installation

Run the automated installation script:

```bash
git clone https://github.com/rzninvo/CNSG.git
cd CNSG
bash scripts/install.sh
```

This script will:

1. Initialize git submodules
2. Create and configure the `habitat` conda environment
3. Build and install habitat-sim with Bullet physics
4. Install habitat-lab
5. Create and configure the `CNSG-meshing` conda environment for the mesh pipeline
6. Download required data
7. Optionally run the segmentation pipeline for ETH HG E floor

---

## Environment Usage

After installation, you'll have two conda environments:

```bash
# For Habitat-Sim (simulation and navigation)
conda activate habitat

# For Mesh Pipeline (3D reconstruction and segmentation)
conda activate CNSG-meshing
```

---

## Running the Demo

After installation, you can run the interactive viewer:

```bash
conda activate habitat
cd habitat-sim
python examples/mr_viewer.py
```

### Optional: OpenAI API Key

If you want to use OpenAI GPT features for instruction generation, create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

---

## Docker Container for Localization

The localization pipeline requires a Docker container. Build it with:

```bash
cd mesh_pipeline/third_party/lamar-benchmark
docker build --target lamar -t lamar:lamar -f Dockerfile ./
```

---

## Download HM3D Dataset (Optional)

To download the Habitat-Matterport 3D Research Dataset:

1. Get a Matterport API Token from [https://my.matterport.com/settings/account/devtools](https://my.matterport.com/settings/account/devtools)
2. Download the minival split:

```bash
conda activate habitat
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```
