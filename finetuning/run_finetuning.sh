#!/bin/bash
#SBATCH --job-name=finetuning_mr
#SBATCH --output=logs/finetuning_mr_%j.out
#SBATCH --error=logs/finetuning_mr_%j.err
#SBATCH --time=03:00:00
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

module purge
module load eth_proxy || true
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1

# Attiva Conda
source /cluster/project/cvg/students/fbondi/miniconda3/etc/profile.d/conda.sh
conda activate muon

export PYTHONNOUSERSITE=1
export PYTHONPATH=

# Debug: show which python
which python
python --version

python -c "import transformers; print('Transformers version:', transformers.__version__)"

nvidia-smi

# -------------------------------
# HUGGINGFACE CACHE CONFIG
# -------------------------------
export HF_HOME=/cluster/project/cvg/students/fbondi/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets

python -c "from huggingface_hub import hf_hub_download; import os; print('HF cache dir:', os.getenv('HF_HOME'))"

python - << 'EOF'
import transformers, accelerate, trl, peft, datasets
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("TRL:", trl.__version__)
print("PEFT:", peft.__version__)
print("Datasets:", datasets.__version__)
EOF

# ------------------------------------------------------------
# Finetuning for MR
# ------------------------------------------------------------
python finetune.py