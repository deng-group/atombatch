#!/bin/bash
# Template for running an atombatch example on one GPU node with SLURM.
# Adapt the partition, account, modules, and paths to your cluster.
#SBATCH --job-name=atombatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# One-time setup (on a login node), for example for an RTX 40-series GPU:
#   python -m venv ~/venvs/atombatch
#   ~/venvs/atombatch/bin/pip install atombatch matscipy numba symmetrix-xl-cuda12-sm89
source ~/venvs/atombatch/bin/activate

# If an older CUDA toolkit is registered with the dynamic linker, point the
# Symmetrix CUDA wheel at its own CUDA 12 libraries (see docs/installation.md).
site=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
export LD_LIBRARY_PATH=$site/cuda_nvrtc/lib:$site/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

export SYMMETRIX_MODEL=$HOME/models/mace-mp-0b-medium.json
symmetrix doctor
python relax_orderings.py
