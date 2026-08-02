#!/bin/bash
#SBATCH --job-name=dense_ext
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /data5/jellyho/Hindsight/HaF
MAX_EP=${MAX_EP:-20000} N_T=${N_T:-30} CHUNK_EP=${CHUNK_EP:-400} \
  .venv/bin/python -u experiments/recoverability/extract/stage1_fractal_dense.py
echo "=== dense_extract done rc=$? MAX_EP=${MAX_EP:-20000} ==="
