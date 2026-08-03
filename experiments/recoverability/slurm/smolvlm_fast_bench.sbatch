#!/bin/bash
#SBATCH --job-name=svlm_fbench
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data5/jellyho/Hindsight/HaF/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data5/jellyho/Hindsight/HaF
/data5/jellyho/Hindsight/HaF/.venv/bin/python -u experiments/recoverability/probes/smolvlm_fast_bench.py "${BSL:-8,16,32,64}"
