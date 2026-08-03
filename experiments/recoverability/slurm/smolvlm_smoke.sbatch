#!/bin/bash
#SBATCH --job-name=svlm_smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /data5/jellyho/Hindsight/HaF
/data5/jellyho/Hindsight/HaF/.venv/bin/python -u experiments/recoverability/probes/smolvlm_smoke.py 8
