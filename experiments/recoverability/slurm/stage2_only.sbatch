#!/bin/bash
#SBATCH --job-name=exp2h_s2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
echo "=== STAGE2 only: DINOv2 latents for N=16000 ==="
env TAG=fractal DEV=cuda /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/measure/stage2_metrics.py || exit 1
echo "=== STAGE2 DONE ==="
