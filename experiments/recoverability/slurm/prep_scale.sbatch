#!/bin/bash
#SBATCH --job-name=exp2h_prep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
echo "=== STAGE1: rebuild transitions_fractal (MAX_EP=$MAX_EP) ==="
env MAX_EP=$MAX_EP /data5/jellyho/Hindsight/HaF/.venv/bin/python -u experiments/recoverability/extract/stage1_fractal.py || exit 1
echo "=== STAGE2: DINOv2 latents ==="
env TAG=fractal /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/measure/stage2_metrics.py || exit 1
echo "=== PREP DONE ==="
