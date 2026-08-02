#!/bin/bash
#SBATCH --job-name=compfix
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /data5/jellyho/Hindsight/HaF
for s in ${SEEDS:-0 1 2}; do
  env SEED=$s TAG=fractal EPOCHS=${EPOCHS:-20} SUBN=${SUBN:-4000} /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/probes/exp2h_compfix.py || echo "seed $s FAILED"
done
echo "=== compfix done ==="
