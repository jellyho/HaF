#!/bin/bash
#SBATCH --job-name=exp2d
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log

export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
echo "host=$(hostname) TAG=$TAG SEED=$SEED"
env TAG=$TAG SEED=$SEED EPOCHS=30 DEV=cuda \
  /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/probes/exp2d_lambda.py
