#!/bin/bash
#SBATCH --job-name=mixki
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
env TAG=$TAG SEED=$SEED EPOCHS=${EPOCHS:-20} SUBN=${SUBN:-6000} RUNTAG=${RUNTAG:-mk} /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/probes/exp2h_mixki.py
