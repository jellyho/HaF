#!/bin/bash
#SBATCH --job-name=conflqual
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
SEED=0 EPOCHS=25 DEV=cuda /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/conflict/exp_conflict_qual.py
echo QUAL_DONE
