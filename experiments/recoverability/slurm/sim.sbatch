#!/bin/bash
#SBATCH --job-name=gensim
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
DEV=cuda /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/generalization/exp_sim.py
