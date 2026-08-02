#!/bin/bash
#SBATCH --job-name=render_test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
cd /data5/jellyho/Hindsight/HaF
export SAPIEN_RENDERER=vulkan
/data5/jellyho/Hindsight/simpler_venv/bin/python -u experiments/recoverability/probes/simpler_render_test.py
