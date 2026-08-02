#!/bin/bash
#SBATCH --job-name=hafw
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log

bash /data5/jellyho/Hindsight/HaF/experiments/recoverability/slurm/worker.sh "$CHUNK"
