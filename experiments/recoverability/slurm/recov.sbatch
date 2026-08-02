#!/bin/bash
#SBATCH --job-name=recovgov
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
V=/data5/jellyho/Hindsight/enc_venv/bin/python
# one job = all 3 regimes x 3 seeds on ONE GPU (9 short runs, sequential)
for R in V L S; do for S in 0 1 2; do
  env REGIME=$R SEED=$S EPOCHS=25 DEV=cuda $V -u experiments/recoverability/conflict/exp_recov_governs.py \
    > /data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/recov_${R}_s${S}.log 2>&1
done; done
echo "RECOV_DONE"
