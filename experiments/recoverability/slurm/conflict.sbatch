#!/bin/bash
#SBATCH --job-name=conflict
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
# LA4VLA preliminary probe: one paired-trained policy per seed, then the 4-condition x 3-modality diagnostic.
# One job = 3 seeds sequentially on ONE GPU (courteous: single GPU, leaves the rest free).
for S in 0 1 2; do
  echo "=== SEED=$S ==="
  env SEED=$S EPOCHS=25 DEV=cuda $V -u experiments/recoverability/conflict/exp_conflict.py \
    > /data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/conflict_s${S}.log 2>&1
done
echo "CONFLICT_DONE"
