#!/bin/bash
#SBATCH --job-name=rt1_jax
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /data5/jellyho/Hindsight/HaF
PY=/data5/jellyho/Hindsight/HaF/.venv/bin/python
for s in ${SEEDS:-0 1 2}; do
  for a in 0 1; do
    echo "=== rt1_jax SEED=$s AUX=$a EPOCHS=${EPOCHS:-10} SUBN=${SUBN:-6000} ==="
    env SEED=$s AUX=$a EPOCHS=${EPOCHS:-10} SUBN=${SUBN:-6000} $PY -u experiments/recoverability/probes/exp2h_rt1_jax.py || { echo "s$s a$a FAILED"; }
  done
done
echo "=== rt1_jax done ==="
