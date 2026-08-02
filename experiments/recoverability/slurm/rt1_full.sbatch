#!/bin/bash
#SBATCH --job-name=rt1_full
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /data5/jellyho/Hindsight/HaF
PY=/data5/jellyho/Hindsight/HaF/.venv/bin/python
for a in 1 0; do
  echo "=== rt1_full AUX=$a MAX_EP=${MAX_EP:-0} EPOCHS=${EPOCHS:-1} ==="
  env AUX=$a MAX_EP=${MAX_EP:-0} EPOCHS=${EPOCHS:-1} N_T=${N_T:-24} BS=${BS:-64} $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "AUX=$a FAILED"
done
echo "=== rt1_full both arms done ==="
