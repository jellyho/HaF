#!/bin/bash
#SBATCH --job-name=rt1_conf
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
for s in 1 2; do
  echo "=== SEED=$s BC ==="; env SEED=$s AUX=0 LAMBDA=0 AUX_OFFSET=30 MAX_EP=0 EPOCHS=1 N_T=24 $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "BC s$s FAILED"
  echo "=== SEED=$s AHA near λ0.1 ==="; env SEED=$s AUX=1 LAMBDA=0.1 AUX_OFFSET=5 MAX_EP=0 EPOCHS=1 N_T=24 $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "AHA s$s FAILED"
done
echo "=== rt1_confirm done ==="
