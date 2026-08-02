#!/bin/bash
#SBATCH --job-name=rt1_sweep
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
# diagnostic: is the AHA hurt from lambda (too strong) or signal-floor (far-future=noise)?
#   near t+5 vs far t+30 ; lambda 0.3 vs 1.0. AUX=1 all. subset for speed.
for cond in "0.3 5" "0.3 30" "1.0 5"; do
  set -- $cond; LAM=$1; OFF=$2
  echo "=== rt1_sweep AUX=1 LAMBDA=$LAM AUX_OFFSET=$OFF MAX_EP=${MAX_EP:-15000} ==="
  env AUX=1 LAMBDA=$LAM AUX_OFFSET=$OFF MAX_EP=${MAX_EP:-15000} EPOCHS=1 N_T=24 $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "cond $cond FAILED"
done
# BC baseline at same subset for fair comparison
env AUX=0 LAMBDA=0 AUX_OFFSET=30 MAX_EP=${MAX_EP:-15000} EPOCHS=1 N_T=24 $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "BC FAILED"
echo "=== rt1_sweep done ==="
