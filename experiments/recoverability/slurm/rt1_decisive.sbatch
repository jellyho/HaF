#!/bin/bash
#SBATCH --job-name=rt1_dec
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
# decisive: full 87k, same budget as BC(+0.118). AHA near t+5, small lambda. vs BC full baseline.
for cond in "1 0.3 5" "1 0.1 5"; do
  set -- $cond; A=$1; LAM=$2; OFF=$3
  echo "=== rt1_dec AUX=$A LAMBDA=$LAM OFFSET=$OFF full ==="
  env AUX=$A LAMBDA=$LAM AUX_OFFSET=$OFF MAX_EP=0 EPOCHS=1 N_T=24 $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "cond $cond FAILED"
done
echo "=== rt1_dec done ==="
