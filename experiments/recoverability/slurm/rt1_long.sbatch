#!/bin/bash
#SBATCH --job-name=rt1_long
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /data5/jellyho/Hindsight/HaF
PY=/data5/jellyho/Hindsight/HaF/.venv/bin/python
E=${EPOCHS:-6}
echo "=== rt1_long BC (AUX=0) EPOCHS=$E full, CKPT=_e6 ==="
env CKPT=_e6 AUX=0 LAMBDA=0 AUX_OFFSET=30 MAX_EP=0 EPOCHS=$E N_T=24 $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "BC FAILED"
echo "=== rt1_long AHA (AUX=1, near t+5, λ0.1) EPOCHS=$E full, CKPT=_e6 ==="
env CKPT=_e6 AUX=1 LAMBDA=0.1 AUX_OFFSET=5 MAX_EP=0 EPOCHS=$E N_T=24 $PY -u experiments/recoverability/probes/exp2h_rt1_full_jax.py || echo "AHA FAILED"
echo "=== rt1_long done — competent-er policies saved to *_e6.msgpack ==="
