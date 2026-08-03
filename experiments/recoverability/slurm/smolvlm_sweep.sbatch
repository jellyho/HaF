#!/bin/bash
#SBATCH --job-name=svlm_sweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data5/jellyho/Hindsight/HaF/src
cd /data5/jellyho/Hindsight/HaF
PY=/data5/jellyho/Hindsight/HaF/.venv/bin/python
S=${SEED:-0}; ST=${STEPS:-1500}; EP=${MAX_EP:-3000}
# BC-only baseline
env TAG=bc_s$S AUX_W=0 SEED=$S STEPS=$ST MAX_EP=$EP $PY -u -m haf_torch.training.train || echo "BC FAILED"
# Family A (structured-hard): mask ratio dials recoverability, target stays learnable+relevant
for R in 0.25 0.50 0.75 0.90; do
  env TAG=maskR${R}_s$S AUX_W=0.1 AUX_FAMILY=mask AUX_MASK=$R SEED=$S STEPS=$ST MAX_EP=$EP $PY -u -m haf_torch.training.train || echo "mask $R FAILED"
done
# Family B (noise-hard): sigma dials recoverability toward the signal floor
for G in 0.5 1.0 2.0; do
  env TAG=noiseS${G}_s$S AUX_W=0.1 AUX_FAMILY=noise AUX_SIGMA=$G SEED=$S STEPS=$ST MAX_EP=$EP $PY -u -m haf_torch.training.train || echo "noise $G FAILED"
done
echo "=== smolvlm_sweep done (A: mask, B: noise) ==="
