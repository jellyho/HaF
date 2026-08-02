#!/bin/bash
#SBATCH --job-name=actiontok
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /data5/jellyho/Hindsight/HaF
PY=/data5/jellyho/Hindsight/HaF/.venv/bin/python     # superset: torch+transformers+sklearn+FAST
for s in ${SEEDS:-0 1 2}; do
  echo "=== actiontok SEED=$s EPOCHS=${EPOCHS:-15} SUBN=${SUBN:-6000} ==="
  env SEED=$s TAG=fractal EPOCHS=${EPOCHS:-15} SUBN=${SUBN:-6000} \
    $PY -u experiments/recoverability/probes/exp2h_actiontok.py || { echo "seed $s FAILED"; exit 1; }
done
echo "=== actiontok all seeds done ==="
