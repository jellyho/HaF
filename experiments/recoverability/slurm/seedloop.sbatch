#!/bin/bash
#SBATCH --job-name=seedloop
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
for S in 0 1 2; do
  echo "=== SEED $S ==="
  env TAG=$TAG SEED=$S EPOCHS=${EPOCHS:-25} SUBN=${SUBN:-4000} RUNTAG=${RUNTAG:-r} \
      MIXSET=${MIXSET:-base} KI_ONLY=${KI_ONLY:-} KBINS=${KBINS:-16} \
      /data5/jellyho/Hindsight/enc_venv/bin/python -u experiments/recoverability/probes/${PYSCRIPT}
done
echo "=== ALL SEEDS DONE ==="
