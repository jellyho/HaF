#!/bin/bash
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
V=/data5/jellyho/Hindsight/enc_venv/bin/python
SCRIPT=experiments/recoverability/probes/exp2b_regularize.py
DS1=${DS1:-fractal}
DS2=${DS2:-droid}
for S in 0 1; do
  env TAG=$DS1 EPOCHS=30 SEED=$S CUDA_VISIBLE_DEVICES=1 $V -u $SCRIPT > /data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/exp2b_${DS1}_s$S.log 2>&1 &
  env TAG=$DS2 EPOCHS=30 SEED=$S CUDA_VISIBLE_DEVICES=2 $V -u $SCRIPT > /data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/exp2b_${DS2}_s$S.log 2>&1 &
  wait
done
echo "TRIMODAL_${DS1}_${DS2}_DONE" > /data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/trimodal_${DS1}_${DS2}.marker
