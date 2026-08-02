#!/bin/bash
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
ENC=/data5/jellyho/Hindsight/enc_venv/bin/python
VENV=/data5/jellyho/Hindsight/HaF/.venv/bin/python
S2=experiments/recoverability/measure/stage2_metrics.py
EXP=experiments/recoverability/probes/exp2b_regularize.py
OUT=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs

# --- goal: latents already encoded; run exp2b seeds 0,1 on GPU 3 ---
for S in 0 1; do
  env TAG=libero_goal EPOCHS=30 SEED=$S DEV=cuda CUDA_VISIBLE_DEVICES=1 $ENC -u $EXP > $OUT/exp2b_libero_goal_s$S.log 2>&1
done

# --- object: wait for extraction, encode latents, then exp2b ---
while [ ! -f $OUT/transitions_libero_object.npz ]; do sleep 10; done
sleep 5
env TAG=libero_object DEV=cuda CUDA_VISIBLE_DEVICES=1 $ENC -u $S2 > $OUT/stage2_libero_object.log 2>&1
for S in 0 1; do
  env TAG=libero_object EPOCHS=30 SEED=$S DEV=cuda CUDA_VISIBLE_DEVICES=1 $ENC -u $EXP > $OUT/exp2b_libero_object_s$S.log 2>&1
done
echo LIBERO_EXP2B_DONE > $OUT/libero_exp2b.marker
