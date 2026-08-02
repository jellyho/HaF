#!/bin/bash
# One slurm job = one GPU processing MANY (exp,dataset,seed) items sequentially.
# Idempotent (skips items whose json already exists) + courteous (yields while other users have pending jobs).
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
V=/data5/jellyho/Hindsight/enc_venv/bin/python
OUT=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs
CHUNK="$1"
declare -A SCRIPT=( [exp2b]=exp2b_regularize.py [exp2c]=exp2c_auxbattery.py [exp2d]=exp2d_lambda.py [exp2e]=exp2e_retro.py [exp2f]=exp2f_fused.py [exp2g]=exp2g_repquality.py )

others_pending() {
  squeue -h -t PD -p debug -o '%u' 2>/dev/null | grep -v '^jellyho$' | grep -c . 2>/dev/null
}

while IFS='|' read -r exp ds seed; do
  [ -z "$exp" ] && continue
  outfile=$OUT/${exp}_${ds}_s${seed}.json
  if [ -f "$outfile" ]; then echo "SKIP(done) $exp $ds s$seed"; continue; fi
  # courtesy: if ANY other user has a pending job on debug, wait until they clear (yield GPUs is via the 6-cap;
  # this pause keeps us from grabbing more work while someone is waiting).
  while [ "$(others_pending)" -gt 0 ]; do
    echo "YIELD: other users pending -> sleep 120s"; sleep 120
  done
  echo "RUN $exp $ds s$seed  host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES  $(date +%H:%M:%S)"
  env TAG=$ds SEED=$seed EPOCHS=30 DEV=cuda $V -u experiments/recoverability/probes/${SCRIPT[$exp]} \
    > $OUT/log_${exp}_${ds}_s${seed}.log 2>&1
  echo "  -> exit $? ($([ -f "$outfile" ] && echo OK || echo FAIL))"
done < "$CHUNK"
echo "WORKER_DONE $CHUNK $(date +%H:%M:%S)"
