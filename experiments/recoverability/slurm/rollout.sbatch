#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
cd /data5/jellyho/Hindsight/HaF
ARM=${ARM:-bc}; PORT=${PORT:-8123}; NT=${NT:-5}
ENC=/data5/jellyho/Hindsight/enc_venv/bin/python
SIM=/data5/jellyho/Hindsight/simpler_venv/bin/python
O=experiments/recoverability/outputs

# 1) train+save the policy if missing
if [ ! -f "$O/policy_${ARM}.pt" ]; then
  echo "=== [1] train+save policy ARM=$ARM ==="
  env ARM=$ARM TAG=fractal EPOCHS=${EPOCHS:-25} SUBN=${SUBN:-0} $ENC -u experiments/recoverability/probes/save_policy.py || exit 1
fi

# 2) start the torch policy server (background)
echo "=== [2] start mini-policy server ARM=$ARM :$PORT ==="
env ARM=$ARM PORT=$PORT $ENC -u experiments/recoverability/probes/mini_policy_server.py &
SRV=$!
sleep 75   # model + MiniLM + DINOv2 load

# 3) run the SimplerEnv client against it
echo "=== [3] SimplerEnv rollout (visual_matching, $NT trials/task) ==="
export SAPIEN_RENDERER=vulkan
$SIM scripts/simpler/main.py --host 127.0.0.1 --port $PORT --task-set visual_matching --num-trials-per-task $NT
RC=$?
kill $SRV 2>/dev/null
echo "=== rollout done (rc=$RC) ARM=$ARM ==="
