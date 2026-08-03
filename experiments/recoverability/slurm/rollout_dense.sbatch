#!/bin/bash
#SBATCH --job-name=roll_dns
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/slurm_%x_%j.log
export HF_HOME=/data5/jellyho/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /data5/jellyho/Hindsight/HaF
ARM=${ARM:-bc}; PORT=${PORT:-8123}; NT=${NT:-5}; MAX_EP=${MAX_EP:-20000}; EPOCHS=${EPOCHS:-10}
VENV=/data5/jellyho/Hindsight/HaF/.venv/bin/python     # cv2+tf+torch+transformers (train)
ENC=/data5/jellyho/Hindsight/enc_venv/bin/python       # server
SIM=/data5/jellyho/Hindsight/simpler_venv/bin/python   # SimplerEnv client
O=experiments/recoverability/outputs

# 1) train the trainable-encoder dense policy if missing
if [ ! -f "$O/policy_${ARM}.pt" ]; then
  echo "=== [1] train dense policy ARM=$ARM MAX_EP=$MAX_EP EPOCHS=$EPOCHS ==="
  env ARM=$ARM MAX_EP=$MAX_EP EPOCHS=$EPOCHS $VENV -u experiments/recoverability/probes/save_policy_dense.py || exit 1
fi

# 2) start the torch policy server (trainable DINO-small; existing server is compatible)
echo "=== [2] start mini-policy server ARM=$ARM :$PORT ==="
env ARM=$ARM PORT=$PORT $ENC -u experiments/recoverability/probes/mini_policy_server.py &
SRV=$!
sleep 75

# 3) SimplerEnv rollout
echo "=== [3] SimplerEnv rollout (visual_matching, $NT trials/task) ARM=$ARM ==="
export SAPIEN_RENDERER=vulkan
$SIM scripts/simpler/main.py --args.host 127.0.0.1 --args.port $PORT --args.task-set visual_matching --args.num-trials-per-task $NT
RC=$?
kill $SRV 2>/dev/null
echo "=== rollout_dense done (rc=$RC) ARM=$ARM ==="
