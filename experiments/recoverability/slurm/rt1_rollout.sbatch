#!/bin/bash
#SBATCH --job-name=rt1_roll
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
VENV=/data5/jellyho/Hindsight/HaF/.venv/bin/python
SIM=/data5/jellyho/Hindsight/simpler_venv/bin/python
NT=${NT:-5}
for ARM in 0 1; do
  PORT=$((8140+ARM))
  echo "=== serve RT-1 JAX ARM=$ARM :$PORT ==="
  env ARM=$ARM PORT=$PORT $VENV -u experiments/recoverability/probes/rt1_server_jax.py &
  SRV=$!
  sleep 110   # jax + params + MiniLM load + jit
  echo "=== SimplerEnv visual_matching ARM=$ARM ($NT trials/task) ==="
  export SAPIEN_RENDERER=vulkan
  $SIM scripts/simpler/main.py --host 127.0.0.1 --port $PORT --task-set visual_matching --num-trials-per-task $NT
  echo "=== rollout ARM=$ARM rc=$? ==="
  kill $SRV 2>/dev/null; sleep 5
done
echo "=== rt1_rollout done (ARM 0=BC, 1=AHA-near-l0.1) ==="
