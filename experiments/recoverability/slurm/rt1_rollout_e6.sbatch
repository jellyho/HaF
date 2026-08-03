#!/bin/bash
#SBATCH --job-name=roll_e6
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
NT=${NT:-8}
for ARM in 0 1; do
  PORT=$((8160+ARM))
  echo "=== serve RT-1 JAX (e6) ARM=$ARM :$PORT ==="
  env CKPT=_e6 ARM=$ARM PORT=$PORT $VENV -u experiments/recoverability/probes/rt1_server_jax.py &
  SRV=$!; sleep 110
  export SAPIEN_RENDERER=vulkan
  echo "=== SimplerEnv visual_matching (e6) ARM=$ARM ($NT/task) ==="
  $SIM scripts/simpler/main.py --args.host 127.0.0.1 --args.port $PORT --args.task-set visual_matching --args.num-trials-per-task $NT
  echo "=== roll_e6 ARM=$ARM rc=$? ==="; kill $SRV 2>/dev/null; sleep 5
done
echo "=== roll_e6 done (competent BC vs AHA) ==="
