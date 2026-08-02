# Exp 3 — HaF objective-mix ablation on RT-1, evaluated in SimplerEnv

Scales the small DINOv2 probes (Exp 2b–2f) up to the real VLA: a **pi05 recipe trained from a raw
PaliGemma checkpoint** on **RT-1 (fractal20220817_data)**, evaluated **closed-loop in SimplerEnv**
(`google_robot`). Tests the thesis at scale: does adding a shortcut-free auxiliary objective to
behavior cloning improve generalization (SimplerEnv success rate) over BC alone?

## The ablation (configs registered in `src/haf/training/config.py`)

KI (`stop_action_to_vlm_grad`) is the **standard LAP/pi05 recipe**, so every aux-bearing arm keeps KI **on**
(backbone shaped by the aux, action head insulated); only the pure-BC baseline turns it off. One soft (no-KI)
mix arm tests the probe finding that soft co-training preserves vision-use better than hard KI.

| config | objectives | KI | role |
|---|---|---|---|
| `exp3_rt1_bc`       | action only             | off | baseline memorizer (action shapes the whole VLM) |
| `exp3_rt1_langact`  | action + langact        | **on** | LAP standard (backbone = language, action insulated) |
| `exp3_rt1_pred`     | action + prediction     | **on** | world-model aux only shapes the backbone |
| `exp3_rt1_mix`      | action + langact + pred | **on** | full HaF mix — DEFAULT KI recipe |
| `exp3_rt1_mix_soft` | action + langact + pred | off | soft co-training — hard-KI vs soft (probe finding) |

All five:
- initialize from **raw PaliGemma** (`weight_loader` default `kind="paligemma"` → `paligemma/pt_224.npz`) —
  *not* a pi0/pi05 pretrained checkpoint;
- use the **pi05** model recipe (`pi05=True`, adaRMS, flow action head), `action_dim=7`, `action_horizon=16`;
- train on **RT-1 only** (`data.data_mix="fractal20220817_data"`).

`batch_size=256`, `num_train_steps=30_000` are modest defaults — tune to your hardware (the stock configs use
2048). Prefer the **B200 node via slurm** (see the `gpu-use-slurm-only` rule); do not grab GPUs directly.

## Run it

```bash
# 1) train (one per ablation arm) — submit via sbatch on the B200 node
python scripts/train.py exp3_rt1_bc   --exp_name a1
python scripts/train.py exp3_rt1_mix  --exp_name d1
# ...

# 2) serve a trained checkpoint (WebSocket policy server on :8000)
python scripts/serve_policy.py \
  --policy.config=exp3_rt1_mix \
  --policy.dir=checkpoints/exp3_rt1_mix/d1/30000 \
  --policy.type=flow

# 3) evaluate in SimplerEnv (separate process; connects to the server)
python scripts/simpler/main.py --task-set visual_matching --num-trials-per-task 25
```

Compare `overall_success_rate` across arms. Thesis prediction: `mix > langact ≈ pred > bc`; the KI arm
probes whether hard stop-gradient helps or (as in the probes) trades vision-use for language-use.

## Requirements & validation notes

- **SimplerEnv**: install from https://github.com/simpler-env/SimplerEnv (SimplerEnv + ManiSkill2). Not a
  repo dependency yet — install in the eval env only.
- `scripts/simpler/main.py` mirrors `scripts/libero/main.py` (WebSocket client → `serve_policy.py`). Three
  sim-specific spots are marked **ASSUMPTION** and must be checked on the first smoke run
  (`--task-set smoke`):
  1. `simpler_obs_to_state` — the proprio→state mapping (tcp pose keys, quaternion order) must match
     `rt1_dataset_transform` (`src/haf/datasets/utils/transforms.py:288`).
  2. `rt1_action_to_simpler` — gripper sign/scale convention.
  3. request image key `base_0_rgb` and `state` must match what the served config's input transform expects.
- If success rate is ~0 on the smoke task, the culprit is almost always the gripper convention or the
  state mapping — fix those two functions first.

## Where this connects

- Probe evidence (Exp 2b–2f) lives in `experiments/exp1_redundancy/` and the artifact report.
- Objective-mix flags: `src/haf/models/haf_config.py` (`enable_{action,langact,prediction,vqa}_training`,
  `*_loss_weight`, `stop_action_to_vlm_grad`), consumed in `src/haf/models/haf.py:compute_loss`.
