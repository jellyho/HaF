# METHODS — technical, reproduction-level

Exactly what each experiment does: data, models, hyper-parameters, splits, metric formulas, procedure. Numbers
are the defaults in the code (env vars override). Paths are relative to the repo root
`/data5/jellyho/Hindsight/HaF`; run everything from there. Outputs (incl. the npz caches) go to
`experiments/recoverability/outputs/` (`OUT`), npz caches under `OUT/cache/`.

## 0. Environments

| use | interpreter | why |
|---|---|---|
| extraction (stage1) | `.venv/bin/python` (uv, JAX/TF) | needs `tensorflow` + `tensorflow_datasets` to read RLDS |
| GPU probes/encode (stage2, exp2*, conflict) | `/data5/jellyho/Hindsight/enc_venv/bin/python` (torch 2.11+cu128) | Blackwell/L40S kernels; `transformers`, `sklearn` |
| aggregation / plots | `.venv/bin/python` | CPU only; `scipy`, `matplotlib` |

GPU jobs go through **slurm** to idle L40S (`node01/node100`, partition `debug`); launchers in `slurm/`. Set
`HF_HOME=/data5/jellyho/.cache/huggingface`. Common env vars: `TAG` (dataset), `SEED`, `EPOCHS`, `DEV=cuda`.

Datasets (RLDS, local): RT-1 = `fractal20220817_data`, DROID, Bridge, LIBERO suites
(`libero_{goal,object,spatial,10}`, openvla/modified_libero_rlds). `TAG` ∈ these.

---

## 1. Exp 1 — Recoverability map (`extract/` → `measure/stage2_metrics.py`)

**Goal.** For ~15 candidate prediction targets, measure how recoverable each is from the current observation
`o_t` — *no policy trained*. Establishes which targets are shortcut-solvable.

### 1a. Extraction (`extract/stage1_<dataset>.py`, run per dataset)
Reads the RLDS shards, iterates the first `MAX_EP` episodes (default 500; skip episodes shorter than
`MIN_LEN=30`). Per episode of length `T`, pick `N_T=8` timesteps evenly spaced as
`t = unique(linspace(3, T−3, 8))`. At each `t` store, all images **resized to 224×224** (INTER_AREA):

- **6 frames** (+1 new): `Fpl`=far-past `o_{t−45}`, `Fps`=near-past `o_{t−5}`, `Ft`=current `o_t`,
  `Ffs`=near-future `o_{t+5}`, `Ffl`=far-future `o_{t+45}`, `F0`=episode-initial `o_0`,
  **`Flast`=episode-final `o_{T−1}`** (added for the fair start↔end control). Horizons are clamped to `[0,T−1]`.
  Constants: `K_SMALL=5, K_LARGE=45` (RT-1 uses 5/30).
- **vectors**: `cart0`/`cart_last`/`cartt` (initial/final/current EEF pose), `gript`, `actt`, `act_fut` (action at
  `t+5`), `grip_fut`, `act_prev` (action at `t−5`), `grip_prev`, `act_chunk` (the next 15 actions, 15×7),
  `progress`=`t/(T−1)`, `ep_id`, `t`, `T`, `instr` (instruction string).

Writes `OUT/cache/transitions_<TAG>.npz` (frames uint8, vectors float32, instr object). ~3–3.6k transitions/dataset.
Run: `SUITE=libero_goal MAX_EP=600 .venv/bin/python experiments/recoverability/extract/stage1_libero.py`.

### 1b. Encode + measure (`measure/stage2_metrics.py`, GPU)
- **Frame encoder**: `facebook/dinov2-base` (768-d), ImageNet-normalized, CLS token; encodes every frame key
  `{Fpl,Fps,Ft,Ffs,Ffl,F0,Flast}` → latents `{z_pl,z_ps,zt,z_fs,z_fl,z0,z_last}`, cached to
  `OUT/cache/dino_latents_<TAG>.npz`; each latent **L2-normalized**. **Language**: `all-MiniLM-L6-v2`,
  mean-pooled over tokens, L2-normalized.
- For each target `Y` (below), with input `X=zt` (current-frame latent) unless noted, compute three losses (MSE):
  - `L_marginal` = predict the training mean (knows nothing).
  - `L_trivial` = the **copy baseline**: copy an *actual input* — for obs targets copy `zt`; for pose/gripper copy
    the current pose/gripper. (The copy rule may only use a real VLA input; the *action is an output*, so action
    targets have no copy rule → `L_trivial=L_marginal`.)
  - `L_probe` = a probe fit from `X` → `Y` (linear ridge; an MLP is also fit and the min taken), evaluated on a
    held-out split.
- **Metrics** (per target): `R_triv = 1 − L_trivial/L_marginal` (shortcut availability);
  `G_obs = 1 − min(L_trivial,L_probe)/L_marginal` (total signal recoverable from `o_t`);
  `probe_beyond_trivial = (L_trivial − L_probe)/L_marginal` (the hard-but-learnable band).
- **Targets** (`family`): retrospective `{R1 initial-obs z0, R2 initial-pose cart0, far/near past-obs, prev-action,
  prev-gripper}`; prospective `{P0 final-obs z_last, P0p final-pose cart_last, near/far future-obs, future-action,
  future-gripper}`; introspective `{gripper-now, progress t/T, instruction}`; policy `{BC action|o_t → 15-step
  chunk}`. **Note:** `P0 final-obs` / `P0p final-pose` are the symmetric prospective anchors added so
  `initial-obs`/`initial-pose` have a fair start↔end control.

Writes `OUT/results_<TAG>.json`. Run: `TAG=fractal DEV=cuda enc_venv/bin/python .../measure/stage2_metrics.py`
(or `sbatch slurm/encode.sbatch`). Delete `OUT/cache/dino_latents_<TAG>.npz` to force a re-encode.

### 1c. Exp 1b — memorization enabler (`measure/exp1b_memorization.py`)
Probes shortcut ② (does the scene reveal the task): fit `zt → instruction embedding` and report
`probe_beyond_trivial`. Strong RT-1, absent DROID. `change_recovery.py` refines obs recoverability to the *change*
`o_t − o_0`.

---

## 2. Exp 2 — regularizer probes (`probes/`)

Shared setup unless noted. **Encoder**: trainable `facebook/dinov2-small` (384-d CLS). **BC head**:
`Linear(384 + state_dim + lang_dim, 512) → GELU → Linear(512, 105)` predicting the **z-scored 15×7 action chunk**
(105-d). **state** = z-scored `[cartt, gript]`; **language** = MiniLM embedding, L2-normalized. **Aux heads** (co-
trained, read the encoder rep only): `Linear(384, 768)` predicting an L2-normalized DINOv2-base target latent.
**Optimizer** AdamW `lr=3e-5, weight_decay=1e-2`, batch 128, `EPOCHS=30`, MSE losses summed. Images ImageNet-
normalized on the fly. Runs are `TAG × SEED`; 5 seeds.

**OOD split** (the key rigor): `KMeans(K=8, n_init=5, random_state=SEED)` on the **per-episode mean instruction
embedding**; the **3 smallest clusters are held out** as the OOD test (unseen task clusters); the rest train.
`GroupKFold`-style — whole episodes stay on one side (retrospective targets are per-episode constant → prevents
leakage).

**Read-outs** (on OOD test): `r2_ood = 1 − MSE(pred,Y)/Var(Y)`; modality **contribution** `contrib_lang =
r2_ood − r2(lang zeroed)`, `contrib_vision = r2_ood − r2(rep replaced by its batch mean)`; modality **sensitivity**
`sensL = ‖Δaction when the instruction is shuffled across the batch‖² / Var(pred)`, `sensV` = same for the image
rep. High sens = the action uses that modality; ≈0 = ignored.

| script | what it adds | key knob / conditions |
|---|---|---|
| `exp2b_regularize.py` | core probe + tri-modal read-out + KI variant | conditions: BC-only, BC+retro, BC+fwd, BC+retro+fwd, and **SG** (KI: `r_bc=r.detach()` — BC gradient blocked from the encoder; encoder shaped by aux only) × the same aux sets. aux targets: retro=`z0`, fwd=`z_fs`. |
| `exp2c_auxbattery.py` | which auxiliary | 7 single-aux targets differing in Exp-1 recoverability: retro-obs`z0`, farpast`z_pl`, dynamics`z_fs−zt`, fwd-obs`z_fs`, farfwd`z_fl`, progress`t/T`, pose0`cart0`. |
| `exp2d_lambda.py` | how strong | aux-weight `λ ∈ {0.25,0.5,1,2,4}` for retro-obs and mix(retro+fwd): `loss = BC + λ·Σaux`. |
| `exp2e_retro.py` | within-retro target & FORM | targets: retro-obs, near/far-past-obs, retro-pose`cart0`, retro-action`act_prev`, displacement`zt−z0` (all MSE) + **semantic-retro** = CrossEntropy to a `KMeans(32)` code of `z0` (discrete-CE form). |
| `exp2f_fused.py` | faithful VLA backbone | image+state+language projected to `d=256`, fused by a 2-layer `TransformerEncoder` (4 heads) → pooled `r_fused`; BC/aux/KI all act on `r_fused`. Read-outs recomputed by swapping the pre-fusion inputs. |
| `exp2g_repquality.py` | representation quality | after training, on the OOD rep: `instr_decod` (5-fold ridge `rep→lang` R²), `task_retrieval@10` (same-instruction rate among 10 nearest neighbours), `silhouette` (by held-out cluster); saves the rep matrix (`OUT/cache/exp2g_reps_*.npz`) for cross-condition **linear CKA**. Conditions: BC-only, BC+retro, BC+fwd, SG+retro. |

Aggregators (`probes/aggregate_*.py`, CPU): mean±std over seeds, correlations with scipy p-values, λ curves, retro
ranking + MSE-vs-CE, CKA + rep-quality→OOD correlation. Run each experiment via `slurm/exp2{b,c,d}.sbatch` or the
`slurm/worker.sh` bundle (one GPU processes a chunk of `exp|TAG|SEED` items sequentially, ≤6 GPUs, yields to others).

---

## 3. Conflict — the general ignoring axis (`conflict/exp_conflict.py`)

**Fully synthetic, no RLDS.** Task: predict a 2-D unit action = one of 8 directions `θ` (`DIRS = arange(8)·π/4`).
Three inputs each independently encode a direction:
- **vision**: a 224×224 image with a Gaussian blob at angular position `θ_v` on a ring (radius/σ/brightness/centre
  jittered so it can't be memorized); encoded by trainable DINOv2-small.
- **language**: `"move {east|northeast|…}"` at `θ_l`; MiniLM embedding (8-phrase vocab, L2-normalized).
- **state**: `[cos θ_s, sin θ_s, 0…]` (8-d) + `N(0,0.05)` noise.

**Reliability knobs** `(p_v,p_l,p_s)` = `RELIAB[REGIME]`: with prob `p` a cue equals the true `θ`, else an
independent random direction (noise). Regimes: `V=(1,0,0)`, `L=(0,1,0)`, `S=(0,0,1)`, `redundant=(1,1,1)`.

**Model/training**: `Model` = DINOv2-small encoder + `Linear(384+lang+8, 256)→GELU→Linear(256,2)`; AdamW
`lr=3e-5, wd=1e-2`, batch 128, `EPOCHS=25`, MSE to the unit-vector action. `N_TRAIN=4000, N_TEST=1600`, 3 seeds.

**Diagnosis** (per trained regime, on a fresh test set): the aligned baseline sets all cues to `θ`. For each
modality `m`, the **conflict** sets `cue_m = θ+π` (opposite) and keeps the others at `θ`; predict the action and
compute `DCS(a, c) = mean cosine(â, [cos c, sin c])`:
- `follow_conflicted = DCS(action, θ+π)` — high ⇒ the policy **exploits** `m` (dragged to the flipped cue);
  ≈0/negative ⇒ **ignores** `m`.
- `follow_rest = DCS(action, θ)`.
**Recoverability(action|cue)**: a ridge probe from each single cue's feature → action, reported as `DCS` on test
(vision cue via frozen DINOv2 CLS features). This should predict which modality is exploited.

Output `OUT/exp_conflict_<REGIME>_s<SEED>.json`. Run `sbatch --export=ALL,REGIME=V slurm/conflict.sbatch` (each job
= one regime × 3 seeds). Aggregate + figure: `conflict/aggregate_conflict.py`, `conflict/plot_conflict.py`
(→ `OUT/fig_conflict.{pdf,png}`).

---

## 4. Reproduce end-to-end (one dataset)

```bash
cd /data5/jellyho/Hindsight/HaF
# 1. extract (CPU/TF)
SUITE=libero_goal .venv/bin/python experiments/recoverability/extract/stage1_libero.py
# 2. Exp-1 encode + recoverability map (GPU via slurm)
sbatch --export=ALL,TAG=libero_goal experiments/recoverability/slurm/encode.sbatch
# 3. a probe (GPU)
sbatch --export=ALL,TAG=libero_goal,SEED=0 experiments/recoverability/slurm/exp2c.sbatch
# 4. conflict (synthetic, GPU) + figure
for R in V L S redundant; do sbatch --export=ALL,REGIME=$R experiments/recoverability/slurm/conflict.sbatch; done
.venv/bin/python experiments/recoverability/conflict/aggregate_conflict.py
.venv/bin/python experiments/recoverability/conflict/plot_conflict.py
```

Theory/interpretation: `../RECOVERABILITY.md`. Results write-up + numbers: `RESULTS.md` and the artifact.
