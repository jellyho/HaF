# HaF preliminary experiments — Recoverability & shortcut-free auxiliaries

This directory holds the **preliminary study** behind HaF (Hindsight-and-Foresight). It validates the
central hypothesis *before* training a full VLA:

> Auxiliary objectives differ in **Recoverability** — how much of the target is already trivially
> recoverable from the current observation. Low-recoverability + task-relevant objectives are shortcut-free;
> co-training them regularizes behavior cloning (BC) away from **memorization** toward a joint
> language–vision–action model that generalizes.

Everything here is a **small controlled probe** (frozen or fine-tuned DINOv2, ~3–3.6k transitions/dataset),
deliberately *not* the full model — so every effect is legible. The scaled test is Exp 3 (see
`../../scripts/simpler/README.md`).

Public report (interactive): the `build_artifact.py` output, currently
<https://claude.ai/code/artifact/c6aa7158-a6e1-4ddb-b88f-25dadfa8afae>.

## Directory layout

```
experiments/
  RECOVERABILITY.md          master design/theory note (the whole argument)
  recoverability/
    README.md RESULTS.md SUMMARY.md EXP3_DESIGN.md
    extract/    stage1_{extract,bridge,fractal,libero}.py   RLDS -> transitions_{TAG}.npz
    measure/    stage2_metrics.py, exp1b_memorization.py, change_recovery.py   (Exp-1 data-property map)
    probes/     exp2b..exp2g_*.py + exp2_transfer.py + aggregate_{trimodal,exp2c,d,e,g}.py  (regularizer probes)
    conflict/   exp_conflict.py, aggregate_conflict.py, plot_conflict.py   (modality-conflict / LA4VLA-generalized)
    viz/        stage3_plot.py, plot_learnable.py, build_artifact.py
    slurm/      worker.sh, worker.sbatch, {exp2b,exp2c,exp2d,encode,conflict}.sbatch, launch_*.sh   [TRACKED code]
    outputs/                                          [GIT-IGNORED — data only]
      cache/    transitions_*.npz, dino_latents_*.npz, exp2g_reps_*.npz   (regenerable; ~14 G — safe to delete)
      (root)    *_analysis.json, exp2*/exp_conflict_* *.json, fig_*.{png,pdf}, *.log
```

**Git tracks only code** (`*.py`, `*.md`, `slurm/*`). `outputs/` is ignored — results and the heavy `cache/`
(regenerable npz) never get committed; `rm -rf outputs/cache` reclaims ~14 G. Every script uses
`OUT=.../recoverability/outputs` and reads/writes npz under `OUT/cache/`.

Run from the repo root (`cd /data5/jellyho/Hindsight/HaF`), e.g.
`sbatch --export=ALL,TAG=fractal,SEED=0 experiments/recoverability/slurm/exp2c.sbatch`. GPU work goes through
slurm to idle L40S (see the memory rule).

---

## Pipeline at a glance

```
 raw RLDS ──stage1──▶ transitions_{TAG}.npz ──stage2──▶ dino_latents_{TAG}.npz + results_{TAG}.json
 (per dataset)         (frames+vectors+instr)   (DINOv2 encode + Exp-1 recoverability metrics)
                                                             │
        ┌────────────────────────────────────────────────────┤
        ▼ (probes: train small model, evaluate OOD)           ▼ (data-property analyses, no policy)
   exp2b/2c/2d/2e/2f/2g_*.py ──▶ exp2*_{TAG}_s{SEED}.json   exp1b_memorization.py, change_recovery.py
        │
        ▼ aggregate_*.py ──▶ exp2*_analysis.json / *_agg.json ──▶ build_artifact.py ──▶ report.html
```

- **TAG** ∈ `{droid, bridge, fractal, libero_goal, libero_object, libero_spatial, libero_10}`
  (`fractal` = RT-1 / Google robot).
- All outputs land in `/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/`.

---

## Key concepts & metrics

| symbol | meaning | where |
|---|---|---|
| `R_triv` | shortcut availability = `1 − L(trivial copy)/L(marginal)`. High = target copyable from an input. | stage2 |
| `G_obs`, `probe_beyond_trivial` | learned-probe recoverability (baseline-free) and how much a probe beats the copy | stage2 |
| `r2_ood` | BC action generalization on held-out task clusters (OOD). Positive = generalizes. | exp2b–g |
| `contrib_lang` / `contrib_vision` | R² lost when language / vision is ablated = how much each modality *buys* | exp2b/c/d/f |
| `sensL` / `sensV` | action movement when instruction / image is swapped = language- / vision-*use* | exp2b/c/d/f |
| `instr_decod`, `task_retrieval`, CKA | representation-quality: is the task decodable / clustered / how far the aux moved the rep | exp2g |

**Two shortcut types (do not conflate):** ① *target redundancy* (R_triv — target copyable from an input;
foresight-obs has it, retrospective doesn't) vs ② *memorization* (policy maps scene→action, ignores
language). Exp 1 measures ①; Exp 1b/2b–g attack ②.

---

## Scripts

### Stage 1 — extraction (RLDS → transitions_{TAG}.npz)
Per-embodiment because each dataset has a different obs/action schema. Each stores 6 frames
(far/near past, current, near/far future, initial) + state/action vectors (`cart*`, `grip*`, `act*`,
`act_chunk` = 15-step chunk, `act_prev`) + `instr`, keyed by `ep_id` for GroupKFold.
Needs TF/tfds → run with the project `.venv` (JAX/TF env).

| script | dataset |
|---|---|
| `stage1_extract.py` | DROID (Franka) |
| `stage1_bridge.py` | Bridge (WidowX) |
| `stage1_fractal.py` | RT-1 / fractal20220817_data (Google robot) |
| `stage1_libero.py` | LIBERO suites (`SUITE=libero_{goal,object,spatial,10}`); globs `*-train.tfrecord-*` (openvla mislabels libero_10 shards as `liber_o10`) |

### Stage 2 — encode + Exp-1 recoverability map
`stage2_metrics.py` — DINOv2-encodes the frames (caches `dino_latents_{TAG}.npz` with
`z_pl,z_ps,zt,z_fs,z_fl,z0`) and computes `R_triv`/`G_obs`/`probe_beyond_trivial` for ~15 prediction
targets → `results_{TAG}.json`. **No policy trained.** Env: `TAG`, `DEV=cuda`. GPU via slurm (below).

### Exp 1b / change-recovery — data-side memorization enablers
- `exp1b_memorization.py` — probes shortcut ② enabler: "is the task readable off the scene" (vision→instruction).
- `change_recovery.py` — refines obs recoverability to *change* (o_t − o_0).

### Exp 2 — the regularizer probes (train a small model, evaluate OOD)
BC head = `[rep, state, language] → 15-step action chunk`; OOD split = KMeans on instruction embeddings,
hold out 3 task clusters. Each writes `exp2X_{TAG}_s{SEED}.json`.

| script | question |
|---|---|
| `exp2_transfer.py` | frozen adapter, in-distribution — **inconclusive** (kept for the record) |
| `exp2b_regularize.py` | **the core probe.** BC vs BC+aux (retro/fwd) vs **SG/KI** (stop-grad). Tri-modal readout (r2_ood, contrib/sens). |
| `exp2c_auxbattery.py` | **which auxiliary?** 7 targets differing in recoverability (retro-obs, farpast, dynamics, fwd-obs, farfwd, progress, pose0). |
| `exp2d_lambda.py` | **how strong?** aux-weight λ ∈ {0.25,0.5,1,2,4} dose-response for retro + mix. |
| `exp2e_retro.py` | **which retrospective form?** retro-obs / near-past / farpast / pose / retro-action / displacement (MSE) + semantic-retro (discrete **CE**). |
| `exp2f_fused.py` | **fused-backbone probe.** image+state+language fused in a shared transformer rep (more faithful VLA), BC/aux/KI act on the multimodal rep. |
| `exp2g_repquality.py` | **the missing middle.** instruction-decodability + same-task retrieval + silhouette + saves reps for cross-condition CKA. |

### Aggregation (JSON → analysis + printed tables)
| script | produces |
|---|---|
| `aggregate_trimodal.py` | `exp2b_trimodal_agg.json` — per-condition mean±std across seeds |
| `aggregate_exp2c.py` | `exp2c_analysis.json` — Axis A (aux recoverability→benefit) + Axis B (vision-legitimacy→benefit), Pearson/Spearman + p-values (scipy) |
| `aggregate_exp2d.py` | `exp2d_analysis.json` — λ dose-response curves, argmax, inverted-U check |
| `aggregate_exp2e.py` | `exp2e_analysis.json` — within-retro ranking + MSE-vs-CE form comparison |
| `aggregate_exp2g.py` | `exp2g_analysis.json` — rep-quality table, CKA-vs-BC, retro-vs-fwd, rep-quality→OOD correlation |

### Plotting / report
`stage3_plot.py`, `plot_learnable.py` (static plots), `build_artifact.py` (interactive themed HTML report;
embeds the aggregated numbers).

### Docs
`RESULTS.md` (full write-up), `SUMMARY.md` (short), `EXP3_DESIGN.md` (the scaled ablation design).

---

## Running (GPU = slurm only)

**Hard rule:** never grab a GPU directly — the login node is a shared B200 box running other jobs. Submit to
idle **L40S** nodes (`node01`/`node100`, partition `debug`). GPU work uses the isolated torch env
`/data5/jellyho/Hindsight/enc_venv` (torch 2.11+cu128, works on L40S & B200); extraction uses the project
`.venv`.

**Worker model (bundled + courteous).** Instead of one slurm job per run, a few *worker* jobs each process a
list of `exp|dataset|seed` items sequentially on **one** GPU, so we cap concurrent GPUs (≈6, leaving the rest
free) and yield while other users have pending jobs. Worker is idempotent (skips items whose JSON exists).

```bash
OUT=/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs
# build a worklist, split into chunks, submit N workers (files live in $OUT):
#   chunk_c.txt lines look like:  exp2c|libero_goal|3
for c in 0 1 2 3 4 5; do
  sbatch --export=ALL,CHUNK=$OUT/chunk_$c.txt --job-name=hafw$c $OUT/worker.sbatch
done
# worker.sh maps exp -> script: exp2b..exp2g, encode(stage2). See $OUT/worker.sh.
```
Single runs (debugging) still go through slurm, e.g. `sbatch --export=ALL,TAG=fractal,SEED=0 $OUT/exp2c.sbatch`.
Aggregators + `build_artifact.py` are CPU-only (safe to run on the login node with the `.venv`).

---

## Headline results (see the artifact for the full, current numbers)

- **BC alone memorizes on all datasets:** `sensL ≈ 0` (ignores language), `sensV ≈ 2` (vision look-up).
- **grad co-training builds the joint model:** raises language-use 10–30× while keeping vision → best OOD gain.
- **hard KI (stop-grad) over-corrects:** language-only, `sensV → 0` (loses vision) — soft > hard for joint grounding.
- **retro-obs is the safest auxiliary** (best mean, only one positive on all datasets).
- **vision-legitimacy law:** how legitimately vision carries the signal predicts aux benefit almost linearly
  (Pearson **r ≈ −0.93, p ≈ 0.007**, 6 datasets, 5 seeds) — the aux helps *iff* vision was a shortcut.
- **Honest limits:** the *objective-level* recoverability→OOD correlation is weak at probe scale
  (dataset effect dominates); the retrospective-vs-form and dose-response stories are direction/ordering, not
  exact magnitudes. Nailing the objective-level law is what Exp 3 (real VLA + SimplerEnv) is for.

**Theory / master design note:** `../RECOVERABILITY.md` — the precise definition of Recoverability, the two
levers (target-side / input-side JEPA), the objective taxonomy, the form decision (continuous embedding vs
discrete CE) and its empirical test, the retro-question WIRING SPEC, and what's solid vs what needs Exp 3.

Related memory: `haf-design-and-codebase-map`, `exp1-redundancy-results`, `gpu-use-slurm-only`,
`exp3-rt1-simpler-scaffold`.
