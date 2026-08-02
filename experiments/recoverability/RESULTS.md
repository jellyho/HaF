# Exp 1 — Target Redundancy Map

**Question (A1/A2):** do prediction targets differ systematically in *shortcut availability*, and is the field's default (forward targets) a cheap shortcut while retrospective targets are not? Measured as a **property of the data — no policy trained.**

- `R_triv = 1 − L_trivial/L_marginal` — shortcut availability (how much a trivial rule already gets)
- `G_obs  = 1 − min(L_trivial,L_probe)/L_marginal` — total signal recoverable from o_t
- `probe_beyond_trivial = (L_trivial − L_probe)/L_marginal` — >0 iff a learned probe beats the cheat
- Encoders: frozen **DINOv2-base** (images), **MiniLM** (instruction). Probes: linear (Ridge) + MLP. **GroupKFold by episode** (no leakage; retrospective targets are per-episode constant).
- **Artifact (web report):** https://claude.ai/code/artifact/07b228cc-da05-4102-a57b-56353cfb85fe

## ⚠️ Methodology correction (applied)
The trivial "shortcut" rule may only copy an **actual VLA input** — image (obs targets) or a state dim (pose/gripper). The **action is an output, not an input**, so "copy the current action" is invalid. With a marginal baseline, **all action targets (BC / future-action / prev-action) sit at R_triv≈0 — action prediction has no copy shortcut; it is the task.** The copy shortcuts are **foresight observation** (copy current frame) and **gripper** (a state input). Tables below show the *original* run; where a row is `P2 future-action` / `Mact prev-action` / `BC` read **R_triv≈0.00** (corrected), not the old copy-action values. (Probe stays image-only = a conservative learnability estimate; adding state/language needs per-target leakage handling.)

## Main run — DROID subset, 500 episodes, 4,000 transitions, 15 targets
Sorted by R_triv (most → least shortcut-solvable):

| target | family | R_triv | G_obs | probe>triv |
|---|---|---|---|---|
| P3 future-gripper | future | **0.934** | 0.934 | 0.00 |
| Mact prev-action | past | **0.931** | 0.931 | 0.00 |
| P2 future-action | future | **0.925** | 0.925 | 0.00 |
| Mgrip prev-gripper | past | **0.924** | 0.924 | 0.00 |
| P1s future-obs k~5 | future | 0.875 | 0.875 | 0.00 |
| Mnear past-obs k~5 | past | 0.864 | 0.864 | 0.00 |
| Mfar past-obs k~45 | past | 0.602 | 0.602 | 0.00 |
| P1l future-obs k~45 | future | 0.572 | 0.572 | 0.00 |
| **R1 initial-obs** | past | **0.444** | 0.459 | **+0.016** |
| R3 instruction | past | 0.009 | 0.009 | 0.00 |
| Iinv act\|both-frames | present | 0.002 | 0.036 | +0.035 |
| BC action\|o_t | policy | 0.002 | 0.002 | 0.00 |
| I-gripper now | present | 0.001 | 0.091 | +0.090 |
| I-progress t/T | present | 0.000 | 0.071 | +0.071 |
| **R2 initial-pose** | past | **−0.664** | −0.587 | +0.077 |

## Findings
1. **The copy shortcut is foresight *observation* (and gripper), not action.** Near-future observation (0.88, copy the current frame) and gripper (0.93, a state input) are the trivially-satisfiable targets — the world-model foresight objectives. **Action prediction has no input to copy → R_triv≈0**; it is the deployment task, not a cheat. (Corrected: an earlier draft used a "copy current action" baseline and wrongly called forward-action the biggest shortcut.)
2. **Redundancy is symmetric in time, decaying with distance.** near-past (0.86) ≈ near-future (0.88); far-past (0.60) ≈ far-future (0.57); action/gripper past≈future≈0.93. Distance, not direction, drives non-redundancy. **Initial-obs (0.44) is the least-redundant observation target** — the retrospective anchor.
3. **Learnability emerges with scale.** At 99 episodes (droid_100) no probe beat the copy anywhere. At 500, **R1 initial-obs crosses over (+0.016)** and R2 initial-pose's image-probe beats copy (+0.08). The retrospective "hard-but-learnable" signal lifts off the diagonal as data grows.
4. **Introspective = extractable, not copyable.** gripper (G_obs 0.09), progress (0.07), inverse-dynamics (0.04): R_triv≈0 but recoverable from o_t. Positive controls.
5. **R3 instruction ≈ 0 on DROID:** image does not determine the instruction (diverse tabletop) — unlike LIBERO's vision→instruction shortcut. Cross-dataset signal.

## Falsification gate — PASSED
R1 non-redundancy is not a short-episode artifact: longer episodes are *less* redundant.
`R_triv(initial-obs): short 0.474, mid 0.472, long 0.368` ; `cos(z0,zt): 0.82→0.77`.

## Honest limits
- **G_obs is stringent for obs targets** — the probe must beat "copy o_t", near-optimal for scene-similar frames, so obs targets sit on the diagonal. Next refinement: measure recoverability of the *change* (o_t−o_0), not the whole frame.
- **Premise, not payoff.** Confirms targets differ in shortcut availability (retrospective escapes, forward doesn't). Whether shortcut-free targets yield better representations = Exp 2.

## Cross-dataset — DROID (Franka) vs Bridge (WidowX) vs RT-1 (Google robot)
Same pipeline, per-embodiment schema adapters (`stage1_bridge.py`, `stage1_fractal.py`). Encoded on **B200** via an isolated **cu128 torch venv** (`/data5/jellyho/Hindsight/enc_venv`) — slurm L40S runs were unreliable (NFS-slow startup hit the --time limit). Cross-dataset artifact: same URL above.

R_triv (shortcut availability) for key targets:
| target | DROID | Bridge | RT-1 |
|---|---|---|---|
| near-future obs (P1s) | +0.88 | +0.60 | +0.60 |
| near-past obs (Mnear) | +0.86 | +0.59 | +0.62 |
| far obs (P1l / Mfar) | ~+0.59 | ~+0.23 | ~+0.19 |
| **initial obs (R1)** | **+0.44** | **+0.20** | **+0.04** |
| next/prev action (P2/Mact) | **+0.93** | **−0.5** | **−0.55** |
| next gripper (P3) | +0.93 | +0.56 | +0.30 |
| instruction (R3) | +0.01 | +0.01 | +0.00 |

probe_beyond_trivial for initial-obs (R1) — learnability beyond the copy shortcut: **DROID +0.02 → Bridge +0.11 → RT-1 +0.57.**

**Robust across all 3 embodiments:**
- Obs redundancy is **symmetric in time** (near-past ≈ near-future) and **decays with distance**; **initial-obs is the least-redundant obs target everywhere.**
- **Initial-obs is the hard-but-learnable target** — the probe beats the copy, by a margin that grows with scene structure (RT-1 kitchen: +0.57, clearly off-diagonal into the "useful" quadrant). This is the retrospective anchor the difficulty axis predicted.
- Falsification gate holds (DROID & Bridge): longer episodes → initial-obs *less* redundant (Bridge long-episode R_triv goes negative −0.18).

**Dataset-dependent (important nuance):**
- **Action targets have no copy shortcut on any robot (R_triv≈0)** — corrected. (The earlier "action flips sign with control smoothness" observation came from the invalid copy-action baseline and is retracted as a *shortcut* claim; action autocorrelation is a data property, not a VLA-input shortcut.) The copy shortcut is foresight observation + gripper, consistently.
- **Vision→instruction shortcut is in the data, dataset-specifically.** DROID (diverse): image doesn't predict instruction (probe +0.00). RT-1 (consistent scenes): it does (+0.22) — mirroring the LIBERO "vision overrides language" finding. Same objective, cheat on one robot, signal on another.

Files: `results_{droid,bridge,fractal}.json`, `exp1_plane_{droid,bridge,fractal}.png`, `exp1_report.html`.

### Change-recoverability (resolves the y-axis / copy-baseline concern)
Instead of predicting the whole past frame (where "copy o_t" dominates), predict the **change** `z_past − z_t` from `z_t` (Ridge, GroupKFold). R² = variance of the change explained — a retrospective-learnability measure free of the copy baseline (`change_recovery.py`).

| dataset | near-past change (t−5) | far-past change (t−45) | initial change (t−0) |
|---|---|---|---|
| DROID | −0.23 | −0.11 | −0.00 |
| Bridge | −0.06 | +0.13 | +0.13 |
| RT-1 | +0.05 | +0.39 | **+0.50** |

**Monotone in both axes:** change-recoverability rises with (1) retrospective distance (near < far < initial) and (2) scene structure (DROID < Bridge < RT-1). The current frame carries more recoverable evidence of *what changed* the farther back you look — quantifying why the far-retrospective target is the useful, hard-but-learnable one. DROID's negatives = the change is not *linearly* recoverable there (needs a deeper model / more data), consistent with its flat probe-beyond-copy.

## Exp 1b — the BC *memorization* shortcut (a different shortcut than R_triv)
R_triv measures "is the target copyable from an input." The **BC memorization shortcut** is different: does the policy learn a lazy lookup and ignore language? Its data-side *enabler* = **is the task readable off the scene** (then language is optional). **Action = the 15-step chunk a VLA outputs** (a single step is trivially ≈ current pose; the chunk needs the whole near-trajectory). `exp1b_memorization.py`, R² held-out by episode:

| signal (R²) | DROID | Bridge | RT-1 |
|---|---|---|---|
| task readable from scene — instr←image (kNN) | **−0.05** | +0.19 | **+0.29** |
| action-chunk ← image+state (linear) | **+0.72** | −0.10 | −0.05 |
| language adds to the chunk (Δ_lang) | −0.05 | −0.10 | −0.00 |

- **Chunk is the honest target:** DROID action-chunk←image+state = **0.72, down from 0.90 for a single action** — a single step is mostly your current pose (proprioceptive redundancy); the 15-step chunk needs the trajectory. Still high on DROID (smooth control → the near-trajectory coasts from image+state), negative on Bridge/RT-1 (delta actions, high-entropy).
- **Language is ~redundant for the chunk (Δ_lang ≈ 0) on all three** — a policy can predict the near-trajectory from image+state without language → BC underweights language (shortcut-prone), even with the chunk.
- **Vision→task enabler, dataset-dependent:** task inferable from scene **strong RT-1 (+0.29), absent DROID (−0.05)** — consistent scenes (RT-1 / LIBERO) let a policy memorize scene→behavior and skip language.
- **Honest limit:** raw-action learnability from **frozen** DINOv2 is a *lower bound* (a trained end-to-end policy memorizes far more). Full BC shortcut needs **Exp 3** (train BC, counterfactual-instruction OOD test, LIBERO-CF style).

**Two shortcut types (don't conflate):** ① target-redundancy (R_triv; foresight-obs has it, retrospective doesn't) vs ② memorization/spurious scene→action (BC has it, needs a trained policy to see fully). Exp 1 measures ①; Exp 1b probes ②'s enabler; Exp 3 measures ② directly.

## Exp 2 — representation transfer (small, controlled) — INCONCLUSIVE for A3 at this scale
Per objective, train a small adapter on **frozen DINOv2** (o_t → 256-d rep) to predict its target on train episodes; freeze; linear-probe downstream readouts {action-chunk, future-gripper, progress, instruction} on **held-out episodes** at {25/50/100}% data. `exp2_transfer.py`. transfer@100 = mean held-out R² over readouts:

| rep (pretraining objective) | RT-1 | DROID |
|---|---|---|
| raw DINOv2 | +0.14 | −0.15 |
| forward-obs (predict future latent) | +0.19 | +0.04 |
| **retrospective (predict initial latent)** | +0.15 | +0.04 |
| BC (action-chunk) | **+0.29** | **+0.27** |
| mixed (retro+BC) | +0.29 | +0.27 |

- **Retrospective ≈ forward (no advantage); BC/mixed transfer best** on both datasets (partly circular — the action readout rewards BC-pretraining). By readout: obs-objectives (fwd/retro) help *instruction* transfer; BC/mixed help *progress*/*action*.
- **This does NOT refute A3 — the setup can't test it:** (1) frozen DINOv2 caps encoder reshaping (retrospective's benefit is *encoder shaping* → needs a trainable encoder); (2) in-distribution held-out ≠ the OOD/generalization the thesis predicts; (3) the mixing→path-of-least-resistance mechanism needs a shared, capacity-limited encoder trained end-to-end.
- **Takeaway:** the frozen-feature probe that was ideal for Exp 1's *data-property* claims is *insufficient* for Exp 2's *representation-quality* claim. The real test = **Exp 3** (trainable PaliGemma-3B encoder + objective mix + OOD eval). See `EXP3_DESIGN.md`.

## Exp 2b — the auxiliary as a REGULARIZER on BC (the corrected, positive result) ★
The right test of the thesis: not "is the aux representation good" (Exp 2) but **"does co-training BC with a shortcut-free aux make the resulting BC policy generalize OOD?"** Needs a **trainable** encoder (so the aux can shape it) + an **OOD** split (the thesis is about generalization, not in-distribution transfer). `exp2b_regularize.py`.

Setup: fine-tune **DINOv2-small** (o_t→rep); BC head = [rep, state, language]→15-step action chunk; aux head (rep only, co-trained) = predict initial-obs (retro) and/or future-obs (fwd) DINOv2 latent. **OOD split = hold out 3/8 instruction (task) clusters** (k-means) — test on *unseen tasks*. 30 epochs. Metric = OOD action-chunk R² (held-out task clusters).

OOD R² on held-out task clusters, **mean ± std over 3 seeds/splits** (`exp2b_agg.json`):

| condition | OOD R² (RT-1) | OOD R² (DROID) |
|---|---|---|
| BC-only | **−0.17 ± 0.07** | +0.02 ± 0.05 |
| BC + retrospective | +0.01 ± 0.02 | +0.10 ± 0.04 |
| BC + forward | −0.01 ± 0.04 | +0.10 ± 0.06 |
| **BC + retro + forward (mixed)** | **+0.04 ± 0.02** | **+0.17 ± 0.03** |

**Robust across 3 seeds × 2 datasets:**
1. **BC-only generalizes worst** — on RT-1 reliably **negative** (all 3 seeds; worse than predicting the mean on unseen tasks).
2. **Mixed (retro+forward) is best on both, with the lowest variance** — most robust.
3. **Ordering mixed > single aux > BC-only holds on both datasets.**
4. retro ≈ forward individually (both beat BC-only); **the win is from *combining* shortcut-free auxiliaries**, exactly the path-of-least-resistance / "more diverse shortcut-free objectives → more shortcuts removed" prediction. (So it's not "retrospective uniquely" at this scale — it's "shortcut-free auxiliaries as regularizers, and more is better.")

This is the **direct confirmation of A3/A4 at small scale** and validates the *regularizer* framing (not MAML): hard shortcut-free auxiliaries, co-trained on a shared trainable encoder, turn an overfitting BC policy into a generalizing one; mixing them helps most. (Caveat: small scale — DINOv2-small, ~3k train, 3/8 task clusters held out; OOD R² absolute values are small, but the sign/ordering are stable across seeds.) Scaling this = `EXP3_DESIGN.md`.

## Scale history
- droid_100 (99 ep, 990 tx, 7 targets): R_triv ordering confirmed; learnability axis flat (per-episode-constant targets data-starved). `results.json`, `exp1_plane.png`.
- DROID subset (500 ep, 4000 tx, 15 targets): above. `results_droid.json`, `exp1_plane_droid.png`, `exp1_report.html`.

## Env / how to run
- Encode on GPU via slurm **L40S (node01/node100)**: `srun -w node01 --gres=gpu:1 bash -lc 'DEV=cuda uv run --no-sync python experiments/recoverability/stage2_metrics.py'` (torch cu126 fails on B200/node200; CPU works but slow).
- Pipeline: `stage1_extract.py` (DROID subset → transitions_droid.npz) → `stage2_metrics.py` (encode+metrics → results_droid.json, cache dino_latents_droid.npz) → `stage3_plot.py` (plot + R1 gate) → `build_artifact.py` (web report).
- Artifacts dir: `/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs/`.

## Next
1. Cross-dataset robustness: **Bridge (WidowX)** and **RT-1 / fractal (Google robot)** — same pipeline, adapted schema. Does the ordering hold across embodiments/scenes?
2. Scale DROID episodes (1000+) as full-download shards land → tighten the G_obs (learnability) axis for retrospective.
3. Refine G_obs to change-recoverability (predict o_t−o_0) so the y-axis separates hard-but-learnable from noise for obs targets.
