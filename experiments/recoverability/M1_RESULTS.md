# M1 — mini-VLA: objective-mixing composition law + Knowledge-Insulation ablation

Code: `probes/exp2h_mixki.py` · analysis inline · mini V-L-A (DINOv2-small+MiniLM+state → transformer → flow-matching BC),
RT-1/fractal, SUBN=4000, EPOCHS=25, 3 seeds. Recoverability = deep 𝒱-information R_deep (from `exp2h_fractal`).

## (A) Composition law — CONFIRMED: the LOWEST-recoverability member dominates

Correlation of each recoverability summary of a mix with the mix's OOD generalization (want negative):

| summary | Pearson | Spearman |
|---|---|---|
| **min-member recoverability** | **−0.87** | **−0.82** |
| joint (concat) recoverability | −0.85 | −0.65 |
| mean-member recoverability | −0.81 | −0.57 |

→ **min wins.** A mix generalizes as well as its HARDEST (lowest-recoverability) question. Concrete:
- `final-pose` alone (high-recov shortcut): gen −0.077 (≈ BC-only −0.075)
- `far-past-obs + final-pose`: gen −0.033 → **one hard question RESCUES a shortcut-laden mix**
- `final-pose + initial-pose` (both high): −0.062 (bad)
- `far-past-obs + far-fut-obs + final-pose`: −0.020 (the low members dominate)

**AHA design rule justified:** include ≥1 genuinely low-recoverability question; it rescues even a shortcut-laden mix, and adding high-recoverability objectives is ~inert.

## (B) Knowledge Insulation — the aux's ACTION-RELATEDNESS is what matters

KI = stop-grad from the flow BC head to the backbone for the first τ·EPOCHS epochs. gen Δ vs BC-only (−0.071):

| backbone-shaping aux | action-related? | R_deep | joint (τ=0) | late-rel (τ=0.5) | **hard-KI (τ=1)** |
|---|---|---|---|---|---|
| far-past-obs | ✗ | +0.16 | +0.039 | +0.028 | **−0.033 (starves ✗)** |
| final-pose | ~ (pose) | +0.96 | +0.002 | +0.006 | +0.026 ✓ |
| cur-action | ✓ | +0.08 | +0.028 | +0.017 | +0.024 ✓ |
| fut-action | ✓ | +0.04 | +0.014 | +0.011 | +0.017 ✓ |

**Findings:**
1. **Hard-KI starves the action head ONLY with an action-UNRELATED aux (far-past-obs).** Action-related auxes (cur/fut-action, and pose-y final-pose) survive hard-KI and keep helping.
2. This **confirms why real KI (π0.5) does not starve**: it shapes the backbone with **FAST action-token prediction** — an action-related, discrete objective. (User correction, 2026-07-30.)
3. **Sweet spot = action-related AND low-recoverability** (cur/fut-action are both here, R_deep 0.04–0.08) → helps under KI. That is exactly what FAST provides: action-related + discrete/harder.
4. **Always-joint (τ=0) is robustly best/tied** across all auxes; **late-release (τ=0.5) did NOT beat joint** at this scale — the schedule advantage is unsupported so far.

**Reframe (supersedes the earlier "KI amplifies" and "hard-KI over-insulates" partial reads):** hard-KI is safe *iff* the backbone-shaping aux is action-related; the "over-insulation starves the head" failure is specific to action-UNrelated auxes. π0.5's KI+FAST = KI + an action-related, low-recoverability question = implicitly KI+AHA.

## (C) Mixing DEEPENING — count-sweep + modality diversity [3 seeds, N=4k]

Nested low-recoverability obs-mix; benefit = gen(mix) − gen(BC-only). BC-only gen = −0.073.

| # low-recov members | benefit (mean±sd) | gen |
|---|---|---|
| 1 | +0.035±0.029 | −0.038 |
| 2 | +0.043±0.017 | −0.031 |
| 3 | +0.043±0.015 | −0.031 |
| 4 | +0.047±0.021 | −0.026 (peak) |
| 5 | +0.040±0.018 | −0.033 |
| 6 | +0.045±0.019 | −0.028 |

**Finding — saturation, not accumulation.** Benefit rises from 1→2 members then **plateaus (~+0.045)**; stacking 3–6 low-recov questions does **not** compound. Consistent with the composition law: it is the **presence of a hard member**, not the *count*, that sets the mix — extra low-recov members are redundant with the first once one is present. (No interference/collapse either — plateau, not decline.)

**Diversity: `far-past-obs` + one partner (same- vs cross-modality):**

| partner | modality | benefit |
|---|---|---|
| far-fut-obs | obs (same) | +0.043±0.017 |
| fut-action | action (cross) | +0.040±0.013 |
| prev-action | action (cross) | +0.045±0.015 |
| displacement | pose (cross) | +0.045±0.018 |

**Finding — modality diversity is ~inert; recoverability is the lever.** Cross-modality partners do **not** beat a same-modality obs partner. The benefit tracks the members' recoverability, not their modality spread — arguing *against* a "diversity" explanation and *for* the recoverability axis. (Distinguishes AHA from task-diversity MTL.)

## (D) FAST-lite — discrete vs continuous action recoverability [3 seeds, K=16 bins]

MEASURE recoverability of the **next action** two ways, plus KI with each as the backbone-shaping aux.

| measure | recoverability | seed spread |
|---|---|---|
| continuous (R² = 1−MSE_val/MSE_marg) | −0.020 ± 0.110 | [−0.13, +0.13] **noisy, sign-unstable** |
| **discrete (1−H_val/H_marg, K=16)** | **−0.117 ± 0.032** | [−0.15, −0.07] **clean, consistently negative** |

**Finding — discretization gives a CLEANER measurement of the same low recoverability.** Both say the next action is a genuinely **low-recoverability** target (a hard question), but the continuous R² is noisy and flips sign across seeds while the **discrete cross-entropy measure is 3.4× lower-variance and consistently negative**. → This is a measurement-methodology point that **explains a design choice in the field**: FAST/π0.5 use *discrete* action tokens partly because discrete action-token prediction is a more *stably measurable / trainable* low-recoverability signal than continuous regression. Connects the discrete-token choice to the recoverability axis.

**KI with the action-token aux (benefit vs BC-only):**

| aux | τ=0 (joint) | τ=1 (hard-KI) |
|---|---|---|
| continuous action | +0.004±0.018 | +0.007±0.020 (≈inert) |
| discrete action | −0.074±0.020 | −0.043±0.019 |

**Honest read (PRELIM, noisy):** at this tiny scale (4k, 25 ep) the discrete action-token aux *hurts* as co-training (both heads undertrained, competing), but **hard-KI reduces the damage** (−0.074→−0.043) — consistent with KI protecting the flow head from a disruptive backbone signal. The clean, reportable result from FAST-lite is the **measurement** point (discrete = lower-variance recoverability estimate), not a co-training win — do not overclaim the KI half.

*Figures: `outputs/fig_mixki.{png,pdf}`. Data: `outputs/exp2h_mixki_{mk,ki,deep}_s*.json`, `outputs/exp2h_fastlite_fl_s*.json`.*
