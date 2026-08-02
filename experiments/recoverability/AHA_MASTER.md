# AHA — master index & honest status  (updated 2026-07-30)

**Paper:** *Regularizing VLA models by asking hard questions.*  **Method:** **AHA** — Asking Hard-question Auxiliaries
(co-train the policy with auxiliary objectives whose answers are HARD to recover from o_t → forces grounding
instead of shortcut → better generalization). "Hard" = quantified as **low recoverability** = low normalized
predictive **𝒱-information** under the policy's own function class (Xu 2020; Ethayarajh 2022).

**Through-line (the arc):** BC takes the most-recoverable shortcut → overfits. (1) We show HOW to measure
recoverability rightly; (2) a LAW: recoverability ↓ ⇒ generalization ↑; (3) a composition rule for mixes;
(4) how it interacts with Knowledge Insulation; (5) a survey unifying prior VLA aux-losses on one axis. Goal = ORAL.

---

## Results & status

| # | result | key numbers | scale | status |
|---|---|---|---|---|
| R1 | **Measurement matters**: recoverability predicts generalization ONLY under the policy function class + learning-dynamics; frozen/linear probe FLIPS the sign | linear +0.78/+0.69/+0.62 (WRONG) · policy-e2e −0.66/−0.64/−0.47 · AULC −0.75/−0.67/−0.63 · **speed@¼ −0.75/−0.72/−0.69 (best)** · selectivity ctrl ≈0 | 4k·8k·16k, 3 seeds, 18 obj | **SOLID** (3-scale replication, V-info-grounded) |
| R2 | **The LAW** recoverability↓⇒gen↑ (single objective) | per-objective R_deep vs gen, negative | 4k–16k | SOLID (same runs as R1) |
| R3 | **Composition law**: a mix generalizes as its LOWEST-recoverability (hardest) member | min r=−0.87/−0.82 > joint −0.85 > mean −0.81/−0.57; one hard Q rescues a shortcut mix | 4k, 3 seeds, 12 mixes | **PRELIMINARY-STRONG** (1 scale) |
| R3b | **Count SATURATES; diversity INERT** — stacking low-recov members: benefit 1→2 rises then plateaus (.035→.043→…→.045), and swapping the 2nd member across modalities (obs/action/pose) is ~flat → the lever is *presence of a hard member* + recoverability, NOT count or modality spread | count-sweep + diversity, 3 seeds | **PRELIMINARY** (strengthens R3 vs task-diversity MTL) |
| R4 | **KI**: hard-KI starves the action head ONLY with an action-UNRELATED aux; action-related (FAST-like) auxes survive | far-past-obs Δ−0.033 (starves) vs cur/fut-action Δ+0.024/+0.017 (help); joint τ=0 best/tied; late-release ≯ joint | 4k, 3 seeds | **PRELIMINARY** (explains π0.5 KI+FAST) |
| R4b | **FAST-lite measurement**: discrete K=16 action-token recoverability is sign-stable + 3.4× lower-variance than continuous R² (both say next-action = low-recov) → explains why FAST/π0.5 use *discrete* action tokens as a cleaner low-recov signal | disc −0.117±0.032 vs cont −0.020±0.110, 3 seeds | **PRELIMINARY** (measurement point solid; co-training half noisy — don't overclaim) |
| R5 | **NTP is a form, not a recov level** (π0.5 subtask = scene-determined = HIGH-recov, not a strong lever) | conceptual + survey placement | — | conceptual, honest |
| R6 | **state-infer measured HIGH-recov** (proprio readable from image) → not all masked objectives are levers | R_deep +0.72(8k)/+0.80(16k) | 8k/16k | measured, honest |
| R7 | **Survey**: ~55 prior VLA aux-losses collapse onto the recoverability axis; field clusters on the cheap side; retrospective lane empty; instr-infer = a new INSTANTIATION (not a clean novelty gap — cf. Hejna 2023) | qualitative | — | QUALITATIVE; novelty corrected below |

**M1 deepening — DONE (2026-08-01):** mixing-deep (count SATURATES, diversity INERT → R3b) + FAST-lite
(discrete = lower-variance recoverability estimate → R4b). Figure: `outputs/fig_deepen.{png,pdf}`.
**Still blocked on GPU contention (user's own RLT/humanoid jobs own the nodes):** mini-VLA SimplerEnv rollout
(closed-loop probe, code ready) + M2 full-VLA (needs B200 sign-off). Not competing for GPU with the user's active jobs.

---

## Honest assessment — strengths, risks, what to shore up

**Strengths.** The measurement contribution (R1) is genuinely novel and rigorous: the sign-flip is striking,
grounded in 𝒱-information, replicated across 3 scales, with a selectivity control. R3/R4 are clean, coherent
mechanism results that also *explain existing practice* (π0.5's KI+FAST).

**Risks / yellow flags (do not hide these).**
1. **Everything is a MINI-VLA proxy.** External validity to real VLAs (π0/π0.5) is UNPROVEN. **M2 (full-VLA +
   SimplerEnv closed-loop) is the make-or-break test** and has not run.
2a. **TINY DATA SUBSAMPLE.** The mini-VLA uses **2,000 of 87,212 RT-1 episodes (~2.3%)** and only N_T=8 frames/episode → **16k of ~700k transitions**. "Scale-up to 16k" is still a small subsample. Combined with the small model + ~500–2,400 grad-steps, the mini-VLA is a proxy in EVERY axis (model, data, compute) — this is the root of the near-zero absolute generalization. Relative recoverability rankings held across 4k/8k/16k (robust to subsample size), but absolute-competence / "matters at scale" claims need the full data (= full-VLA M2, which trains on full RT-1). Lever: raise MAX_EP + re-extract (stage1+stage2) for stronger mini stats (costly).
2. **Near-zero absolute generalization.** BC-only OOD action R² ≈ −0.07…0 — we measure small differences
   (~0.02–0.1) in a regime where the policy barely generalizes. Signal-vs-noise concern.
3. **Correlations soften at scale — RESOLVED (2026-07-30, `analyze_stats` + scale decomposition): NOT a law washout.** gen-spread GROWS with scale (std 0.011→0.020, range 0.041→0.084) and recov-spread is stable — so the axes do NOT compress. What softens is the ASYMPTOTIC measure's predictive power: with more data even 'hard' low-recov targets get fit asymptotically (mae-mask/future-obs R_deep −0.01→+0.13/0.16 at 16k), so asymptotic recoverability SATURATES and loses discrimination — while the learning-DYNAMICS measure (speed@¼) stays robust (−0.69, CI excl 0). → another point FOR 'measure recoverability as dynamics, not asymptotics.' (BC-only→0 at 16k: policy nearly generalizes; shortcut headroom shrinks but the aux effect's RANGE grows.)
4. **Offline OOD action R² ≠ closed-loop success.** The metric that matters (SimplerEnv success rate) is untested.
5. **Recoverability is scale/context-dependent** (more data ⇒ everything more recoverable). Frame as a
   *relative, measured-in-context* quantity, never absolute.
6. **Stats (now with CIs — `analyze_stats.py`).** Sign-flip robust all 3 scales (CI excl 0, p≤.018); **speed@¼ robust all scales** (p≤.003); BUT **asymptotic e2e loses significance at 16k** (CI[−0.90,+0.34], p=.048). Mixing: min robust, mean not (Spearman p=.057). CIs WIDE (N=12–18). Refinement: the DYNAMICS measure is the robust anchor; asymptotic washes out at scale.

**What to shore up (priority order).**
- **(P1) M2 — full-VLA closed-loop (SimplerEnv).** The critical external-validity test. Needs checkpoint saving,
  action/proprio-convention validation, a policy wrapper, B200 training. Highest leverage on oral-worthiness.
- **(P2) Does the lever survive at scale?** Re-examine why |r| softens 4k→16k; test whether a harder OOD split
  (where BC actually generalizes > 0) restores clean, large effect sizes.
- **(P3) Replicate R3/R4 at ≥2 scales + report CIs.** Currently single-scale.
- **(P4) FAST-lite** (running): does discretizing action prediction lower its recoverability? Ties KI+FAST↔AHA
  quantitatively — but report honestly if recoverability does NOT explain FAST's benefit.

**Verdict.** Progressing well as a clean mini-VLA science project with a strong, novel *measurement* core. The
mechanism story (mixing, KI) is coherent and confirmed at proxy scale. Oral-worthiness now hinges on M2 (scale +
closed-loop transfer) and on showing the effect doesn't wash out at scale. Treat R1 as the anchor; treat R3/R4 as
promising-but-preliminary; do not overclaim external validity until M2.

---

## Durable file index (repo — survives /tmp cleanup)
- `AHA_MASTER.md` (this) — index + honest status.
- `RECOVERABILITY_MEASUREMENT.md` — R1/R2 methods-comparison writeup (3-scale, 𝒱-info).
- `M1_RESULTS.md` — R3 composition law + R4 KI (mixing/KI).
- `SURVEY_vla_auxlosses.md` — R7 survey master table + boundary rulings.
- `M2_DESIGN.md` — full-VLA+SimplerEnv design + mini-rollout readiness. `PAPER_OUTLINE.md` — oral paper skeleton. `probes/analyze_stats.py` — bootstrap CIs.
- Code: `probes/{exp2h_law, exp2h_mixki, exp2h_fastlite, analyze_measures, plot_measures, plot_mixki}.py`.
- Data: `outputs/exp2h_{fractal,frac4k,frac8k}_s*.json`, `outputs/exp2h_mixki_*_s*.json`, `outputs/measure_comparison_*.json`.
- Figures: `outputs/fig_measures_{frac4k,frac8k,fractal}.{png,pdf}`, `outputs/fig_mixki.{png,pdf}`, `outputs/fig_law*.{png,pdf}`.
- Clean wandb: `jellyho_/aha-recoverability` (summary_frac4k/8k/fractal + measure_compare).
- Artifacts (claude.ai; note: /tmp sources were cleaned — rebuild dashboard from these durable files when needed):
  combined 6-tab hub last at artifact/5d115976-4941-415e-9238-0af859d27c31.
- Memory: MEMORY.md index → recoverability-measurement-study, ki-and-mixing-framing, oral-unifying-ambition,
  core-claim, llm-ntp-connection, project-naming, project-boundary-rlt-not-aha, exp3-rt1-simpler-scaffold.

*Note on older md files in this dir (RESULTS.md, SUMMARY.md, METHODS.md, EXP3_DESIGN.md, PROPOSAL_*.md): predate the
AHA rename / current results — treat AHA_MASTER.md as the current source of truth.*

---
## NOVELTY assessment (3 web-verified agents, 2026-07-30) — honest positioning
| claim | verdict | closest prior art / threat |
|---|---|---|
| **R1 measurement** (recoverability as a predictive aux-selection criterion; inverse law; frozen-probe SIGN-FLIP) | **PARTIALLY-NOVEL (strong)** | ancestors used descriptively: Xu2020/Ethayarajh2022 (𝒱-info = DATA difficulty, opposite framing), Voita-Titov/Blier-Ollivier (MDL=effort, no objective-selection). No prior work: selects aux by recoverability under the POLICY function class, states the inverse law, or the sign-flip. |
| **R3 composition** (mix ≈ its min-recoverability member; one hard Q rescues) | **NOVEL (specific form)** | worst-case/minimax MTL (aggregates LOSS for robustness ≠ our BENEFIT aggregation); task-diversity bounds (reward coverage). Not preempted. |
| **R4 KI × action-relatedness** | **PARTIAL** | POSITIVE half (action-related/FAST survives KI) = π0.5's design (preempted). NEGATIVE ablation (action-UNrelated aux starves under hard-KI ⇒ KI needs action-related grounding) = OURS (π0.5/GR00T never ran it). |
| **AHA method core** (regularize a VLA by low-recoverability auxiliaries) | **PARTIAL (novel synthesis)** | the RECOVERABILITY AXIS as a unifying, measurable, predictive design principle is the novel core. |

**CORRECTION (supersedes the survey's "instr-infer = novelty gap"):** instruction-prediction-as-auxiliary to force
grounding is ALREADY PUBLISHED — **Hejna, Abbeel, Pinto, AAAI 2023, "Improving Long-Horizon Imitation Through
Instruction Prediction" (arXiv 2306.12554)** [verify before citing]. Our `instr-infer` is a NEW INSTANTIATION
(within-VLA, masked-input, selected-by-recoverability), NOT a new idea. Do NOT claim instruction-inference itself as novel.

**Strongest defensible oral contribution:** the **recoverability axis as a single quantitative design principle** that
(a) UNIFIES the scattered zoo of prior VLA auxiliaries — explaining which help (retrospective, inverse-dynamics,
instruction-recovery) vs which quietly become the shortcut (future-frame prediction), (b) must be measured under the
policy's own function class + as learning dynamics (else the frozen probe FLIPS the sign — the sharpest ablation),
(c) yields a composition law (min-member dominates). Backed by survey (unification) + measurement (R1) + mixing (R3).

**Biggest prior-art threat (web-verified 2026-08-01, `CITATIONS_THREATS.md`) = dilution-by-ASSEMBLY, MODERATE — no direct pre-emption exists.** A skeptic assembles adjacent pieces: 𝒱-info/PVI measures *data* difficulty (Xu 2020/Ethayarajh 2022) + PI-as-aux (PI-QT-Opt arXiv:2210.08217) + aux-weighting-by-alignment (Lin NeurIPS 2019; ITTS NeurIPS 2020; AANG; SLGrad) + future-pred regularizes BC (BYOL-γ arXiv:2506.10137) ⇒ "recoverability is just a repackage." **Defuse explicitly:** they measure *data* difficulty (not select auxes), weight by *main-task alignment* (not target recoverability under the policy class), and treat auxes as *helpers to MAXIMIZE* — AHA's thesis is that *low* recoverability is the desirable property, plus the probe-flip + composition law none of them have. Secondary/shared-motivation line = Hejna 2023 (instruction-pred aux, MINOR) + causal-confusion (de Haan 2019 / OREO NeurIPS 2021 / GABRIL IROS 2025, MINOR). Foreground the SELECTION principle + measurement content; NOT "we add an aux loss."

*Foundational cites VERIFIED: Xu 2020 (arXiv:2002.10689), Ethayarajh 2022 (2110.08420), FAST (2501.09747), π0.5-KI (2505.23705), Hejna 2023 (2306.12554). Fix before submit: cite Voita-Titov via ACL Anthology (arXiv id unverified). GR-1/2 (future-frame prediction) is a FOIL (= high-recoverability shortcut), not a precedent.*
