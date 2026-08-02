# Paper outline — "Regularizing VLA models by asking hard questions" (AHA)  [draft 2026-07-30]

**One-line contribution.** A single, measurable axis — **recoverability** (usable information of a target under the
policy's own function class) — that (i) unifies the scattered zoo of VLA auxiliary objectives, (ii) predicts which
regularize behavior cloning vs. quietly become its shortcut, and (iii) tells you to *ask hard (low-recoverability)
questions* — with the crucial caveat that recoverability must be measured under the policy's function class and as a
learning-dynamics quantity, else a naive probe flips the sign.

**Status legend:** [SOLID] proven here · [PRELIM] shown at proxy scale · [DESIGN] planned (M2) · [THREAT] prior-art to defuse.

---
## Abstract (draft)
Behavior cloning learns the cheapest input→action mapping and generalizes poorly. Prior work adds auxiliary
objectives (future prediction, inverse dynamics, subtask/VQA, instruction prediction) to fix this, case by case.
We show these are points on one axis — **recoverability**, the usable information of the target under the policy —
and that a single law governs them: the *less* recoverable an auxiliary's answer is from the current input, the
*more* co-training on it improves out-of-distribution generalization. We show recoverability must be measured under
the policy's own end-to-end function class and as a learning-dynamics quantity: a frozen linear probe — the surrogate
one would naturally reach for — **inverts the law's sign**. We give a composition rule (a mix inherits the benefit of
its hardest member), connect the axis to Knowledge Insulation (which needs an action-related grounding signal), and
survey ~55 prior VLA auxiliaries onto the axis. [M2] We validate closed-loop in SimplerEnv.

---
## 1. Introduction
- Problem: BC = shortcut → poor OOD (causal confusion, LCBC, "ignores language"). [THREAT: de Haan 2019 / causal-confusion line — cite as the problem, our cure differs.]
- Prior fixes = a ZOO of auxiliaries, each justified ad hoc. → thesis: they lie on ONE axis (recoverability).
- Contributions: (C1) the recoverability axis + the law [R1/R2]; (C2) how to MEASURE it right — the sign-flip [R1, headline]; (C3) composition law [R3]; (C4) KI interaction [R4]; (C5) unifying survey [R7]; (C6) [M2] closed-loop.
- Frame the CORE claim per memory: **generalization failure, not "ignores language"** — ignoring is the diagnostic.

## 2. Related work (defuse threats explicitly) — all ids web-verified 2026-08-01, see `CITATIONS_THREATS.md`
**THE threat to lead with = dilution-by-ASSEMBLY (MODERATE), not direct pre-emption.** No paper uses 𝒱-information under the policy function class as an auxiliary-SELECTION axis, nor states the inverted law (lower recov ⇒ better OOD). Defuse the assembly of adjacent pieces explicitly:
- Usable/𝒱-information (Xu 2020 ICLR, arXiv:2002.10689; Ethayarajh 2022 ICML Outstanding, arXiv:2110.08420): our measure primitive, but they measure DATA difficulty — we invert into an auxiliary-SELECTION criterion. [THREAT-defused]
- MDL/prequential probing (Voita-Titov 2020 EMNLP — cite **ACL Anthology**, arXiv id unverified; Blier-Ollivier 2018): learning-effort as informativeness; never select objectives / predict co-training benefit.
- **Auxiliary-task WEIGHTING (Lin, Baweja, Kantor, Held NeurIPS 2019 — gradient-cosine alignment; AANG arXiv:2205.14082; SLGrad arXiv:2306.04519):** weight by main-task ALIGNMENT, not target recoverability. [assembly-threat — distinguish]
- **Info-theoretic TASK selection (ITTS, NeurIPS 2020):** selects training tasks by MI-difference, not aux-objective difficulty; no policy-class 𝒱-info. [assembly]
- **Predictive-Information aux (PI-QT-Opt, arXiv:2210.08217):** PI as a fixed helper aux to MAXIMIZE — not a difficulty measure to CHOOSE among auxes; opposite direction. [assembly]
- **Future-rep prediction regularizes BC (BYOL-γ, arXiv:2506.10137, 2025):** nearest conceptual neighbor — but one method, no recoverability axis, no probe-flip, no 𝒱-info selection. [assembly, closest]
- Instruction prediction as aux (**Hejna, Abbeel, Pinto AAAI 2023, arXiv:2306.12554** — VERIFIED) [MINOR, one aux on our axis]: our instr-infer is a new instantiation; the novelty is the SELECTION principle, not the instruction aux.
- Causal-confusion / anti-shortcut (de Haan NeurIPS 2019; **OREO NeurIPS 2021 arXiv:2110.14118**; **GABRIL IROS 2025 arXiv:2507.19647**) [MINOR]: same disease we treat; each a specific mechanism (interventions/object-dropout/gaze), none an info-theoretic selection criterion. Shared-motivation line.
- Simplicity bias / shortcut learning (Geirhos 2020; Shah 2020): the MECHANISM our criterion operationalizes (not competing prior art).
- Knowledge Insulation (π0.5-KI arXiv:2505.23705) + FAST (Pertsch 2025 arXiv:2501.09747) + GR00T: source of the KI setup; we add the missing ablation [R4].
- Future-pred VLA line (VPP arXiv:2412.14803; VLA surveys 2405.14093/2512.11362) [MINOR]: "future prediction helps" is in the air — do NOT claim it novel; none proposes the organizing axis.

## 3. The recoverability axis (framework)
- Def: recoverability R(o_t→y) = normalized predictive 𝒱-information under the policy class = 1 − L_val/L_marg
  (continuous ⇒ R²; discrete ⇒ 1 − H_val/H_marg). Cite Xu Prop 1.5.
- The taxonomy (prospective/retrospective/introspective) is SECONDARY — the axis is recoverability, not type. [R5: NTP is a form, not a recov level.]
- Why low recoverability forces grounding (simplicity-bias mechanism).

## 4. Measuring recoverability — you must measure it right (the sharpest result) [R1, SOLID]
- 6 measures × 3 scales (4k/8k/16k), 3 seeds, 18 objectives; selectivity control.
- **Headline 1: the sign flip.** Frozen/linear probe correlates +0.78/+0.69/+0.62 with generalization (WRONG); policy-class end-to-end −0.66/−0.64/−0.47; **learning-dynamics speed@¼ −0.75/−0.72/−0.69 (BEST, CI excludes 0 at every scale).** [stats: bootstrap CIs + perm-p in analyze_stats.]
- **Headline 2: asymptotic saturates, dynamics holds.** At scale the asymptotic measure loses significance (16k CI crosses 0) while the dynamics measure stays robust — because asymptotic recoverability saturates with data; the shortcut is still grabbed FIRST. → measure as learning dynamics.
- Why linear inverts (linear-decodability ⊥ policy-recoverability). Why not Shannon MI (Tschannen/McAllester-Stratos/Poole).

## 5. The law + composition [R2 SOLID / R3 PRELIM]
- The LAW: recoverability↓ ⇒ generalization↑ (per-objective).
- **Composition law [R3, NOVEL]:** a mix generalizes as its MIN-recoverability (hardest) member (min r=−0.87, CI excl 0 > mean); one hard question rescues a shortcut-laden mix. Contrast worst-case MTL (aggregates loss) & task-diversity (rewards coverage). [THREAT-defused]

## 6. Interaction with Knowledge Insulation [R4, PARTIAL]
- KI insulates the flow head; the backbone is shaped only by the aux (π0.5 uses FAST action tokens).
- **Our ablation [the novel half]:** hard-KI STARVES the action head ONLY with an action-UNrelated aux; action-related (cur/fut-action, FAST-like) survives. → KI needs an action-related grounding signal (LP-FT connection). Credit π0.5 for the positive design; ours is the missing contrastive ablation. Ties to R3 (FAST = the action-related, hard member that survives KI AND rescues the mix).

## 7. Unifying survey [R7, QUALITATIVE]
- ~55 prior VLA auxiliaries placed on the axis: field clusters on the cheap (shortcut) side; gains come from prior-transfer/planning, not target difficulty; retrospective lane nearly empty; future-frame prediction = the high-recoverability shortcut (GR-1/2 is a FOIL). Qualitative map — honest labeling.

## 8. Experiments
- 8.1 Mini-VLA (RT-1/fractal): the measurement study, law, composition, KI. [SOLID/PRELIM]
- 8.2 [M2, DESIGN — make-or-break] Full-VLA (pi05 from PaliGemma) + SimplerEnv closed-loop: does recoverability
  (amortized, ~free from each arm's aux val-loss) predict success? does a hard-question aux beat BC? [P1]
- 8.3 [PRE-STEP] mini-VLA SimplerEnv rollout (cheap closed-loop probe; code ready).

## 9. Limitations (be first to say them)
- Mini-VLA is a proxy in EVERY axis: small model, ~500–2,400 grad-steps, **2.3% of RT-1 (2k/87k episodes, 16k/~700k transitions)** → near-zero absolute generalization; relative rankings robust to subsample size but absolute-competence needs full data (= M2).
- Offline OOD action R² ≠ closed-loop success (M2 addresses).
- Recoverability is scale/context-dependent (relative, measured-in-context; never absolute).
- Small N (12–18 points) ⇒ wide CIs. Survey is qualitative. R4 half-preempted (π0.5). instr-infer not a novelty gap (Hejna 2023).

## 10. Conclusion
- Recoverability = the axis; ask hard questions; measure them the way the model experiences them (its function class, early).

---
## What we can claim NOW vs. what needs M2
- **NOW (defensible):** the measurement result + sign-flip (R1) [SOLID], composition (R3) [PRELIM], KI ablation (R4) [PRELIM], the unifying survey (R7). These stand on the mini-VLA.
- **NEEDS M2 (make-or-break for oral):** external validity — does recoverency predict closed-loop success on a real VLA, and does AHA beat BC. Until M2, do NOT claim external validity.
- **Single strongest oral pitch:** "the field's VLA auxiliaries are one axis; here's the measurable law, why the obvious way to measure it lies to you, and how to ask the right hard question."

*Sources of truth: `AHA_MASTER.md` (status), `RECOVERABILITY_MEASUREMENT.md` (R1+stats), `M1_RESULTS.md` (R3/R4), `SURVEY_vla_auxlosses.md` (R7), `M2_DESIGN.md` (M2). Verify all citations (esp. 2025/26 arXiv ids, Hejna 2306.12554) before submission.*
