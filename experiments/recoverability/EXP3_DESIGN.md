# Exp 3 — design (the real test of A3/A4; needs a trained encoder + OOD, not autonomous-runnable)

**Why a design, not a run:** Exp 2 (frozen DINOv2 + adapter, in-distribution) could **not** separate retrospective from forward (both ≈+0.04 transfer; BC/mixed win, partly circular). Reasons: (1) a frozen backbone caps how much an objective can reshape the representation — retrospective's benefit is *encoder shaping*, which needs a **trainable encoder**; (2) it tested in-distribution transfer, but the thesis predicts a **OOD/generalization** benefit; (3) the mixing→path-of-least-resistance mechanism needs a **shared, capacity-limited encoder trained end-to-end**. Exp 3 supplies all three. It is a multi-day, multi-GPU training study → run under user sign-off, not autonomously.

## Hypothesis (A3 → A4)
Adding **shortcut-free retrospective objectives** to the pretraining mix yields (a) **lower memorization-shortcut reliance** (measured, from Exp 1b) and (b) **better OOD generalization + data-efficiency** than the forward-only LAP baseline — largest gap in low-data / OOD.

## Ablation (maps to HAF config flags — `src/haf/models/haf_config.py`)
Backbone: PaliGemma-3B, full-FT, action head SG-isolated (KI). Toggle objectives via:
`enable_langact_training` · `enable_action_training` · `enable_prediction_training` · `enable_vqa_training` + `{language,action,prediction,vqa}_loss_weight` + the mix weights in `question_types.py::QuestionConfig` / `datasets/utils/mixtures.py`.

| config | objectives | how |
|---|---|---|
| **(a) LAP baseline** | policy (language-action + action FM) + VQA/motion (forward+inverse) | current `haf` config (langact+action+prediction with existing motion-VQA) |
| **(b) +forward-hard** | (a) + far-future world-model (predict future-obs code @ large k) | new `QuestionType.FORWARD_OBS`, discrete-semantic target (below) |
| **(c) +retrospective** | (a) + instruction-inference + initial-obs/"what-changed" recovery | new `QuestionType.INSTRUCTION_INFER`, `RETRO_OBS`; **recoverable** past only |
| **(d) full** | (a)+(b)+(c) bidirectional | all on |

## What Exp 1/1b force into the implementation (design constraints already learned)
- **Retrospective target = the *recoverable* past, not pixel o₀.** Exp 1: initial-obs is learnable-beyond-copy (RT-1 +0.57) and change-recoverability rises with distance/structure. Use **discrete-semantic codes** of the target obs (VQ/k-means over frozen DINOv2/V-JEPA latent) predicted with **CE on the shared backbone** (KI-consistent; §6-corrected). Not continuous JEPA-into-backbone (KI risk), not pixels.
- **Keep a language-floor in the mix.** Exp 1b: language is ~redundant for the action chunk (Δ_lang≈0) → BC underweights language; the "language-dropout hurts grounding" evidence says a language-free auxiliary flood makes it worse. Enforce a minimum fraction of language-required samples.
- **Symmetric difficulty knob = |temporal distance|** (Exp 1: redundancy symmetric, decays with distance) — schedule targets by distance, not a hand-curriculum (curriculum was rejected earlier).

## Eval (the OOD the frozen probe couldn't test)
1. **OOD generalization:** LIBERO-Plus (perturbation), **RoboCasa365** (novel scenes/objects, controlled, sim), held-out-scene/object, cross-embodiment zero-shot. Primary metric: **success-gap vs LAP baseline, as a function of OOD severity** (expect gap ↑ with severity).
2. **Memorization-shortcut, directly:** counterfactual-instruction grounding (LIBERO-CF style — different feasible instruction in the same scene). Expect retrospective-mix to **raise instructed-object grounding** (reduce the Exp 1b shortcut).
3. **Data-efficiency curves:** 10/25/50/100% downstream data; expect (d)≈(c) > (b) > (a), gap largest at low data.
4. **Real-robot capstone:** RoboArena pairwise (Ours vs LAP-3B) — DROID platform (our data), relative win = the claim's shape.

## Sequencing (cost-aware)
1. **Implement** the two new objective families in the data pipeline (`question_types.py` + `sample_handlers.py`) + the discrete-semantic obs head (offline VQ/k-means over DINOv2, cache codes) + world-model CE branch in `HAF.compute_loss` (`pred_mask` path).
2. **Smoke-test at LIBERO scale** (`haf_libero`, fast) — verify losses go down, BC not starved (PCGrad/GradNorm; BC-OOD as the model-selection metric, per earlier).
3. **Ablation (a)–(d) at LIBERO/LIBERO-Plus scale** — first real A3/A4 signal, cheap.
4. **Scale winners to OXE + RoboCasa365**, then RoboArena.

## Cost
LIBERO-scale ablation: hours–1 day/config on 8×B200. OXE-scale: days/config. Start at LIBERO scale (cheap, 4 configs) → decide.

**Gate:** only proceed past step 2 if the LIBERO-scale ablation shows the predicted ordering; otherwise revisit the retrospective target design (recoverability / discretization).
