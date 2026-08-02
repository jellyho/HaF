# Recoverability — the central design principle of HaF

*Master design note. Captures the theory, the taxonomy, the experiments, and the open decisions. Written to
be the single place a newcomer (or future-us) can reconstruct the whole argument.*

---

## 1. The claim (one paragraph)

Auxiliary objectives for a robot policy differ in **Recoverability**: how much of the target is already
obtainable from the model's input by a **cheap / shortcut solution**. Behavior cloning (BC) alone takes the
path of least resistance and **memorizes** (maps scene→action, ignores language) — a *high-recoverability
shortcut* that works in-distribution and breaks OOD. Co-training a **low-recoverability, task-relevant**
auxiliary makes that shortcut uneconomical and forces a representation that jointly grounds language, vision,
and action, improving OOD generalization. **Recoverability — not "retrospection" — is the contribution.**
Retrospective prediction is simply the first practical instance of a low-recoverability objective.

---

## 2. Precise definition of Recoverability

**Recoverability(Y | X) = the extent to which the target `Y` can be predicted from the given input `X` by a
*low-complexity (cheap / shortcut) predictor*.**

Two things must be pinned down:

1. **From what input `X`?** `X` is *what the model is actually given*. This unifies the two levers (§4):
   choosing a distant target lowers `I(X; Y)`; masking an input (JEPA-style) also lowers it.
2. **Recoverable by what?** Not the information-theoretic ceiling `I(X;Y)` (any function), but a
   **complexity-indexed** quantity: *how cheap a predictor suffices*. A far-future frame may have non-trivial
   `I(o_t; o_{t+45})` yet not be copyable — so its **cheap-recoverability is low** even if total MI isn't.

So Recoverability is a **spectrum indexed by predictor complexity**, estimated at two points:

| estimator | predictor | reads out |
|---|---|---|
| `R_triv = 1 − L(copy)/L(marginal)` | identity/copy (cheapest) | shortcut availability |
| probe (`G_obs`, `probe_beyond_trivial`) | linear/MLP on a fixed encoder | richer recoverability |
| **gap = probe − triv** | — | **"cheap can't, but structure can" = the hard-but-learnable band we want** |

> Reviewer point 1 asks us to treat Recoverability as a *fundamental property*, not a heuristic tied to one
> baseline. Resolution: it is `I_cheap(X; Y)` — a complexity-indexed recoverability; `R_triv` is just its
> cheapest end. The paper should lead with the probe/MI estimate, not the copy baseline.

**The good-auxiliary conditions (both required):**
1. **Low cheap-recoverability** — no shortcut solution.
2. **High task-relevance** — `Y` is determined by the hidden task variables (goal, initial state, instruction),
   so it is *learnable-with-structure*, not noise. (Condition 2 is the anti-collapse guard — the analog of
   JEPA's EMA/predictor that stops the degenerate constant solution.)

---

## 3. Why one lens explains both the problem and the fix

- **Problem = BC memorization** = "the action is recoverable from the scene by a cheap memorize-map." High
  (spurious, in-distribution) recoverability → BC cheats → fails OOD.
- **Fix = low-recoverability auxiliary** co-trained on the shared backbone → the cheap map no longer minimizes
  total loss → the representation must encode task structure.

Same axis, both directions. This is what makes Recoverability a *taxonomy* rather than a single trick.

---

## 4. Two levers that lower Recoverability

Recoverability = `I_cheap(X_given ; Y_target)`. You can lower it two ways:

| lever | how | examples | flavor |
|---|---|---|---|
| **Target-side** | pick a `Y` the current input doesn't cheaply determine | retro-obs (initial frame), displacement-from-start, "how did I get here" | temporal / causal |
| **Input-side (JEPA/MAE/masking)** | remove part of `X`, predict it back | mask the instruction → infer the task; state-dropout; masked patches | occlusion / masking |
| **Both** | give only behavior (mask instruction) and ask the goal | `task_inference_from_history` | strongest |

This is the unification: **JEPA/MAE/contrastive manipulate Recoverability on the *input* side; hindsight
manipulates it on the *target/time* side. They are two handles on one axis.** Retrospection is the first
target-side instance.

---

## 5. Objective taxonomy (ordered by Recoverability — *measure it, don't assume it*)

```
 high recoverability  ── shortcut-prone ──▶                 ◀── shortcut-free ── low recoverability
   future frame (≈copy)        progress?*        far-future        retro initial-obs, "what changed",
   near-future gripper         next action       inverse-dynamics  displacement, task-inference (masked)
```

`*` **Caveat from our own data:** semantics ≠ recoverability. `progress` (t/T) *sounds* introspective/low, but
we measured it **highly recoverable** (G_obs≈0.88 — readable off the scene) and it was a *weak* auxiliary
(exp2c rank near the bottom). This is direct evidence for "categorize by measured recoverability, not by
semantic label" (reviewer point 5).

---

## 6. The Form question (discrete CE vs continuous embedding) — resolved with a test

Predicting an **image** target in a text-output VLM (PaliGemma) raises an architecture tension:

- A **continuous regression head** (predict SigLIP/DINOv2 embedding, JEPA-style) is *structurally* like pi0/pi05's
  **action expert** — a continuous module bolted onto a discrete-token LLM. But its **role is opposite**: the
  action expert is KI-*insulated from* the backbone; the retro head is *meant to shape* the backbone.
- The real risk: shaping a **discrete-token LLM backbone with a continuous MSE gradient** may reintroduce the
  instability KI avoids (KI shapes the backbone with discrete CE only). Our probe used continuous MSE and it
  worked — but the probe's backbone was DINOv2 (a continuous vision model), **not** a pretrained discrete LLM,
  so it doesn't settle the question.

**Empirical update (exp2e, 6 datasets × 5 seeds):** the **continuous latent-MSE** retrospective target
(`retro-obs`, +0.083 mean ΔOOD) beats the **discrete-CE** target (`semantic-retro`, −0.168) by ΔR²≈0.25 on
*every* dataset. So at probe scale **continuous latent retrospection >> discrete-code retrospection** — the data
*favors* embedding prediction over discrete CE (caveat: coarse k-means-32 codes; a fine VQ tokenizer is
untested). This both supports the decision below and sharpens the tension: the *best-performing* form
(continuous) is exactly the action-expert-like head whose backbone gradient we're unsure about → hence the test.

**Decision (user):** go with **embedding prediction** (describing vision in language is unnatural / lossy), and
**empirically test whether the gradient flowing into the backbone actually hurts** rather than assume it.

Test plan:
- **Preliminary (cheap, already running): `exp2f_fused.py`** — a small transformer backbone shaped by a
  continuous latent auxiliary (predict z0/z_fs). First read on "does continuous MSE stably shape a transformer
  backbone." Caveat: from-scratch transformer, not a pretrained LLM.
- **Definitive: Exp 3 arm** — add a continuous **retro-embedding head** to the HaF VLA (LLM hidden → SigLIP
  embedding of the retro frame), with a **`stop_retro_to_vlm_grad`** toggle (mirrors `stop_action_to_vlm_grad`):
  - `soft` (grad flows) — backbone shaped by retro ← the thing we want
  - `ki` (stop-grad) — control, backbone untouched
  - `none` — no aux
  Measure SimplerEnv success + representation quality + training stability → answers "does the gradient help
  or hurt."

Discrete alternatives kept as optional arms: **VQ tokens** (discretize target image, predict codes with CE —
fits KI cleanly, heavy) and **language description** (native, coarse — deprioritized as unnatural for vision).

**Implementation status (retro-embedding head):**
- **DONE** in `src/haf/models/haf.py` (compiles; inert unless `retro_embedding_loss_weight>0`): the head module
  `self.retro_head = nnx.Linear(paligemma_width, paligemma_width)` (created in `__init__` only when enabled) and
  the method `retro_embedding_loss(llm_hidden, retro_image)` — computes `target = stop_grad(mean_pool(
  PaliGemma.img(retro_image)))` (reuses the model's own SigLIP; no external model), `pred = retro_head(h)` with
  `h = stop_grad(llm_hidden)` iff `stop_retro_to_vlm_grad`, and returns cosine distance. Flags copied to
  `self.retro_embedding_loss_weight` / `self.stop_retro_to_vlm_grad`.
- **REMAINING (needs JAX forward/backward smoke):**
  1. **Call site in `compute_loss`:** pick a pooled VLM hidden `h` (e.g. mean of the prefix pre-logits over the
     prompt/image tokens), call `self.retro_embedding_loss(h, retro_image)`, add
     `self.retro_embedding_loss_weight * loss` to the total (guard on `weight>0`).
  2. **`retro_image` plumbing:** carry the retrospective anchor frame (the same ≈2 s start anchor as the
     pipeline retro questions) into the observation/batch and hand it to the call site.
  3. **Exp-3 arms** (add once the call site exists): `exp3_rt1_retroemb_soft` (weight>0,
     `stop_retro_to_vlm_grad=False`), `exp3_rt1_retroemb_ki` (stop-grad True); baseline `exp3_rt1_bc`.
     Compare SimplerEnv success + rep-quality + training stability → **does the continuous gradient into the
     discrete-LLM backbone help or hurt.** Run a single forward/backward smoke step before any full run.

---

## 7. The retrospective objectives we have

**Probe-level (`exp2e_retro.py`, latent-MSE / discrete-CE, for measuring recoverability):**
retro-obs (z0), near-past-obs (z_ps), farpast-obs (z_pl), retro-pose (cart0), retro-action (act_prev),
displacement (z_t − z_0), semantic-retro (discrete code of z0, **CE** — form contrast).

**Pipeline-level (`src/haf/policies/question_types.py`, language-QA, for the real VLA):**
- target-side: `RETRO_MOTION`, `DISPLACEMENT_FROM_START`, `PROGRESS_ESTIMATION`, `PAST_GRIPPER_RECALL`
- input-side (masked): `TASK_INFERENCE`, `TASK_INFERENCE_FROM_HISTORY`
- + the continuous **retro-embedding** visual objective (§6), added to the model.

**Anchor sampling (user):** for start/end references (initial pose, initial scene, displacement origin) sample
the anchor **randomly from a ~2 s window** at the relevant end of the episode, not the exact first/last frame —
matches the existing `~2.5·control_frequency` horizon scale and avoids over-fitting one crisp frame.

**More retro ideas (backlog):** "what changed since start" (scene-change, language or latent), temporal-JEPA
(mask a past frame, predict its embedding), retro-action *chunk* (full a_{t−K:t}), event localization ("when did
it first grasp"), sub-goal recall (long-horizon), object-interaction history.

---

## 8. What is solid vs what needs Exp 3 (be honest)

**Solid (5 datasets/6 datasets, 5 seeds):**
- BC memorizes everywhere (`sensL≈0`, `sensV≈2`).
- shortcut-free aux → better OOD + more language grounding.
- **Vision-legitimacy law (dataset-level Recoverability):** how legitimately vision carries the signal predicts
  aux benefit, Pearson **r ≈ −0.93, p ≈ 0.007** — the aux helps *iff* the target was cheaply recoverable from
  the scene. This is Recoverability working at the dataset level.
- retro-obs is the safest single auxiliary.

**Weak / honest at probe scale (the gaps):**
- **Objective-level** "Recoverability predicts benefit" — the direct `R_triv → ΔOOD` correlation is ~0
  (dataset effect dominates); only the language-grounding link survives (Spearman ≈ +0.31, p≈0.06).
- **λ dose-response** is monotone-increasing to λ=4 (no inverted-U yet) — regularization not saturating at this
  scale.
- **Form (exp2e):** obs-based continuous retrospection (retro-obs / near-past / far-past, all ≈+0.085) is the
  best family; **discrete-CE (semantic-retro) is much worse (−0.17)** — see §6.
- **Fused backbone (exp2f):** the aux still improves OOD, but the hard-KI "vision collapse" **softens** with a
  multimodal rep (vision-use 0.4–1.0, not ≈0) — that collapse was partly a vision-only-representation artifact.
- **Representation quality ≠ control (exp2g):** the aux raises instruction-decodability / task-clustering
  (hard-KI most, retrieval@10≈1.0), **but decodability does NOT predict OOD control** (instr_decod vs OOD-R²:
  r=0.26, p=0.34); KI maximizes decodability while often hurting control. *The reviewer's "missing middle" is
  subtle: a more decodable representation is not a better one for acting — the causal middle is behavioral
  (contrib_lang/vision), not representational.*

**Therefore Exp 3 is the referee:** put multiple objectives on the Recoverability axis, train the real VLA on
RT-1, and test **Recoverability(objective) vs SimplerEnv OOD success**. If it holds → Recoverability is a design
principle (strong paper). If not → the dataset-level law still stands (narrower but real). *Either way, not
empty-handed* — that is the point of this preliminary design.

---

## 9. Literature anchors (Recoverability as a unifying explanation)

- **Shortcut learning** (Geirhos 2004.07780; ObjectNet 1909.03450): every objective has a cheapest cheat;
  Recoverability = whether that cheat exists.
- **SSL** (SimCLR 2002.05709, BYOL 2006.07733, MAE 2111.06377, **I-JEPA 2301.08243**): good pretext tasks remove
  trivial prediction pathways — i.e. lower Recoverability on the *input* side.
- **Information Bottleneck** (1612.00410; Tishby): low-recoverability targets force compression into
  task-relevant latents.
- **Predictive coding / Free energy** (Rao–Ballard 1999; Friston 0905.2546): predictable signal contributes
  little; surprise drives abstraction.
- **Causal representation learning** (IRM 1907.02893; 2102.11107): retrospective prediction requires inferring
  hidden causes (goal, initial state, instruction), not surface correlations.
- **MDL**: if the target isn't cheaply recoverable, the cheapest optimizer becomes a reusable latent.

Positioning line: *"Recoverability generalizes, to the target/temporal axis and to auxiliary co-training for
control, the design principle that contrastive/MAE/JEPA instantiate on the input side."*

---

## 9b. WIRING SPEC — plumbing the pipeline retro questions (status + remaining work)

**Status (done, safe, additive):**
- Vocabulary/prompts/answer-generators + weight preset `RETRO_MIXED_QUESTION_WEIGHTS` in `question_types.py`.
- Handler branches in `sample_handlers.py::_format_question_answer` for all new types.
- **`TASK_INFERENCE` / `TASK_INFERENCE_FROM_HISTORY` are fully wired and work now** with no pipeline change:
  prediction samples already overwrite the prompt with the generated question, so the instruction is naturally
  masked, and the branch returns `answer = TextParser.parse_prompt(data)` (the instruction). Activate by adding
  them to `DataConfig.question_type_weights`.

**DONE (implemented + unit-smoke passed — `scripts/test_retro_wiring.py`):**
- `TrajectoryOutputBuilder.retro_fields(traj_id, state, traj_len, control_frequency, seed)` in
  `output_schema.py` — per-frame `progress`, `displacement_cm`, `start_gripper` with a **≈2 s start anchor**
  (seeded by traj id, per user spec).
- Wired into both robot restructure callers via `extra_fields`: `robot/oxe_datasets.py::restructure` and
  `robot/droid_dataset.py`. Batch-key consistency: same keys defaulted in `build_vqa_frame`.
- Handler branches in `sample_handlers.py` for `PROGRESS_ESTIMATION`, `DISPLACEMENT_FROM_START`,
  `PAST_GRIPPER_RECALL`, `TASK_INFERENCE(/_FROM_HISTORY)`. Unit smoke prints correct (Q, A) pairs.
- **Activate:** set `DataConfig.question_type_weights` to `question_types.RETRO_MIXED_QUESTION_WEIGHTS`
  (or include the retro keys) with `enable_prediction_training=True, pred_prob>0`.

**REMAINING (validate/implement with eyes open):**
1. **Full-batch validation** (the one thing the unit smoke can't cover): load a real batch with the retro
   weights set and confirm `progress`/`displacement_cm`/`start_gripper` actually reach the handler's `data`
   (survive restructure→prediction→repack→batch) and the (prompt, answer) pairs look right. Do NOT launch a 3B
   run until verified — wrong answers train silently.
2. **`RETRO_MOTION`** (the one retro type still inactive): needs a *retrospective image pair* `[past, current]`.
   Build it in `add_prediction_pairs` (a past-direction variant of the existing future pair, tagged
   `pred_direction`) and set `language_actions ← retro cumulative motion`; then gate the handler on
   `data.get("pred_direction")`. Deferred because it changes the image-pair direction (higher risk).

## 10. Pointers

- Probes + metrics: `experiments/recoverability/` (see its `README.md`).
- Question types (retro + task-inference): `src/haf/policies/question_types.py`.
- Data wiring: `src/haf/datasets/base_dataset.py::add_prediction_pairs`,
  `src/haf/policies/transforms/sample_handlers.py`.
- Retro-embedding head + Exp-3 arms: `src/haf/models/haf.py`, `src/haf/training/config.py`.
- Scaled eval: `scripts/simpler/` (RT-1 → PaliGemma-init pi05 → SimplerEnv).
- Report: `build_artifact.py` → the artifact URL.
