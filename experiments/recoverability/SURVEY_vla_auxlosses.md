# VLA auxiliary-objective survey — unified on the recoverability axis

**Claim this evidences (oral target).** The field has proposed dozens of *ad-hoc* self-supervised / auxiliary objectives for VLAs. They are **not a zoo**: each is a point on ONE axis — **recoverability** (how cheaply the target is obtainable from the current input o_t) — and one law (recoverability ↓ ⇒ generalization ↑) predicts which ones actually force grounding. This survey classifies prior objectives into our taxonomy and places them on the axis.

**Taxonomy (defined on the TARGET's position relative to input time t):**
- **prospective** — target lies in the FUTURE of t (future frames/latents, goal/subgoal images, future actions, value/time-to-goal, next subtask).
- **retrospective** — target lies in the PAST of t (past frames, previous actions, initial obs/state, displacement).
- **introspective** — MASK/corrupt part of the CURRENT o_t and reconstruct it (MAE, masked language/state, augmentation-invariance, VQA-on-present, progress t/T).
- (**policy-BC** — the action objective itself; listed separately.)

**Recoverability rating**: H = target near-trivially obtainable from o_t (shortcut regime) · M = partial · L = requires genuine grounding/inference (forcing function). Our LAW predicts H→overfit, L→generalize.

---

## Boundary-case rulings (these decide the survey's rigor)

1. **Inverse dynamics** (predict a_t from o_t **and** o_{t+1}). The target is the *present* action, but a *future* frame is in the input. As a **labeler / reconstruction** (both endpoints given) it is **introspective, H** — that is exactly why it works as a cheap auto-labeler (VPT, BCO, GR00T-IDM). The moment the future frame is removed and the model must predict from o_t alone, it flips to **prospective, L**. → *Classify by which stage is exposed to the policy.*

2. **Latent-action models** (LAPA, Genie, Moto, LAPO, IGOR, UniAct, GO-1) are **two-stage objects spanning both ends**: Stage-1 tokenizer/LAM (frame pair → latent) is **introspective, H**; Stage-2 policy (o_t + language → latent) is **prospective, L**. The pretraining signal that matters for recoverability is Stage-2. *(LAP-3B in this codebase descends from this family.)*

3. **Forward dynamics splits by output space**: pixel next-frame (GR-1/2, Genie, Seer-foresight, WorldVLA, NWM) is **prospective but H in vision** (consecutive frames barely differ) — the shortcut our thesis flags; latent/feature next-state (SPR, DynaMo, DINO-WM, LaWM) deliberately removes the pixel shortcut → **prospective, M–L**.

4. **Hindsight relabeling** (HER, GCSL, Play-GCBC, RT-Trajectory). Two clocks are conflated: the *operation* is retrospective (look back over a finished rollout), but on our axis (target-time) the achieved goal sits at t′>t = **prospective**. Because the goal lies ON the executed trajectory it is **maximally redundant with the actions (H, the cheapest possible "future")** — a supporting instance of the shortcut thesis, not a forcing function.

5. **"Temporal contrastive" is not uniformly prospective**: ATC matches a short-horizon *future* obs (prospective, M); CURL contrasts augmentations of the *same* frame (introspective, H); TCN/R3M contrast temporal *distance/order* symmetrically (hybrid, encodes task-phase, M).

---

## Master table (consolidated, deduped across slices)

### Introspective — mask/reconstruct the present (mostly H = shortcut regime)
| work | aux objective | type · recov | conf |
|---|---|---|---|
| MVP / RRL-MVP (Xiao/Radosavovic 2022) | MAE pixel reconstruction of masked current-frame patches | intro · H | high |
| VC-1 (Majumdar 2023) | MAE on Ego4D+ImageNet | intro · H | high |
| MAE / MWM / Crossway-Diffusion | current-frame reconstruction (aux to policy) | intro · H | high/med |
| CURL (Srinivas 2020) | InfoNCE on two augmentations of same frame | intro · H | high |
| BYOL / DINO / DINOv2 / MoCo / SimCLR (as control pretrain) | augmentation-invariance of current image | intro · H | high/med |
| I-JEPA (Assran 2023) | predict latent of masked blocks within one image | intro · H | med |
| Voltron (Karamcheti 2023) | language-conditioned masked recon + caption current frame | intro · H–M | high |
| RT-2 (2023) | co-train internet VQA with action tokens | intro · H | high |
| PaLM-E (2023) | VQA + captioning + planning co-train | intro (VQA) · H | high |
| ChatVLA (2025) | VQA / multimodal understanding co-train | intro · H | high |
| RoboVQA (2024) | long-horizon video VQA (planning/success) | intro/hybrid · H–M | high |
| Magma – Set-of-Mark (2025) | label actionable objects in current frame | intro · H | high |
| ECoT – bbox/gripper (2024) | predict object boxes + gripper pos in current frame | intro · H | high |
| MolmoAct – depth tokens (2025) | depth perception tokens from RGB | intro · H | high |
| VPT (Baker 2022), BCO (Torabi 2018) | inverse-dynamics label a_t from frame pair | intro/borderline · H | high |
| MTM (Wu 2023), MaskDP (Liu 2022), RPT (Radosavovic 2023) | masked trajectory-token reconstruction (some tokens forced future) | intro/hybrid · M | high |
| SMART (Sun 2023) | forward + inverse + random-masked control | hybrid · M (fwd = L) | high |
| progress / phase t/T heads | regress progress from current frame | intro · M | med |
| **OUR mae-mask 25/50/75** | mask current image, reconstruct z_t | intro · M–L | — |
| **OUR state-infer** | mask proprio, infer pose+gripper from image+lang | intro · L | — |
| **OUR instr-infer** | mask language, infer instruction from image+state | intro · L | — |

### Prospective — predict the future (vision-future is H; abstract/long/semantic is L)
| work | aux objective | type · recov | conf |
|---|---|---|---|
| GR-1 (2023), GR-2 (2024) | predict future RGB frame(s) + action | prosp · H (vision) | high |
| VPP – Video Prediction Policy (2024) | condition on video-diffusion predicted-future features | prosp · H | high |
| Seer / PIDM (2024) | predict future frame → inverse-dynamics to action | prosp · H → invert | high |
| CoT-VLA (2025) | autoregressive future subgoal frames before action | prosp · H–M | high |
| WorldVLA (2025), Navigation World Models (2024) | action-conditioned next-frame (forward dynamics) | prosp · H | high/med |
| Genie (2024) | next-frame dynamics given inferred latent action | prosp · H | high |
| SuSIE (2023) | synthesize near-future subgoal image | prosp · H–M | high |
| MimicPlay (2023) | predict future 3D hand trajectory | prosp · M | high |
| Magma – Trace-of-Mark (2025); RT-Affordance (2024); MolmoAct-trace | future point-trace / future gripper poses | prosp · M–L | high |
| ATC (Stooke 2021) | contrast o_t with short-future o_{t+k} | prosp · M | high |
| RT-H (2024) | predict imminent language motion, then act | prosp · M | high |
| SPR (2021), DynaMo (2024), DINO-WM (2024), LaWM (2026) | predict future *latent/feature* state | prosp · M–L | high/med |
| LAPA (2024), Moto (2024), LAPO (2024), IGOR, UniAct, GO-1 | predict latent-action from o_t+lang (Stage-2) | prosp · L | high/med |
| **π0.5 – subtask (2025)** | predict next semantic subtask (discrete language) | prosp · **H–M** (NTP *form* but scene-determined low-entropy target → easy; NOT a strong forcing function) | high |
| DiffusionVLA (2025) | self-generated task-decomposition reasoning | prosp · M–L | high |
| UniPi (2023), Video-Language-Planning (2023), RoboDreamer (2024), 3D-VLA (2024) | text→future-video / goal-image generation as planner | prosp · L (long-horizon) | high |
| VIP (Ma 2023), LIV (2023) | value / time-to-goal (implicit) | prosp · L | high |
| Decision/Trajectory Transformer, PACT | next state+action token sequence modeling | prosp/hybrid · H–M | high |
| **OUR near/far future-obs, fut-action, final-obs** | predict future latent/action from o_t | prosp · M–L | — |
| **OUR final-pose (the shortcut)** | predict last-frame arm pose | prosp · **H (0.92)** | measured |

### Retrospective — predict the past (GENUINELY RARE in prior work — the open lane)
| work | aux objective | type · recov | conf |
|---|---|---|---|
| HER (2017), GCSL (2021), Play-GCBC (2019), RT-Trajectory (2023) | hindsight goal relabeling (operation retro; **target-time prospective**, H redundant) | retro-op / prosp-target · H | high |
| DIAL (2022), CAST (2025), Röder (2022) | hindsight *language* relabeling — describe achieved trajectory | retro (language) · H from full clip | high |
| HIR (2023, language domain) | relabel the instruction to fit the produced output | retro · H | high |
| **OUR far/near past-obs, prev-action, initial-obs/pose, displacement** | predict past latent/action from o_t | retro · M–L | measured |

### Policy-BC only — NO auxiliary loss (the field's flagships)
π0 (2024), GR00T N1/N1.5 (2025), Diffusion Policy (2023), VQ-BeT (2024), Decision Transformer (2021), ACT (weak CVAE reg.), RoboFlamingo (2023), OpenVLA. → generalization currently comes from **data scale + VLM/video priors**, leaving target-recoverability **unexamined**.

---

## Key findings (for positioning)
1. **The field clusters on the HIGH-recoverability (cheap) side.** MAE/contrastive/VQA (introspective) and pixel next-frame/hindsight-goal (prospective) are all H. Reported gains "almost never come from target difficulty" — they come from **prior transfer** (GR-1/2, VPP, GR00T-Dreams) or **using the prediction as a planner/data-generator** (UniPi, SuSIE, RoboDreamer, DreamGen).
2. **The recurring architecture is predict-future-frame-then-invert-to-action** (Seer, CoT-VLA, UniPi, AVDC, WorldVLA): the visual target is the easy part; the true signal is the inverse-dynamics action map.
3. **The genuinely LOW-recoverability levers are underused**: value/time-to-goal (VIP/LIV), long-horizon/goal generation, and open-vocabulary language grounding. **Caveat (important refinement): NTP is a FORM, not a recoverability level.** LLM NTP is low-recoverability because open web text is high-entropy; π0.5's subtask NTP predicts a *scene-determined, low-entropy* subtask vocabulary → it is actually HIGH-recoverability ("an easy task"), NOT a strong forcing function. Recoverability = entropy of the target given the input, independent of the objective's form (NTP/MAE/etc.). This both sharpens the thesis and pre-empts the reviewer objection "π0.5 already does NTP."
4. **Retrospective is a nearly empty lane** — prior "retrospective" work is almost entirely hindsight *relabeling*, which is prospective-by-target and maximally redundant. Genuine "predict the past from o_t" objectives are unexplored.
5. **Instruction-recovery aux — NOT a clean novelty gap (corrected 2026-07-30):** instruction-prediction as an auxiliary to force grounding is already published (Hejna, Abbeel, Pinto, AAAI 2023, arXiv 2306.12554 — VERIFY). Our `instr-infer` is a new INSTANTIATION (within-VLA, masked-input, recoverability-selected), not a new idea. The novel contribution is the recoverability-SELECTION principle, not the instruction aux itself.

*Confidence: works marked "high" web-verified this session; "med" from model knowledge, not re-fetched. A few 2026 preprints (ProgVLA, PALM, LaWM) surfaced in search but individual losses were not verified — do not cite without checking.*
