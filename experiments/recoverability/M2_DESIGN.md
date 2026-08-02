# M2 — full-VLA + SimplerEnv closed-loop (the external-validity test of AHA)

**Goal.** Show the recoverability law transfers from the mini-VLA (offline OOD R²) to a **real VLA** measured by
**closed-loop SimplerEnv success rate**. The make-or-break test (AHA_MASTER P1). Central question:

> **Does the measured recoverability of an auxiliary objective PREDICT its closed-loop SimplerEnv success — and
> does a low-recoverability "hard question" beat BC-only?**

If yes → the law + AHA hold on a real VLA in closed loop (oral evidence). If the recov↔success correlation is
flat/positive → the mini-VLA law was a proxy artifact (report honestly).

## Model & data (existing scaffold)
- **pi05 recipe from RAW PaliGemma** (weight_loader kind="paligemma"), flow-matching action expert, `action_dim=7`,
  `action_horizon=16`, adaRMS. Configs in `src/haf/training/config.py` (`exp3_rt1_*`). Trained on **RT-1
  (fractal20220817_data)** — same data as the mini-VLA, so recoverability numbers are comparable.
- **KI** = `stop_action_to_vlm_grad` (stop-grad flow head → VLM backbone). Backbone shaped by the aux (+ web/FAST).

## IMPORTANT — how the full VLA does "prediction" (grounds the whole design)
The full-VLA aux is **NOT continuous future-frame/latent regression.** It is a set of **discrete language-QA
objectives** (`src/haf/policies/question_types.py::QuestionType`), answered as tokens with cross-entropy on the
VLM backbone — **KI-consistent by construction** (discrete, VLM-modality; same philosophy as π0.5's FAST tokens):
- foresight: `DELTA_MOTION` ("what's the next motion?"), DIRECTION/GRIPPER/MAGNITUDE/TASK_PREDICTION/TEMPORAL_ORDERING
- retrospective: `RETRO_MOTION`, `DISPLACEMENT_FROM_START`, `PROGRESS_ESTIMATION`, `PAST_GRIPPER_RECALL`
- input-masked: `TASK_INFERENCE` (= instr-infer; mask the instruction, recover the task) — **already implemented**.

So **"future prediction" here = "predict the next motion in language" (discrete CE)**, not pixel/latent regression.
Two consequences:
1. **The mini-VLA "hard-KI starves" finding (R4) does NOT transfer directly**: it was measured with a *continuous*
   action-unrelated aux (far-past-obs MSE). The full VLA never regresses continuous latents into the backbone, so
   the starvation failure mode is designed away. → **corrected prediction (supersedes P4): all full-VLA auxes are
   KI-safe (discrete QA); the real axis is the RECOVERABILITY of each discrete QA ↔ success, not starvation.**
2. A genuine "future-**obs**" hard question (not just motion) must be discrete too: **VQ/k-means quantize the future
   DINOv2/V-JEPA latent → predict codes with CE** ("future-obs token"). Designed (old `FORWARD_OBS`) but **NOT in the
   current enum → must be built.** This is where the FAST-lite discrete-recoverability method (Phase-2, running) applies.

**Code-readiness (honest):** EXISTS — the VQA/QuestionType framework (12+ discrete questions incl. instr-infer &
retrospective) + SimplerEnv eval scaffold + exp3 5-arm configs. MISSING / UNVERIFIED — (a) never run end-to-end
(train+eval unverified); (b) FAST-token arm; (c) future-obs VQ-code aux; (d) full-VLA recoverability probe. → build + verify before any claim.

## Arms (recoverability-framed; reuse existing 5 + 2 AHA additions)
Map the existing HaF-era arms to the AHA axis and add the two the M1 findings call for:

| arm | aux shaping the backbone | action-related? | expected recov | KI | AHA role |
|---|---|---|---|---|---|
| A0 `bc` | none (action shapes whole VLM) | — | — | off | shortcut baseline (memorizer) |
| A1 `langact` | language-action tokens | ✓ | mid | on | action-related discrete (LAP standard) |
| A2 `pred` | DELTA_MOTION (discrete motion-QA) | ✓ (motion) | low–mid | on | discrete ⇒ KI-safe (not continuous world-model) |
| A3 `mix` | langact + pred | mixed | min-dominated | on | composition (should ≈ its hardest member) |
| A4 `mix_soft` | langact + pred | mixed | — | **off** | KI on/off contrast |
| **A5 `fast` (ADD)** | **FAST action-token NTP** (discrete, freq-compressed) | ✓ | **low(?)** | on | π0.5-standard; M1 sweet-spot (action-related + low-recov) |
| **A6 `hardq` (ADD)** | **genuine low-recov aux** (masked-obs latent / far-future-obs code, discrete-semantic CE) | ✗–~ | **low** | on/off | the pure AHA "hard question" |

## Measuring aux recoverability WITHOUT extra full-VLA runs (addresses the cost concern)
Do NOT train a separate full VLA per aux (prohibitive), and do NOT use a frozen-backbone linear probe (R1 proved
it MISranks). Two cheap paths:
- **(primary) AMORTIZED — ~free.** Each arm is trained anyway; the aux head's **held-out (val) loss during that
  training IS its recoverability**: R = 1 − L_val/L_marg (discrete QA ⇒ 1 − CE_val/CE_marg). The dynamics version
  (speed/AULC) = the aux val-loss learning curve — also free. Only extra cost = a cheap marginal/∅-model baseline
  (answer-frequency predictor). So recoverability adds ≈0 cost on top of the arm training that the experiment
  requires anyway. (The real cost is training the arms, not measuring recoverability.)
- **(cross-check) mini-VLA as the instrument.** Use the already-measured mini-VLA recoverability as the predictor
  and correlate with full-VLA success — sidesteps full-VLA recoverability entirely. Caveat: mini=continuous-latent,
  full=discrete-QA, and recoverability is scale-dependent → ranking may not transfer perfectly (that is itself part
  of what M2 tests).

## The money analysis (the AHA cross-validation)
1. **Get each arm's aux recoverability for free** from its training run (amortized aux val-loss, above) — R_deep per aux.
2. **Closed-loop success** per arm on SimplerEnv google_robot (below).
3. **Correlate R_deep(aux) ↔ SimplerEnv success across arms.** Prediction (the law): **negative** — lower-recov aux
   ⇒ higher success. This is the transfer of R1/R2 to a real VLA + closed loop = the oral result.
4. **Composition (min-dominance):** A3 `mix` success ≈ the success of its lowest-recov member (A2 or A5).
5. **KI × action-relatedness (M1 R4):** predict A2 `pred` (action-UNrelated) UNDER hard-KI underperforms its
   KI-off counterpart / langact — i.e., world-model aux starves the flow head under KI; A1/A5 (action-related) do not.

## Closed-loop eval (scaffold is ready: `scripts/simpler/`)
- **Serve:** `python scripts/serve_policy.py --policy.config=exp3_rt1_<arm> --policy.dir=checkpoints/.../<step> --policy.type=flow`
  (openpi websocket server, already supports the Checkpoint dataclass).
- **Eval:** `python scripts/simpler/main.py --task-set visual_matching` (websocket client; google_robot).
  - tasks = `GOOGLE_ROBOT_TASKS["visual_matching"]` (~4 pick/move/drawer tasks), `num_trials_per_task=25`,
    `replan_steps=5` (execute 5 of the 16-step chunk, then re-plan), `resize_size=224`.
  - obs→policy: `image_tools.resize_with_pad(224)`, instruction from env; proprio via `simpler_obs_to_state`
    (mirrors `rt1_dataset_transform`). action→env via `rt1_action_to_simpler` (world_vector[3]+rotation_delta[3]+gripper[1]).
  - **metric = success rate** per task + aggregate, over 25 trials × seeds.

## Gaps to resolve (validate FIRST, before the full sweep)
| gap | action |
|---|---|
| action/proprio convention | **first rollout of A0 `bc`**: confirm `rt1_action_to_simpler` sign/scale + `simpler_obs_to_state` mapping give non-zero success (README explicitly flags this — a wrong convention ⇒ success 0). |
| SimplerEnv install | `pip install simpler-env` + ManiSkill2 in the serving/eval env; verify `simpler_env.make(task)` runs. |
| checkpoints | openpi saves during training; pick a step (e.g., 30–40k) with converged BC + non-starved aux. |
| A5/A6 not yet configs | add `exp3_rt1_fast` (FAST tokenizer over the 16-step chunk) and `exp3_rt1_hardq` (discrete-semantic low-recov obs aux, KI-consistent CE head) to `config.py` — mirror `pred`/`langact` wiring. |
| compute | B200 via **slurm only**; **node200 = user's B200 (rlt5) — needs user sign-off / coordination**, do NOT grab. |

## Sequencing (cost-aware; each gate before the next)
1. **Wrapper + smoke (cheap, 1 arm).** Train/checkpoint `bc` briefly (or reuse any existing pi05 ckpt), serve,
   run a 5-trial SimplerEnv smoke → **validate action/proprio convention + success computes**. (Blocks everything.)
2. **Recoverability probe.** Implement/reuse the aux-recoverability measurement on the pi05 backbone for each aux.
3. **Train the arm set** (A0–A6) to 30–40k on B200 (sequential if 1 slot; parallel if more) — BC-OOD as
   model-selection metric (don't let BC starve).
4. **Eval all arms on SimplerEnv** (25×tasks×seeds) → success table.
5. **Analyze:** recov↔success correlation (the law), composition (A3≈min), KI×action-relatedness (A2 starve test).
6. Write into `AHA_MASTER.md` (R8) + the combined artifact.

## Predictions (pre-registered — test, don't assume)
- **P1 (the law):** across arms, R_deep(aux) ↔ SimplerEnv success is **negative**.
- **P2 (AHA beats BC):** A5/A6 (low-recov, action-related) success > A0 `bc`.
- **P3 (composition):** A3 `mix` ≈ its lowest-recov member.
- **P4 (corrected):** all full-VLA auxes are discrete-QA ⇒ KI-safe; success is driven by each aux's RECOVERABILITY, not by starvation. (The mini-VLA starvation was continuous-aux-specific.)

## Risks (honest)
- **Convention mismatch ⇒ success 0** — mitigated by step-1 smoke (fail-fast).
- **B200 contention / cost** — 40k×~7 arms is multi-day; needs user sign-off on node200.
- **Small success gaps / high variance** — SimplerEnv success is noisy; need enough trials×seeds + CIs.
- **Recoverability of full-VLA aux may not rank like the mini-VLA** — that's exactly what M2 tests; if it doesn't,
  the mini-VLA law doesn't transfer (a real, publishable negative worth knowing).
- SimplerEnv sim-to-real gap (it's a proxy for real success, but the standard RT-1 closed-loop benchmark).

## Relation to prior design
Supersedes the SimplerEnv portion of `EXP3_DESIGN.md` with the AHA framing + M1 findings (recoverability-ranked
arms, the recov↔success cross-validation, KI×action-relatedness, FAST/composition). The OOD-generalization and
real-robot (RoboArena) parts of EXP3_DESIGN remain as later, heavier capstones.

*Status: DESIGN (not yet run). Needs user sign-off for B200 (node200) + the step-1 convention smoke before the sweep.*

---
## PRE-STEP (cheaper): mini-VLA SimplerEnv rollout — CODE READY (2026-07-30), blocked on GPU
Before the expensive full-VLA M2, a cheap closed-loop probe: roll out the (already-trained-able) mini-VLA in
SimplerEnv. Tests offline-R²↔success + AHA-aux vs BC WITHOUT B200. All code + env is ready; only GPU is missing.
- **Env ready:** `/data5/jellyho/Hindsight/SimplerEnv` + `simpler_venv` (maniskill+sapien); `openpi-client` now
  installed in `simpler_venv` (client) and `enc_venv` (server). NOTE: installing openpi-client downgraded enc_venv
  numpy→1.26.4 — verified torch 2.11 + transformers still work (harmless).
- **Code ready:** `probes/save_policy.py` (train+save mini-VLA ckpt + norm stats), `probes/mini_policy_server.py`
  (torch websocket policy server, openpi_client protocol), `scripts/simpler/main.py` (patched: state→8-d
  tcp_pose+gripper), `slurm/rollout.sbatch` (1 GPU: train→serve→SimplerEnv eval), `slurm/render_test.sbatch`.
- **Gate:** `render_test` (33174, queued) — does SAPIEN vulkan render on a GPU node? (login node fails; unknown on
  GPU). If it passes → launch `sbatch --export=ALL,ARM=bc rollout.sbatch` then `ARM=aha`. If it fails → SimplerEnv
  rendering blocked on this cluster (needs infra fix).
- **UNVALIDATED assumptions (validate on first rollout, per README):** (a) state = tcp_pose(7 pos+quat)+gripper vs
  training cartt+grip; (b) gripper action sign (main.py flips [0,1]→[1,-1]); (c) mini-VLA may be too weak (4k–16k
  data) → success ~0 (then: mini-rollout inconclusive, go full-VLA M2).
- **Blocker:** GPU cluster saturated (node01/node100 full of other users' multi-day jobs) — nothing runs yet.
