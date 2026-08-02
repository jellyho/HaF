# CONTINUE HERE — AHA project handoff (resume on any cluster / new session)

**Paper:** "Regularizing Vision-Language-Action Models by Asking Hard Questions".
**Method:** **AHA** (Asking Hard-question Auxiliaries) — NOT in the title, method name only.
**One-line thesis:** *Recoverability* = normalized predictive 𝒱-information of an auxiliary target y given the
policy's observation o_t, under the policy's own function class (`1 − L_val/L_marg`). BC takes the most-recoverable
shortcut ⇒ overfits; co-training a LOW-recoverability ("hard question") auxiliary forces grounding ⇒ better OOD
generalization. Goal = an **ORAL** that UNIFIES all prior VLA auxiliary-loss work onto this one axis.
**Core claim = GENERALIZATION failure, not "ignores language"** (ignoring is the diagnostic/mechanism).

---

## 0. Status snapshot (2026-08-02)
| id | result | status |
|---|---|---|
| R1 | Measurement decides the answer: frozen linear probe INVERTS the sign (+0.78) vs policy-class dynamics (−0.75); 3 scales, CIs | **SOLID** |
| R2 | the LAW: recoverability↓ ⇒ generalization↑ (per objective) | **SOLID** |
| R3 | composition: mix ≈ its lowest-recoverability member (min r=−0.87, CI excl 0) | PRELIM |
| R3b | count-sweep "saturation" + "diversity inert" → **DIAGNOSED AS A LOSS-NORMALIZATION ARTIFACT** (see §4). Corrected experiment `exp2h_countfix.py` is the live task. | ⚠ under revision |
| R4 | KI: hard-KI starves an action-UNrelated aux; action-related (FAST-like) survives | PRELIM |
| R4b | FAST-lite: discrete action-token recoverability is sign-stable, 3.4× lower-variance than continuous | PRELIM |
| R7 | survey: ~55 prior VLA auxiliaries collapse onto the axis (qualitative) | DONE |
| M2 | full-VLA (π0.5 from PaliGemma) + SimplerEnv closed-loop | DESIGN (make-or-break, not run) |

**Live dashboard:** https://claude.ai/code/artifact/f2689f8b-8565-4e3b-a775-de2b07328a1f (source
`dashboard/index.html`). **Truth source:** `AHA_MASTER.md`. **Full prose draft:** `PAPER_DRAFT.md`. **ICML LaTeX:**
`paper/` (Overleaf-ready, compiles).

---

## 1. Environments (three venvs on the original cluster, under /data5/jellyho/Hindsight/)
Recreate with the same package sets; paths below are the originals.
- **`.venv`** (repo root `/data5/jellyho/Hindsight/HaF/.venv`) — **has EVERYTHING**: cv2, tensorflow, tensorflow_datasets,
  torch 2.7, transformers 4.53, sklearn. Used for: dense extract (`stage1_fractal_dense.py`), dense training
  (`save_policy_dense.py`), figures. **uv-managed (no `pip` module — use `uv pip`).**
- **`enc_venv`** (`/data5/jellyho/Hindsight/enc_venv`) — torch + transformers + sklearn + openpi_client + websockets.
  Used for: exp2h measurement/mixki/fastlite/countfix, the policy server (`mini_policy_server.py`). No cv2. uv-managed.
- **`simpler_venv`** (`/data5/jellyho/Hindsight/simpler_venv`) — SimplerEnv + ManiSkill2 + sapien(vulkan) + tyro.
  Used for the SimplerEnv client (`../../scripts/simpler/main.py`). No tensorflow.
- **GOTCHA (protobuf):** a protobuf version override fixes a tfds import crash — see the `env-setup` note if tfds
  import fails.

## 2. Data (must be present on the new cluster)
- **RT-1 / fractal**: `/data5/jellyho/Hindsight/fractal_rlds/fractal20220817_data/0.1.0` — 1024 TFRecord shards,
  112 GB, 87k episodes. Copy this (or re-download the RLDS) to the new cluster; update `DATA=` in
  `extract/stage1_fractal*.py` if the path differs.
- **Caches** live in `outputs/cache/` and are **git-ignored (regenerable)**:
  - `transitions_fractal.npz` (raw frames, sparse anchors) + `dino_latents_fractal.npz` — built by
    `extract/stage1_fractal.py` (MAX_EP, N_T=8) then `measure/stage2_metrics.py` (DINOv2-base encode). Needed by
    exp2h_law / exp2h_mixki / exp2h_countfix / exp2h_fastlite.
  - `dense_fractal_20000.npz` — built by `extract/stage1_fractal_dense.py` (JPEG frames + z_fl latents, 20k eps).
    Needed by `save_policy_dense.py` (the closed-loop rollout).
- **Disk warning:** original /data5 was 98% full. Dense extract stores Ft as JPEG (~15 GB) not raw (~120 GB).

## 3. How to run each experiment (env vars → venv → script; all GPU via slurm)
All sbatch files in `slurm/`. **GPU RULE: slurm only, never grab a GPU on a shell node.** L40S ~6–8 GB VRAM is
plenty (see below); jobs wait on GPU *count* allocation, not VRAM.

- **Regenerate caches** (once): `sbatch slurm/prep_scale.sbatch` (stage1 raw+anchors, `.venv`) then
  `sbatch slurm/encode.sbatch` (stage2 DINOv2-base, `enc_venv`). TAG=fractal.
- **R1 measurement study** (sign-flip, 6 measures × scales): `slurm/exp2h.sbatch` → `probes/exp2h_law.py`
  (env SUBN∈{4000,8000,16000}, SEED, EPOCHS). Analyze: `probes/analyze_stats.py` (bootstrap CI + perm p).
- **R3/R4 mixing + KI**: `slurm/exp2h_mixki.sbatch` → `probes/exp2h_mixki.py` (env MIXSET=base|deep, RUNTAG, SEED,
  SUBN=4000). Figures: `probes/plot_mixki.py`, `probes/plot_deepen.py`.
- **R3b CORRECTED count-sweep (THE LIVE TASK)**: `slurm/countfix.sbatch` → `probes/exp2h_countfix.py`
  (env SEEDS="0 1 2", EPOCHS=20, SUBN=4000, KMAX=6). See §4 for design + how to read it.
- **R4b fastlite**: `slurm/exp2h_fastlite.sbatch` → `probes/exp2h_fastlite.py`.
- **20k dense closed-loop rollout** (mini-VLA, trainable encoder):
  1. `sbatch slurm/dense_extract.sbatch` (env MAX_EP=20000, N_T=30) → `extract/stage1_fractal_dense.py` → writes
     `outputs/cache/dense_fractal_20000.npz`. **Uses `.venv`. TF is forced CPU-only (`tf.config.set_visible_devices([],'GPU')`)
     so torch/DINO gets the GPU — do NOT remove that line (removing it → the 44 GB OOM we already hit).**
  2. `sbatch --export=ALL,ARM=bc slurm/rollout_dense.sbatch` then `ARM=aha` → trains `save_policy_dense.py`
     (JPEG DataLoader → trainable DINOv2-small), serves `mini_policy_server.py` (enc_venv), runs SimplerEnv
     (`scripts/simpler/main.py`, simpler_venv, task-set visual_matching). Output success rate in the slurm log.
  - The AHA arm's auxiliary = far-future observation latent (`z_fl`, a low-recoverability hard question).

## 4. R3b — the count-sweep artifact + the corrected experiment (`exp2h_countfix.py`)
**What went wrong (user caught it):** the old deep-mixing "saturation"/"diversity-inert" panels showed benefits all
clustered ~0.04 within noise. Root cause is mechanical, not scientific: a mix was ONE linear head over the
CONCATENATION of member targets with a single `nn.MSELoss()` (mean over all dims) at fixed 1:1 BC:aux weight ⇒ the
aux gradient budget into the backbone is CONSTANT in the member count K; each member gets 1/K of it. Saturation was
baked in by loss normalization. (Diversity panel had the same flaw at fixed K=2.)

**The fix (`probes/exp2h_countfix.py`, `slurm/countfix.sbatch`):** for K=1..6 diverse low-recov members (ordered
hardest-first, distinct groundings: fut-action → far-fut-obs → far-past-obs → near-fut-obs → initial-obs →
displacement):
- **SUM**: K independent heads, EACH weight 1 → total aux pressure = K (the honest count-sweep).
- **MATCHED**: 1 head on the single hardest member, weight = K → SAME total pressure, one question (fair control).
- **MEAN**: old concat+mean-MSE, weight 1 → reproduces the flat ~0.04 artifact (sanity).
**Read:** SUM rising with K AND SUM>MATCHED ⇒ diverse hard questions add complementary grounding (NOT saturation,
supports the user's intuition). SUM≈MATCHED ⇒ only total weight mattered. SUM rises-then-falls ⇒ over-regularization.
**Analysis TODO when 3-seed json (`outputs/exp2h_countfix_s{0,1,2}.json`) lands:** aggregate SUM/MATCHED/MEAN vs K
across seeds; if signal, rewrite `PAPER_S5_law_composition.md` + fig_deepen panel A from "saturates" → the corrected
story; update `M1_RESULTS.md`, `AHA_MASTER.md`, dashboard. If no signal, demote honestly to a bounded-null sentence.

## 5. Known gotchas / lessons (don't re-hit these)
- **TensorFlow grabs the WHOLE GPU** on init → torch OOM. Any script importing tf + torch must call
  `tf.config.set_visible_devices([], 'GPU')` right after `import tensorflow` (done in stage1_fractal_dense.py).
- **`torch.load` weights_only**: checkpoints hold numpy arrays → load with `weights_only=False` (mini_policy_server).
- **`tyro`** must be installed in simpler_venv (SimplerEnv client arg parsing).
- **SimplerEnv state**: `scripts/simpler/main.py` `simpler_obs_to_state` was patched to 8-d = tcp_pose[7]+gripper[1]
  to match training state.
- **openpi_client is CLIENT-only** (no server) → `mini_policy_server.py` is a custom torch websocket server
  (msgpack_numpy protocol; sends metadata on connect, returns {"actions"}).
- **VRAM**: mini-VLA (trainable DINOv2-S, BS=128, 224²) peaks ~6–8 GB; fits any ≥12 GB card. BS=64 → ~3–4 GB.
- **Do NOT touch** the user's other projects sharing the cluster: RLT (`/data5/jellyho/ACRFT/openpi`, node200 B200,
  jobs `rlt*`,`c7_*`,`rollsmoke`) and humanoid_bench (`lh_*`). Separate projects — never cancel/preempt them.

## 6. Pending tasks (priority order)
1. **countfix 3-seed** → analyze → fix the composition section (§4 above). *In flight.*
2. **20k dense rollout** (bc + aha) → BC-vs-AHA closed-loop success → replace the `\prov{}` placeholders in
   `paper/main.tex` Table 3 + abstract with measured numbers.
3. **M2 full-VLA** (π0.5 from PaliGemma + SimplerEnv) — the make-or-break external-validity test. Needs a big GPU.
4. Verify all 2025/26 arXiv ids before submission (esp. Voita-Titov → cite ACL Anthology; Hejna 2306.12554 OK).
   See `CITATIONS_THREATS.md` (biggest threat = "assembly", defused there).

## 6b. Experts + NTP + dynamic KI (added 2026-08-02, mirroring real VLA training)
New reusable modules in `experts/` (all CPU-smoked; scale via d_model/depth/nhead → mini to pi0.5):
- **`experts/latent_expert.py` — `LatentPredictionExpert`**: a future-frame-LATENT expert, analogous to the
  pi0/pi0.5 flow-matching ACTION expert. Cross-attends the backbone features; heads = `flow` (flow-matching, same
  interface as the action expert: `flow_loss`/`sample`), `mse` (continuous embedding regression), `vq` (discrete
  codebook cross-entropy = VQ-image / discrete frame tokens). This is the AHA low-recoverability aux as a proper
  expert at pi0.5 scale. Gradients ground the backbone; detach the ctx to apply KI.
- **`experts/token_ntp_expert.py` — `TokenNTPExpert`** (generic): autoregressive NTP over ANY discrete token
  stream, cross-attending the backbone (causal self-attn + cross-attn + LM head). Serves BOTH: ACTION via the
  **official FAST tokenizer** (`AutoProcessor("physical-intelligence/fast", trust_remote_code=True)`, DCT+BPE,
  vocab 2048, near-lossless) = OpenVLA/pi0.5-FAST style, AND TEXT objectives (instruction/subtask/CoT) via a text
  tokenizer = RT-2/ECoT style. Recoverability the "real VLA" way: `R = 1 - CE_val/CE_marg`.
- **`probes/exp2h_dynki.py` — dynamic KI**: release the BC-head→backbone stop-grad when the BC head's gradient into
  the backbone has DECAYED (EMA < RELFRAC×peak), i.e. once the flow expert warmed up — vs fixed tau. Logs release epoch.
- **`probes/exp2h_actiontok.py` (+ `slurm/actiontok.sbatch`)**: measures where each TARGET FORM sits on the
  recoverability axis — action {cont-MSE, FAST-NTP, OpenVLA-256bin-NTP} + instruction {embed, text-NTP}, one shared
  backbone. Runs in **`.venv`** (superset: torch+transformers+sklearn+FAST). SEEDS/EPOCHS/SUBN env.

**Data plan (decided 2026-08-02): (a) then (b).**
- **(a) now** — run the new expert/NTP experiments on the existing **16k cache** (recoverability rankings are stable
  across 4k/8k/16k per R1, so this is scientifically sufficient for the *measurement*). Jobs queued.
- **(b) next** — scale to bigger/full data with **DYNAMIC DINO**: do NOT pre-cache DINO latents (that caused the
  98%-disk problem). Hold a frozen DINO-base in-process and encode target frames on-the-fly per batch; input encoder
  is the trainable DINO-small on raw/JPEG frames. Then only FRAMES are needed (JPEG ~60GB for full, or stream
  TFRecords) — no latent cache, scales to full 87k. Apply this to `save_policy_dense.py` / a dynamic variant.

Still-open builds: **VQ-image** experiment (diffusers `VQModel` VQGAN → frame codes → `LatentPredictionExpert` vq
head; diffusers 0.39 confirmed in `.venv`) and wiring the FAST-NTP/latent experts into the full pi0.5 (M2).


## 6c. RT-1-style lightweight real-VLA (added 2026-08-02) — the first real-architecture test
Instead of jumping to pi0.5, first test the thesis on a genuine but tiny VLA. RT-1 is ideal: our data IS RT-1's
(fractal) and RT-1 is a SimplerEnv baseline. Backbone = image encoder -> FiLM(language) -> TokenLearner (K tokens)
-> Transformer -> tokens [B,K,d]; head-agnostic so our experts plug in as the action/aux heads.
- **`experts/rt1_vla.py`** (torch, .venv): EfficientNet-B0 (pretrained) + FiLM + TokenLearner + Transformer;
  `BinActionHead` (RT-1/OpenVLA 256-bin). Smoked: TokenNTPExpert(FAST) and LatentPredictionExpert(latent aux)
  both attach to the RT-1 tokens.
- **`experts/rt1_vla_jax.py`** (JAX/Flax linen, matches src/haf backbone style; SigLIP-swap noted): self-contained
  ConvStem+FiLM+TokenLearner+Transformer + BinActionHeadJAX. Smoke: `JAX_PLATFORMS=cpu python rt1_vla_jax.py`.
  This is the path to integrate with the user's JAX stack (src/haf, openpi/pi0.5) for M2.
Plan: train RT-1 on fractal, eval in SimplerEnv (visual_matching); arms = BC(256-bin) / +low-recov latent aux /
swap action tokenizer {256-bin, FAST, VQ}. First "real VLA architecture x recoverability" check. GPU via slurm.

## 7. Doc index
`AHA_MASTER.md` (status truth) · `PAPER_DRAFT.md` (full prose) · `PAPER_S*.md` (per-section) · `PAPER_OUTLINE.md` ·
`RECOVERABILITY_MEASUREMENT.md` (R1) · `M1_RESULTS.md` (R3/R4) · `M2_DESIGN.md` · `SURVEY_vla_auxlosses.md` (R7) ·
`CITATIONS_THREATS.md` (verified refs) · `paper/` (ICML LaTeX) · `dashboard/index.html` (live status).
