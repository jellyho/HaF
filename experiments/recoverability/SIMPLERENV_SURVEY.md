# SimplerEnv — Google Robot (fractal / RT-1) suite: exhaustive results + training-setup survey

*Compiled 2026-08-03 by a web-search sweep that screened 469 papers citing arXiv:2405.05941. Numbers were read
from the source tables; anything unverified is marked **UNVERIFIED**. This file exists to calibrate our own runs —
see §4 for what it means for us.*

---

## 0. READ THIS FIRST — three incompatible "average" conventions

| Convention | Definition | RT-1 example |
|---|---|---|
| **A — 4-task plain mean** | mean of {pick coke can, move near, open/close drawer, **put in drawer**} | mean(85.7, 44.2, 73.0, 6.5) = **52.4** |
| **B — 3-task "6-sub-cell" mean** (SpatialVLA / RoboVLM style) | mean of **six sub-cells**: coke-horizontal, coke-vertical, coke-standing, move-near, open-drawer, close-drawer — NOT the mean of the three task columns | mean(96, 90, 71, 44.2, 60.1, 86.1) = **74.6** |
| **B′ — 3-task plain mean** | mean of the three task columns | mean(85.4, 67.5, 60.6) = **71.2** |

**The same model differs by >20 points between A and B.** Worse, some papers mix conventions inside one table
(Discrete Diffusion VLA Tab.2 keeps convention-B numbers for copied baselines but computes plain 3-task means for
its own rows). InstructVLA's printed average pools Google Robot **and** WidowX. Magma's "52.3" is a VM+VA blend.

**Official protocol** (arXiv:2405.05941, App. D) — Visual Matching trial counts:
pick coke can **75** · move near **60** · open/close drawer **54** · place apple in closed top drawer **27**.

**Real-world reference** (from `tools/calc_metrics_evaluation_videos.py` in the official repo):

| Policy | coke (H/V/S) | move near | drawer open / close | apple→drawer |
|---|---|---|---|---|
| RT-1 (converged) | .96/.88/.72 | .633 | .815 / .926 | .185 |
| RT-1 (15%) | 1.0/.96/.80 | .583 | .704 / .889 | .185 |
| RT-1-X | .88/.56/.84 | .450 | .519 / .741 | .407 |
| RT-2-X | .92/.80/1.00 | .733 | .333 / .630 | .074 |
| RT-1 (begin) | .20/.00/.20 | .017 | .000 / .000 | .000 |
| Octo-Base | .44/.20/.24 | .350 | .148 / .519 | .000 |

---

## 1. Master table — Visual Matching per task, VM avg, VA avg

`ZS` = trained on **real** robot data only, evaluated in sim with no further training.
`FT-fractal` = additionally post-trained on the **real** RT-1/fractal dataset (still no sim data).
`FT-sim` = actually trained on simulator-collected data. "—" = not reported.

| Model | Params | Base init | Training data | ZS / FT | Action head | coke | move near | drawer | put-in-drawer | VM avg (conv.) | VA avg | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real robot (RT-1 conv.)** | 35M | — | — | real | — | 85.3 | 63.3 | 87.0 | 18.5 | 63.5 (A) | — | repo `calc_metrics` |
| **RT-1 (converged)** | 35M | **scratch** (EffNet-B3 + USE) | **fractal only** (~130k demos) | ZS | 256 discrete bins, chunk 1 | 85.7 | 44.2 | 73.0 | 6.5 | **52.4 (A)** / 74.6 (B) | 43.7 (A) / 63.3 (B) | CogACT T1; SpatialVLA T-X |
| RT-1 (15%) | 35M | scratch | fractal, 15% ckpt | ZS | ″ | 71.0 | 35.4 | 56.5 | UNVERIFIED | — / 60.2 (B) | — / 56.2 (B) | SpatialVLA T-X |
| RT-1 (begin) | 35M | scratch | fractal, early ckpt | ZS | ″ | 2.7 | 5.0 | 13.9 | UNVERIFIED | — / 6.8 (B) | — / 4.2 (B) | SpatialVLA T-X |
| **RT-1-X** | 35M | scratch | OXE mixture | ZS | 256 bins | 56.7 | 31.7 | 59.7 | 21.3 | **42.4 (A)** / 53.4 (B) | 30.2 (A) / 39.6 (B) | CogACT T1; SpatialVLA T-X |
| **RT-2-X** | **55B** | pretrained VLM **PaLI-X** | OXE + **web VQA co-fine-tune** | ZS | text tokens | 78.7 | 77.9 | 25.0 | 3.7 | **46.3 (A)** / 60.7 (B) | 54.4 (A) / 64.3 (B) | CogACT T1; SpatialVLA T-X |
| **Octo-Base** | 93M | scratch transformer + frozen T5 | OXE 25 datasets / 800k traj | ZS | diffusion head | 17.0 | 4.2 | 22.7 | 0.0 | **11.0 (A)** / 16.8 (B) | 1.2 (A) / 1.1 (B) | CogACT T1; SpatialVLA T-X |
| **OpenVLA** (SpatialVLA run) | 7B | pretrained VLM **Prismatic-7B** | OXE 970k traj | ZS | 256 bins, chunk 1 | 16.3 | 46.2 | 35.6 | 0.0 | 27.7 (B) | 39.8 (B) | SpatialVLA T-X |
| **OpenVLA** (CogACT run) | 7B | ″ | ″ | ZS | ″ | 18.0 | 56.3 | 63.0 | 0.0 | **34.3 (A)** | 39.3 (A) | CogACT T1 |
| **OpenVLA** (ThinkAct run) | 7B | ″ | ″ | ZS | ″ | 15.3 | 47.1 | 49.5 | — | 37.3 (B′) | — | ThinkAct T |
| OpenVLA-OFT | 7B | pretrained **VLA OpenVLA** | OXE → fractal | FT-fractal | parallel decode + L1 | 72.3 | 69.6 | 47.2 | — | 63.0 (B′) | 45.5 | DD-VLA T2; SOMA T4 |
| **TraceVLA-7B** | 7B | pretrained **VLA OpenVLA-7B** | 150k Bridge+RT-1 w/ visual-trace prompts | FT (real) | 256 bins | 28.0 | 53.7 | 57.0 | 0.0¹ | 42.0 (B) | 45.0 (B) | TraceVLA T1 |
| TraceVLA-Phi3-4B | 4B | pretrained VLM Phi-3-Vision | ″ | FT (real) | 256 bins | 52.2 | 50.4 | 31.0 | — | 44.0 (B′) | 43.5 (B′) | TraceVLA T1 |
| **RoboVLM (zero-shot)** | ~2B (KosMos-2) | pretrained VLM | OXE pretrain | ZS | continuous head, history | 72.7 | 66.3 | 26.8 | — | 56.3 (B) | 46.3 (B) | SpatialVLA T-X |
| **RoboVLM (fine-tuning)** | ~2B | ″ | OXE → fractal | FT-fractal | ″ | 77.3 | 61.7 | 43.5 | 24.1 | 63.4 (B) / 51.7 (A) | 51.3 (B) | SpatialVLA T-X |
| **SpatialVLA (zero-shot)** | 3.5B | pretrained VLM **PaliGemma2** | OXE + RH20T **1.1M demos**; 64×A100 ×10d, bs 2048 | ZS | spatial action **grids** (8194 bins), chunk 4 | 81.0 | 69.6 | 59.3 | — | **71.9 (B)** | 68.8 (B) | SpatialVLA T-I/T-X |
| **SpatialVLA (fine-tuned)** | 3.5B | ″ | + full FT on fractal | FT-fractal | ″ | 86.0 | 77.9 | 57.4 | 0.0¹ | **75.1 (B)** / 55.3 (A) | 70.7 (B) | SpatialVLA T-X |
| **CogACT (DiT-B)** | 7.6B (Prismatic + 89M DiT) | pretrained VLM Prismatic | OXE **22.5M frames / 25 datasets** | ZS | **DiT diffusion**, DDIM-10, chunk 15 | 91.3 | 85.0 | 71.8 | 50.9 | **74.8 (A)** | 61.3 (A) | CogACT T1 |
| CogACT (DiT-L 308M) | 7.8B | ″ | ″ | ZS | ″ | — | — | — | — | ~76.7 (A)² | 64.8 (A)² | CogACT abl. |
| **MemoryVLA** | 7B + ~300M DiT + memory bank | pretrained **VLA** (CogACT-style) | **RT-1 only**, 80k steps, 8×A100, bs 256 | FT-fractal | DiT diffusion, chunk 16 | 90.7 | 88.0 | 84.7 | 47.2 | **77.7 (A)** | 67.7 (A) | MemoryVLA T2 |
| **Magma-8B** | 8.6B | pretrained VLM (LLaMA-3 + ConvNeXt) | OXE 970k + UI + video + VQA | ZS | 256 bins | 75.0 | 53.0 | 58.9 | 8.3 | 48.8 (A)³ | 57.5 (A)³ | AsyncVLA T3 |
| **Dita** | 334M | **scratch** transformer (frozen CLIP/DINOv2) | OXE | ZS | in-context **diffusion**, DDIM-20, chunk 16 | 83.7 | 76.0 | 46.3 | — | 68.7 (B′) | 65.3 (B′) | Dita T1 |
| **Moto-GPT** | ~100M | scratch (latent-motion video pretrain) | action-free video → RT-1 | FT-fractal | MLP | 74.0 | 60.4 | 43.1 | — | 59.2 (B′) | — | DD-VLA T2 |
| **HPT** | small trunk | scratch trunk (frozen encoders) | 52-dataset pretrain → few-shot | FT | MLP | 56.0 | 60.0 | 24.0 | — | 46.0 (B) | — | SpatialVLA T-X |
| **π₀ (DD-VLA repro)** | ~3.3B PaliGemma | pretrained VLM | OXE | FT | **flow matching** | 72.7 | 65.3 | 38.3 | — | 58.8 (B′) | 54.8 | DD-VLA T2 |
| **π₀ (EO-1/AsyncVLA repro)** | ~3.3B | ″ | ″ | FT-fractal | flow matching | 97.9 | 78.7 | 62.3 | 46.6 | **71.4 (A)** | 54.7 (A) | AsyncVLA T3 |
| **π₀ (LatBot repro)** | ~3.3B | ″ | ″ | FT | flow matching | 87.3 | 35.0 | 72.6 | 16.0 | 52.7 (A) | 46.0 (A) | LatBot T1 |
| **π*₀ = open-pi-zero** ⭐ | ~3.6B | pretrained VLM **PaliGemma**, **fractal only** | community π₀ reimpl. | FT-fractal | flow matching | 88.0 | 80.3 | 56.0 | — | **70.1** | — | SpatialVLA T-I + footnote |
| **π₀-FAST** | ~3B PaliGemma | pretrained VLM | OXE → fractal | FT | **FAST tokens** | 75.3 | 67.5 | 42.9 | 0.0 | 61.9 (B′) / 46.4 (A) | 59.0 (B′) | DD-VLA T2 |
| **π₀.₅** (so labelled) | ~3B | pretrained VLM | OXE | FT | flow matching | 72.7 | 65.3 | 38.3 | — | 58.8 (B′) | — | SOMA T4 ⚠ identical to "π₀" row above |
| π₀.₅ (StarVLA-α baseline) | ~3B | ″ | ″ | — | ″ | — | — | — | — | 72.7 | 68.4 | StarVLA-α T1 |
| **GR00T-N1 / N1.5** ⚠ label conflict | ~3B (Eagle-2) | pretrained VLM | OXE + sim + neural traj. | FT | flow-matching DiT | 47.0 | 70.0 | 18.1 | — | 45.0 (B′) | 51.5 / 42.4 | DD-VLA T2; SOMA T4 |
| GR00T-N1.6 | 3B | ″ | ″ | — | ″ | — | — | — | — | 67.7 | 65.3 | StarVLA-α T1 |
| **Discrete Diffusion VLA** | Prismatic-7B based | pretrained VLM/VLA | OXE → **fractal**, 4×A800, 100k steps | FT-fractal | **discrete diffusion**, chunk 8 | 85.4 | 67.5 | 60.6 | — | 71.2 (B′) | 56.9 (B′) | DD-VLA T2 |
| **MolmoAct-7B-D (ZS)** | 7B (SigLIP2 + Qwen2.5) | pretrained VLM | OXE subset 26.3M samples | ZS | 256-bin **BPE tokens**, chunk 8 | 71.3 | 73.8 | 66.5 | — | 70.5 (B′) | 59.3 | MolmoAct T |
| MolmoAct-7B-D (FT) | 7B | ″ | + fractal post-train | FT-fractal | ″ | 77.7 | 77.1 | 60.0 | — | 71.6 (B′) | 72.1 | MolmoAct T |
| **ThinkAct** | Qwen2.5-VL-7B + 432M DiT | pretrained VLM | OXE + human video + QA | FT | DiT diffusion | 92.0 | 72.4 | 50.0 | — | 71.5 (B′) | 65.1 (B′) | ThinkAct T |
| **CronusVLA-7B** | 7B | pretrained **VLA** (OXE) | + Bridge+fractal 148k eps, 50k steps | FT-fractal | DiT **diffusion**, 6 past frames | 95.7 | 76.0 | 77.8 | 64.8 | **78.6 (A)** | 73.8 (A) | CronusVLA T1 |
| **EO-1** | 3B Qwen2.5-VL | pretrained VLM | 1.2M robot eps + 5.7M web (**135B tokens**) | ZS (fractal FT UNVERIFIED) | flow matching + AR language | 98.0 | 83.8 | 71.3 | 52.8 | **76.5 (A)** | 63.0 (A) | EO-1 T4 |
| **InternVLA-M1** | 4.1B (Qwen2.5-VL-3B + 86M policy) | pretrained VLM | 3M+ multimodal + OXE; 16×A100, 50k steps | FT | diffusion, chunk 16 | 95.3 | 90.0 | 75.5 | 62.0 | **80.7 (A)** | 76.0 (A) | InternVLA-M1 T |
| **AsyncVLA** | 4.08B (Qwen2.5-VL-3B + FM head) | pretrained VLM | OXE → LIBERO/Bridge/**fractal** | FT-fractal | asynchronous **flow matching** | 98.0 | 82.3 | 70.5 | 50.4 | **75.3 (A)** ⚠ | 64.3 (A) ⚠ | AsyncVLA T3 |
| **DAM-VLA** | DINOv2+SigLIP+LLaMA-2, dual DiT | pretrained VLA (CogACT lineage) | OXE subsets 1M+ traj, 8×H100 | ZS | **dual diffusion heads** | 96 | 84 | 75 | 78 | **83 (A, integers)** | 81 (A) | DAM-VLA T |
| **LatBot** | π₀.₅ student, InternVL3.5-2B + SANA-1.6B | pretrained **VLA π₀.₅** | 1M video eps, latent-action distillation | **FT-sim** | continuous action expert | 96.7 | 91.7 | 90.4 | 33.3 | **78.0 (A)** | 70.1 (A) | LatBot T1 |
| **Xiaomi-Robotics-0** | 4.7B Qwen3-VL-4B | pretrained VLM | ~200M robot timesteps + 80M VL; bs **32768** | FT-fractal | flow-matching **DiT**, chunk 4 | **98.7** | **88.8** | **79.6** | **75.0** | **85.5 (A)** | **74.7 (A)** | Xiaomi T |
| **Hume** | VLM + System-2/System-1 | pretrained VLM | OXE | FT | flow matching + value-guided diffusion | 97.0 | 80.4 | 58.8 | — | 78.7 (3T) | 74.1 | Hume T |
| **villa-X** (LatBot's copy) | ~3.6B PaliGemma | pretrained VLM | OpenX + AgiBot + Ego4D/SSv2 → fractal | FT-fractal | joint diffusion, latent+action | 81.7 | 55.4 | 38.4 | — | — | — | LatBot T1 |
| **DeFI** ⭐aux | SVD GFDM + GIDM + DiT-B | pretrained **video diffusion (SVD)** | OXE/CALVIN/SSv2/Ego4D **fwd+inv dynamics** → fractal | FT-fractal | diffusion adapter | 54.2 | 60.7 | 38.6 | — | 51.2 (B′) | 45.4 (B′) | DeFI T2 |
| **SOMA** | GR00T-N1.5 based | pretrained **VLA GR00T-N1.5** | RoboCasa Tabletop + demos | FT | DiT | 85.0 | 73.0 | 31.5 | — | 63.2 (B′) | 52.5 (B′) | SOMA T4 |
| **GeoAlign** | ~3.6B (GR00T-N1.6 + geometry) | pretrained **VLA** | LIBERO + **SimplerEnv demos** + 580k RGB-D | **FT-sim** | flow-matching DiT | 100.0 | 85.5 | 70.3 | — | 85.3 (own protocol) | — | GeoAlign App.T3 |
| ↳ its RGB-only control | ″ | ″ | ″ | ″ | ″ | 97.5 | 75.5 | 65.8 | — | 79.6 | — | ″ |
| DiT-Policy (baseline) | — | — | — | — | diffusion | 64.3 | 58.9 | 44.9 | — | 56.0 (B′) | — | ThinkAct T |

¹ put-in-drawer = 0.0 per AsyncVLA's 4-task table. ² per-task not printed. ³ Magma's own paper prints per-task only in a figure.

### Additional models (discovery sweep, per-task verified from arXiv HTML)

**Convention A (4-task):** FutureVLA-GT 92.3/74.2/68.5/85.2 → **80.1** VM, 75.6 VA (Qwen3-VL-4B + frozen WAN 3D-VAE, OXE+LIBERO 15.6M frames, flow matching chunk 16) · FutureVLA-OT 97.0/83.8/55.6/74.1 → 77.6/49.4 · StarVLA-OFT 95.3/75.0/68.8/66.1 → 76.0/70.2 (Qwen3-VL-4B, Bridge+fractal, **no sim data**, 100k steps) · RoboInter-IC-E2E 94.0/88.0/73.6/48.1 → 75.9/71.3 (3B Qwen2.5-VL, explicit **ZS cross-embodiment**) · VideoVLA 92.3/82.9/66.2/50.9 → 73.1/62.8 (**CogVideoX-5B as the policy**) · VLA-JEPA ⭐aux 88.3/64.1/59.3/49.1 → **65.2** (Qwen3-VL-2B + **frozen V-JEPA2**, held-out ZS) · CoWVLA 92.3/67.6/42.8/40.7 → 60.9 (**Emu3-8.5B**, fractal only) · FLOWER 56.3/43.3/27.8/**0.0** → 31.9 (**950M** Florence-2, ZS, ~200 GPU-h) · InstructVLA-Expert 87.7/68.3/47.2/61.1 (Eagle2-2B + VQA co-training, ZS — printed avg pools WidowX, do not cite).

**Convention 3-task:** NORA-1.5 94.0/88.0/66.4 → **82.8** (ZS 76.9 → FT 77.9 → **DPO 82.8**) · E-TTS on MolmoAct → 80.0 (training-free test-time scaling; base 71.05) · ST-π 85.5/84.1/68.2 → 79.3 · World-Guidance ⭐aux 89.0/82.5/62.5 → **78.0** (rectified-flow DiT, frozen DINOv2 + Wan VAE) · SimVLA 87.4/65.2/75.9 → 76.1 (**0.5B SmolVLM, no robotic pretraining** ⚠ table labelled VA) · VOTE 78.7/86.7/57.9 → 74.4/62.6 (**179M LoRA on OpenVLA-7B, vision frozen**) · TTT-VLA 85.0/71.7/60.6 → 72.4 (π₀.₅ base 67.5) · Green-VLA R0/R1/R2 → 71.4/66.9/69.9 VM (R2 = RL-aligned) · **Vlaser-2B 85.0/76.3/44.9 → 68.7 — 🔴 trains on demos collected inside SimplerEnv** · Dream-VLA 80.3/78.3/40.7 → 66.5 · LARA 82.3/83.7/29.5 → 65.2 (post-trained on eval datasets) · VITA 57.5/55.8/58.9 → 57.4.

**Training-free / efficiency wrappers on CogACT** (canonical base 74.8 VM / 61.3 VA): Retrieve-then-Steer 79.5 · ElegantVLA 77.59 · AC²-VLA 76.8 · EfficientVLA 76.13 · UAOR 75.7 · LAC 75.5 · SAFE-Pruner 74.5 · VLA-Cache 74.4 · ActDistill 74.08 · EcoVLA 73.6 (its own "vanilla CogACT" = 73.3 ≠ 74.8 ⚠).

---

## 2. Grouped summary

**(a) From scratch (no pretrained VLM/VLA).** Only three genuine members: **RT-1** (35M, fractal-only),
**Dita** (334M, OXE), **Moto-GPT** (~100M). Range **52.4 (A) / 59–75 (B)**. Headline: RT-1 — 35M, fractal only,
256 discrete bins, 2022 — still scores **74.6 (B)**, beating OpenVLA-7B (27.7), Octo (16.8), RT-2-X-55B (60.7),
RoboVLM (63.4). Scale and VLM pretraining buy surprisingly little here; the 4th task (put-in-drawer) is where
modern models actually separate.

**(b) Pretrained VLM → robot data.** The largest group (RT-2-X, SpatialVLA, π₀-family, OpenVLA/CogACT/DD-VLA,
EO-1/InternVLA-M1/AsyncVLA, Xiaomi, MolmoAct, Magma, FLOWER, SimVLA, CoWVLA, VideoVLA, VLA-JEPA…).
Range **11 → 85.5 (A)**; frontier ≈ **76–86 (A)**. Backbone choice matters far less than action head +
post-training: PaliGemma-based models span 31.9 (FLOWER) to 79.3 (ST-π).

**(c) Pretrained VLA → fine-tuned.** OpenVLA → TraceVLA/OFT/VOTE; CogACT → MemoryVLA (77.7), DAM-VLA (83);
GR00T → SOMA/GeoAlign; π₀.₅ → LatBot (78.0), TTT-VLA. Typical gain from a second stage: **+3 to +8**.
Inference-time wrappers move CogACT by **−1 to +5**, i.e. within re-run noise.

**(d) Zero-shot vs sim-finetuned.** Nearly the whole field is **zero-shot to simulation** (trained on real
OXE/fractal/Bridge, evaluated in SimplerEnv with no sim data). What papers call "fine-tuned" usually means
*post-trained on the **real** fractal dataset*, worth **+3 to +7** (SpatialVLA 71.9→75.1 B; RoboVLM 56.3→63.4 B;
MolmoAct VA 59.3→72.1). Genuine sim-trained exceptions: **Vlaser, GeoAlign, LARA, LatBot** — treat separately.

---

## 3. Comparability and reproducibility

**Comparable:** rows within one paper's table. Across papers, two reliable clusters: (i) convention-B family
seeded by SpatialVLA Tab.X (RT-1 variants, RT-1-X, RT-2-X, Octo, OpenVLA, HPT, TraceVLA, RoboVLM, SpatialVLA);
(ii) convention-A family seeded by CogACT Tab.1 (RT-1, RT-1-X, RT-2-X, Octo, OpenVLA, CogACT, MemoryVLA, EO-1,
Xiaomi, InternVLA-M1, AsyncVLA, LatBot, CronusVLA).

**Not comparable:** A vs B (20-pt artifact) · B vs B′ · 3-task vs 4-task (omitting put-in-drawer inflates by
5–15 pts) · Magma 52.3 and Fast-ThinkAct 68.7 are VM/VA blends · InstructVLA's average includes WidowX ·
GeoAlign 85.3 uses SimplerEnv training demos · GR00T official repo uses a non-standard 6-env set (UNVERIFIED).

**Reproducibility problems (this benchmark has real ones):**
- **OpenVLA has no canonical number** — coke 15.3/16.3/18.0, drawer 35.6/49.5/63.0 for one released checkpoint (~27-pt spread).
- **π₀ has four mutually inconsistent rows** (58.8 / 71.4 / 52.7 / 70.1, ~19-pt spread). Neither π₀ nor π₀-FAST reports SimplerEnv Google Robot in its own paper — every number is third-party.
- **Label collisions**: 72.7/65.3/38.3 is called π₀ by DD-VLA/Hume and π₀.₅ by SOMA; 47.0/70.0/18.1 is GR00T-N1 in one table, N1.5 in another.
- **RT-1-X put-in-drawer**: 21.3 (CogACT) vs 40.7 (Xiaomi) — 40.7 is RT-1-X's *real-world* score, so that looks like a transcription error.
- **Wrapper papers re-run their base and get different numbers** (EcoVLA's CogACT 73.3 vs canonical 74.8) — the same size as the effects they claim.

---

## 4. Calibration for US (haf_torch SmolVLM-VLA on fractal)

1. **Our closest published analogue is `open-pi-zero`** (allenzren) — **verified from the repo README, not just
   a citing table**, because the "scratch vs pretrained" question matters:
   - initialises from a **pretrained PaliGemma 3B VLM** (2.291B fine-tuned) **+ a newly initialised action expert
     (0.315B)** — it does **NOT** load Physical Intelligence's π₀ weights; it re-implements the π₀ architecture.
   - **no large-scale robot pretraining**: README says *"I have only trained with either fractal or bridge dataset
     ... so far"*. So it is **not "from scratch"** and **not "robot-pretrained"** — exactly our regime:
     pretrained VLM + fresh action expert + fractal-only + zero-shot to sim.
   - training scale: **fractal 30k gradient steps ≈ 8 epochs**, ~1.5–2 days on one L40 node (8–12 h on H100s).
   - **⚠ number conflict**: the repo README reports **Pick-up-Coke 97.9 / Open-Drawer 49.5**, while SpatialVLA's
     table cites it as **88.0 / 80.3 / 56.0 → 70.1**. Both are in circulation; cite the source explicitly.
   Practical read for us: this is the right band to aim at, but note the gap in scale (3.6B vs our 0.33B) and in
   training (their 30k steps ≈ 8 epochs vs our current 20k steps ≈ 0.8 epoch).

   **Distinguish from official π₀** (Physical Intelligence): that one *is* pretrained on large robot mixtures
   (OXE etc.) before fine-tuning — a different regime, and its SimplerEnv numbers are all third-party repros
   spanning ~19 points.
2. **Second anchor: RT-1 itself** (52.4 A / 74.6 B) — same data, same tasks, scratch, 35M. If our VLM-init model
   on fractal-only lands far below RT-1 on the same convention, suspect the harness/training, not the objective.
3. **🔴 Our harness is not protocol-standard.** `scripts/simpler/main.py` enumerates five env ids
   (`pick_coke_can`, `move_near`, `open_drawer`, `close_drawer`, `place_apple_in_closed_top_drawer`) at
   `num_trials_per_task=25` and takes a plain mean. The official VM protocol is **75 / 60 / 54 / 27** trials with
   the coke-can **orientation sweep** (horizontal / vertical / standing). Fix this before quoting any number —
   otherwise our results sit on neither published axis.
4. **Report BOTH conventions.** Publish the six sub-cells (coke H/V/S, move-near, open-drawer, close-drawer) plus
   put-in-drawer, so convention A and B can both be printed from one run.
5. **Prior art directly relevant to AHA (aux objectives).** Several auxiliary/world-model objectives land *below*
   plain BC baselines, which is itself evidence for the recoverability framing:
   - **DeFI** (explicit forward + inverse dynamics pretraining) → **51.2** VM
   - **VLA-JEPA** (frozen V-JEPA2 predictive objective) → **65.2** (A)
   - **World-Guidance** → 78.0 (B′) · **FutureVLA** (future prediction aux) → **80.1** (A)
   - **CoWVLA / VideoVLA** (world-model / video-generation objectives) → 60.9 / 73.1
   These are the concrete data points for "aux objectives on the recoverability axis".

**Coverage caveat.** The sweep screened 469 citing papers and returned an explicit negative list (X-VLA, F1,
UniVLA, VLM4VLA, GR-3, VPP, SimpleVLA-RL, BitVLA, VLA-Adapter, NORA-original, WALL-OSS, RynnVLA and ~40 more
report WidowX/LIBERO/CALVIN only). Still unverified: SmolVLA, MiniVLA, WorldVLA, Seer, ChatVLA, DiffusionVLA,
Emma-X, LAPA, GO-1/AgiBot, RoboDual, VLAS, RDT, TinyVLA, RoboMamba, RoboFlamingo — none surfaced in any
Google-Robot table read here, which *suggests* they do not report on this suite, but that is **UNVERIFIED**.
