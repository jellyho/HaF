# Overnight log — Exp 1 (target-redundancy) + cross-dataset

**Artifact (read this first):** https://claude.ai/code/artifact/07b228cc-da05-4102-a57b-56353cfb85fe
**Full details:** `RESULTS.md`. **Code:** `stage1_*.py` (per-dataset extract) → `stage2_metrics.py` → `stage3_plot.py` / `plot_learnable.py` → `build_artifact.py`.

## What was done
1. **Exp 1 on DROID** (500 ep, 4000 tx, **15 objectives**) — property-of-data, no policy trained. Frozen DINOv2 + MiniLM probes, GroupKFold by episode.
2. **Cross-dataset replication** on **Bridge (WidowX)** and **RT-1 (Google robot)** — same 15 objectives.
3. Built an **isolated cu128 torch venv** so the **B200s are usable for torch** (`/data5/jellyho/Hindsight/enc_venv`) — slurm L40S was unreliable (NFS-slow startup hit the time limit).

## The result (robust across 3 embodiments)
- **Observation redundancy is symmetric in time and decays with distance:** near-past ≈ near-future; far-past ≈ far-future; **initial observation is the least shortcut-solvable observation target on every robot.**
- **The retrospective initial-observation is the "hard-but-learnable" target** — a probe on the current frame beats the trivial copy by a margin that **grows with scene structure: +0.02 (DROID) → +0.11 (Bridge) → +0.57 (RT-1).** It moves into the useful corner (low shortcut, high learnability). This is exactly the hindsight target the difficulty-axis thesis predicts.
- **Falsification gate passed:** longer episodes make initial-obs *less* redundant (Bridge long-episode R_triv negative), so it is not a short-episode artifact.

## Two honest, dataset-dependent nuances (good science, worth discussing)
- **The forward-action shortcut is control-dependent.** On smooth DROID, next/prev action is the MOST copyable target (R_triv 0.93) — the policy's own objective is the biggest cheat. On delta-action Bridge/RT-1 it flips *anti*-redundant. "Forward target = shortcut" is real but mediated by control smoothness.
- **Vision→instruction shortcut lives in the data.** DROID (diverse): image can't predict the instruction (+0.00). RT-1 (consistent kitchen): it can (+0.22) — mirrors the LIBERO "vision overrides language" finding.

## State of long-running jobs (as of overnight)
- **DROID full download**: resumed, running (~11 MiB/s, ~140/2048 shards). Won't finish by morning; resumable.
- Bridge subset (~20 shards) and RT-1/fractal subset (10 shards) downloaded and used.

## Suggested next steps (for when you're back)
1. **Exp 2 (the payoff):** does training a representation on shortcut-free targets (initial-obs) transfer better / more data-efficiently than forward-only? (frozen-encoder probe on a held-out task).
2. Refine the y-axis to **change-recoverability** (predict o_t−o_0) so obs targets separate off the diagonal on DROID too.
3. Scale DROID episodes (shards keep landing) to see if DROID's initial-obs learnability grows like the others.
