# HaF — proposal talk: speaker script & figure pack

The whole talk is one idea — **recoverability** — used three times: it explains why policies ignore language, why
that overfits, and how to fix it. Say the **bold** line for each slide; the rest is support.

**Figure files** (drop into your slides — 300-dpi PNG + vector PDF in `outputs/`):
- **Evidence 1 (real data, MAIN):** `fig_realdata_recov` — near-obs = copy shortcut (R_triv≈0.7); BC reads vision (sensV≈2) not language (sensL≈0)
- results (1 message each): `fig_p1_ignore` · `fig_p2_overfit` · `fig_p3_cure`
- "how it works" setup schematics: `fig_m_probe` (evidence 1) · `fig_m_sim` (evidence 2–3)
- differentiation (vs LA4VLA Fig 5): `fig_decod_gen` — representation separation ≠ generalization (r=0.26, n.s.)
- motivation (the LLM lesson): `fig_llm_lesson` — NTP is a low-recoverability objective; HaF supplies that forcing function for robots
- qualitative (LA4VLA Fig 5 left analog): `fig_conflict_qual` — predicted action reverses under vision-flip, unchanged under language/state-flip
- concept (why hindsight): `fig_funnel` — entropy funnel; past diverges (erased) vs future converges (goal); near-term symmetric

**Deliverables:**
- Slide deck (arrow-key, 13 slides): https://claude.ai/code/artifact/323384e6-26a3-4eb2-aba1-393e056bc757
- One-pager handout: https://claude.ai/code/artifact/f922a4ee-9582-424e-8373-1cb74f96d59d
- Full spoken script (Korean): `PROPOSAL_SCRIPT_KR.md`

---

### 1 · Title
> **"Robot policies ignore what you tell them. I'll argue that's a shortcut-learning problem — and propose a fix."**
Set the frame: this is about instruction-following in generalist (vision–language–action) robot policies.

### 2 · The problem
> **"A policy trained by imitation often does the same thing no matter what you command — it reads the scene and acts."**
"VLAs don't follow language" is a widely reported failure. If it won't listen, it isn't a general robot — it just
replays memorized demonstrations. (Cite LA4VLA / instruction-following papers here.)

### 3 · The idea (the one concept)
> **"It's not about language — it's about shortcuts. The policy uses the cheapest input to read and ignores the rest."**
Define it once, clearly: **Recoverability = how cheaply a target can be obtained from an input.** Imitation finds
the cheapest input→action mapping, so it exploits the most recoverable cue (usually vision) and drops the others.
Language is ignored because it is *harder to read* than vision — not because it is language.

### 4 · Evidence 1 — ignoring is general, recoverability picks the victim  · `fig_p1_ignore`
> **"Which input gets ignored is set by recoverability, not by which input it is."**
Controlled probe (generalizing LA4VLA §3): train a policy where vision, language, AND state all carry the command.
It commits to just one and ignores the others *though they are correct*. Then make a **different** modality the
easiest to read — the exploited one **rotates** vision → language → state. So "ignore language" is just the special
case where vision happened to be cheapest. *(Say: all three always carry the true command — nothing is a trick.)*

### 5 · Evidence 2 — why we should care  · `fig_p2_overfit`
> **"Taking a shortcut is overfitting: perfect on training, failing on new situations."**
A controlled simulation where the cheap shortcut *breaks* out-of-distribution. The policy fits train and
in-distribution data perfectly, then the error explodes on new situations. The disease isn't cosmetic — it's the
generalization failure that blocks generalist robots.

### 6 · Evidence 3 — the cure  · `fig_p3_cure`
> **"Add an objective the shortcut can't solve, and generalization comes back — the lower its recoverability, the better."**
Co-train a shortcut-free objective; it forces a genuine representation. Left: OOD error drops monotonically as the
objective's recoverability drops. Right — the honesty slide — **the same law holds on 6 real robot datasets
(r = −0.93)**: the auxiliary helps most exactly where vision was a shortcut. Controlled proof + real-data
confirmation.

### 7 · The proposal — HaF
> **"Hindsight and Foresight: give the policy objectives a shortcut can't fake."**
Three levers: (1) **hindsight/foresight targets** — predict the past, the goal, progress (not recoverable from the
present frame); (2) **input-side masking (JEPA)** — hide the shortcut cue to lower recoverability; (3) **validate at
scale** — train a real VLA on RT-1, measure instruction-following generalization in SimplerEnv.

### 8 · Close
> **"One idea — recoverability — explains the failure and prescribes the fix: from memorizing demonstrations to understanding tasks."**

---

## Anticipated questions
- **"Isn't the sim artificial?"** — Yes, deliberately: it *isolates* the mechanism with one cause per effect. The
  claim isn't a magnitude; it's the mechanism, and the **r = −0.93 on real datasets** shows it isn't a toy artifact.
  Exp 3 (real VLA) is exactly the scaled test.
- **"Why does vision win?"** — It's usually the most recoverable path. Fig 1 proves it's recoverability, not vision:
  degrade vision and language or state takes over.
- **"Isn't this just auxiliary losses / data augmentation?"** — The point is *which* auxiliary: only
  **low-recoverability** objectives help (Fig 3, monotone). Recoverability tells you which auxiliaries are worth
  adding — that's the contribution.
- **"How is recoverability measured?"** — cheap-probe loss vs a trivial copy baseline (Exp 1). See `METHODS.md`.
- **"Isn't predicting the past the same difficulty as predicting the future? Why hindsight over foresight?"** (`fig_funnel`) — Concede first: **at short horizon, yes** — near-past ≈ near-future ≈ a copy of the present (our R_triv is identical, ~0.7), so both are shortcuts. The axis isn't the arrow of time, it's **recoverability**. The real asymmetry appears at **long horizon** because goal-directed demos are **irreversible**: forward is many-to-one (converges to the goal → recoverable), backward is one-to-many (the origin is washed out by contact/occlusion → not recoverable). And the strongest hindsight targets have **no prospective analog** — the instruction/latent task, displacement-from-start, progress — the scene doesn't determine them. So we claim "low-recoverability forces grounding," not "past is harder than future." Honest scope: the asymmetry scales with task **irreversibility** (strong for contact-rich manipulation, weak for free-space motion).
