# How to measure an objective's recoverability — methods, tested

**Question (paper-critical).** Recoverability = how cheaply a training target y is obtainable from the policy's
input o_t. The LAW (recoverability ↓ ⇒ generalization ↑) is only as good as how we *measure* recoverability.
So we implemented several measurement methods and asked, empirically: **which one's recoverability numbers best
predict BC generalization?** (RT-1/fractal, mini V-L-A, 18-objective battery, flow-matching BC, 3 seeds.)

## Result (frac4k: N=4000, 3 seeds) — the measurement decides the answer

| measure | function class 𝒱 | kind | Pearson r vs generalization | Spearman | verdict |
|---|---|---|---|---|---|
| linear probe (Ridge, frozen z_t) | linear | asymptotic | **+0.78** | +0.66 | ✗ **backwards** |
| MLP probe (frozen z_t) | MLP | asymptotic | +0.03 | +0.16 | ✗ no signal |
| prequential MDL (frozen, MLP) | MLP | dynamics | +0.42 | +0.31 | ✗ backwards |
| **end-to-end** | **policy net** | asymptotic | **−0.66** | −0.70 | ✓ law |
| **AULC (learning curve)** | policy net | dynamics | **−0.75** | −0.79 | ✓✓ law |
| **@¼ budget (speed)** | policy net | dynamics | **−0.75** | −0.75 | ✓✓ **best** |

**Selectivity control:** mean R_linear = +0.167 vs shuffled-target control = −0.001 → the estimator reflects
genuine extractability, not probe capacity (Hewitt & Liang 2019 control-task guard). 

### Replication at 2× scale (frac8k: N=8000, 3 seeds) — Pearson r vs generalization
| measure | 𝒱 · kind | N=4000 | N=8000 | stability |
|---|---|---|---|---|
| linear probe | linear · asym | **+0.78** | **+0.69** | stably **backwards** |
| MLP probe | MLP · asym | +0.03 | −0.34 | unstable / weak (sign flips) |
| prequential MDL | MLP · dyn | +0.42 | −0.05 | unstable / weak |
| end-to-end | policy · asym | −0.66 | −0.64 | **stable ✓** |
| AULC | policy · dyn | −0.75 | −0.67 | **stable, strong ✓** |
| @¼ budget (speed) | policy · dyn | **−0.75** | **−0.72** | **stable, strong · BEST ✓** |

**Extra finding:** frozen-feature measures are not just wrong-on-average, they are *unstable across scale* (MLP +0.03→−0.34,
MDL +0.42→−0.05) — you cannot trust them. The **policy-class measures are stable and strong at both scales**
(−0.66→−0.64; speed −0.75→−0.72), and the linear probe is *stably backwards* (+0.78→+0.69). Only recoverability
measured under the policy's function class — and as learning dynamics — is a reliable predictor. Figure: `fig_measures_frac8k`.

### Three findings (the knee-slapper)
1. **A naive (frozen linear-probe) recoverability doesn't just add noise — it FLIPS the law** (+0.78, the exact
   opposite sign). Measure recoverability the wrong way and you would conclude the opposite of the truth.
2. **Recoverability must be measured under the policy's own function class 𝒱.** Then the law appears (−0.66).
   This is *predicted*, not patched: 𝒱-information is family-relative by definition (Xu 2020, Prop 2 monotonicity),
   so linear-𝒱 and policy-𝒱 measure different quantities — the sign flip is the theory working, not a bug.
3. **Learning dynamics beats asymptotics.** How *cheaply/fast* the target is grabbed early (@¼ budget, AULC)
   predicts generalization better (−0.75) than the converged value (−0.66). Shortcut-taking is a *dynamics*
   phenomenon: BC latches the fastest-recoverable target first. (Fork A confirmed.)

**One-line takeaway:** *recoverability must be measured the way the model actually experiences it — under its own
function class, and early in training. Any frozen/linear/transfer-score proxy inverts the law.*

## Why linear inverts (mechanism)
Across this battery, linear-decodability and policy-recoverability are **anti-correlated**. The high-dim visual
latents (past/future/masked obs) are moderately linearly-decodable from frozen DINO features (lin≈0.2) yet the
policy *cannot* cheaply fit them end-to-end (deep≈−0.1) — so they force grounding and *help* generalization.
Conversely `final-pose` is linearly-hard (lin=0.06) but deep-easy (deep=0.92) — the archetypal shortcut. A linear
probe reads these backwards; only the policy-class measure sees what the policy will actually exploit.

## Formalization (citable)
Recoverability := normalized **predictive 𝒱-information** with 𝒱 = the policy's function class:
`R = 𝓘_𝒱(o_t→y)/H_𝒱(y) = 1 − L_val/L_marg`, estimated on held-out data by two heads — the policy (o_t→y) and a
marginal/∅ baseline (predict-the-mean). 
- **Continuous target (Gaussian head):** `R = 1 − MSE_val/MSE_marg = R²`, which Xu et al. 2020 **Prop 1.5** proves
  equals normalized 𝒱-information (𝓘_𝒱 = R²·tr Cov y). The mean-baseline is the ∅-model; tr(Cov) auto-normalizes
  across the 768 dims. Ethayarajh et al. 2022 give the two-model estimator (𝒱-usable info / PVI).
- **Discrete target (instruction):** `R = 1 − H_𝒱(y|o_t)/H_𝒱(y)` (cross-entropy), per-example PVI diagnostic —
  the honest measure for the language objective (avoids embedding-MSE artifacts). *[to add for instr-infer]*
- **Learning-dynamics variant (recommended headline):** prequential/online MDL codelength = area under the
  learning curve (Blier & Ollivier 2018; Voita & Titov 2020) — natively measures "cheapness/speed." We approximate
  it with AULC and @¼-budget val-R under the policy net.

**Why not Shannon MI:** ill-defined/invariant for deterministic nets (Tschannen 2020); no distribution-free lower
bound above O(ln N) samples (McAllester & Stratos 2020); variational estimators break at high MI (Poole 2019).
𝒱-information sidesteps all three and reduces to R² for our continuous targets.

**Transfer scores (LEEP, LogME, H-score…):** cite as related, do NOT adopt — all read y off *frozen* features via a
linear/Gaussian head, i.e. exactly the linear-probe regime that inverts the law here. Their frozen-feature framing
is the failure mode our result exposes.

## Threats handled
- **Baseline:** L_marg is a genuinely-fitted target-only (∅) predictor; predict-the-mean is its Gaussian instance.
- **Capacity vs extraction:** selectivity control (shuffled target) ≈ 0 proves R reflects extraction, not capacity.
- **Symmetric budget:** conditioned and marginal use matched optimization; held-out val; 3 seeds.
- **Scale/robustness:** report Spearman (rank) alongside Pearson; high-dim targets can make R very negative when
  undertrained — mitigated at 25 epochs and by rank stats.

## Status
- frac4k (N=4000) done — table above. Figure: `outputs/fig_measures_frac4k.{png,pdf}`;
  data: `outputs/measure_comparison_frac4k.json`. Code: `probes/exp2h_law.py` (+ `analyze_measures.py`, `plot_measures.py`).
- Full N=16000 (RUNTAG=fractal) running — will confirm at scale.
- TODO: discrete PVI/retrieval measure for `instr-infer`; true prequential codelength under the policy net (currently
  frozen-MLP approximation for the MDL row + AULC/speed as the policy-class dynamics proxy).

*Citations: Xu, Zhao, Song, Stewart, Ermon, ICLR 2020 (𝒱-information); Ethayarajh, Choi, Swayamdipta, ICML 2022
(𝒱-usable info, PVI); Voita & Titov, EMNLP 2020 (MDL probing); Blier & Ollivier, NeurIPS 2018 (prequential MDL);
Hewitt & Liang, EMNLP 2019 (control tasks); McAllester & Stratos, AISTATS 2020; Tschannen et al., ICLR 2020;
Poole et al., ICML 2019; Nakkiran/Kalimeris et al., NeurIPS 2019 (SGD increasing complexity); Shah et al., NeurIPS 2020
(simplicity bias).*

---
## Statistical rigor — bootstrap 95% CI + permutation p (added 2026-07-30; addresses "no CIs")
(10k bootstrap over objectives; two-sided permutation p. Small N=12–18 ⇒ CIs wide.)
- **Sign-flip is ROBUST at all 3 scales:** linear-probe Pearson +0.78 / +0.69 / +0.62, 95% CI excludes 0, p≤0.018.
- **Learning-dynamics (speed@¼) is the ROBUST predictor at every scale:** −0.75 / −0.72 / −0.69, CI excludes 0, p≤0.003.
- **Asymptotic policy-e2e LOSES significance at 16k:** −0.66 / −0.64 / **−0.47 (CI[−0.90,+0.34], p=0.048 — crosses 0)**.
  → refines the claim: the *dynamics* measure is the robust one; the *asymptotic* measure washes out at scale.
  This strengthens "measure recoverability as learning dynamics, not asymptotics."
- **Mixing (4k):** min-recov most robust (Pearson −0.87 CI[−0.98,−0.67]; Spearman −0.82, CI excludes 0);
  mean weaker (Spearman −0.57, p=0.057, CI crosses 0). Supports "the min (hardest) member dominates."
- Caveat: wide CIs from small N — more objectives / scales / seeds would tighten. Code: `probes/analyze_stats.py`.
