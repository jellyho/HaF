# §4 draft — Measuring recoverability: you must measure it right

*Prose draft of the paper's anchor (SOLID) result. Numbers are the verified 3-scale study (4k/8k/16k transitions, 3 seeds, 18 objectives). Sources: `RECOVERABILITY_MEASUREMENT.md`, `analyze_stats.py`. Status: R1 SOLID — the one result that stands without M2.*

---

## 4. Measuring recoverability: you must measure it right

Section 3 defined the recoverability of an auxiliary target $y$ as the normalized predictive $\mathcal{V}$-information it carries about itself given the observation $o_t$, under the policy's own function class $\mathcal{V}$: $R_\mathcal{V}(o_t\!\to\!y)=1-\mathcal{L}^*_\mathcal{V}(y\mid o_t)/\mathcal{L}^*_\mathcal{V}(y\mid\varnothing)$, estimated as $1-L_\text{val}/L_\text{marg}$ (an $R^2$ for continuous targets, $1-H_\text{val}/H_\text{marg}$ for discrete). The law we test is that *lower* recoverability predicts *better* out-of-distribution (OOD) generalization when $y$ is used as a co-training auxiliary. But the definition hides a decision that turns out to be the whole story: **under which function class, and at what point in training, do you measure $\mathcal{L}^*$?** We show the natural, convenient choice — a frozen linear probe on pretrained features — does not merely add noise: **it inverts the law's sign.** Getting the measurement right is the result.

### 4.1 Setup

We train a mini vision–language–action policy (trainable DINOv2-S image encoder, frozen MiniLM language encoder, proprioceptive state, fused by a 2-layer transformer into a 256-d representation, with a flow-matching action-chunk head) on a 2.3% subsample of RT-1/`fractal` (2k/87k episodes). For each of 18 auxiliary targets — prospective (near/far future obs and actions, final obs/pose), retrospective (near/far past obs, previous action, initial obs/pose, displacement), and introspective (masked-image reconstruction at 25/50/75%, state-infer, instruction-infer) — we measure recoverability six ways and, independently, measure the target's effect on OOD generalization by co-training it with behavior cloning and evaluating held-out action $R^2$. We repeat the entire study at three dataset scales (4k, 8k, 16k transitions) and three seeds. We report Pearson and Spearman correlations between each recoverability measure and generalization, with 10k-sample bootstrap 95% CIs and two-sided permutation $p$-values over the objective set.

The six measures cross two axes — **function class** (linear probe / MLP probe / the policy network itself) × **read-out time** (asymptotic converged loss / learning-dynamics). A *selectivity control* re-runs each estimator against a label-shuffled target; a faithful estimator should collapse to zero.

### 4.2 Headline 1 — the sign flip

The measure you reach for first is a frozen linear probe on the pretrained image features, because it is cheap and standard. **It is not merely weak — it is backwards.** Frozen linear-probe recoverability correlates with generalization at Pearson **+0.78 / +0.69 / +0.62** across the three scales (95% CI excludes 0, $p\le0.018$): it confidently ranks the *most* linearly-decodable targets as the *best* auxiliaries, which is the exact opposite of the law. Moving to the policy's own function class flips the sign to where the theory predicts: end-to-end recoverability correlates **−0.66 / −0.64 / −0.47**. The selectivity control confirms the estimator measures extraction, not capacity: mean linear recoverability +0.167 on real targets versus −0.001 on shuffled targets.

This is not an artifact of one estimator. The frozen-*feature* measures are not just wrong on average, they are **unstable across scale** — the frozen MLP probe swings +0.03 → −0.34, prequential MDL on frozen features +0.42 → −0.05 — so no reading of them can be trusted. Only measures taken **under the policy function class** are both correctly-signed and stable across scale.

### 4.3 Headline 2 — measure it as a learning dynamic, not an asymptote

Given the right function class, *when* you read the loss still matters. A shortcut is defined by being grabbed *first*, so the learning-dynamics estimators — area under the validation-loss learning curve (AULC) and the validation recoverability at a quarter of the training budget (speed@¼) — should predict generalization better than the converged asymptotic loss. They do. Speed@¼ is the single best and most stable predictor: **−0.75 / −0.72 / −0.69**, CI excluding 0 at every scale ($p\le0.003$); AULC tracks it closely (−0.75 / −0.67). By contrast the asymptotic end-to-end measure **loses significance at the largest scale**: −0.66 / −0.64 / **−0.47, CI [−0.90, +0.34], $p=0.048$ (crosses 0)**.

The reason sharpens the thesis rather than threatening it. As data grows, even genuinely hard, low-recoverability targets eventually get fit — asymptotic recoverability *saturates* and loses its power to discriminate auxiliaries (mae-mask / future-obs $R_\text{deep}$ climbs from ≈−0.01 to +0.13–0.16 at 16k). What does *not* wash out is how *cheaply and early* the target is grabbed: the shortcut is still taken first. The generalization spread across objectives actually **grows** with scale (std 0.011 → 0.020) while the recoverability spread stays stable, so the axes do not compress — only the asymptotic *read-out* degrades. The dynamics measure holds. Measure recoverability as a learning dynamic.

### 4.4 Why the linear probe inverts (mechanism)

Across the full battery, linear-decodability and policy-recoverability are **anti-correlated**, and the reason is concrete. The high-dimensional visual targets (past/future/masked observation latents) are *moderately* linearly decodable from frozen DINO features (linear ≈ 0.2) yet the policy *cannot* cheaply fit them end-to-end (deep ≈ −0.1) — precisely why they force grounding and *help* generalization. The archetypal shortcut runs the other way: `final-pose` is linearly *hard* (linear = 0.06) but deep-*easy* (deep = 0.92). A linear probe therefore measures a different quantity than the policy experiences; ranking auxiliaries by it is not a noisy version of the right answer, it is the wrong answer. This also dispatches the natural alternative of Shannon mutual information: MI is function-class-free and would not exhibit — or explain — the flip that the policy actually undergoes.

### 4.5 Takeaway

Recoverability predicts generalization, but only when measured **the way the model experiences it**: under the policy's own function class, and early in training. Every convenient proxy — frozen linear probe, frozen MLP, transfer score — either inverts the law or destabilizes across scale. The practical rule for the rest of the paper: estimate an auxiliary's recoverability with a policy-class learning-dynamics measure (speed@¼ / AULC), and prefer the auxiliaries it ranks *lowest*.

---

*Caveats to carry into Limitations: N=12–18 objectives ⇒ wide CIs; mini-VLA at 2.3% data ⇒ near-zero absolute generalization (relative rankings are the claim); offline action-$R^2$ ≠ closed-loop success (M2). The sign-flip and speed@¼ robustness survive all three scales; the asymptotic-measure significance loss at 16k is reported as itself a finding (measure as dynamics), not hidden.*
