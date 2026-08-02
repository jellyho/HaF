# §5 draft — The law and its composition rule

*Prose draft. Numbers: mini-VLA RT-1/fractal, 3 seeds. R2 SOLID (same runs as R1); R3/R3b PRELIM (4k, 3 seeds). Sources: `M1_RESULTS.md`, `analyze_stats.py`. Measure recoverability with the policy-class dynamics estimator established in §4.*

---

## 5. The law, and how auxiliaries compose

Section 4 fixed *how* to measure recoverability. We now state the law it obeys and, more usefully, what happens when you co-train on several auxiliaries at once — the regime every real VLA is in.

### 5.1 The law

**Per objective, lower recoverability predicts better OOD generalization.** Regressing each of the 18 auxiliaries' held-out action $R^2$ (co-trained with behavior cloning) on its policy-class recoverability gives a negative slope at every scale (§4.2: −0.66 / −0.64 / −0.47 asymptotic, −0.75 / −0.72 / −0.69 by the dynamics measure). The interpretation is the simplicity-bias mechanism made quantitative: a *highly* recoverable auxiliary is one the policy can satisfy with a cheap function of the input, so it reinforces the same shortcut behavior cloning already takes; a *low*-recoverability auxiliary cannot be satisfied without building the grounded representation the shortcut skips, and that representation is what transfers. Recoverability is the knob that says, in advance and by measurement, which side a given auxiliary falls on.

Two measured points anchor the extremes. `final-pose` — predict the arm's last-frame pose — has recoverability 0.92: the policy fits it almost for free, and it does not help. The masked/future observation-latent targets sit near or below zero recoverability under the policy class, and they are the ones that regularize. Neither is obvious a priori; both are read off the same estimator.

### 5.2 Composition: a mix inherits its hardest member

Auxiliaries are used in combination, so the operative question is how recoverability *aggregates*. We form 12 mixes of 2–3 auxiliaries and ask which summary of a mix's member recoverabilities predicts the mix's generalization. The answer is clean: **the minimum.** The *lowest*-recoverability (hardest) member predicts mix generalization at Pearson **−0.87 (95% CI [−0.98, −0.67]; Spearman −0.82, CI excludes 0)**, beating the mean (Spearman −0.57, $p=0.057$, CI crosses 0) and roughly tying the joint estimate (−0.85). Concretely, adding a single genuinely hard question **rescues a shortcut-laden mix**: `far-past-obs + final-pose` — one hard member, one pure shortcut — generalizes at −0.033, near the hard member's own value and far better than `final-pose + initial-pose` (two shortcuts, −0.062). One hard question is enough; a second shortcut is nearly inert.

This is a different statement than the multi-task-learning literature makes. Worst-case / minimax MTL aggregates *loss* for robustness; task-diversity bounds reward *coverage*. Our composition law aggregates *benefit* by the hardest member's *recoverability* — a property of the targets, measurable before training, not of the loss landscape or the task distribution.

### 5.3 It is presence, not count or diversity

If the minimum dominates, two predictions follow, and both hold.

**Stacking more hard members saturates.** Nesting additional low-recoverability observation targets into a mix, the benefit over behavior-cloning-only rises from one member to two (+0.035 → +0.043) and then **plateaus** across three, four, five, and six members (+0.043, +0.047, +0.040, +0.045) — no further accumulation, and no interference/collapse either. Once one hard member is present, additional hard members are redundant with it. This is exactly what "the minimum sets the mix" predicts, and it is a practical design rule: you need *one* well-chosen hard question, not a large auxiliary suite.

**Modality diversity is inert.** Holding one member fixed (`far-past-obs`) and swapping the second across modalities — another observation target, a future/previous *action*, a *pose* displacement — moves the benefit negligibly (+0.043 / +0.040 / +0.045 / +0.045, flat within noise). The lever is the members' recoverability, not how many modalities they span. This distinguishes AHA from diversity- or coverage-driven auxiliary selection: spreading auxiliaries across modalities buys nothing per se; lowering recoverability is what buys generalization.

### 5.4 Takeaway

The law is per-objective (lower recoverability ⇒ better generalization) and it composes by the **minimum**: a mix is only as good as its hardest question, one hard question rescues an otherwise shortcut-laden mix, and neither piling on more hard members nor diversifying their modality adds beyond that. For a practitioner the rule is short — **add one low-recoverability auxiliary, chosen by the §4 estimator, and stop.**

---

*Caveats for Limitations: R3/R3b are single-scale (4k), 12 mixes, N small ⇒ CIs wide (min-recov CI robust, mean crosses 0). Absolute generalization is near zero at 2.3% data; the claims are about *relative* ordering, which is what recoverability predicts. Replicating composition at 8k/16k is the obvious next strengthening (P3).*
