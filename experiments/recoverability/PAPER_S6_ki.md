# §6 draft — Interaction with Knowledge Insulation

*Prose draft. Numbers: mini-VLA RT-1/fractal, 3 seeds, 4k (R4) + K=16 fastlite (R4b). PRELIM. Sources: `M1_RESULTS.md`. Defuses the reviewer objection "modern VLAs stop-grad the BC head, so BC can't be a shortcut" and connects the recoverability axis to π0.5's KI+FAST design.*

---

## 6. Recoverability under Knowledge Insulation

A sharp objection targets the whole premise. Many modern VLAs — π0.5 with Knowledge Insulation (KI), and others — **stop the gradient from the action head into the backbone**: the flow/diffusion action expert is trained, but it does not shape the shared representation. If behavior cloning cannot write to the backbone, how can it install a shortcut there? We take this seriously, reproduce the KI setup, and find that it does not dissolve the recoverability story — it *sharpens* it, and in doing so explains a design choice π0.5 made but did not ablate.

### 6.1 Setup

KI insulates the backbone from the action head; the backbone is then shaped only by whatever auxiliary objective is co-trained through it. We implement KI as a stop-gradient from the flow head to the shared representation for the first $\tau\cdot E$ epochs (a schedule: $\tau=0$ = always joint, $\tau=1$ = always insulated), and we vary *which* auxiliary shapes the backbone. The question is which auxiliaries still deliver their generalization benefit when the action head can no longer help build the representation.

### 6.2 The result: hard-KI starves an action-*unrelated* aux; an action-*related* one survives

Under full insulation ($\tau=1$) the outcome splits cleanly by the auxiliary's **relatedness to action**, not by its recoverability alone:

| backbone-shaping aux | action-related? | benefit at $\tau{=}0$ | at $\tau{=}0.5$ | **at $\tau{=}1$ (hard-KI)** |
|---|---|---|---|---|
| far-past-obs | ✗ | +0.041 | +0.035 | **−0.030 (starves)** |
| cur-action | ✓ | +0.028 | +0.017 | +0.024 (survives) |
| fut-action | ✓ | +0.021 | +0.011 | +0.018 (survives) |
| final-pose (pose-y) | ~ | +0.012 | +0.009 | +0.028 |

A low-recoverability but **action-unrelated** auxiliary (`far-past-obs`) helps when it can co-shape the backbone jointly ($\tau=0$), but under hard insulation it **starves the action head** and goes negative: with the flow head cut off, the only signal reaching the backbone is a task orthogonal to control, and the representation drifts away from what the action head needs. **Action-related** auxiliaries (`cur/fut-action`) survive hard-KI and keep helping — they shape the backbone toward the action manifold even while insulated.

This is precisely why π0.5's KI does **not** collapse in practice: its backbone is co-trained with **FAST discrete action-token prediction** — an action-related objective. KI is safe *iff* the insulated backbone is shaped by an action-related signal; the "over-insulation starves the head" failure is specific to action-*unrelated* auxiliaries. π0.5 supplies the positive design (KI + FAST works); ours is the missing contrastive ablation showing *why* — and that its safety is conditional. (We also find the KI *schedule* offers no advantage here: always-joint $\tau=0$ is best or tied for every auxiliary; late-release $\tau=0.5$ never beats it at this scale.)

The sweet spot is the intersection: **action-related AND low-recoverability.** `cur/fut-action` sit there (recoverability 0.04–0.08, action-related) and help under KI. That is exactly the profile FAST provides — an action-related, discretized, harder-to-recover target.

### 6.3 FAST-lite: discretization is a cleaner recoverability measurement

Why *discrete* action tokens, specifically? We measure the next action's recoverability two ways — continuous ($R^2 = 1-\text{MSE}_\text{val}/\text{MSE}_\text{marg}$) and discretized into $K=16$ quantile bins ($1-H_\text{val}/H_\text{marg}$). Both agree the next action is a genuinely **low-recoverability** target. But the continuous estimate is noisy and *sign-unstable* across seeds (−0.020 ± 0.110, range [−0.13, +0.13]), while the **discrete estimate is sign-stable and 3.4× lower-variance** (−0.117 ± 0.032). Discretization does not change the underlying quantity; it makes it *measurable* — the cross-entropy read-out has stable variance where the MSE-$R^2$ does not.

This reframes FAST on the recoverability axis: beyond tokenization convenience, a discrete action-token objective is a *more stably measurable and trainable* low-recoverability signal than continuous regression — which is what an insulated backbone needs. (Honesty: the *co-training* half of FAST-lite is noisy at this scale — a discrete action-token aux slightly hurts when both heads are undertrained on 4k, though hard-KI reduces the damage, consistent with KI protecting the flow head. We report the measurement point, which is clean, and do not overclaim a co-training win.)

### 6.4 Takeaway

KI does not refute "BC takes the shortcut" — it relocates the question. With the action head insulated, generalization is set by whatever shapes the backbone, and the recoverability axis still governs it, now with an added constraint: the shaping auxiliary must be **action-related** or hard-KI starves the policy. π0.5's KI+FAST is, on our axis, exactly KI + an action-related low-recoverability question — implicitly, KI+AHA.

---

*Caveats for Limitations: R4/R4b are single-scale (4k), 3 seeds, PRELIM; the positive "action-related survives KI" half overlaps π0.5's design (the negative ablation is ours). The FAST-lite co-training result is noisy — the reportable claim is the measurement (discrete = lower-variance recoverability), not a co-training improvement.*
