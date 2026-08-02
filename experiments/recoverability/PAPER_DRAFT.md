# AHA — full paper draft (assembled)

*"Regularizing VLA models by asking hard questions" · method = AHA (Asking Hard-question Auxiliaries).*
*Assembled 2026-08-02 from section drafts. Each section keeps its provenance footer (data sources, status tags). This is a working draft on the mini-VLA proxy; §8.2 (M2) is the unrun make-or-break. Outline of record: PAPER_OUTLINE.md.*

---

# Abstract + §1 Introduction — draft

*The framing that carries an oral. Sources: `PAPER_OUTLINE.md`, `AHA_MASTER.md`, `core-claim` memory (claim = GENERALIZATION failure, not "ignores language"). Honest status tags kept out of the prose; see Limitations.*

---

## Abstract

Behavior cloning learns the cheapest input-to-action mapping that fits the demonstrations, and that cheapness is exactly why it generalizes poorly out of distribution. A large body of work repairs this with auxiliary objectives — future-frame prediction, inverse dynamics, latent actions, subtask or VQA prediction, instruction prediction — each introduced and justified case by case. We show these are not a zoo but points on a single measurable axis, **recoverability**: the usable ($\mathcal{V}$-)information an auxiliary target carries about itself given the policy's observation, measured under the policy's own function class. One law governs the axis — *the less recoverable an auxiliary's answer is from the current input, the more co-training on it improves generalization* — and it composes simply: a mixture inherits the benefit of its single hardest member. Crucially, recoverability must be measured the way the model experiences it: under the policy function class and as a learning dynamic. The convenient surrogate — a frozen linear probe — does not approximate the law, it **inverts its sign**, confidently ranking the worst auxiliaries as the best. We connect the axis to Knowledge Insulation (which we show is safe only when the insulated backbone is shaped by an action-related signal, explaining π0.5's KI+FAST), and we place ~55 prior VLA auxiliaries on the axis, revealing that the field clusters on the cheap, shortcut side. From all of this a one-line design rule falls out: **ask one hard question** — add a single low-recoverability auxiliary, chosen by measuring it correctly. We name the scheme AHA (Asking Hard-question Auxiliaries).

---

## 1. Introduction

Imitation-learned robot policies fail out of distribution in a characteristic way. Given expert demonstrations, behavior cloning (BC) finds *a* mapping from observation to action that fits them — and by the simplicity bias of gradient descent, it finds the *cheapest* such mapping. The cheapest mapping keys on whichever input feature is most predictive of the action on the training distribution, which is frequently a spurious correlate rather than the causal structure of the task: the policy latches onto a shortcut and breaks the moment the shortcut shifts. This is the well-documented causal-confusion / shortcut failure of imitation, and it is a failure of **generalization**. (It is often described as the policy "ignoring language" or "ignoring the goal"; we take the ignoring to be a *diagnostic symptom* of the shortcut, not the disease. The disease is that BC took the cheap path.)

The field's response has been to add auxiliary objectives that co-train the representation to encode more than the shortcut needs. The catalog is large and growing: predict future frames (GR-1/2, VPP), predict future *latents* (SPR, DINO-WM), invert dynamics (VPT, BCO), predict latent actions (LAPA, Moto), reconstruct masked inputs (MVP, VC-1), predict subtasks or answer visual questions (RT-2, π0.5, ECoT), predict the instruction (Hejna et al.). Each is proposed with its own motivation and its own ablation. What is missing is a principle that says, *before* training and *by measurement*, which of these will actually regularize BC and which will quietly reinforce its shortcut — and why they are the same kind of thing at all.

**We provide that principle.** Our claim is that every one of these auxiliaries is a point on a single axis — **recoverability**, the usable information the auxiliary target carries about itself given the policy's observation, under the policy's own function class (§3) — and that one law relates the axis to generalization: *lower recoverability co-trains to better out-of-distribution generalization* (§5). The mechanism is exactly the simplicity bias the field has been fighting by hand: a target the policy can cheaply recover reinforces cheap features; a target it *cannot* cheaply recover forces the grounded representation the shortcut skips.

The result that makes this a usable design tool — and our sharpest finding — is that **how you measure recoverability decides whether you see the law at all** (§4). The natural instrument, a frozen linear probe on pretrained features, is not a noisy estimate of the right quantity; it is anti-correlated with it, and it *inverts the law's sign* (Pearson +0.78 where the truth is −0.75), stably across dataset scales. Only when recoverability is measured under the policy's own function class, and read out as a *learning dynamic* (how cheaply the target is grabbed early) rather than an asymptote, does the law appear and stay stable. A practitioner who measures the obvious way would select precisely the wrong auxiliaries.

**Contributions.**
1. **The recoverability axis and its law** (§3, §5): a single, measurable, policy-relative quantity that unifies VLA auxiliary objectives and predicts, per objective, which regularize BC (lower is better).
2. **How to measure it — the sign flip** (§4, the headline): recoverability must be estimated under the policy function class and as a learning dynamic; the frozen-linear-probe surrogate inverts the law. Robust across three dataset scales with bootstrap CIs.
3. **A composition law** (§5): a mixture of auxiliaries inherits the benefit of its lowest-recoverability member — one hard question rescues a shortcut-laden mix, and neither stacking more hard members nor diversifying their modality adds beyond that.
4. **The Knowledge Insulation interaction** (§6): when the action head is insulated from the backbone, generalization is governed by the shaping auxiliary, and hard-KI is safe only if that auxiliary is action-related — a contrastive ablation that explains π0.5's KI+FAST as implicitly KI+AHA.
5. **A unifying survey** (§7): ~55 prior VLA auxiliaries placed on the axis, showing the field clusters on the cheap (shortcut) side and that the genuinely low-recoverability lane is underused.
6. **[M2] Closed-loop validation** (§8): whether recoverability predicts closed-loop success on a full-scale VLA.

The design rule is one sentence: **ask one hard question** — add a single low-recoverability auxiliary, and measure recoverability the way the model experiences it, or the measurement will lie to you.

---

*Positioning guardrails (from `CITATIONS_THREATS.md`, keep in Related Work not Intro): the novel core is $\mathcal{V}$-information under the policy function class as an auxiliary-*selection* axis, in the low-is-better direction, plus the sign-flip and composition law. Pre-empt the "assembly" threat (PVI measures data difficulty; aux-weighting weights by alignment; PI-as-aux maximizes; BYOL-γ is one method) explicitly in §2. Do not claim future-prediction itself is novel. Core claim framing = generalization failure, not "ignores language."*

---

# §2 Related Work + §9 Limitations — draft

*Sources: `CITATIONS_THREATS.md` (web-verified ids), `AHA_MASTER.md`, all §-drafts' caveat footers. §2 leads with the assembly threat; §9 is deliberately first-to-say.*

---

## 2. Related work

Our contribution is a *selection principle*, so we organize related work by the pieces a skeptic might assemble to argue the principle is not new, and distinguish each.

**Usable information and dataset difficulty.** Predictive $\mathcal{V}$-information (Xu et al., ICLR 2020) generalizes mutual information to a computationally-constrained function class; $\mathcal{V}$-usable information (Ethayarajh et al., ICML 2022) uses it to measure *dataset* difficulty, and MDL/prequential probing (Voita & Titov, EMNLP 2020; Blier & Ollivier 2018) reads informativeness off a learning curve. These are our measurement primitive, but they diagnose the difficulty of *data given fixed labels*; we invert the object — hold the input fixed, vary the auxiliary *target*, and use recoverability to *select* the target — and we invert the direction — *low* usable information is the desirable property. Neither the selection use nor the low-is-better direction appears in this line.

**Auxiliary-task selection and weighting.** A body of work chooses or weights auxiliaries, but by *relevance to the main task*: gradient cosine-alignment (Lin et al., NeurIPS 2019), learned/searched weightings (AANG; SLGrad), and information-theoretic *training-task* selection for meta-RL (ITTS, NeurIPS 2020) via mutual-information difference. None selects by the target's recoverability under the policy class, and all treat auxiliaries as helpers to be *aligned and maximized* — the opposite of our claim that a *hard*, low-recoverability target is what regularizes.

**Predictive-information and future-prediction auxiliaries.** PI-QT-Opt adds predictive information as an auxiliary to *maximize*; self-predictive representations improve BC out-of-distribution (BYOL-γ, 2025) — the nearest neighbor to our thesis in spirit. These are single methods that instantiate one point of our axis; they neither propose the axis, nor the measurement caveat (sign-flip), nor the composition law. Future-frame prediction as a VLA objective (GR-1/2, VPP) is, on our axis, a *high*-recoverability shortcut — a foil, not a precedent.

**Anti-shortcut / causal-confusion regularization.** Causal confusion in imitation (de Haan et al., NeurIPS 2019) and its cures — object dropout (OREO, NeurIPS 2021), gaze regularization (GABRIL, IROS 2025) — target the same disease we do (BC's shortcut), but each supplies a specific architectural or data mechanism rather than a measurable criterion for choosing an auxiliary. We share their motivation and differ in kind: a policy-class recoverability measurement that ranks *any* candidate auxiliary.

**Instruction prediction.** Predicting the instruction as an auxiliary to improve long-horizon imitation is established (Hejna, Abbeel & Pinto, AAAI 2023). Our `instr-infer` is a new instantiation (within-VLA, masked-input, recoverability-selected), not a new idea; the novelty is the selection principle, not the instruction auxiliary.

**Knowledge Insulation and action tokenization.** π0.5's Knowledge Insulation stop-grads the action head from the VLM backbone and co-trains it with FAST discrete action tokens (Pertsch et al., 2025). We adopt this setup and add the missing contrastive ablation (§6): KI is safe only when the insulated backbone is shaped by an action-related signal — which is what FAST is — recasting KI+FAST as implicitly KI+AHA.

*In sum, every adjacent piece exists, but the combination we claim — recoverability under the policy function class as a low-is-better auxiliary-selection axis, with the measurement sign-flip and composition law — is not assembled anywhere in prior work.*

---

## 9. Limitations

We state these plainly; several bound the strength of specific claims.

**The evidence is a proxy in every axis except the one under test.** All results in §4–§6 come from a mini vision–language–action policy trained on **2.3% of RT-1/fractal** (2k of 87k episodes; 16k of ~700k transitions) for a few hundred to a few thousand gradient steps. Absolute out-of-distribution generalization is therefore near zero; our claims are about the *relative ordering* of auxiliaries, which is what recoverability predicts, and which we verify is stable to subsample size. Whether the ordering holds at full scale and competence is the purpose of M2 (§8) and is not established here.

**Offline $R^2$ is not closed-loop success.** §4–§6 measure held-out action $R^2$, an offline surrogate. It need not track closed-loop task success under compounding error. The mini-VLA SimplerEnv rollout (§8.3) and M2 (§8.2) address this; until they report, external validity is unproven.

**Small objective sets ⇒ wide intervals.** Correlations are over 12–18 objectives; bootstrap CIs are correspondingly wide. The sign-flip and the speed@¼ predictor exclude zero at all three scales; the asymptotic end-to-end measure *loses* significance at 16k (CI crosses 0) — which we report as itself a finding (measure as a dynamic) rather than suppress. Composition (§5) is single-scale (4k); its minimum-member correlation is robust but the mean-member is not.

**Two results are partially preempted.** The positive half of the KI interaction (an action-related aux survives insulation) is π0.5's design; ours is the *negative* ablation (an action-unrelated aux starves under hard-KI). The FAST-lite *co-training* effect is noisy at 4k; the reportable claim there is the measurement (discrete = lower-variance recoverability estimate), not a co-training win. Instruction prediction as an auxiliary is not a novelty gap (Hejna et al., 2023).

**Recoverability is relative, not absolute.** It is defined under a function class and measured in context (scale, budget); it is not a scale-free property of a target. This is a feature for a *design* criterion — it is measured for the policy you are training — but it means numbers are not portable across model classes without re-measurement.

**The survey is qualitative.** §7 places ~55 prior objectives on the axis by argument and boundary-case rulings, not by re-measuring each under a common policy; recoverability ratings there are H/M/L estimates. One case we *did* measure contradicted our prior estimate (`state-infer` measured high-recoverability), which we report as a caution against reading the qualitative map as measured.

**Citations to verify.** A small number of 2025–2026 arXiv identifiers are used; all foundational and threat-line citations were web-verified, but the Voita–Titov arXiv id is unconfirmed and is cited via the ACL Anthology entry.

---

*These limitations scope the claims: the measurement result (§4) and the mechanism story (§5–§6) stand on the mini-VLA at proxy scale; external validity is explicitly deferred to M2. We prefer to state this than to let a reviewer find it.*

---

# §3 draft — The recoverability axis (framework)

*Prose draft of the framework/definition section. Conceptual; grounds Xu 2020 / Ethayarajh 2022. Sources: `AHA_MASTER.md`, `PAPER_OUTLINE.md`, `CITATIONS_THREATS.md`. Establishes the object §4–§6 measure and test.*

---

## 3. The recoverability axis

### 3.1 Definition

Let a policy observe $o_t$ (image, language, proprioception) and let $y$ be any auxiliary target we might co-train it to predict — a future frame, a past action, a masked-out modality, an instruction. We define the **recoverability** of $y$ from $o_t$ as the normalized predictive $\mathcal{V}$-information $y$ carries about itself given $o_t$, *under the policy's own function class* $\mathcal{V}$:

$$R_\mathcal{V}(o_t \to y) \;=\; 1 - \frac{\mathcal{L}^*_\mathcal{V}(y \mid o_t)}{\mathcal{L}^*_\mathcal{V}(y \mid \varnothing)},$$

where $\mathcal{L}^*_\mathcal{V}$ is the best achievable predictive loss within $\mathcal{V}$, with the conditioning input ($o_t$) or without (the marginal baseline $\varnothing$). For a continuous target this is an $R^2$ ($1-\text{MSE}_\text{val}/\text{MSE}_\text{marg}$, Xu et al.'s Prop. 1.5); for a discrete target it is $1-H_\text{val}/H_\text{marg}$. Recoverability is 1 when $y$ is trivially readable from $o_t$ within $\mathcal{V}$, 0 when $o_t$ helps no more than the prior, and can go negative under finite data when conditioning hurts.

Two properties are load-bearing. First, $R_\mathcal{V}$ is **relative to the function class** $\mathcal{V}$: it is not a property of the data alone but of what *this policy* can cheaply extract. §4 shows this is not a technicality — evaluated under a linear class it inverts, under the policy class it obeys the law. Second, it is **measured in context**, at a given scale and training budget; §4 shows the read-out must be taken as a *learning dynamic*, not an asymptote.

### 3.2 Relation to usable information — and the inversion that makes it a design tool

Predictive $\mathcal{V}$-information (Xu et al., ICLR 2020) and its dataset-difficulty instantiation, $\mathcal{V}$-usable information (Ethayarajh et al., ICML 2022), measure how much usable signal a *dataset's inputs* carry about *its labels* — a property of data, used to diagnose difficulty. We invert the framing. We hold the *input* fixed (the policy's observation) and treat the *auxiliary target* as the free variable, then use recoverability not to diagnose data but to **select the auxiliary**: among candidate targets, prefer the ones the policy can *least* cheaply recover. To our knowledge this inversion — $\mathcal{V}$-information under the policy function class as an auxiliary-*selection* criterion, in the counterintuitive *low-is-better* direction — is not present in prior work (see §2; the nearest neighbors weight auxiliaries by main-task gradient alignment, or add predictive-information as a helper to maximize, not select-by-difficulty).

### 3.3 Why low recoverability forces grounding

Behavior cloning is subject to simplicity bias: among functions that fit the demonstrations, gradient descent finds the cheapest, which latches onto whatever input feature is most predictive of the action on the training distribution — often a spurious shortcut that breaks under distribution shift (the causal-confusion failure mode). An auxiliary target acts on this bias through its recoverability. A *high*-recoverability target is, by definition, one the policy can satisfy with a cheap function of $o_t$ — the same kind of cheap function the shortcut already is — so co-training on it reinforces rather than corrects the shortcut. A *low*-recoverability target cannot be satisfied by any cheap function of $o_t$; fitting it *forces* the backbone to build a representation that genuinely grounds in the scene, and that representation is what survives distribution shift. Recoverability is thus a direct, measurable proxy for "does this auxiliary fight the simplicity bias or feed it."

### 3.4 Type is secondary; recoverability is the axis

It is tempting to organize auxiliaries by *type* — prospective (predict the future), retrospective (predict the past), introspective (reconstruct a masked part of the present). We use this taxonomy descriptively, but it is **not** the axis. Type does not determine recoverability: predicting the *next video frame* is prospective yet near-trivially recoverable (consecutive frames barely differ — a shortcut), while predicting a *long-horizon goal* is also prospective yet has low recoverability. The same dissociation appears within every type. A particularly instructive case is next-token prediction: NTP is a *form*, not a recoverability level. In open-web language modeling NTP is low-recoverability because the target is high-entropy; π0.5's subtask NTP predicts a *scene-determined, low-entropy* subtask and is therefore **high**-recoverability — an easy task wearing the NTP costume. What predicts generalization is recoverability, measured under the policy class; the type taxonomy only organizes where in observation-time the target lives.

### 3.5 What the rest of the paper does

§4 shows recoverability must be measured under the policy's function class and as a learning dynamic — the frozen-probe measurement inverts the law's sign. §5 establishes the per-objective law and its composition rule (a mix inherits its hardest member). §6 connects the axis to Knowledge Insulation. §7 places ~55 prior VLA auxiliaries on the axis, showing the field's "zoo" is one spectrum. §8 [M2] tests whether recoverability predicts closed-loop success on a full-scale VLA.

---

*Notation note: we write $R_\mathcal{V}$ for recoverability and reserve "recoverability" (unqualified) for the policy-class, learning-dynamics estimator of §4, since §4 proves other estimators of the same definition disagree in sign.*

---

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

---

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

---

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

---

# §7 Survey + §8 Experiments + §10 Conclusion — draft

*Sources: `SURVEY_vla_auxlosses.md` (R7), `M2_DESIGN.md` (§8), `M1_RESULTS.md`. §7 is qualitative (honestly labeled). §8 states what is done vs. what M2 must show.*

---

## 7. The field on one axis (survey)

We place ~55 prior VLA auxiliary objectives on the recoverability axis, classifying each by where its target sits relative to the current time — prospective (future), retrospective (past), introspective (masked present) — and rating its recoverability H/M/L by argument. Three boundary rulings do most of the work and expose how *type* dissociates from *recoverability*.

**Inverse dynamics** (predict $a_t$ from $o_t$ and $o_{t+1}$) is introspective and near-trivially recoverable *as a labeler* — both endpoints are given — which is exactly why it works as a cheap auto-labeler (VPT, BCO, GR00T-IDM); the moment the future frame is withheld and the policy must predict from $o_t$ alone, it flips to prospective and low-recoverability. **Latent-action models** (LAPA, Genie, Moto) are two-stage: the Stage-1 tokenizer (frame-pair → latent) is introspective/H, but the Stage-2 signal that pretrains the policy (o_t + language → latent) is prospective/L — the stage exposed to the policy is what counts. **Hindsight relabeling** (HER, RT-Trajectory) *feels* retrospective, but on the target-time axis the achieved goal sits in the future and lies *on* the executed trajectory — maximally redundant with the actions, the cheapest possible "future," i.e. a shortcut. Its "retrospective" character is only the backward data-generation procedure.

Two findings shape the field's picture. First, **the field clusters on the high-recoverability (cheap) side**: masked reconstruction, contrastive, and VQA (introspective) and pixel-next-frame and hindsight-goal (prospective) are all H, and reported gains "almost never come from target difficulty" — they come from *prior transfer* (GR-1/2, VPP) or from *using the prediction as a planner/data-generator* (UniPi, SuSIE). Second, **the genuinely retrospective lane is nearly empty**: prior "retrospective" work is almost entirely hindsight relabeling, which is prospective-by-target; predicting the *past* from $o_t$ is largely unexplored. A caution the survey itself surfaced: next-token prediction is a *form*, not a recoverability level — π0.5's scene-determined subtask NTP is high-recoverability, contradicting the reflex that "NTP is always a strong objective."

*(This section is a qualitative map: ratings are H/M/L estimates argued from each objective's construction, not re-measured under a common policy. One measured case, `state-infer`, came out high-recoverability against our prior estimate — read the map as a hypothesis-organizer, not a measurement.)*

---

## 8. Experiments

### 8.1 Mini-VLA measurement study (§4–§6)
The mini-VLA on RT-1/fractal supplies the measurement result (§4), the per-objective law and composition (§5), and the KI interaction (§6): the sign-flip is replicated at three dataset scales with bootstrap CIs; composition and KI are shown at 4k across 3 seeds. These stand on the proxy and establish *relative* orderings.

### 8.2 Full-VLA closed-loop (M2 — make-or-break, DESIGN)
The external-validity test co-trains a full VLA (π0.5-class from a PaliGemma backbone) with behavior cloning plus a single auxiliary of chosen recoverability, and evaluates **closed-loop success in SimplerEnv**. Two questions: (i) does recoverability — estimated cheaply from each arm's auxiliary validation loss, amortized, without a separate probe sweep — predict closed-loop success across auxiliaries? (ii) does a low-recoverability ("hard question") auxiliary beat BC-only closed-loop? A positive answer converts the offline law into a closed-loop, full-scale claim; this is the result the oral hinges on and is not yet run (gated on compute).

### 8.3 Mini-VLA closed-loop probe (PRE-STEP, code ready)
As a cheap precursor we serve the mini-VLA behind the SimplerEnv client and roll out BC vs. an AHA arm (auxiliary = far-future observation latent, a low-recoverability target). The pipeline is verified end-to-end — headless Vulkan rendering passes, both policies train and serve, the client connects — and is queued; its purpose is to validate the closed-loop plumbing and surface any BC-vs-AHA closed-loop difference at proxy scale, not to reach high absolute success.

---

## 10. Conclusion

The scattered auxiliary objectives of the VLA literature are one axis. Recoverability — the usable information of a target under the policy's own function class — measures where each objective sits, and a single law relates the axis to generalization: ask the policy a question it *cannot* cheaply answer from what it already sees, and co-training on it forces the grounded representation that behavior cloning's shortcut skips. The catch, and our sharpest result, is that this only works if you measure recoverability the way the model experiences it — under its function class and early in training — because the obvious surrogate inverts the law. The design rule is one sentence: **ask one hard question, and measure it right.**

---

*Assembly note: full draft = `PAPER_S1_abstract_intro` (Abstract+§1) · `PAPER_S2_S9_related_limits` (§2, §9) · `PAPER_S3_framework` (§3) · `PAPER_S4_measurement` (§4) · `PAPER_S5_law_composition` (§5) · `PAPER_S6_ki` (§6) · this file (§7, §8, §10). Figures: `fig_measures_*` (§4), `fig_mixki` + `fig_deepen` (§5–§6), survey dot-plot (§7). Outline of record: `PAPER_OUTLINE.md`.*

---

