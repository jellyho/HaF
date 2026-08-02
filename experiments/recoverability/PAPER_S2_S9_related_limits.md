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
