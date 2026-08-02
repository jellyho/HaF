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
