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
