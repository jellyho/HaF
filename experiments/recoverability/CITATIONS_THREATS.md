# Verified citations & prior-art threats — AHA / "Regularizing VLA models by asking hard questions"

*Web-verified 2026-08-01 (agent, WebSearch/WebFetch). Verdict scale: none / minor / moderate / severe.*

## Verified foundational citations (safe to cite)
| key | citation | id | note |
|---|---|---|---|
| Xu 2020 | Xu, Zhao, Song, Stewart, Ermon. "A Theory of Usable Information Under Computational Constraints." **ICLR 2020** | arXiv:2002.10689 | predictive 𝒱-information — our measure primitive |
| Ethayarajh 2022 | Ethayarajh, Choi, Swayamdipta. "Understanding Dataset Difficulty with 𝒱-Usable Information." **ICML 2022 (Outstanding Paper)**, PMLR 162:5988–6008 | arXiv:2110.08420 | PVI; measures DATASET difficulty (we invert → aux selection) |
| Voita-Titov 2020 | Voita, Titov. "Information-Theoretic Probing with Minimum Description Length." **EMNLP 2020**, ACL Anthology 2020.emnlp-main.14 | arXiv id **UNVERIFIED** (2003.12298 unconfirmed) → **cite the ACL Anthology entry** | MDL/prequential probing |
| FAST 2025 | Pertsch et al. "FAST: Efficient Action Tokenization for Vision-Language-Action Models." 2025 | arXiv:2501.09747 | DCT+quantize+BPE action tokens |
| π0.5-KI 2025 | "Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better." (Physical Intelligence) 2025 | arXiv:2505.23705 | stop-grad backbone↔action-expert + non-action co-train (our R4 source) |
| π0.5 2025 | π0.5 "open-world generalization via heterogeneous co-training" 2025 | arXiv:2504.16054 | companion |

*(Author lists for FAST/π0.5-KI not fully enumerated — fetch PDFs before camera-ready if exact strings needed.)*

## Threat table
| # | work | verified id | what it does | THREAT |
|---|---|---|---|---|
| 1 | **Hejna, Abbeel, Pinto. "Improving Long-Horizon Imitation Through Instruction Prediction." AAAI 2023** 37(7):7857–7865 | arXiv:2306.12554 (DBLP conf/aaai/HejnaAP23) ✔ | instruction-prediction aux improves long-horizon imitation (BabyAI/Crafter) | **MINOR** — one aux = a data point ON our axis; no selection criterion, no info-difficulty, no BC-shortcut analysis. Cite as closest instruction-pred prior; AHA generalizes the *why* (low recoverability). |
| 2a | de Haan, Jayaraman, Levine. "Causal Confusion in Imitation Learning." **NeurIPS 2019** | ✔ | more info → worse under shift; fix via targeted interventions/expert queries | **MINOR** — same disease we treat; mechanism differs (interventions, not aux selection). |
| 2b | Park et al. "OREO: Object-Aware Regularization…" **NeurIPS 2021** | arXiv:2110.14118 ✔ | VQ-VAE object codes + random object dropout | **MINOR** — architectural mechanism, no recoverability criterion. |
| 2c | Banayeeanzade et al. "GABRIL: Gaze-Based Regularization…" **IROS 2025** | arXiv:2507.19647 ✔ | gaze-prediction regularizer | **MINOR** — data mechanism, no axis/law. |
| 3a | Lin, Baweja, Kantor, Held. "Adaptive Auxiliary Task Weighting for RL." **NeurIPS 2019** | ✔ | weights auxes by **gradient cosine-sim to main task** (alignment) | **MODERATE (assembly)** — weights by relevance, NOT target recoverability. |
| 3b | "Information-Theoretic Task Selection for Meta-RL" (ITTS). **NeurIPS 2020** | ✔ | selects **training tasks** via MI-difference + relevance | **MODERATE (assembly)** — task selection, not aux-difficulty; no policy-class 𝒱-info. |
| 3c | PI-QT-Opt | arXiv:2210.08217 ✔ | **Predictive Information** as an added aux to help multi-task robot RL | **MODERATE (assembly)** — PI is a fixed helper aux to MAXIMIZE, not a difficulty measure to CHOOSE among auxes. |
| 3d | "Self-Predictive Representations for Combinatorial Generalization in BC" (BYOL-γ) 2025 | arXiv:2506.10137 ✔ | future-representation-prediction aux improves BC OOD | **MODERATE (assembly)** — nearest neighbor conceptually; one method, no recoverability axis, no probe-flip, no 𝒱-info selection. |
| 5 | future-pred VLA line (VPP arXiv:2412.14803; FoMoVLA; UP-VLA) + VLA surveys (2405.14093; 2512.11362) | ✔ | future prediction helps VLAs / taxonomize architectures | **MINOR** — "future prediction helps" is in the air (don't claim it novel); none proposes the organizing axis. |

## Bottom line (for Related Work)
**Safely novel (found nowhere):** (1) recoverability = normalized predictive 𝒱-information **under the policy's own function class**, used as an auxiliary-**selection** axis; (2) the **counterintuitive direction** — *lower* recoverability ⇒ better OOD (hard-to-recover targets as regularizers), opposite the field's reflex to add relevance-aligned *helpful* auxes; (3) the **learning-dynamics requirement + frozen-linear-probe SIGN-FLIP**; (4) the **composition law** (mix ≈ its lowest-recoverability member).

**#1 threat = dilution-by-ASSEMBLY, not direct pre-emption.** A skeptic assembles: PVI/𝒱-info measures *data* difficulty (Xu/Ethayarajh) + PI-as-aux (PI-QT-Opt) + aux weighting by alignment (Lin 2019, ITTS, AANG arXiv:2205.14082, SLGrad arXiv:2306.04519) + future-pred regularizes BC (BYOL-γ) ⇒ "recoverability is just a repackage." **Defuse explicitly:** those measure difficulty of *data* (not select *auxiliaries*), weight by *main-task alignment* (not target recoverability under the policy class), and universally treat auxes as *helpers to maximize* — whereas AHA's thesis is that **low** recoverability is the desirable property, plus the probe-flip and composition-law results none of them have.

**Related-Work citation plan:** anti-shortcut/shared-motivation line = Hejna 2023 + de Haan 2019 + OREO + GABRIL; nearest info/prediction-aux neighbors to distinguish = Lin 2019, ITTS, PI-QT-Opt, BYOL-γ. **Fix before submit:** cite Voita-Titov via ACL Anthology (arXiv id unverified).
