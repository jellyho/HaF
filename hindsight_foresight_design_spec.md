# Hindsight and Foresight: Pretraining Generalist Robot Policies
### 설계 명세서 / Claude Code 핸드오프 문서

> 이 문서는 지금까지의 설계 논의를 구현용으로 정리한 것이다. 목표는 **LAP-3B 코드베이스 위에** 우리만의 기여(retrospective + world-model 예측 task)를 얹어 학습·평가하는 것.

---

## 0. 한 줄 요약 (Thesis)

대부분의 VLA는 **forward 예측(미래 action)** 에만 집중한다. 우리의 claim은 —
**현재 입력만으로는 underdetermined한 target을 예측하도록 강제하면(특히 먼 미래 *그리고* 과거), 모델이 더 general한 control 표현을 학습한다.** forward와 retrospective를 *동시에* 요구하는 것이 forward-shortcut 학습을 차단하고 진짜 인과·dynamics 표현을 강제한다.

- **Hindsight** (과거 복원: instruction, 초기 관측 등) = 우리의 novelty
- **Foresight** (미래 예측: action, 미래 관측) = 기존 VLA도 하는 것
- 핵심 원리: 표현 품질 ∝ 예측 target의 난이도(underdetermination). NTP가 강한 이유("last token만으론 next token이 안 정해져서 history를 압축하게 강제됨")를 *시간 양방향*으로 확장한 것.

---

## 1. 출발점: LAP-3B (baseline이자 코드베이스)

- **LAP (Language-Action Pre-training)**, arXiv 2602.10556, `github.com/lihzha/lap`
- 우리가 도달한 설계의 상당 부분을 LAP이 이미 구현함:
  - 백본: **PaliGemma-3B**
  - **KI (Knowledge Insulation)**: 백본은 언어-action을 CE로 예측, 연속 flow-matching action expert는 SG로 격리 (π0.5 MoT 구조)
  - **언어-action** 예측 target (FAST 대신)
  - **VQA co-training** (motion-prediction task = inverse dynamics 1종)
  - cross-embodiment (OXE), JAX/TPU, openpi 기반
- **따라서 LAP은 우리 baseline이자 시작 코드.** 처음부터 만들지 않는다. LAP 위에 아래 §4의 novel task들을 얹는 게 작업의 핵심.

---

## 2. 백본: PaliGemma 3B (변경 없음)

- **왜 dVLM이 아니라 PaliGemma인가**: ≤3B 제약에서 native full-bidirectional masked diffusion VLM은 존재하지 않는다(전부 7~8B 또는 block-diffusion). 그리고 아래 §3의 "question at the end" 트릭 덕분에 **완전 양방향 백본이 필요 없다** — causal(prefix-LM)로 충분.
- PaliGemma는 **prefix-LM**: prefix(image·prompt·state)는 bidirectional attention, suffix(생성 답)는 causal. 이 구조가 우리 설계와 native하게 맞음.
- ≤3B, 강한 언어 prior, JAX/TPU, openpi 생태계 호환.

---

## 3. 핵심 트릭: "Question at the end" (retrospective를 causal로)

**문제**: 과거를 예측(retrodictive)하려면 시간 역행이 필요 → 보통 완전 양방향 백본을 요구.

**해결**: 예측 대상을 **시퀀스 맨 뒤에 "질문(question)"으로 배치**한다.
- 맥락(과거·현재)은 prefix(양방향 인코딩), 질문 + 답은 suffix(causal).
- "시간상 과거를 예측" → "시퀀스상 앞쪽 맥락을 조건으로 한 예측"으로 변환됨.
- **시간 역행이 시퀀스 순행이 된다** → causal PaliGemma로 충분.
- 모든 task가 `[context prefix] + [question] + [answer]` 형식 = VQA와 동일한 형식.

이 트릭이 "3B + 완전 양방향" 딜레마를 해소함. (LAP의 attention mask가 이미 이 구조 — prefix 양방향, 언어-action은 prefix에 full attention.)

---

## 4. Task 매트릭스 (우리의 novelty)

모든 task를 `[context] + [question] → [answer]` 로 통일. 답의 supervision은 modality별로 다름.
**답으로 물을 modality는 prefix에서 빼고 나머지를 prefix에 넣는다** (이게 그 샘플의 조건/예측을 결정).

| Task | Context (prefix) | Question | Answer | Loss | 방향 |
|---|---|---|---|---|---|
| Policy | image, current, instruction | "next action?" | language-action | CE (+FM head) | foresight |
| Forward dynamics | image, current, action | "obs at t+k?" | future obs latent | JEPA align | foresight (난이도 = k) |
| Inverse dynamics | current, future obs | "what action?" | language-action | CE (+FM) | (LAP에 있음) |
| **Instruction inference** | image, current, action | "what instruction?" | instruction (text) | CE | **hindsight (novel)** |
| **Retrodictive state** | current, (action) | "what was initial obs?" | initial obs latent | JEPA align | **hindsight (novel)** |
| Progress estimation | image, current, goal | "progress?" | scalar | regression | - |
| VL rehearsal | (standard VQA) | - | text | CE | anti-forgetting |

- **난이도 축 = 질문의 시간 거리.** "다음 스텝 action"(쉬움) ↔ "k스텝 뒤 관측"(어려움) ↔ "초기 상태"(과거, 관성 shortcut 불가).
- LAP은 policy + inverse-dynamics VQA만 함. **우리는 hindsight(instruction/state 복원)와 forward-hard, world-model을 체계적 매트릭스로 확장.**

### 마스킹/샘플링 믹스 (기본값, 데이터셋별 재정규화)
- Policy 40% / Forward dynamics 15% / Inverse dynamics 12% / **Instruction inference 10%** / **Retrodictive state 8%** / Progress 8% / VL rehearsal 7%
- hindsight 계열(instruction infer + retrodictive state ≈ 18%)이 claim의 핵심.

---

## 5. Action head: KI 구조 (LAP과 동일)

- **백본**: 언어-action 토큰을 CE로 예측 → 통합 task 형식 참여 + 백본 길들이기 (KI의 이산 신호 역할).
- **연속 FM head**: 백본 hidden을 조건으로 받아(π0 토폴로지) 연속 action 출력. **SG로 백본과 격리** (KI). 추론 시 이 head만 사용.
- **입력은 백본 hidden** (FAST/latent 토큰이 아니라). π0 방식으로 확정됨.
- 우리는 정밀 제어보다 *일반화*가 목표 → 연속 head는 LAP 기본값 유지하되, 필요시 순수 언어-action만으로도 가능.

---

## 6. World model: JEPA-style latent alignment

- 미래/과거 관측 예측을 **생성이 아니라 latent 정렬**로. frozen **V-JEPA 2** 인코더를 target으로, SG로 격리.
- 이유: 정밀 픽셀 생성 불필요(일반화 목표), 가볍고 강건, PaliGemma는 native 이미지 생성이 없음.
- 구현: forward/retrodictive obs question의 답을 latent 공간에서 예측 → V-JEPA target에 정렬 loss.
- (선택) target 인코더: 외부 V-JEPA 2(강건, 정렬 비용) vs 백본 EMA 사본(self-contained, collapse 방지 필요). 초기엔 외부 V-JEPA 2 권장.

---

## 7. 시간·타입 인코딩

- **상대 temporal position embedding** (current = t=0 기준). sequence-position, modality-type embedding과 **별도 축**으로 더함.
- 질문이 "언제를 예측하는지"를 명시 → 시간 방향이 명시적, causal과 충돌 없음.
- modality-type embedding: image / state / action / instruction / progress 구분.
- goal은 시간축 밖 별도 마커 토큰으로 (특정 timestamp가 아니라 조건).

---

## 8. Cross-embodiment / 다양한 데이터

- **언어-action은 embodiment-agnostic** → cross-embodiment 통합이 거의 공짜. action 차원 불일치가 언어 수준에서 흡수됨.
- **결측 modality**: prefix에서 누락 + 해당 question 비활성화로 흡수. 데이터셋별로 가능한 question 집합이 다름 → 믹스 비율 재정규화.
- **embodiment 토큰**을 prefix 맨 앞에.
- **action-less 인간 비디오**: policy/inverse는 못 켜도 forward-dynamics + retrodictive-state는 켤 수 있음 (언어-action은 라벨 불필요, 프레임 전이에서 언어로 describe 가능). world model·표현 학습의 통로.
- 데이터: OXE / cross-embodiment. 데이터셋 크기 편차 → 다양성 보존 샘플링 가중(작은 데이터셋 upweight).

---

## 9. 학습 디테일

- **loss 균형**: CE(text/language-action) · FM(action) · JEPA(latent align) 스케일 정규화 (GradNorm류 고려).
- **SG 경계**: 연속 head·JEPA head → 백본 gradient 차단(KI). 백본은 CE 신호로만 학습.
- **Curriculum**: 초반 policy + VL rehearsal 비중↑로 백본 안정화 → 점진적으로 hindsight·world-model·forward-hard 비중↑.
- **prior 보존 (no LoRA)**: full-FT + SG 격리 + VL rehearsal(anti-forgetting) 조합. LAP이 이미 이 방식.
- 인프라: JAX, openpi 기반, TPU (LAP과 동일).

---

## 10. 실험 계획 (claim 증명)

### 10.1 난이도-통제 ablation (메인)
예측 target을 난이도별로 점증 학습 후 일반화 측정:
- (a) forward-easy만 (다음 스텝 action) = **LAP baseline**
- (b) + forward-hard (먼 미래)
- (c) + retrodictive (instruction, initial state)
- (d) 양방향 전체 (우리 claim)

**측정**: OOD 일반화(새 객체·환경·embodiment) + **데이터 효율 곡선**(10/25/50/100%).
**기대**: (d) > (c) ≈ (b) > (a), 특히 소량 데이터·OOD에서 격차 최대.

### 10.2 표현 품질 probing
frozen 표현 + linear probe on new tasks. 양방향 학습 표현이 forward-only보다 *적은 데이터로* downstream을 풀면 "더 general한 표현" 직접 입증. (MTM 방식)

### 10.3 Shortcut 차단 증명
forward-only가 관성 shortcut에 의존하는지 진단(관측 교란에도 예측 불변 → shortcut). 양방향 모델이 덜 의존하면 메커니즘 확증.

### 벤치마크 / 베이스라인
- 벤치: LIBERO, **LIBERO-Plus**(OOD/perturbation), CALVIN, SimplerEnv, 실로봇(in-dist/task-OOD/layout-OOD), **cross-embodiment zero-shot**(LAP-3B식).
- 베이스라인: **LAP-3B**(최근접), π0/π0.5, GR00T-N1.5, OpenVLA-OFT, Discrete Diffusion VLA.

---

## 11. 주의점 / 정직한 경계

- **"적절히 어려운" target만 신호가 됨.** 완전히 예측 불가능한 target(예: 1000스텝 뒤 관측)은 noise만 학습. underdetermined *하지만 partially predictable*한 sweet spot을 찾는 게 실험의 일부.
- **retrodiction은 auxiliary(학습용)**, 배포 롤아웃엔 forward policy만 사용. 프레이밍: "retrodiction이 표현을 개선해 forward policy를 좋게 한다."
- **LAP과의 관계 명시**: LAP의 인프라(PaliGemma+KI+언어-action+VQA co-train)를 채택. 우리 기여는 그 위의 **retrospective + world-model + 난이도별 양방향 예측 매트릭스**와 그 과학적 검증.

---

## 12. Claude Code 첫 세션 To-Do (제안)

1. **LAP-3B 코드베이스 클론·셋업** (`github.com/lihzha/lap`), JAX/openpi 환경 구성, 기존 학습·평가 파이프라인 재현(baseline 확인).
2. **데이터 인터페이스 파악**: LAP의 언어-action 생성 방식, VQA(motion-prediction) task 구현 위치 확인. → 여기가 우리 task를 얹을 진입점.
3. **Task 매트릭스 구현** (§4): question 포맷 정의, prefix 구성 로직(답 modality 제외), 각 task별 answer head 라우팅.
4. **hindsight task 추가** (instruction inference, retrodictive state): LAP의 motion-prediction VQA를 확장하는 형태.
5. **temporal position embedding 추가** (§7).
6. **World-model head** (§6): V-JEPA 2 target + latent alignment loss.
7. **믹스 샘플러** (§4, §8): 데이터셋별 가능 task 집합 + 재정규화.
8. **§10.1 ablation 스캐폴딩**: (a)~(d) config 스위치.

---

## 부록: 핵심 설계 결정 로그 (왜 이렇게 됐나)

- **dVLM 안 씀**: ≤3B에 native full-bidirectional 없음 + "question at the end"로 causal 충분.
- **FAST 안 씀**: 언어-action이 cross-embodiment·언어 prior·VQA 통합·tokenizer-free에서 우월.
- **연속으로 안 감(전부 이산 language)**: 단, KI 연속 head는 정밀 필요시 유지 (LAP 기본).
- **World model = JEPA(생성 아님)**: 일반화 목표 + PaliGemma native 생성 없음 + SG 철학 정합.
- **expert 분리 vs 통합**: question 스킴으로 world model도 백본 안에서 처리 → 별도 world-model expert 불필요(action 연속 head만 KI로 분리).
- **논문 제목**: *Hindsight and Foresight: Pretraining Generalist Robot Policies*
