# Grouped preference 수식 검증과 현재 SFPS 적용 검토

검토일: 2026-09-11.

> 후속 적용: 사용자 승인에 따라 upper/lower와 wide/narrow 두 그룹을 구현하고,
> 현재 checkpoint의 anchor-inclusive convention으로 MD를 수정했다. P/M/C는 toy에서 제외했다.
> [구현·실험 결과](../../docs/research/2026-09-11-grouped-preference-results.md).
> 아래 본문은 grouped 구현 이전 시점의 검토 기록이며, 이전 flat pilot과 후속 구현을 구분해서 읽는다.

**핵심 판정:** 제공된 grouped MD의 핵심 수식은 명시한 가정 아래 일관된다.
현재 pilot은 signed 2D overall BT와 단일 미래 경로 재선택을 구현했으며,
grouped parameter 추정과 conditional expected-utility guidance는 아직 구현하지 않았다.
특히 유한 continuation 수 `L`의 편향과 실제 feature의 식별 가능성을 보강해야 한다.

## 1. 확인한 원문과 확인하지 못한 부분

직접 읽은 수식 원문은 작업 폴더의
[vrhandover_grouped_few_shot_guidance_formulation.md](../../vrhandover_grouped_few_shot_guidance_formulation.md)다.
아래 식 번호 (1)–(19)는 그 문서를 가리킨다. 현재 구현은 기존 B2 worktree의
[runner](../../env/run_preference.py),
[fitter](../../env/preference.py),
[continuation simulator](../../env/preference_rollout.py),
[features](../../env/features.py)를 대조했다.

별도 프로젝트 `Vrhandover-streaming-policy`의 원래 규약은 아직 확보하지 못했다.
로컬 경로 검색과 연결된 GitHub에서 정확한 이름의 저장소를 찾지 못했고, 접근 가능한
`jjanixe/VRHandover`의 기본 브랜치 검색에서도 preference 규약은 확인되지 않았다.
사용자에게 경로·저장소·브랜치를 요청했다. 제공된 MD가 인용한 D1의 다른 대화와
D2 `vrhandover_streaming_flow_policy_integration_review.md`,
D3 `streaming_flow_policy_summary.md`도 이번 작업 공간에는 없었다.
따라서 **MD 내부의 수학적 일관성 검증**과 **별도 원본 규약 준수 확인**을 구분한다.
후자는 원문 확보 후 추가 대조가 필요하다.

## 2. 수식별 판정

| 식 | 판정 | 필요한 조건·해석 |
|---|---|---|
| (1) complete option | 정의로 타당 | `complete`는 staging 성공 후 handover 범위다. Grasp 획득·staging 실패 비용까지 평가하려면 별도 outcome을 포함해야 한다. |
| (2) 실행 prefix | 타당 | Anchor 제외 convention을 generator와 맞춰야 한다. 현재 구현의 horizon은 아래에 따로 기록한다. |
| (3)–(5) 그룹 utility·simplex | 타당 | 비음수 weight는 공통 desirability/basis feature를 전제한다. 그룹화 자체가 비선형 선호나 차원 축소를 만들지는 않는다. |
| (6) group-specific BT | 타당 | 해당 그룹만 평가하라는 질문이다. `alpha_r`를 넣지 않는다. |
| (7) overall BT | 타당 | 그룹 중요도와 내부 weight가 모두 들어간다. Pair에서 한 그룹만 변화시켰다는 이유로 (6)으로 바꾸면 안 된다. |
| (8) joint loss | 유효한 추정 목적함수 | 일반적으로 `alpha,w`에 대해 jointly nonconvex다. 같은 pair의 여러 응답을 곱하는 것은 조건부 독립 가정 또는 composite likelihood다. 비유일해를 허용하면 `omega_hat ∈ argmin`이 더 정확하다. |
| (9) regularizer | 타당 | 고정 reference에 대한 shrinkage이며 데이터가 식별하지 못하는 선호를 학습했다는 근거가 아니다. |
| (10)–(12) expected group features | 기대값의 선형성으로 정확 | 모든 그룹에 동일한 base continuation 법칙을 사용하고, 성공·중단을 포함한 종료 outcome에서 feature가 정의되어야 한다. |
| (13)–(14) KL와 Gibbs target | 정확 | 유한 목적함수의 정의역과 적분 가능성이 필요하다. Bounded feature는 간단한 충분조건이다. |
| (15) pushforward | 타당 | Latent 분포가 실행 prefix 분포로 변환된다. 일반적인 latent KL과 measured-trajectory KL의 동일성은 주장할 수 없다. |
| (16) resampling | 유한 후보의 가중 선택식으로 정확 | `p_Z`에서 뽑았으므로 Gaussian density를 다시 곱하지 않는다. 유한 `L`의 오차는 `M`만 늘려도 없어지지 않는다. |
| (17) latent gradient | 미분 가능성 아래 정확 | `-z + beta grad Q`. Black-box simulator의 scalar 반환값만으로 이 gradient가 생기지는 않는다. |
| (18) 고정 grasp 항 | 정확 | Additive grasp utility는 소거된다. 다만 simplex의 `alpha_G`는 남은 그룹의 guidance 강도에 영향을 준다. |
| (19) outer grasp 선택 | 정의로 타당 | Guided handover와 staging 불확실성·실패 outcome을 평가해야 한다. Base continuation으로 바꾸면 별도 근사다. |

SFP의 auxiliary latent ODE 해석은 원 논문의 해당 variant와 부합한다.
Conventional chunk-flow의 noisy action tensor와 SFP trajectory point를 구분한 것도
적절하다. [SFP 원문 Appendix B](https://arxiv.org/html/2505.21851v2#A2)

## 3. 핵심 Gibbs 수식은 왜 맞는가

한 decision의 context를 고정하고

\[
Q(z)=\sum_r\hat\alpha_r\hat w_r^\top
\mathbb E_0[\widetilde f_r(\chi)\mid\bar c_k,B_k(z)]
\]

라 하자. `p=p_Z`, `Z=E_p exp(beta Q)`이고
`q*(z)=p(z) exp(beta Q(z))/Z`이면, 유한한 항들에 대해

\[
\mathbb E_q Q-\frac1\beta\mathrm{KL}(q\|p)
=\frac{\log Z}{\beta}-\frac1\beta\mathrm{KL}(q\|q^*).
\]

KL의 비음수성으로 식 (14)가 식 (13)의 최적해다. 지수의 부호와 `1/beta` 위치가 맞다.
이 구조는 relative-entropy policy improvement와도 연결되지만, 해당 논문 전체
알고리즘을 구현했다는 뜻은 아니다.
[Peters et al., REPS, AAAI 2010](https://ojs.aaai.org/index.php/AAAI/article/download/7727/7588)

`Z<infinity`는 density 정규화를 보장한다. 목적함수의 두 항까지 유한하게 다루려면
예를 들어 `E_p[exp(beta Q)|Q|]<infinity`도 확인하거나, 이번 첫 구현처럼
**모든 종료 outcome에서 유한하고 bounded인 feature**를 명시하면 된다.
정확한 full-trajectory tilted marginal 또는 반복 실행의 전역 최적해라는 주장은
이 유도에서 나오지 않으며, 원 MD도 이를 올바르게 제한하고 있다.

## 4. 가장 중요한 차이: exp(E U)와 E exp(U)

문서가 의도한 current-latent target은

\[
q^*(z)\propto p_Z(z)\exp\{\beta\mathbb E_0[U(\chi)\mid z]\}.
\]

현재 코드는 서로 다른 현재 latent마다 **미래 latent schedule 하나**를 붙여 전체 경로를
시뮬레이션한 뒤 그 utility를 지수 가중한다. 독립적인 schedule 후보 수가 커질 때,
현재 latent의 marginal은 대략 다음 대상으로 간다.

\[
q_{\mathrm{old}}(z)\propto p_Z(z)
\mathbb E_0\!\left[
\mathbf1_{\mathrm{full\ future\ valid}}
\exp\{\beta U(\chi)-C_{\mathrm{goal}}(\chi)\}\mid z
\right].
\]

현재 유한 32개 후보가 이 density의 exact sampler라는 뜻은 아니다.
**기대값 밖과 안의 exp 위치가 다르므로 두 목표는 일반적으로 다르다.**
현재 결과는 유용한 full-schedule resampling baseline으로 보존할 수 있지만,
그대로 식 (14)의 expected-value guidance라고 해석하면 안 된다.

더 일반적으로 각 `z`에 대해 IID continuation `L`개로
`Q_hat_L=(1/L) sum_l U_l`을 추정해 지수 가중하면, gate와 task cost를 생략한
독립 후보·평가 설정에서 `M→infinity`, 고정 `L`의 대상은

\[
q_L(z)\propto p_Z(z)\mathbb E[e^{\beta\widehat Q_L}\mid z]
=p_Z(z)\left(\mathbb E[e^{\beta U/L}\mid z]\right)^L.
\]

적절한 cumulant 조건에서 log-weight는

\[
\log\frac{q_L(z)}{p_Z(z)}
=\beta Q(z)+\frac{\beta^2}{2L}\operatorname{Var}(U\mid z)
+O(L^{-2})+\text{normalizing constant}.
\]

따라서 기대 utility가 같아도 **미래 결과의 분산이 큰 후보를 더 선호**할 수 있다.
이는 `M`만 증가시켜 해결되지 않는다. Bounded utility 아래 `L`을 늘려 inner bias를
줄이고 `M`도 충분히 늘려야 한다. Nested Monte Carlo의 이런 문제에 대한 일반적 근거는
[Rainforth et al., ICML 2018](https://proceedings.mlr.press/v80/rainforth18a.html)에 있다.
위 구체적 식은 이 formulation에 대해 직접 유도했다.

검증한 반례: Gaussian latent를 확률 1/2씩인 A/B 영역으로 나눈다. A의 utility는 항상
0.5이고, B는 0 또는 1이 각각 확률 1/2다. 두 영역의 기대 utility는 같다.
`beta=4`일 때 식 (14)는 B를 50% 선택하지만, 고정 `L`의 large-`M` 극한은 다음과 같다.

| L | B 선택 확률 |
|---:|---:|
| 1 | 79.00% |
| 2 | 70.42% |
| 4 | 61.79% |
| 16 | 53.11% |
| 64 | 50.78% |
| infinity | 50% |

이는 실제 pilot에서 측정한 비율이 아니라, **서로 다른 두 수식을 구별하는 bounded
수치 반례**다. 후보가 딱 두 개인 경우의 평균 softmax 확률과도 구분한다.

## 5. Grouped 모델의 학습과 식별 가능성

### 질문 종류를 데이터로 보존해야 한다

\[
\ell_r=\operatorname{softplus}(-\rho_r y\,w_r^\top\Delta f_r),\qquad
\ell_O=\operatorname{softplus}(-\rho_O y\sum_r\alpha_r w_r^\top\Delta f_r).
\]

Group 질문에는 `alpha`가 없다. Overall 질문에는 있다. 현재 query의 `direction`,
`width`, `tradeoff` 태그는 경로쌍 생성 방식이고, 실제 label은 모두 overall BT에서 왔다.
이를 group-specific 응답으로 소급 변환하면 likelihood가 달라진다.
새로운 데이터에는 `(option_A, option_B, scope, label, prompt_version, response_id)`를
저장해야 한다. 동일 영상쌍에 그룹별 3개 질문과 overall 1개 질문을 하면 **4개 판단**이다.

### 그룹화 자체는 파라미터 수를 줄이지 않는다

`v_rj=alpha_r*w_rj`로 놓으면 전체 utility는 `v^T f`이고 `v`는 하나의 simplex에 있다.
`alpha_r>0`이면 `alpha_r=sum_j v_rj`, `w_r=v_r/alpha_r`로 복원된다.
전체 자유도는 원 MD대로 `sum_r d_r-1`이다. Overall contrast 행렬을 simplex tangent
공간에 제한했을 때 충분한 rank가 있어야 하며, 단순한 label 개수만으로 보장되지 않는다.
Group-only 응답만 있으면 `alpha`의 추정값은 데이터가 아니라 regularizer에 의해 정해진다.

각 그룹에 complementary basis를 사용하면 group mass와 내부 contrast 강도를 분리하지
못해, hierarchy가 달라도 모든 overall 비교가 같을 수 있다. 두 그룹의 feature가 중복일
필요는 없다. Direction basis `[(1+d)/2,(1-d)/2]`, narrow/wide basis `[1-e,e]`에서
다음 두 모델은 모든 전체 경로 utility 차이가 같다.

| 모델 | alpha | w_direction | w_narrow/wide |
|---|---|---|---|
| A | `(0.5,0.5)` | `(0.8,0.2)` | `(0.6,0.4)` |
| B | `(0.6,0.4)` | `(0.75,0.25)` | `(0.625,0.375)` |

두 모델 모두 가변 부분이 `0.15*d - 0.1*e`이기 때문이다. 수치 체크의 최대 contrast
차이는 `1.11e-16`이었다. Group-specific 질문이 별도 식별 정보를 제공할 수 있다.

### Simplex는 feature 설계와 최적화를 바꾼다

Positive weight에 raw speed 하나만 주면 느림 선호를 표현할 수 없다. 공통 `[fast,slow]`
basis가 필요하다. 단순 `[x,1-x]`는 양 방향의 단조 선호를 표현하지만, 중간의 특정 속도나
pose를 선호하려면 여러 공통 center의 desirability 등 비선형 basis가 필요하다.
Normalization은 사용자·query·후보 사이에서 고정해야 한다.

Joint objective는 `alpha*w` 때문에 일반적으로 nonconvex다. 초기 구현은 group별
simplex-constrained BT를 먼저 fit하고, 이를 고정해 overall 응답으로 `alpha`를 fit하는
두 단계 추정을 명시적인 baseline으로 둘 수 있다. 각 블록은 고정된 나머지 파라미터에
대해 convex지만, 이것이 식 (8)의 joint 최적해를 보장하지는 않는다. Joint refinement는
초기값 여러 개, 목적함수 값, 수렴과 파라미터 변동을 보고해야 한다.

## 6. 기존 두 feature를 그대로 활용할 수 있는 범위

현재 경로 feature의 raw 값을
`d=mean tanh(y/.35)`, `e=mean tanh(y/.35)^2`라고 두면

\[
U_{\mathrm{old}}=a\frac{d-\mu_d}{s_d}+b\frac{e-\mu_e}{s_e}.
\]

현재 환경에서 바로 정의할 수 있는 surrogate group은 **Direction D**와 **Excursion E**다.
실제 Presentation/Motion/Path-comfort 또는 Grasp로 이름만 바꾸어 해석하지 않는다.

\[
f_D=[(1+d)/2,(1-d)/2],\quad f_E=[e,1-e].
\]

`A=2|a|/s_d`, `B=|b|/s_e`, `S=A+B`, `alpha=(A/S,B/S)`로 두고,
within-group weight는 각 계수 부호가 선호하는 basis의 one-hot으로 두면

\[
U_{\mathrm{old}}=S\,U_{\mathrm{group}}+\text{경로와 무관한 상수}.
\]

완료된 고정 길이 경로에 대한 정확한 대수적 변환이다. 0-weight group의 내부 weight는
임의이고, 두 계수가 모두 0인 경우는 별도 no-guidance convention으로 처리한다.

현재 저장된 normalizer와 네 oracle 사용자 `(±2,±1)`에 적용하면 모두

\[
S=16.20097051,\qquad \alpha=(0.5121974,0.4878026)
\]

이 된다. 즉 **기존 네 사용자는 이 표현에서 그룹 중요도가 같고, 그룹 내부 선호 방향만
다르다.** 기존 결과는 서로 다른 그룹 우선순위를 학습했다는 검증이 아니다.

기존 logits를 보존하려면 oracle 비교 모델의 `rho`와 preference exponent의 `beta`에도
이 scale을 반영해야 한다. 기존 `rho=beta=1`의 oracle과 맞추는 값은 각각 `16.20097`이다.
이는 **대수적 parity 값**이며 추천 최적 guidance 강도가 아니다. 임의 fitted signed
weight에는 서로 다른 S가 생기므로, 사용자마다 rho를 바꾸어 원 MD의 고정·공유 rho 규약을
어기면 안 된다. 새 grouped 실험의 공통 feature scale과 rho를 먼저 고정해야 한다.

Float64 feature 계산의 대수 오차는 최대 `2.67e-15`였고, 원래 저장된 float32 contrast와의
차이는 최대 `1.43e-6`였다. 이 변환은 기존 결과를 재해석하는 검사이며 새로운 grouped
feedback을 얻거나 사용자별 두 수준의 weight를 식별한 실험은 아니다.

## 7. 현재 환경에서 실제로 식별할 수 없는 그룹

| 문서의 의미 | 현재 데이터·환경 | 적용 판단 |
|---|---|---|
| Presentation: 최종 전달 위치·자세 | 80개 query option의 endpoint가 모두 정확히 `(1,0)`. Orientation·receiver가 없음 | 선호 대비가 없다. 별도 유효 docking 위치·자세와 demonstration support가 필요하다. |
| Motion: duration | 완주 경로는 모두 64 action, 2초 | Duration 선호를 추정할 수 없다. |
| Motion: mean speed | `path_length/2초`와 동일 | Path-comfort의 길이와 독립된 정보라고 취급할 수 없다. |
| Motion: smoothness·speed profile | 변할 수 있지만 현재 analytic family에서는 amplitude와 연결됨 | 상수라고 단정할 수는 없다. Geometry와 timing을 달리하는 query와 contrast conditioning 검사가 필요하다. |
| Path comfort: clearance | Obstacle이 없음 | Lateral excursion을 물리적 clearance로 부를 수 없다. |
| Grasp | 상태·행동 공간에 grasp가 없음 | Handover의 outer discrete selection을 별도로 구현할 때 다룬다. |

Policy 출력의 작은 최종 위치 오차는 사용자에게 허용된 다양한 presentation의
demonstration support와 다르다. 기존 single-detour 환경의 D/E surrogate와 문서의
Fork-and-Dock P/M/C 설계를 분리해야 한다.

## 8. 현재 SFPS에 적용할 구체적 순서

### A. 현재 checkpoint로 grouped 학습·평가 연결을 먼저 검증

1. D/E surrogate와 bounded 공통 basis를 고정한다. Schema version, normalization,
   실제 질문 scope를 저장한다. 현재 overall label을 group label로 재사용하지 않는다.
2. 동일 within-group 취향에 대해 다른 `alpha`를 갖는 synthetic 사용자도 추가한다.
   예를 들어 direction-priority `(0.8,0.2)`와 excursion-priority `(0.2,0.8)`가 같은
   tradeoff pair에서 다르게 응답하는지 본다. 값은 제안 실험값이며 검증된 기본값이 아니다.
3. K=20이면 D 질문 6개, E 질문 6개, overall 8개와 같은 판단 budget을 사전 고정할 수 있다.
   실제 pair contrast rank, scope별 held-out log-loss/calibration과 여러 label seed를 평가한다.
4. Simplex 두 단계 추정을 baseline으로 하고 필요하면 식 (8)의 joint refinement를 비교한다.
   `w`와 `alpha`의 추정 오차를 별도로 보고한다.

### B. 현재 latent와 미래 latent를 분리

```text
현재 latent z:          [batch, M, 2]
미래 base latent들:     [batch, M, L, remaining_chunks-1, 2]
완전한 option feature:  f_r[batch, M, L, d_r]
기대 feature:          Psi_r = mean_L(f_r)
personalized value:    Q = sum_r alpha_r * (Psi_r @ w_r)
선택 logits:           beta * Q
```

동일 후보의 L개 continuation은 **현재 latent와 그로부터 실행되는 8개 행동을 공유**하고,
그 이후의 base-policy randomness만 다르게 한다. 현재 prefix는 이미 실행한 상태를
보존한다. 기존 32개 경로는 현재 latent부터 다르므로 이를 단순 평균해 L개 continuation으로
대체할 수 없다. 그룹마다 다른 continuation을 사용하기보다 같은 rollout에서 모든 feature를
계산해야 weighting과 기대값의 비교가 명확하다.

첫 계산 예산 비교로 `M=8,L=4`, `M=16,L=2`, `M=32,L=1`을 고려할 수 있다.
각각 32개 future outcome을 계산하지만 **M 변화에 따른 후보 coverage도 함께 바뀐다.**
L 효과를 분리하려면 소수의 고정 decision에서 M을 고정한 `L=1/4/16` 진단도 필요하다.
Prefix 계산 공유로 연산량이 줄 수 있으나 같은 wall-clock latency를 보장하지 않는다.
마지막 8-step decision은 현재 deterministic toy에서 이후 randomness가 없으므로 L=1이면 된다.

### C. Task cost, gate와 실패 convention을 명시

원 MD의 gate는 현재 실행 prefix에 대한 독립 gate다. 현재 코드는 미래 전체의 실패로
후보를 제거하므로 동일한 gate가 아니다. 미래 실패는 정의한 종료 outcome의 feature와
별도 task cost에 반영하고, 성공한 continuation만 골라 평균하지 않는다.
현재 feature 함수가 NaN/미완료 경로를 받지 못하므로 이 terminal convention을 먼저
정해야 한다. 임의 padding이나 실패 샘플 누락은 그 정의를 대체할 수 없다.

기존 goal penalty를 유지한다면 명시적으로 다음 확장으로 기록한다.

\[
q_{\mathrm{task+pref}}(z)\propto
p_Z(z)\mathbf1_{\mathrm{prefix\ safe}}(z)
\exp\{\beta Q_{\mathrm{pref}}(z)-\lambda_{\mathrm{task}}C_{\mathrm{task}}(z)\}.
\]

여기서 예를 들어 `C_task=E[(goal_error/0.1)^2 + failure_cost | z]`다.
Goal error의 평균을 제곱하는 것과도 구분한다. `beta=0`이면 **task-conditioned base**로
돌아가며 원래 p_Z로 돌아간다는 식 (14)의 설명은 이 확장에 그대로 적용되지 않는다.
현재 candidate-0 fallback은 오류를 기록하는 기존 동작이지, 안전한 fallback을 검증한 것이
아니다. 새 gate 실험에서는 명시적으로 확인한 hold/stop 등의 동작을 정의해야 한다.

### D. Horizon을 현재 checkpoint에 맞게 명시

현재 pilot은 **관측 2 / 생성 future command 8 / 실행 8**이다.
`num_actions=9`로 anchor와 future 8개를 받고 anchor를 버린다.
Checkpoint의 `pred_horizon=16`은 anchor-inclusive 학습 시간 convention이다.
원 MD의 **future 16, anchor 별도**와 같지 않다.
Grouped 검증은 현재 8/8 convention으로 먼저 가능하며, future16 비교는 별도
data/time convention 검토 후 수행해야 한다. 숫자만 바꾸면 같은 policy가 되지 않는다.

### E. P/M/C와 실제 G/P/M으로 확장

P/M/C를 검증하려면 task-valid docking alternatives, 독립적으로 바뀌는 motion profile,
물리적으로 정의된 path-comfort feature가 필요하다. 고정 2초에서도 fast-start/slow-finish
등 speed-profile을 바꿀 수는 있지만, duration 선호를 검증하려면 duration을 실제로 바꾸어야 한다.
새 demonstration bank와 필요시 관측·환경을 정의하고 base policy를 한 번 학습한 뒤 동결한다.
그 후에 grouped few-shot 개인화를 평가한다.

실제 VRHandover에서 고정 grasp의 additive utility는 latent 선택에서 소거된다.
`alpha_G<1`일 때 `alpha'_r=alpha_r/(1-alpha_G)`로 정의하면 원래 exponent는
`beta*(1-alpha_G)*sum_{r=P,M} alpha'_r Q_r`로 다시 쓸 수 있다. 따라서 원래 목적함수의
유효 강도는 `beta_effective=beta*(1-alpha_G)`이고, 큰 grasp 중요도는 같은 상대적 P/M
weight와 고정 beta에서 handover steering을 약하게 만든다. Grasp weight를 버린 뒤
정규화한 P/M weight에 기존 beta를 그대로 쓰면 다른 목적함수가 된다.
Grasp 자체의 개인화는 원 MD의 outer discrete selector와 guided handover value가 필요하다.

## 9. 검증 산출물과 이번 작업 범위

[수치 검증 코드](grouped_formulation_checks.py)와
[실행 결과 JSON](2026-09-11-grouped-formulation-checks.json)을 저장했다.
실행 명령은 저장소 루트에서 다음과 같다. 기존 pilot NPZ가 필요하다.

```bash
.venv/bin/python docs/research/grouped_formulation_checks.py
```

Gibbs 목적함수 항등식 100개 분포, 유한-L 반례, 서로 다른 hierarchy의 동일 overall
contrast, 기존 oracle utility의 simplex 변환, 실제 query endpoint·속도/길이 관계를 확인했다.
이 검사는 수식·데이터 사실을 검증하며 grouped policy의 학습 성능을 검증한 실험은 아니다.
이번 요청에서는 기존 policy·실험 runner를 변경하거나 새 학습·inference를 실행하지 않았다.
