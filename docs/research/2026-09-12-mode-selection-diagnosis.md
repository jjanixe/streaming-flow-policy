# Grouped preference: mode 선택 실패 진단과 수정 순서

2026-09-12. 기존 `pilot-seed0`의 저장 후보와 실제 실행 결과를 재분석했다. 이번 분석에서는 policy를 새로 실행하거나 학습하지 않았다. 현재 문제는 **선호 feature와 mode 의미의 불일치, 주어진 상태에서의 후보 부족, 확률적 선택의 강도**로 나누어야 한다. Few-shot weight 추정만 개선해서 해결될 문제는 아니다.

기존 구현/실험: [grouped preference 결과](2026-09-11-grouped-preference-results.md).
재현 스크립트: [mode_selection_diagnosis.py](../../env/artifacts/grouped_preference/analysis/mode_selection_diagnosis.py).
새 원자료: [mode_selection_diagnosis.json](../../env/artifacts/grouped_preference/analysis/mode_selection_diagnosis.json).

## 1. 현재 oracle의 목적부터 mode 이름과 다르다

현재 구현은 `r_t=tanh(y_t/.35)`, `d=mean(r_t)`, `e=mean(r_t²)`로

\[
f_D=[(1+d)/2,(1-d)/2],\qquad f_W=[e,1-e]
\]

를 계산한다. `upper` 점수는 위쪽으로 멀리 갈수록 커지고, `narrow` 점수는 직선에 가까울수록 커진다. 따라서 방향과 폭이 분리되어 있지 않고, narrow는 시연의 narrow 경로에 대한 적합도가 아니다.

동일한 시작/종점의 analytic 경로 5개(진폭 +.27, +.53, -.27, -.53, 0)를 현재 oracle utility로 평가하면 다음과 같다. 숫자는 실제 `grouped_features`와 `grouped_utility`로 계산했다.

| Oracle profile | Upper-narrow | Upper-wide | 직선 | 가장 선호하는 경로 |
|---|---:|---:|---:|---|
| upper+narrow, direction 우선 | .6677 | **.6948** | .5800 | Upper-wide |
| upper+narrow, width 우선 | .7299 | .5898 | **.8200** | 직선 |

Lower도 대칭적으로 동일하다. 이전 dense amplitude grid 검사에서도 narrow/direction의 최대는 ±.58, narrow/width의 최대는 약 ±.029였다. 이 결과는 policy와 무관한 목적함수 진단이다. 현재 synthetic feedback은 바로 이 utility에서 생성했으므로, 학습기가 응답을 잘 맞추더라도 mode 이름대로 실행하지 않을 수 있다.

**첫 수정안:** toy mode benchmark에서는 먼저 평가 기준과 일치하는 feature로 이 문제를 격리한다. `y_32` 기준으로 기존과 같은 네 mode bin을 정의하고,

\[
f_D=[\mathbf1\{upper\},\mathbf1\{lower\}],\qquad
f_W=[\mathbf1\{wide\},\mathbf1\{narrow\}]
\]

를 사용한다. `|y_32|<.12`인 other는 네 점수를 모두 0으로 둔다. Upper/lower와 wide/narrow의 두 그룹 및 그룹 내부 weight/그룹 간 alpha simplex는 유지한다. 현재 합성 profile처럼 각 그룹의 목표 속성 weight가 .9, 반대가 .1이고 두 alpha가 양수이면, 목표 조합의 utility는 .9로 모든 다른 mode보다 높다. 그룹 중요도는 한 속성씩 상충하는 두 경로의 순서를 결정한다.

이 indicator는 mode 정의가 맞는지 검증하기 위한 기준이다. 이후 연속 steering에는 방향이 충분히 정해지면 포화되는 direction 점수와 narrow/wide 시연 폭에 각각 최대를 갖는 부드러운 점수를 검토한다. Width를 단순히 `1 - excursion`으로 정의하지 않는다. 새로운 feature의 최적 경로가 원하는 bin에 있는지 먼저 검사한다. 관측된 경로의 평가와 실행에 hard mode label을 입력하는 것은 별개의 설계다.

Feature를 바꾸면 기존 synthetic BT label과 oracle도 새 의미로 다시 생성해야 한다. 기존 수치와 같은 ground truth라고 취급하면 안 된다. Other 또는 폭 prototype 점수 때문에 feature의 합이 항상 1이 아니게 되면, 기존 complementary-feature 전제의 identifiability 결론도 다시 검산한다. P/M/C 확장은 필요하지 않다.

## 2. 처음부터 원하는 mode가 없는 후보 bank가 있다

선호가 아직 실제 상태에 영향을 주지 않은 첫 결정의 bank를 사용했다. 각 regime의 16 episode seed × 현재 latent 8개 × base 미래 4개 = 512개 미래 경로다. 같은 episode의 모든 oracle profile에 current/future latent와 expected feature가 정확히 같은지도 확인했다. 미래 4개는 현재 chunk를 공유하므로 512개의 독립 현재 후보로 세면 안 된다.

| 첫 결정의 완주 미래 경로 | Upper-narrow | Upper-wide | Lower-narrow | Lower-wide | Other |
|---|---:|---:|---:|---:|---:|
| Centered | 78 | **0** | 98 | **0** | 336 |
| Gaussian | 52 | 164 | 36 | 252 | 8 |

Centered는 512개 모두 task 성공이다. Gaussian은 모두 65-state 완주했지만 task 성공은 470개이며, 성공한 경로만의 counts는 52/150/36/224/8이다.

Gaussian의 전체 합계에 네 mode가 있어도 **동일한 초기 상태에서 네 mode를 선택할 수 있다는 뜻은 아니다.** GIF의 episode 0은 시작점 `(-.9479999, .0569164)`에서 32개 미래가 모두 upper-wide였다. 이 bank에서는 lower 선호의 weight를 정확히 알아도 lower 후보를 고를 수 없다. 16개 Gaussian 시작점 중 성공한 upper-narrow 미래가 하나라도 있는 것은 3개, upper-wide는 6개, lower-narrow는 3개, lower-wide는 9개뿐이다.

학습 데이터도 확인했다. Train 시연은 mode별 128개로 균형 잡혔지만, 512개 모두 시작점은 정확히 `(-1,0)`이다. Gaussian 시작점은 초기 context의 학습 분포 밖이다. 이는 초기 위치에 따라 branch가 강하게 정해지는 현상의 원인 후보다. 이 관찰만으로 학습 분포 차이가 유일한 원인이라고 결론내리지는 않는다. Centered에서도 wide가 부족하므로 Gaussian augmentation만으로 모든 문제가 해결된다고 할 수 없다.

또한 **유한한 base-continuation bank에서 wide가 0인 것과, 동결한 policy의 모든 latent sequence로 wide가 불가능한 것은 다르다.** 이후 모든 chunk에 guidance를 적용하거나 다른 latent 영역을 탐색하면 도달할 가능성은 별도로 검사해야 한다.

**다음 수정/진단 순서:**

1. 수정한 feature에서 true preference를 고정하고, 같은 초기 상태/seed에 대해 M=8/32/128과 soft/argmax를 비교한다. L은 먼저 고정해 현재 후보 수 효과를 분리한다. 전체 episode에서 성공하며 목표 mode에 들어갔는지를 측정한다. 지금 저장 bank에서 계산한 수치를 이 새 closed-loop 실험 결과로 대신하지 않는다.
2. Random proposal의 coverage가 부족하면 **동결한 policy의 latent를 inference에서 탐색**한다. 현재 chunk의 latent만 바꾸는 탐색과 여러 미래 chunk의 latent sequence를 함께 바꾸는 탐색을 구분한다. 탐색은 원래 prior 주변에서 시작하고, 후보의 실제 action 한계와 goal 도달을 검사한다. 현재 base-future 평균 Q를 쓰는 selector와 sequence 최적화 controller는 서로 다른 방법으로 보고한다.
3. 충분히 넓은 탐색에서도 coverage가 부족한 경우 base 학습을 별도 단계로 고친다. Centered에서 latent별 네 mode 분포를 검증하고, perturbation을 포함한 **동일한 시작 context를 네 mode 모두에 제공하는 시연**을 만든다. 이후 prefix에서도 복구 가능한 학습 경로가 필요하다. Checkpoint 선정에는 validation loss뿐 아니라 동일 context의 mode coverage를 포함한다. 이렇게 수정한 base를 다시 동결하고 사용자별 학습은 few-shot preference에 한정한다.

Mode-conditioned base는 별도의 비교 기준으로 둘 수 있지만, 이것만으로 현재 unconditioned SFPS의 inference steering이 해결됐다고 주장하지 않는다.

## 3. Beta만 높이면 어떤 사용자에게는 더 나빠진다

현재 selector는 `P(j) ∝ exp(beta * Q_j - C_j)`다. 기존 learned closed-loop 평균 ESS는 8개 중 약 7.36–7.86으로 후반을 포함한 많은 decision에서 재가중이 약했다. 다만 첫 결정의 oracle ESS는 그보다 작으므로 모든 decision이 거의 uniform했다고 일반화하지 않는다.

같은 첫 결정 bank에 대해 oracle weight를 고정하고 beta만 바꿨다. 아래는 **선택된 현재 후보 뒤에 저장된 base 미래를 균등하게 따를 때의, 성공 및 목표 mode에 해당하는 경험적 확률**이다. 새로운 실제 rollout 성공률이 아니다. 각 후보의 L=4 mode 빈도에 selector 확률을 곱한 뒤 episode별 값을 평균했다.

| Centered 목표 profile | beta=16 | beta=64 | beta=256 | beta=16 score의 argmax |
|---|---:|---:|---:|---:|
| Upper-narrow, direction 우선 | 30.5% | 58.2% | 69.1% | 70.3% |
| Upper-narrow, width 우선 | 11.9% | 4.4% | **0.1%** | **0.0%** |
| Upper-wide, direction 우선 | 0.0% | 0.0% | 0.0% | 0.0% |

Direction 우선 narrow는 이 bank에서 가장 큰 upper 후보를 고르면 narrow에 머물러 숫자가 좋아진다. Width 우선 narrow는 잘못 정의된 목적을 더 강하게 최적화하면서 other로 향한다. Wide는 bank에 후보가 없어 바뀌지 않는다. 이 결과는 beta 조절 전에 feature와 coverage를 확인해야 함을 보여준다. Beta가 커지면 고정된 task cost에 대한 preference의 상대 비중도 커지므로, 목표 도달과 action 한계를 함께 평가한다.

## 실행할 비교와 통과 기준

우선 두 그룹의 mode feature를 교정한 뒤 **oracle 제어가 먼저 가능한지** 확인한다. 비교 순서는 `(old feature, oracle) → (mode-aligned feature, oracle) → (mode-aligned feature, learned)`로 하고, 각 단계에서 후보/미래 seed를 맞춘다. M, L, beta, selector를 한 번에 바꾸지 않는다.

- Feature 검사: 네 대표 mode와 straight를 포함한 analytic bank에서 각 합성 profile이 목표 mode를 가장 높게 평가한다.
- Coverage 검사: 같은 초기 context에서 각 목표 mode로 성공하는 proposal의 빈도를 보고한다. Centered와 Gaussian을 구분한다.
- Control 검사: 후보 bank의 예측 mode뿐 아니라 실제 전체 episode의 `task success AND target mode`를 평가한다. Oracle soft/argmax와 learned를 나란히 둔다.
- Few-shot 검사: 수정한 의미의 scoped feedback으로 K=5/10/20/40을 다시 비교한다. Oracle과 learned의 차이를 preference 학습 손실로 분리한다.

Anchor는 실제 prefix 끝을 근거로 하는 기존 SFPS 초기좌표를 유지하고, 9개 반환점 중 8개 command를 실행한다. Latent 탐색도 동일한 anchor에서 수행한다. 현재 audit에서 anchor 실행 규약 오류가 mode 실패의 원인이라는 근거는 없었다.

이번에 확인한 것은 원인 분리와 수정 우선순위다. 새 feature, latent 탐색, 학습 데이터 보완의 closed-loop 효과는 아직 실행하지 않았다.

재현:

```bash
PYTHONPATH=. .venv/bin/python \
  env/artifacts/grouped_preference/analysis/mode_selection_diagnosis.py
```


후속 상태: [후속 구현·실험 결과](2026-09-12-mode-aligned-steering-results.md)에서 feature/selector/후보 탐색 비교와 실제 inference GIF를 확인할 수 있다.
