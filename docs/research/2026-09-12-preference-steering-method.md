# Grouped few-shot preference steering: method와 구현

2026-09-12. 현재 구현과 실제 실행 설정을 설명하는 문서다. [결과](2026-09-12-mode-aligned-steering-results.md), [산출물 목록](2026-09-12-preference-steering-artifacts.md), [이전 continuous 실험](2026-09-11-grouped-preference-results.md), [논문·공개 코드 조사](2026-09-11-preference-publication-audit.md)를 함께 참조한다.

## 1. 구현 범위와 전체 흐름

현재 방법은 **동결한 stochastic Streaming Flow Policy(SFPS)의 latent 후보를 inference 때 평가·선택하는 방식**이다. 사용자별로 학습하는 것은 3자유도의 작은 preference model이다. Flow network의 in-context learning, network gradient guidance, preference-conditioned policy 재학습을 구현한 것은 아니다.

```text
네 mode의 synthetic demonstration → base SFPS 학습 → checkpoint/stats 동결
scope가 붙은 A/B 선호 응답 → grouped preference의 세 파라미터 fit
실제 관측 prefix → M개 현재 latent → 후보별 L개 base 미래
→ full-episode utility와 task cost → 현재 후보 선택 → 8 command 실행 → 재관측
```

Toy의 두 그룹은 `D=(upper, lower)`, `W=(wide, narrow)`다. P/M/C는 포함하지 않는다. 별도 VRHandover 원본 저장소 규약과의 대조가 끝났다는 뜻은 아니며, 제공된 [grouped formulation](../../vrhandover_grouped_few_shot_guidance_formulation.md)에서 이 두 그룹만 적용했다. [수식 검토와 조건](2026-09-11-grouped-preference-formulation-review.md)을 보존했다.

환경은 position-command 2D reach다. 시작점은 centered `(-1,0)` 또는 해당 점 주변 Gaussian(표준편차 .04), goal은 `(1,0)`, episode는 64 command, 최대 한 step 거리는 .075, 최종 goal tolerance는 .1이다. 관측은 `(x,y,normalized physical time)`이며 policy는 최근 두 관측을 조건으로 사용한다. Base의 train set은 centered 시작점의 512 episode(네 mode별 128개)다.

## 2. Frozen SFPS와 anchor

물리 실행 step을 $q$, SFPS local trajectory time을 $t$라고 구분한다. 정규화된 action $a$와 보조 latent $z\in\mathbb R^2$에 대해 동결 network는 다음 joint ODE를 정의한다.

$$
\frac{d}{dt}\begin{bmatrix}a_t\\z_t\end{bmatrix}
=v_{\bar\theta}\!\left(t,\begin{bmatrix}a_t\\z_t\end{bmatrix},c_q\right),
\qquad a_0=N_{\rm obs}(o_{q-1:q})_{-1,:2},\quad z_0=z_j.
$$

Checkpoint의 prediction horizon 16에는 anchor가 포함된다. 현재 호출은 9점을 반환하고, local time $0,1/15,\ldots,8/15$ 중 index 1..8만 action 통계로 역정규화하여 실행한다. 다음 결정은 실제로 도달한 prefix 끝을 다시 anchor로 삼는다. 후보별 또는 사용자별로 anchor를 이동시키지 않는다.

```python
# env/preference_rollout.py: _predict / simulate_continuations
prediction = policy.predict_batch(
    nobs, num_actions=9, integration_steps_per_action=6, latents=current_latents)
points = prediction.detach().cpu().numpy()
commands = unnormalize_actions(points[:, 1:, :], stats)
```

위 코드는 핵심 호출을 요약한 것이다. 실제 helper는 tensor→NumPy 변환, shape/dtype 검사와 numerical exception의 row별 재시도를 포함한다. [SFPS 구현](../../env/sfp_policies.py)의 `predict_batch`는 `dopri5`(atol=rtol=1e-4)를 사용한다. `integration_steps_per_action=6`은 반환을 요청하는 시간 grid의 설정이며, adaptive solver의 내부 function-evaluation 수가 정확히 6이라는 의미는 아니다.

Observation/action normalization의 x 최소값 차이로 **버려지는** 모델 anchor에는 최대 약 $7.45\times10^{-5}$의 물리 좌표 차이가 있다. 기존 checkpoint 규약을 보존하고 실제 관측 anchor와 반환 모델 anchor를 각각 저장한다. [확인 값](../../env/artifacts/grouped_preference/analysis/anchor_check.json).

Base 학습은 사용자 preference와 별도다. [벡터화된 training target](../../env/drake_trajectory.py)의 `sample_sfps_training_targets`에서, 정규화된 demonstration의 first-order-hold 경로 $\xi(t)$에 대해

$$
\sigma_r=\sqrt{\sigma_1^2-\sigma_0^2},\quad z_0\sim\mathcal N(0,I),\quad
\epsilon_{a0}\sim\mathcal N(0,\sigma_0^2 I),
$$
$$
a_t=\xi(t)+\epsilon_{a0}+\sigma_r t z_0,\qquad
z_t=[1-(1-\sigma_1)t]z_0+t\xi(t),
$$
$$
\dot a_t=\dot\xi(t)+\sigma_r z_0,\qquad
\dot z_t=\xi(t)+t\dot\xi(t)-(1-\sigma_1)z_0.
$$

`StreamingFlowPolicyStochastic.loss`는 joint velocity prediction과 위 두 target을 합친 tensor 사이의 평균 MSE다. Training target은 Torch float32로 벡터화하고 Drake first-order-hold와의 일치는 별도 테스트한다. 현재 preference 실험 중에는 이 학습을 다시 수행하지 않았다.

## 3. Grouped utility와 mode feature

전체 경로를 $\tau=(p_0,\ldots,p_{64})$, $y=y_{32}$라 하자. 현재 `feature_kind=mode`의 정의는 다음과 같다.

| Midpoint mode | $y_{32}$ 범위 | $f_D$ (upper,lower) | $f_W$ (wide,narrow) |
|---|---|---|---|
| upper-narrow | $[.12,.4)$ | (1,0) | (0,1) |
| upper-wide | $[.4,\infty)$ | (1,0) | (1,0) |
| lower-narrow | $(-.4,-.12]$ | (0,1) | (0,1) |
| lower-wide | $(-\infty,-.4]$ | (0,1) | (1,0) |
| other | $(-.12,.12)$ | (0,0) | (0,0) |

$$
U_\omega(\tau)=\alpha_D w_D^T f_D(\tau)+\alpha_W w_W^T f_W(\tau),
\quad w_D,w_W,\alpha\in\Delta^1.
$$

구현의 자유도는 $\eta=(d,w,a)\in[0,1]^3$, $w_D=(d,1-d)$, $w_W=(w,1-w)$, $\alpha=(a,1-a)$다. Feature와 preference 계산은 float64이며, policy와 rollout은 float32다. 저장된 midpoint도 float64로 변환한 값에 경계를 적용한다. 중도 실패 경로는 feature 0, 완주했지만 goal을 놓친 경로는 geometric feature를 유지하며 별도 cost를 받는다.

```python
# env/grouped_preference.py: grouped_utility의 계산
within = np.sum(features * preference.weights, axis=-1, dtype=np.float64)
utility = np.sum(within * preference.alpha, axis=-1, dtype=np.float64)
```

이전 `continuous` feature는 $r_t=\tanh(y_t/.35)$의 평균 $d_c$와 제곱 평균 $e_c$를 이용해 $f_D=((1+d_c)/2,(1-d_c)/2)$, $f_W=(e_c,1-e_c)$로 정의한다. 이 정의에서는 방향 점수가 높이와 함께 증가하고 narrow가 직선을 선호하여 named mode와 어긋났다. 기본값은 호환성을 위해 `continuous`로 남겼으며, 현재 실험은 명시적으로 `mode`를 선택한다.

Hard-bin feature는 mode 판정 일치 실험이다. Wide 경로가 .4 경계 가까이 모일 수 있으며, 시연의 대표 폭·매끄러움까지 최적화하는 연속 스타일 점수는 아니다.

## 4. Few-shot 학습식

응답 $i$는 두 complete trajectory, scope $s_i\in\{D,W,O\}$와 label $b_i\in\{-1,+1\}$로 구성한다. $\Delta f_i=f(\tau_i^A)-f(\tau_i^B)$일 때 logit은

$$
\ell_i(\eta)=\rho\begin{cases}
w_D^T\Delta f_{i,D},&s_i=D,\\
w_W^T\Delta f_{i,W},&s_i=W,\\
\alpha_Dw_D^T\Delta f_{i,D}+\alpha_Ww_W^T\Delta f_{i,W},&s_i=O.
\end{cases}
\qquad P(b_i=+1)=\operatorname{sigmoid}(\ell_i).
$$

**그룹 내부 비교에는 alpha가 들어가지 않는다.** Overall 비교만으로 complementary feature의 세 자유도를 분리 식별했다고 주장하지 않는다. Scoped 비교를 함께 사용한다.

현재 optimizer가 최소화하는 정확한 목적은 존재하는 scope별 **평균** logistic loss의 합이다.

$$
\hat\eta=\arg\min_{\eta\in[0,1]^3}
\left[\sum_{s:n_s>0}\frac1{n_s}\sum_{i:s_i=s}
\log(1+\exp[-b_i\ell_i(\eta)])
+\lambda\|\eta-(.5,.5,.5)\|_2^2\right].
$$

`rho=16`, `lambda=l2=.1`. 구현은 `np.logaddexp`와 analytic Jacobian을 쓴다. 그룹별 scalar fit→alpha fit으로 만든 초기값, uniform 초기값, .15/.85 corner 조합 8개에서 bounded L-BFGS-B를 실행한다. 최종 projected-gradient norm과 optimizer 상태를 기록한다. Joint objective는 비볼록이며, 다중 초기화가 전역 최적성을 보장하지 않는다. [구현: `fit_grouped_preference`, `_objective_state`](../../env/grouped_preference.py).

K=5/10/20/40은 같은 응답열의 prefix다. K20은 direction 6개, width 6개, overall 8개다. Mode feature로 바꿀 때 query/heldout feature와 synthetic BT 응답도 새로 생성했다. 네 선호 조합 × 방향/폭 중요도 두 설정의 8개 synthetic user를 사용한다. 그룹 내 정답 weight는 .9/.1, 우선 그룹 alpha는 .8이다. 이 실험은 실제 사람의 few-shot 성능 검증이 아니다.

## 5. Inference-time steering

실제 관측 prefix $h_q=(p_0,\ldots,p_q)$를 고정한다. 현재 latent와 가상 미래 latent를 구분한다.

$$
z_j=\sigma_{\rm prop}\epsilon_j,\quad\epsilon_j\sim\mathcal N(0,I),
\qquad z^{j\ell}_{q+8:}\sim\mathcal N(0,I).
$$

각 현재 후보는 8 command를 한 번 생성한다. 이 현재 chunk를 L개 미래가 공유하며, 이후는 independent base-policy latent로 끝까지 진행한다. 과거 prefix도 full-episode feature에 포함한다.

$$
\hat f_j=\frac1L\sum_{\ell=1}^L f(h_q\oplus B_q(z_j)\oplus\tau_{\rm base}^{j\ell}),
\qquad \hat Q_j=U_{\hat\omega}(\hat f_j),
$$
$$
\hat C_j=\frac1L\sum_{\ell=1}^L
\left[(\mathrm{goalError}_{j\ell}/.1)^2+25\mathbf1\{\neg\mathrm{success}_{j\ell}\}\right],
\qquad S_j=\beta\hat Q_j-\hat C_j.
$$

Utility가 feature에 선형이므로 $U(E[f])=E[U(f)]$다. 미래 실패도 분모 L에 포함한다. 현재 chunk가 invalid인 후보만 선택 대상에서 제외한다.

$$
P_{\rm soft}(j)=\frac{\mathbf1\{\mathrm{valid}_j\}e^{S_j}}{\sum_{k:\mathrm{valid}_k}e^{S_k}},
\qquad j_{\rm best}=\arg\max_{j:\mathrm{valid}_j}S_j.
$$

```python
# 실제 핵심 연결을 요약. 모든 helper는 아래 소스 표에 있다.
streams = draw_latents(rollout_seeds, chunk=chunk, candidates=M, continuations=L)
current = streams['current_latents'] * np.float32(proposal_std)
batch = evaluate_conditional_continuations(
    policy, stats, prefixes, current, streams['future_latents'],
    environment_config=environment_config, feature_kind='mode')
scores = conditional_scores(batch, learned_preference, beta=16.)
# select_candidates가 valid mask와 soft/best를 처리한다.
# 선택한 current_commands만 실제 Gym에서 실행하고 prefix를 갱신한다.
```

모든 현재 후보가 invalid이면 실제 현재 위치를 유지하는 8-command fallback을 기록한다. `best`의 동점은 NumPy argmax 순서로 해소한다. Raw base는 원래 unit-normal current latent를 그대로 실행한다. Task-only는 $\beta U$ 없이 task cost로 선택하므로 raw base와 다르다.

**현재 주 실험:** `mode / best / M128 / L4 / beta16 / proposal_std8 / learned20`. Std8은 guided 현재 proposal만 넓히며 base와 가상 미래는 std1을 유지한다. 원래 unit-normal prior에 대한 importance correction은 없다.

이상적인 KL-regularized target $q^*(z)\propto p(z)\exp[\beta Q(z)-C(z)]$와 현재 구현을 구분한다. 유한 M/L의 resampling, 추정 평균의 지수화, proposal 변경, 그리고 best의 argmax는 동일한 연산이 아니다. 현재 결과를 원래 prior의 정확한 Gibbs sampling이라고 부르지 않는다. $\exp(\beta\operatorname{mean}U)$와 $\operatorname{mean}\exp(\beta U)$도 다르다.

## 6. 핵심 코드 위치

| 파일 | 핵심 구현과 책임 |
|---|---|
| [grouped_preference.py](../../env/grouped_preference.py) | `GroupedPreference`, `classify_modes`, `grouped_features`, scoped BT, `fit_grouped_preference`, synthetic query/user |
| [grouped_rollout.py](../../env/grouped_rollout.py) | `evaluate_conditional_continuations`: 현재 chunk 공유, L개 미래, 평균 feature/cost; `conditional_scores` |
| [preference_rollout.py](../../env/preference_rollout.py) | `simulate_continuations`: anchor 제외 8 command, 실제 환경 규칙과 동일한 rejection; `select_candidates` |
| [run_grouped_preference.py](../../env/run_grouped_preference.py) | `draw_latents`, `_closed_loop`, `_path_metrics`, few-shot fit와 oracle/learned/base/task-only 실행·저장 |
| [run_mode_ablation.py](../../env/run_mode_ablation.py) | M/selector oracle 비교, paired seed, 조건별 atomic NPZ·marker, 제한된 source manifest 기반 resume |
| [sfp_policies.py](../../env/sfp_policies.py) | SFPS joint ODE, `predict_batch`, float32·shape·device·numerical 검사 |
| [drake_trajectory.py](../../env/drake_trajectory.py), [train_stage_b.py](../../env/train_stage_b.py) | 벡터화된 base training targets와 checkpoint 학습; preference 실험에서는 재학습하지 않음 |
| [chunk_data.py](../../env/chunk_data.py), [evaluate_stage_b.py](../../env/evaluate_stage_b.py) | normalization·materialized dataset, batched base 평가와 실제 rollout |
| [analysis scripts](../../env/artifacts/grouped_preference/analysis) | coverage, fit 반복, actual replay audit, 비교 집계, GIF 렌더링, 실패 조건의 two-chunk 진단 |
| [tests](../../env/tests) | feature 경계/순서, scoped fit, seed/latent, anchor/fallback, actual task+mode, resume 무결성 회귀 |

`env/preference.py`, `env/run_preference.py`, `env/preference_plots.py`는 앞선 flat-preference baseline과 공유 helper다. 현재 grouped method의 선호 모델은 `grouped_preference.py`가 담당한다.

## 7. 결과와 남은 한계

| Learned20 목표 mode+task 성공 | Centered, 두 우선순위 각각 | Gaussian 방향 우선 | Gaussian 폭 우선 |
|---|---:|---:|---:|
| Upper-narrow | 16/16 | 16/16 | 16/16 |
| Upper-wide | 16/16 | 13/16 | 12/16 |
| Lower-narrow | 16/16 | 16/16 | 16/16 |
| Lower-wide | 16/16 | 15/16 | 15/16 |

Oracle도 이 주 실험의 각 조건에서 같은 목표 mode 성공 횟수를 보였다. 16개 seed의 탐색적 pilot이며 profile/방법 사이에 재사용된 episode를 독립 표본으로 합치지 않는다. 16/16의 Wilson 95% 하한은 약80.6%다. 같은 pilot context로 탐색 폭을 선택했으므로 held-out 성능이 아니다.

Feature만 교정하거나 std1에서 M을 128로 늘려도 centered wide는 0/16이었다. Std8에서 centered 네 mode가 모두 선택됐다. Gaussian wide의 남은 oracle 실패 9건에 한정하여 첫 chunk의 8-candidate beam과 다음 M128 후보를 함께 탐색한 진단은 3건을 해결했다. 이후 실행 future는 미리 정한 sample0이며 성공한 미래를 사후 선택하지 않았다. 이 결과는 별도 oracle 진단으로, 위 learned20 결과에 합치지 않았다. [two-chunk 구현](../../env/artifacts/grouped_preference/analysis/two_chunk_probe.py).

전체 코드 테스트와 실제 replay 검증 기록, sample coverage, GIF, latency 및 모든 조건의 표는 [결과 보고서](2026-09-12-mode-aligned-steering-results.md)에 있다. Gaussian wide와 일반적인 guidance-aware lookahead, 실제 인간 preference 및 로봇 실시간성은 아직 해결·검증되지 않았다.

## 8. 새 checkout에서 재현

Repo의 `pyproject.toml`/`uv.lock` 환경을 사용한다. 아래 경로의 시연 bank, 동결 checkpoint와 stats는 이번 publication에 포함된다. 기존 결과 폴더는 읽기용으로 두고 **새 출력 폴더**를 쓴다.

```bash
uv sync --locked
MPLCONFIGDIR=/tmp/preference-mpl .venv/bin/python -m env.run_grouped_preference \
  --checkpoint-path env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt \
  --stats-path env/artifacts/stage_b/b2-seed0-optimized/pusht_stats.npz \
  --demonstrations-path env/artifacts/demonstrations.npz \
  --output-dir env/artifacts/grouped_preference/reproduce-mode-best-M128-std8 \
  --device cpu --feature-kind mode --selection-method best \
  --proposal-std 8 --candidates 128 --continuations 4 --rollout-count 16 --seed 0
```

GPU를 사용할 수 있으면 `--device cuda:0` 등으로 명시한다. 기존 GPU 수치의 bitwise 동일 재현을 다른 device/version에 보장하지 않는다. CPU sandbox에서 전체 테스트 389개 통과·2개 skip과 별도의 실제 GPU 실험을 확인했다. `uv sync --locked`로 신규 설치까지 이 publication에서 재실행한 것은 아니다.

```bash
.venv/bin/python -m env.run_mode_ablation \
  --checkpoint-path env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt \
  --stats-path env/artifacts/stage_b/b2-seed0-optimized/pusht_stats.npz \
  --demonstrations-path env/artifacts/demonstrations.npz \
  --output-dir env/artifacts/grouped_preference/reproduce-oracle-std1 \
  --device cpu --feature-kind mode --proposal-std 1 \
  --candidates 8 32 128 --methods soft best --continuations 4 --rollout-count 16
```

실행이 끝난 전체 후보 NPZ에 대해서는 다음 audit를 실행할 수 있다.

```bash
PYTHONPATH=. .venv/bin/python env/artifacts/grouped_preference/analysis/audit_mode_inference.py \
  env/artifacts/grouped_preference/reproduce-mode-best-M128-std8
```

Git에 있는 축약 actual rollout은 그림·mode count와 실제 실행을 확인하기 위한 것이다. 모든 가상 미래 배열을 대체하지 않으며, 위 full audit에는 재생성한 전체 NPZ 또는 manifest의 원래 로컬 NPZ가 필요하다. 기존 JSON에는 당시 절대 경로가 provenance로 남아 있다. 이동한 checkout에서 재실행할 때는 위 CLI 경로를 사용한다.
