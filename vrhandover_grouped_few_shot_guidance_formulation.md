# VRHandover — Grouped Few-Shot Preference Learning and Inference-Time Flow Guidance

> 작성일: 2026-09-11<br>
> 문서 성격: 현재 대화의 grouped utility·weighting formulation을 정리한 연구 설계 문서<br>
> 핵심 흐름: **base flow 학습 → parameter 고정 → grouped few-shot preference 추정 → inference-time steering**<br>
> 범위: conventional chunk-flow와 stochastic Streaming Flow Policy(SFP)에 공통으로 적용할 preference model을 정의하고, SFP의 latent reweighting을 중심으로 실행 방법을 연결한다.<br>
> 상태: 제안 formulation이다. 구현 완료, few-shot 성능, 안전성 또는 실시간성을 검증했다는 뜻은 아니다.

> **2026-09-11 수식·현 구현 검토:** [검증 보고서](docs/research/2026-09-11-grouped-preference-formulation-review.md).
> 핵심 식은 명시한 가정 아래 일관되지만, Section 8의 유한 continuation 수 L에 따른 지수 가중 편향,
> 실제 feature의 식별 가능성, 현재 checkpoint의 anchor/horizon convention을 추가로 구분해야 한다.
> 별도 `Vrhandover-streaming-policy` 원본 규약과의 대조는 해당 경로·ref 확보 후 확인이 필요하다.

> **승인된 toy 적용 범위:** `direction=(upper, lower)`, `width=(wide, narrow)` 두 그룹만 사용한다.
> 그룹 내부 weight와 그룹 간 alpha를 scope가 구분된 비교 응답으로 추정한다. P/M/C는 toy에 추가하지 않는다.
> Anchor/horizon은 아래 Section 2.3처럼 현재 SFPS checkpoint와 맞춘다.
> [구현·실험 기록](docs/research/2026-09-11-grouped-preference-results.md).
> 두 그룹 toy의 구현·테스트와 576개 실제 Gym episode 평가를 완료했다. 일반 VRHandover 적용이나 모든 mode 제어를 검증한 결과는 아니다.

## 0. 목적과 표기 원칙

새로운 사용자마다 flow network를 재학습하지 않고, 소수의 trajectory 비교로 추정한 preference를 inference에 반영한다.

$$
\boxed{
\mathcal D_{\mathrm{demo}}
\longrightarrow\bar\theta\ \text{(freeze)},
\qquad
\mathcal D_u\longrightarrow\hat\omega_u,
\qquad
(\bar\theta,\hat\omega_u)\longrightarrow\pi_u^{\mathrm{guide}}.
}
$$

**학습하는 것과 바꾸지 않는 것을 구분한다.** Demonstration으로 base flow를 먼저 학습한다. 개인화할 때는 사용자 preference parameter만 추정하며, 학습된 flow parameter는 고정한다. 따라서 “학습이 전혀 없다”가 아니라 **“사용자별 flow-network 재학습이 없다”**고 표현한다.

이 문서는 직전 대화의 group utility, group 간 weighting, 두 종류의 comparison likelihood를 유지한다. 기존 SFP 적용 검토 문서의 full-episode evaluator와 latent reweighting을 연결한다. [D1, D2]

표기만 다음과 같이 통일한다.

- $r$: preference group index.
- $k$: 물리적인 control-step index.
- $s\in[0,1]$: conventional chunk-flow의 artificial generation time.
- $\tau\in[0,1]$: SFP의 normalized local trajectory time. Group index $r$와 중복하지 않는다.
- $\mathcal Q$: preference action-value. Physical command window와 기호를 구분한다.
- $\hat\omega_u$: few-shot 추정 이후의 parameter. 실제 guidance에는 추정값을 사용한다.

현재 정의되지 않은 learned population preference prior나 Bayesian user posterior는 도입하지 않는다. 아래의 $p_Z$는 generator noise 분포이고, $q_{\hat\omega_u,k}^{\star}$는 inference에 사용할 latent 분포다. 둘 다 사용자 preference parameter의 population prior 또는 posterior가 아니다.

---

## 1. Task와 평가 대상

### 1.1 Three-stage handover

실제 VRHandover episode는 다음과 같이 구성된다.

$$
\text{Discrete grasp selection and acquisition}
\;\longrightarrow\;
\text{Predefined staging}
\;\longrightarrow\;
\text{Flow-based handover}.
$$

물체 $o$와 grasp 선택 전 scene state $s_{\mathrm{pre}}$에 대한 feasible candidate set에서 grasp를 선택한다.

$$
g\in\mathcal G_{\mathrm{feas}}(o,s_{\mathrm{pre}}).
$$

선택된 grasp를 획득한 뒤 predefined staging configuration $\bar{\mathbf q}^{\mathrm S}$로 이동한다. **Staging 목표 configuration은 고정이지만, 그곳까지의 경로와 object-in-gripper transform은 grasp에 따라 달라질 수 있다.** Flow policy의 실행 범위는 staging 완료 이후의 handover다.

### 1.2 Complete option

사용자가 비교하는 complete option을

$$
\boxed{
\chi=(o,c_0^g,g,\xi_{0:T}^{\mathrm H})
}
\tag{1}
$$

로 정의한다.

$u$는 사용자 index이며, 현재 손 위치나 velocity가 아니다. $c_0^g$는 선택된 grasp 아래 staging이 완료되었을 때의 초기 handover context다. $\xi_{0:T}^{\mathrm H}$는 그 시점부터 종료까지 실제로 실행된 closed-loop rollout이며, $T$는 성공·중단·시간 제한 등 사전에 정한 종료 규칙에 따른 마지막 control index다.

Grasp가 없는 Fork-and-Dock toy에서는 $o,g$를 생략하고 $\chi=(c_0,\xi_{0:T})$로 쓴다. A/B가 서로 다른 context에서 생성되었다면 각 option에 해당 context를 보존한다.

**Preference의 평가 대상은 complete rollout이며, policy가 매번 생성하는 것은 local command window다.** 두 대상을 연결하기 위해 Section 6의 action-value를 사용한다.

---

## 2. Preference-free base flow와 windowed execution

### 2.1 Base policy 학습 후 고정

선택한 base generator의 demonstration loss를 $\mathcal L_{\mathrm{base}}$라고 두면

$$
\theta^\star
=\arg\min_\theta
\mathcal L_{\mathrm{base}}(\theta;\mathcal D_{\mathrm{demo}})
$$

로 학습한다. 이후 학습된 parameter 또는 그 EMA를 $\bar\theta$로 고정한다. 이 학습 loss와 사용자 preference loss는 별개다.

Conventional chunk-flow와 SFP는 서로 다른 conditional flow를 학습하므로, 기존 checkpoint나 training target을 그대로 같은 것으로 취급하지 않는다. Base training construction의 상세 내용은 기존 적용 검토 문서에 두고, 여기서는 frozen generator의 interface를 정의한다. [D2, Sections 3–5]

### 2.2 Policy condition과 generator

$c_k^g$는 현재 decision에서 관측 가능한 windowed condition이다. Handover에서는 앞서 정의한

$$
c_k^g=
\left(
\mathcal H_k,\mathbf h_k,
\mathbf q_k^{\mathrm{meas}},
\dot{\mathbf q}_k^{\mathrm{meas}},
\zeta_g
\right)
$$

를 사용한다. $\mathcal H_k$는 hand history, $\mathbf h_k$는 현재 손 관측, $\zeta_g$는 선택된 grasp에 따른 object-in-gripper pose의 표현이다. 미래 ground truth는 condition에 넣지 않는다. Toy에서는 로봇·receiver 관측과 관측 가능한 map으로 대응시킨다.

고정된 generator를 다음과 같이 추상화한다.

$$
A_k(z)=\mathsf G_{\bar\theta}(c_k^g,z)
\in\mathbb R^{H_p\times d_a},
\qquad
z\sim p_Z=\mathcal N(0,I).
$$

$\mathsf G_{\bar\theta}$는 flow 적분과 physical-command decoding을 포함한다. $z$의 Gaussian 성분은 서로 독립이며, 차원은 generator에 따라 다르다.

Conventional chunk-flow에서는 $z$를 초기 Gaussian action tensor로 reshape한다.

$$
X_0=\operatorname{reshape}(z),
\qquad
\frac{dX_s}{ds}
=v_{\bar\theta}^{\mathrm{chunk}}(X_s,s,c_k^g),
\qquad
A_k=\operatorname{Dec}_{c_k^g}(X_1).
$$

Stochastic SFP에서는 초기 action anchor는 고정하고 latent를 sampling한다.

$$
Y_k(0)=
\begin{bmatrix}a_k^{\mathrm{anchor}}\\z\end{bmatrix},
\qquad
\frac{dY_k(\tau)}{d\tau}
=v_{\bar\theta}^{\mathrm{SFP}}(Y_k(\tau),\tau,c_k^g).
$$

적분 경로의 action 성분에서 command를 얻는다. SFP의 action point와 conventional flow의 noisy command tensor는 같은 변수가 아니다. [D2, Sections 3, 5]

### 2.3 실행 prefix

실제로 실행할 command prefix를

$$
\boxed{
B_k(z)=\operatorname{Prefix}_{H_e}
\left[\mathsf G_{\bar\theta}(c_k^g,z)\right]
}
\tag{2}
$$

로 둔다.

현재 toy checkpoint의 convention은 **observation 2, anchor-inclusive prediction horizon 16, execution 8**이다. 이번 inference에서는 `num_actions=9`로 anchor와 future command 8개를 생성하고, **anchor를 제외한 index 1..8만 실행한다.** Local integration time은 `0..8/15`다. 이전의 “future prediction 16, anchor 별도” 서술을 현재 checkpoint에 적용하지 않는다.

원본 SFPS처럼 `a_k^{anchor}`는 최신 normalized observation의 위치 성분으로 고정하고, 후보 사이에서는 latent만 바꾼다. Toy의 실제 최신 관측은 마지막 accepted position command와 같다. 다만 observation/action 정규화 통계가 조금 달라, 역정규화한 모델 anchor와 실제 관측 위치를 기록할 때는 구분한다. 다음 replan은 실제 실행 후의 최신 관측에서 시작한다. [원본 SFPS](https://github.com/siddancha/streaming-flow-policy/blob/main/streaming_flow_policy/pusht/sfps.py), [원본 rollout](https://github.com/siddancha/streaming-flow-policy/blob/main/streaming_flow_policy/pusht/dp_state_notebook/rollout.py).

정상적으로 8개를 모두 실행하면 다음 decision index는 $k^+=k+8$이다. 다음 관측으로 다시 sampling하며, 조기 종료 또는 gate 개입은 실행 기록에 반영한다.

---

## 3. Grouped utility와 두 수준의 weighting

### 3.1 Preference group과 공통 feature

실제 handover와 첫 toy의 group은 각각 다음과 같다.

$$
\mathcal R_{\mathrm{handover}}=\{\mathrm G,\mathrm P,\mathrm M\},
\qquad
\mathcal R_{\mathrm{toy}}=\{\mathrm P,\mathrm M,\mathrm C\}.
$$

| Group | 의미 | 공통 feature 예시 |
|---|---|---|
| Grasp $\mathrm G$ | 선택된 grasp의 특성 | Receiver-accessible surface, grasp overlap, affordance |
| Presentation $\mathrm P$ | 최종 전달 상태 | Relative docking position, orientation, receiver comfort |
| Motion $\mathrm M$ | 전체 전달 동작 | Duration, smoothness, speed profile, replan continuity |
| Path comfort $\mathrm C$ | Toy의 경로 특성 | Path length, clearance |

$\mathrm C$를 실제 grasp group과 같은 의미라고 해석하지 않는다. 위 항목은 앞서 논의한 후보 feature이며, 최종 계산법과 차원은 실험 설정에 명시해야 한다. [D1, D2]

각 group의 normalized feature vector를

$$
\widetilde{\mathbf f}_r(\chi)\in\mathbb R^{d_r}
$$

로 둔다. Handover에서 그 인자를 풀어 쓰면

$$
\widetilde{\mathbf f}_{\mathrm G}(\chi)
=\widetilde{\mathbf f}_{\mathrm G}(g,o),
$$

$$
\widetilde{\mathbf f}_{\mathrm P}(\chi)
=\widetilde{\mathbf f}_{\mathrm P}
(g,\xi_{0:T}^{\mathrm H},c_0^g),
\qquad
\widetilde{\mathbf f}_{\mathrm M}(\chi)
=\widetilde{\mathbf f}_{\mathrm M}
(\xi_{0:T}^{\mathrm H},c_0^g).
$$

Feature의 정의와 normalization은 사용자에게 공유한다. 사용자별로 아직 모르는 preferred pose나 speed를 feature 함수 안에 미리 넣지 않는다. 서로 반대인 취향이 필요하면, 위쪽·아래쪽 docking 또는 slow·fast에 대한 공통 basis를 만들어 weight로 그 차이를 표현한다.

### 3.2 Group 내부 utility

사용자 $u$의 group 내부 weight를 $\mathbf w_{u,r}$라 하면

$$
\boxed{
U_{u,r}(\chi)
=\mathbf w_{u,r}^{\top}\widetilde{\mathbf f}_r(\chi)
}
\tag{3}
$$

로 group utility를 정의한다.

$w_{u,r,j}$는 **group $r$ 안에서 feature $j$의 상대적인 중요도**다. 예를 들어 Presentation group에서는 position, orientation, wrist-comfort feature가 각각 어느 정도 반영되는지 결정한다.

### 3.3 Group 사이의 weighting

Group 간 trade-off weight를

$$
\boldsymbol\alpha_u=(\alpha_{u,r})_{r\in\mathcal R}
$$

로 두면 전체 utility는

$$
\boxed{
U_{\omega_u}(\chi)
=\sum_{r\in\mathcal R}\alpha_{u,r}U_{u,r}(\chi)
=\sum_{r\in\mathcal R}
\alpha_{u,r}\mathbf w_{u,r}^{\top}
\widetilde{\mathbf f}_r(\chi)
}
\tag{4}
$$

다. 사용자 parameter 전체는

$$
\omega_u=
\left(\boldsymbol\alpha_u,\{\mathbf w_{u,r}\}_{r\in\mathcal R}\right).
$$

두 weighting 단계는 다음처럼 구분된다.

$$
\widetilde{\mathbf f}_r(\chi)
\xrightarrow{\ \mathbf w_{u,r}\ }
U_{u,r}(\chi),
\qquad
\{U_{u,r}(\chi)\}_{r\in\mathcal R}
\xrightarrow{\ \boldsymbol\alpha_u\ }
U_{\omega_u}(\chi).
$$

### 3.4 Parameter constraint와 해석상 주의

현재 grouped formulation은 다음 simplex constraint를 사용한다.

$$
\boldsymbol\alpha_u\in\Delta^{|\mathcal R|-1},
\qquad
\mathbf w_{u,r}\in\Delta^{d_r-1}.
$$

여기서

$$
\Delta^{m-1}
=\left\{x\in\mathbb R^m:x_i\ge0,\ \sum_{i=1}^{m}x_i=1\right\}.
$$

따라서 허용되는 parameter 공간은

$$
\boxed{
\Omega=
\Delta^{|\mathcal R|-1}
\times\prod_{r\in\mathcal R}\Delta^{d_r-1}.
}
\tag{5}
$$

Feature와 group score의 scale을 비교 가능하게 정의해야 $\alpha_{u,r}$를 group importance로 해석할 수 있다. 이 문서의 simplex weight는 공통 desirability/basis feature를 전제로 하며, 모든 raw scalar를 그대로 사용해도 임의의 선호를 표현할 수 있다는 뜻은 아니다.

**기존 리뷰에서 구분한 제한:** 두 simplex 제약은 $\alpha$와 $w$의 단순 rescaling 자유도를 제거하지만, 유한한 비교 데이터로 모든 parameter가 식별된다는 보장은 아니다. $\alpha_{u,r}=0$이면 해당 $\mathbf w_{u,r}$는 전체 utility에서 식별되지 않는다. 같은 parameterization의 자유도는 $\sum_r d_r-1$이므로, 실제 query budget에 맞게 feature 차원을 제한해야 한다.

Within-group weight를 고정하고 $\boldsymbol\alpha_u$만 추정하는 방법은 별도 ablation이다. 이를 사용한다면 group 내부 취향까지 개인화한다고 주장하지 않는다.

---

## 4. Group-specific 및 overall comparison

사용자가 어떤 기준으로 응답했는지에 따라 likelihood를 구분한다. $+1$은 A 선호, $-1$은 B 선호로 통일한다.

### 4.1 Group-specific feedback

“Motion만 보면 어느 쪽이 좋은가?”와 같은 group-specific feedback을

$$
\mathcal D_{u,r}
=\{(\chi_{n,r}^A,\chi_{n,r}^B,y_{n,r})\}_{n=1}^{N_{u,r}}
$$

로 저장한다. 해당 group에 대한 Bradley–Terry likelihood는

$$
\boxed{
p_r(y_{n,r}\mid\chi_{n,r}^A,\chi_{n,r}^B,\mathbf w_{u,r})
=\sigma\!\left(
\rho_r y_{n,r}
[U_{u,r}(\chi_{n,r}^A)-U_{u,r}(\chi_{n,r}^B)]
\right)
}
\tag{6}
$$

다. $\sigma(x)=(1+e^{-x})^{-1}$이며 $\rho_r>0$는 choice sharpness다.

이 응답은 해당 group 내부 weight $\mathbf w_{u,r}$에 대한 정보다. **Group 자체의 전체 중요도 $\alpha_{u,r}$는 이 likelihood에 들어가지 않는다.**

### 4.2 Overall feedback

“전체적으로 어느 쪽이 좋은가?”에 대한 응답을

$$
\mathcal D_u^{\mathrm O}
=\{(\chi_n^A,\chi_n^B,y_n)\}_{n=1}^{N_u^{\mathrm O}}
$$

로 둔다. 전체 likelihood는

$$
\boxed{
p_{\mathrm O}(y_n\mid\chi_n^A,\chi_n^B,\omega_u)
=\sigma\!\left(
\rho_{\mathrm{BT}}y_n
\sum_{r\in\mathcal R}\alpha_{u,r}
[U_{u,r}(\chi_n^A)-U_{u,r}(\chi_n^B)]
\right)
}
\tag{7}
$$

다. Group 간 trade-off를 포함한 overall comparison은 $\boldsymbol\alpha_u$를 추정하는 데 사용한다. Joint fitting을 하면 overall 응답은 $\mathbf w_{u,r}$에도 영향을 준다.

따라서 “특정 group의 feature만 다르게 만든 pair”와 “그 group만 평가하라고 지시한 질문”은 구분한다. 질문은 overall인데 pair만 controlled했다면 식 (7)을 사용한다.

---

## 5. Regularized few-shot estimation

### 5.1 두 종류의 응답을 결합한 loss

전체 응답 집합을

$$
\mathcal D_u=
\left(\mathcal D_u^{\mathrm O},\{\mathcal D_{u,r}\}_{r\in\mathcal R}\right)
$$

로 둔다. 후보 parameter $\omega=(\boldsymbol\alpha,\{\mathbf w_r\})$를 사용하여

$$
\mathcal L_{\mathrm O}(\omega)
=-\sum_{n=1}^{N_u^{\mathrm O}}
\log p_{\mathrm O}(y_n\mid\chi_n^A,\chi_n^B,\omega),
$$

$$
\mathcal L_r(\mathbf w_r)
=-\sum_{n=1}^{N_{u,r}}
\log p_r(y_{n,r}\mid\chi_{n,r}^A,\chi_{n,r}^B,\mathbf w_r)
$$

로 정의한다. 최종 추정식은

$$
\boxed{
\hat\omega_u
=\arg\min_{\omega\in\Omega}
\left[
\mathcal L_{\mathrm O}(\omega)
+\sum_{r\in\mathcal R}\eta_r\mathcal L_r(\mathbf w_r)
+\lambda_{\mathrm{reg}}R(\omega)
\right].
}
\tag{8}
$$

$\eta_r\ge0$는 응답 종류별 loss 기여도를 정하는 **설계 계수**다. 사용자 group importance $\alpha_{u,r}$와 다른 변수다. Overall 응답만 받는 실험에서는 group-specific 데이터 항을 생략한다.

### 5.2 Regularizer

앞서 논의한 고정 reference의 한 예는

$$
\boldsymbol\alpha^{\mathrm{ref}}
=\frac1{|\mathcal R|}\mathbf1,
\qquad
\mathbf w_r^{\mathrm{ref}}=\frac1{d_r}\mathbf1
$$

이다. 이에 대한 regularizer를

$$
\boxed{
R(\omega)
=\|\boldsymbol\alpha-\boldsymbol\alpha^{\mathrm{ref}}\|_2^2
+\sum_{r\in\mathcal R}
\|\mathbf w_r-\mathbf w_r^{\mathrm{ref}}\|_2^2
}
\tag{9}
$$

로 둘 수 있다. 이것은 사용자 집단에서 학습한 prior가 아니라 정해 둔 regularization 기준이다. Constraint나 regularization만으로 few-shot 정확도 또는 최적해의 유일성이 보장된다고 주장하지 않는다.

### 5.3 계수의 역할

| 기호 | 역할 | 사용자 선호 parameter인가? |
|---|---|---|
| $\mathbf w_{u,r}$ | Group 내부 feature weight | 예 |
| $\alpha_{u,r}$ | Group 사이의 trade-off | 예 |
| $\eta_r$ | Group-specific 응답 loss의 상대적 계수 | 아니오 |
| $\lambda_{\mathrm{reg}}$ | Regularization 강도 | 아니오 |
| $\rho_r,\rho_{\mathrm{BT}}$ | 비교 모델의 choice sharpness | 여기서는 고정·공유 설정 |
| $\beta$ | Inference-time preference guidance 강도 | 별도 inference 설정 |

응답 budget은 사용자가 실제로 제공한 판단 수로 기록한다. 같은 영상 pair에 group 질문 세 개와 overall 질문 하나를 했다면, 네 개의 판단을 받은 것이다.

---

## 6. Group별 long-term action-value와 전체 weighting

### 6.1 Evaluator context와 continuation distribution

Evaluator가 사용하는 context를

$$
\bar c_k=(o,g,c_0^g,c_k^g,\rho_k)
$$

로 둔다. 여기서 $\rho_k$는 이미 실행된 prefix 또는 선택한 full-episode feature를 계산하는 데 충분한 누적 statistic이다. 충분한 statistic을 정의하지 않았다면 전체 prefix를 보존한다. 이 정보는 evaluator의 입력이며, base policy에 동일한 길이의 history를 추가한다는 뜻은 아니다.

다음 분포를 정의한다.

$$
\mathsf P_0^{\mathrm{cont}}(\cdot\mid\bar c_k,B_k(z)).
$$

이 분포에서는 이미 실행된 prefix를 고정하고, 현재 candidate $B_k(z)$를 controller와 environment에서 실행한 다음, 이후에는 preference-free base policy의 재계획 규칙으로 종료까지 진행한다. Reference continuation의 gate·fallback·종료 규칙도 일관되게 정해야 한다.

**평가 대상은 항상 complete option $\chi$다.** Full-episode utility를 suffix에 다시 적용하지 않으며, command를 measured trajectory에 그대로 이어 붙이지 않는다. 실제 state 변화는 tracking dynamics와 environment transition을 거쳐 얻는다. [D2, Section 6.3]

### 6.2 Group별 action-value

추정된 within-group weight를 사용하여

$$
U_{\hat{\mathbf w}_{u,r},r}(\chi)
:=\hat{\mathbf w}_{u,r}^{\top}
\widetilde{\mathbf f}_r(\chi)
$$

라고 쓰겠다. 각 group의 action-value는

$$
\boxed{
\mathcal Q_{u,r}^{0}(\bar c_k,z)
:=
\mathbb E_{\chi\sim
\mathsf P_0^{\mathrm{cont}}(\cdot\mid\bar c_k,B_k(z))}
\left[
\hat{\mathbf w}_{u,r}^{\top}\widetilde{\mathbf f}_r(\chi)
\right].
}
\tag{10}
$$

즉, 현재 latent가 만드는 prefix를 실행했을 때, **group $r$ 관점에서 기대되는 full-episode utility**다. 위 첨자 $0$은 이후 base policy를 따른다는 의미다.

### 6.3 Group별 expected feature

공통 future-feature quantity를

$$
\boxed{
\boldsymbol\Psi_r^{0}(\bar c_k,z)
=
\mathbb E_{\chi\sim
\mathsf P_0^{\mathrm{cont}}(\cdot\mid\bar c_k,B_k(z))}
\left[\widetilde{\mathbf f}_r(\chi)\right]
}
\tag{11}
$$

로 정의하면

$$
\mathcal Q_{u,r}^{0}(\bar c_k,z)
=\hat{\mathbf w}_{u,r}^{\top}
\boldsymbol\Psi_r^{0}(\bar c_k,z)
$$

다. 같은 context, dynamics와 reference continuation을 고정하면 $\boldsymbol\Psi_r^0$는 preference parameter에 의존하지 않는다. Receiver의 실제 움직임이나 관측 차이까지 모든 사용자에게 같다고 가정하는 표현은 아니다.

### 6.4 Group 간 weighting을 적용한 전체 action-value

Expectation의 선형성으로

$$
\boxed{
\begin{aligned}
\mathcal Q_{\hat\omega_u}^{0}(\bar c_k,z)
&=\sum_{r\in\mathcal R}
\hat\alpha_{u,r}\mathcal Q_{u,r}^{0}(\bar c_k,z)\\
&=\sum_{r\in\mathcal R}
\hat\alpha_{u,r}
\hat{\mathbf w}_{u,r}^{\top}
\boldsymbol\Psi_r^{0}(\bar c_k,z).
\end{aligned}
}
\tag{12}
$$

따라서 두 수준의 weighting이 long-term guidance에서도 유지된다.

$$
\underbrace{\boldsymbol\Psi_r^0}_{\text{미래 group feature}}
\xrightarrow{\ \hat{\mathbf w}_{u,r}\ }
\underbrace{\mathcal Q_{u,r}^{0}}_{\text{group action-value}}
\xrightarrow{\ \hat\alpha_{u,r}\ }
\underbrace{\mathcal Q_{\hat\omega_u}^{0}}_{\text{전체 personalized value}}.
$$

---

## 7. KL-regularized inference-time guidance

### 7.1 현재 decision에서의 latent 분포 개선

현재 context에서 높은 personalized value를 만드는 latent를 더 자주 선택하되, base noise 분포에서 지나치게 벗어나지 않도록 한다.

$$
\boxed{
q_{\hat\omega_u,k}^{\star}
=
\arg\max_{q\ll p_Z}
\left\{
\mathbb E_{z\sim q}
\left[\sum_{r\in\mathcal R}
\hat\alpha_{u,r}\mathcal Q_{u,r}^{0}(\bar c_k,z)\right]
-\frac1\beta D_{\mathrm{KL}}(q\|p_Z)
\right\},
\qquad\beta>0.
}
\tag{13}
$$

$q\ll p_Z$는 base latent 분포에 대해 absolute continuity를 요구한다. $q$ 자체를 새로운 neural network로 학습한다는 뜻은 아니다.

### 7.2 Closed-form target distribution

정규화 상수가 유한하다는 조건에서 해는

$$
\boxed{
q_{\hat\omega_u,k}^{\star}(z)
=
\frac{
p_Z(z)
\exp\!\left[
\beta\sum_{r\in\mathcal R}
\hat\alpha_{u,r}
\hat{\mathbf w}_{u,r}^{\top}
\boldsymbol\Psi_r^{0}(\bar c_k,z)
\right]
}{Z_{\hat\omega_u,k}}
}
\tag{14}
$$

이며,

$$
Z_{\hat\omega_u,k}
=
\mathbb E_{z\sim p_Z}
\left[
\exp\!\left(
\beta\sum_{r\in\mathcal R}
\hat\alpha_{u,r}\hat{\mathbf w}_{u,r}^{\top}
\boldsymbol\Psi_r^{0}(\bar c_k,z)
\right)
\right].
$$

**식 (14)가 grouped utility, group 내부 weight, group 간 weight, long-term evaluation, inference-time guidance를 함께 나타내는 핵심 식이다.**

$\beta\to0^+$이면 gate 적용 전 latent 분포는 $p_Z$로 돌아간다. $\beta$와 comparison model의 $\rho_{\mathrm{BT}}$는 역할이 다르다.

이 식은 현재 decision에서 정의한 KL-regularized objective의 해다. Base continuation을 평가한 한 번의 local improvement이며, guided policy를 반복 실행한 결과가 globally optimal하다는 주장이나 full-trajectory target의 exact marginal이라는 주장은 하지 않는다. [D2, Section 6.4]

### 7.3 유도되는 personalized policy

현재 context에서 $B_k$를 고정하면

$$
\boxed{
\pi_{0,k}^{\mathrm{prefix}}=(B_k)_\#p_Z,
\qquad
\pi_{u,k}^{\mathrm{guide}}=(B_k)_\#q_{\hat\omega_u,k}^{\star}.
}
\tag{15}
$$

$\#$는 latent를 command prefix로 변환하여 얻는 pushforward distribution이다. Network는 그대로지만 latent 선택 확률이 달라져 실행 후보 분포가 달라진다. 이 latent-space formulation에서 별도 논증 없이 일반적인 latent-space KL과 trajectory-space KL을 동일하다고 놓지 않는다.

---

## 8. 첫 구현: Monte Carlo evaluation과 safe weighted resampling

### 8.1 Candidate와 continuation 생성

$$
z_k^{(m)}\overset{\mathrm{i.i.d.}}{\sim}p_Z,
\qquad m=1,\ldots,M.
$$

각 candidate를 평가할 때 같은 이미 실행된 prefix를 사용하고, 해당 candidate의 실행 결과와 base-policy continuation을 포함하는 complete option $\chi^{(m,\ell)}$을 만든다. $L$개 continuation으로

$$
\widehat{\boldsymbol\Psi}_r^{0}(\bar c_k,z_k^{(m)})
=\frac1L\sum_{\ell=1}^{L}
\widetilde{\mathbf f}_r(\chi^{(m,\ell)})
$$

를 계산한다. Personalized score는

$$
s_m
=\sum_{r\in\mathcal R}
\hat\alpha_{u,r}\hat{\mathbf w}_{u,r}^{\top}
\widehat{\boldsymbol\Psi}_r^{0}(\bar c_k,z_k^{(m)}).
$$

Toy simulator를 직접 사용하는 이 방법은 별도 critic network를 요구하지 않는다. 대신 inference에 continuation 계산이 필요하다.

### 8.2 Independent safety gate

안전한 candidate index 집합을

$$
\mathcal I_{\mathrm{safe}}
=\left\{m:
\Gamma_{\mathrm{safe}}(c_k^g,B_k(z_k^{(m)}))=1
\right\}
$$

로 둔다. 이 gate는 preference와 상쇄할 수 있는 scalar reward가 아니라 별도의 실행 조건이다.

### 8.3 Preference-weighted sampling

$\mathcal I_{\mathrm{safe}}\ne\varnothing$일 때

$$
\boxed{
P(m^\star=m)
=\frac{\exp(\beta s_m)}
{\sum_{j\in\mathcal I_{\mathrm{safe}}}\exp(\beta s_j)},
\qquad m\in\mathcal I_{\mathrm{safe}}.
}
\tag{16}
$$

선택된 $B_k(z_k^{(m^\star)})$를 실행하고 새 관측에서 다시 계획한다. 수치적으로는 exponentiation 전에 후보 점수의 최대값을 빼도 같은 선택 확률이다.

후보는 이미 $p_Z$에서 sampling되었으므로 위 weight에 $p_Z(z_k^{(m)})$를 다시 곱하지 않는다. 다른 proposal로 sampling하는 확장은 여기서 다루지 않는다.

이는 finite-candidate approximation이며, score estimator의 오차도 포함한다. $m^\star=\arg\max_m s_m$인 best-of-$M$은 별도의 greedy baseline이다. 안전한 후보가 없으면 사전에 정한 fallback을 사용한다. Gate·projection·fallback 이후 실제 execution distribution을 gate 적용 전의 식 (14)와 동일하다고 두지 않는다. [D2, Section 6.6]

---

## 9. Gradient guidance — 선택적 확장

이 절은 앞서 논의한 선택적 확장이다. **Main formulation의 latent reweighting과, latent-gradient optimization 또는 vector-field guidance를 구분한다.** [D1, D2, Section 7]

### 9.1 Differentiable group-feature evaluator

$\widehat{\boldsymbol\Psi}_r^0$가 $z$에 대해 미분 가능하면

$$
\widehat{\mathcal Q}_{\hat\omega_u}^{0}(\bar c_k,z)
=\sum_r
\hat\alpha_{u,r}\hat{\mathbf w}_{u,r}^{\top}
\widehat{\boldsymbol\Psi}_r^0(\bar c_k,z)
$$

의 gradient를 사용할 수 있다. Gaussian $p_Z$에 대한 목표 log-density gradient의 근사는

$$
\boxed{
\nabla_z\log q_{\hat\omega_u,k}^{\star}(z)
\ \approx\
-z+\beta\sum_r\hat\alpha_{u,r}
\nabla_z\!\left[
\hat{\mathbf w}_{u,r}^{\top}
\widehat{\boldsymbol\Psi}_r^0(\bar c_k,z)
\right].
}
\tag{17}
$$

Exact $\boldsymbol\Psi_r^0$를 쓰면 위 관계도 정확하다. 이 식을 이용한 gradient ascent는 mode-seeking latent optimization이며, 그 자체가 $q^\star$의 exact sampler는 아니다.

Gradient 계산에는 differentiable continuation 또는 별도로 준비한 differentiable evaluator가 필요하다. Black-box Monte Carlo scalar score만으로 gradient가 자동 제공되지는 않는다.

공유 predictor $\boldsymbol\Psi_{\psi,r}$를 offline 학습한다면 새 사용자마다 $\bar\theta$와 $\psi$를 고정하고 preference parameter만 추정할 수 있다. 그러나 이 경우 공통 predictor 학습까지 없었다고 표현하지 않는다.

### 9.2 Chunk-flow와 SFP의 guidance를 혼용하지 않는다

Conventional straight-path chunk-flow의

$$
\widehat X_1=X_s+(1-s)v_{\bar\theta}(X_s,s,c_k^g)
$$

는 clean action tensor의 one-step estimate다. SFP에서는 경로상의 action point에 같은 공식을 적용해 full future trajectory를 얻었다고 해석하지 않는다.

SFP vector field 자체에 utility-gradient residual을 더하는 방법은 별도 제안이다. 식 (13)의 latent KL objective를 정확히 푸는 sampler라고 자동으로 주장할 수 없으며, score의 정의·미분·안전성·지연을 별도로 검증해야 한다.

---

## 10. Grasp group과 outer discrete selection

Grasp가 선택된 뒤 handover를 수행하는 동안에는 $g,o$가 고정된다. 따라서

$$
U_{u,\mathrm G}(\chi)
=\mathbf w_{u,\mathrm G}^{\top}
\widetilde{\mathbf f}_{\mathrm G}(g,o)
$$

는 $z$에 대해 상수다.

$$
\boxed{
\nabla_zU_{u,\mathrm G}(g,o)=0,
\qquad
\mathcal Q_{u,\mathrm G}^{0}(\bar c_k,z)
=\hat{\mathbf w}_{u,\mathrm G}^{\top}
\widetilde{\mathbf f}_{\mathrm G}(g,o).
}
\tag{18}
$$

따라서 식 (14)의 grasp-only 항은 normalizer에서 소거된다. 선택된 grasp가 presentation과 motion의 기하학적 결과에 미치는 영향은 다른 group의 feature와 continuation에 남는다.

**Grasp까지 개인화하려면 별도의 discrete selection이 필요하다.** 앞서 논의한 outer objective를 다음처럼 둔다.

$$
\boxed{
g_u^\star\in
\arg\max_{g\in\mathcal G_{\mathrm{feas}}(o,s_{\mathrm{pre}})}
J_{\hat\omega_u}(g\mid c^{\mathrm G}).
}
\tag{19}
$$

$c^{\mathrm G}$는 grasp 선택 시점의 관측이다. $J_{\hat\omega_u}$는 그 grasp를 획득하고 staging한 뒤 guided handover를 수행했을 때의 expected full-option utility다.

Grasp 선택 시 아직 staging 완료 context가 관측되지 않았다면, 앞서 리뷰에서 논의한 대로 그 불확실성도 포함한다.

$$
J_{\hat\omega_u}(g\mid c^{\mathrm G})
=
\mathbb E_{c_0^g\sim\nu_{\mathrm{stage}}(\cdot\mid c^{\mathrm G},g)}
\mathbb E_{\xi^{\mathrm H}\sim
P^{\pi_u^{\mathrm{guide}}}(\cdot\mid c_0^g,g)}
\left[U_{\hat\omega_u}(o,c_0^g,g,\xi^{\mathrm H})\right].
$$

$\nu_{\mathrm{stage}}$는 **예측하거나 가정한 staging 완료 context의 분포**를 뜻한다. 이 분포를 실제로 어떻게 얻을지는 별도 구현 항목이다. 이미 알려진 $c_0^g$에 조건화하는 실험에서는 첫 expectation을 생략할 수 있다.

이 objective는 grasp-only score만 최대화하는 것이 아니라 이후 Presentation·Motion utility도 포함한다. 다만 $J$ 평가에는 guided continuation이 필요하므로, base-continuation $\mathcal Q^0$와 계산 가정을 구분한다. Personalized grasp selector가 없는 실험은 고정된 grasp 아래의 handover steering만 검증한 것으로 보고한다.

---

## 11. Formulation이 주장하는 것과 주장하지 않는 것

| 항목 | 현재 문서의 범위 |
|---|---|
| Base model 학습 | Demonstration으로 먼저 학습하고 고정한다. |
| Few-shot 학습 | User-specific group/feature weight의 regularized point estimation이다. |
| Population preference prior | 가정하지 않는다. |
| Group weighting | Within-group $\mathbf w_{u,r}$와 between-group $\alpha_{u,r}$를 모두 명시한다. |
| Long-term preference | Executed prefix를 포함한 full-option utility로 평가한다. |
| Main steering | KL-regularized latent target을 finite-candidate reweighting으로 근사한다. |
| Gradient steering | Differentiable evaluator가 필요한 별도 확장이다. |
| 새로운 사용자별 network 학습 | Base flow 재학습은 하지 않는다. 공통 evaluator 학습 여부는 별도로 밝힌다. |
| 전역 최적성·안전 보장 | 이 formulation만으로 주장하지 않는다. |
| Real-time 성능 | Candidate generation뿐 아니라 continuation과 safety 처리 지연까지 측정해야 한다. |

Few-shot 비교는 “무엇을 선호하는가”를 학습한다. $\boldsymbol\Psi_r^0$ 또는 $\mathcal Q^0$는 “현재 행동이 어떤 미래를 만드는가”를 평가한다. 전자가 정의되었다고 후자가 자동으로 구현되는 것은 아니다.

Feature의 구체적 계산식·차원, normalization, response budget, $\eta_r$, $\lambda_{\mathrm{reg}}$, choice sharpness, $\beta$, candidate 수 $M$, continuation 수 $L$, safety gate와 fallback 규칙은 실제 실험 설정에서 고정해 보고해야 한다. 현재 대화에서 확정되지 않은 수치를 이 문서에서 검증된 기본값으로 추가하지 않는다.

---

## 12. 핵심 수식 요약

**Group 내부 utility**

$$
U_{u,r}(\chi)=\mathbf w_{u,r}^{\top}\widetilde{\mathbf f}_r(\chi).
$$

**Group 간 weighting**

$$
U_{\omega_u}(\chi)
=\sum_r\alpha_{u,r}U_{u,r}(\chi).
$$

**Grouped few-shot estimation**

$$
\hat\omega_u
=\arg\min_{\omega\in\Omega}
\left[
\mathcal L_{\mathrm O}(\omega)
+\sum_r\eta_r\mathcal L_r(\mathbf w_r)
+\lambda_{\mathrm{reg}}R(\omega)
\right].
$$

**Group별 미래 feature와 personalized action-value**

$$
\boldsymbol\Psi_r^0(\bar c_k,z)
=\mathbb E_{\mathsf P_0^{\mathrm{cont}}}
[\widetilde{\mathbf f}_r(\chi)],
\qquad
\mathcal Q_{\hat\omega_u}^0
=\sum_r\hat\alpha_{u,r}
\hat{\mathbf w}_{u,r}^{\top}\boldsymbol\Psi_r^0.
$$

위 expectation의 조건은 Section 6의 $\bar c_k,B_k(z)$와 동일하다.

**Inference-time latent guidance**

$$
\boxed{
q_{\hat\omega_u,k}^{\star}(z)
\propto
p_Z(z)\exp\!\left[
\beta\sum_{r\in\mathcal R}
\underbrace{\hat\alpha_{u,r}}_{\text{group 간 weight}}
\underbrace{\hat{\mathbf w}_{u,r}^{\top}}_{\text{group 내부 weight}}
\underbrace{\boldsymbol\Psi_r^0(\bar c_k,z)}_{\text{expected full-episode feature}}
\right].
}
$$

선택된 latent를 같은 frozen generator에 통과시키고, 안전 검사를 통과한 prefix만 실행한다.

$$
\boxed{
\mathcal D_u
\longrightarrow
\big(\hat{\boldsymbol\alpha}_u,\{\hat{\mathbf w}_{u,r}\}_r\big)
\longrightarrow
\mathcal Q_{\hat\omega_u}^0
\longrightarrow
q_{\hat\omega_u,k}^{\star}
\longrightarrow
\text{frozen-generator prefix execution}.
}
$$

---

## 13. 근거 문서와 출처 범위

- **[D1] 현재 대화:** “Grouped utility와 그 사이 weighting”에 대한 직전 formulation 및 그에 앞선 수식 리뷰. Group별 utility, 두 수준의 weight, group-specific/overall BT likelihood, joint estimation loss와 group별 action-value의 근거다.
- **[D2] 제공된 적용 검토 문서:** `vrhandover_streaming_flow_policy_integration_review.md`. Sections 3–5의 base generator 정의, Section 6의 full-episode evaluation·KL latent reweighting, Section 7의 gradient guidance 구분, Section 8의 anchor convention을 참조했다.
- **[D3] 제공된 SFP 요약:** `streaming_flow_policy_summary.md`. Conventional flow와 SFP의 변수·시간 해석 차이를 설명하는 배경 문서다.

원 방법의 참고 링크: [Streaming Flow Policy 논문](https://arxiv.org/abs/2505.21851), [공식 저장소](https://github.com/siddancha/streaming-flow-policy).

원 논문은 base SFP의 근거이며, 이 문서의 grouped few-shot personalization이 원 논문에서 이미 제안·검증된 기능이라는 뜻은 아니다. 이 문서 작성은 대화와 제공된 문서의 정리이며, 새로운 외부 조사나 코드·성능 재검증을 수행한 것은 아니다.
