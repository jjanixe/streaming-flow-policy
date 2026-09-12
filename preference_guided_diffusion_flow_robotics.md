# Preference-Guided Diffusion / Flow Policies for Robotics

> 정리 기준: **robotics trajectory/action policy**, **human preference**, **few-shot 가능성**, **inference-time guidance**, **code availability**<br>
> 확인일: 2026-09-11

> **게재·코드 상태 재확인:** [공식 출판 기록 및 GitHub 소스 감사](docs/research/2026-09-11-preference-publication-audit.md). FPL의 CoRL 2026 게재는 확인되지 않았으며, UF-OPS도 이 조사에서는 preprint로 분류한다. FlowPRO는 저자 페이지에 Under review / Code coming soon으로 표시되어 있다. 공개 코드가 있다는 사실과 동료심사 게재 여부는 별개이다.

> **구현·실험 완료:** 기존 B2 SFPS를 동결하고, 전체 경로 비교 5/10/20/40개로 BT utility를 fit한 뒤 남은 전체 경로 후보를 선택하는 pilot을 실행했다. [측정 결과와 한계](docs/research/2026-09-11-preference-steering-results.md), [구현 사용법](env/README.md). 합성 사용자 seed-0 결과이며, 현재 SFPS의 conditional mode coverage가 주요 제약이다.

## 목표

관심 있는 연구 설정은 다음과 같다.

1. demonstration으로 diffusion / flow-matching robot policy를 먼저 학습한다.
2. 사용자는 소수의 **전체 trajectory pair/ranking**에 preference를 제공한다.
3. preference로부터 user-specific utility/reward를 추정한다.
4. base policy를 가능한 한 고정한 채, action chunk sampling 과정에서 preference guidance를 준다.

예를 들면,

\[
\tau_i \succ_u \tau_j
\]

로부터

\[
U_u(\tau)=w_u^\top f(\tau)
\]

를 추정하고, diffusion에서는

\[
\epsilon^u
=
\epsilon_\theta
-
\gamma_k\nabla_{x_k}U_u(\hat{x}_0,c),
\]

flow matching에서는

\[
v^u(x_t,t,c)
=
v_\theta(x_t,t,c)
+
\gamma_t\nabla_{x_t}U_u(\hat{\tau},c)
\]

형태로 sampling trajectory를 steer하는 구조를 생각할 수 있다.

---

# 1. DynaGuide

**DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance**<br>
Maximilian Du, Shuran Song. NeurIPS 2025.

- Paper: https://arxiv.org/abs/2506.13922
- NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2025/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html
- Project: https://dynaguide.github.io/
- GitHub: https://github.com/MaxDu17/DynaGuide

### 핵심

Pretrained diffusion policy의 weight를 변경하지 않고 **diffusion denoising 중 external guidance gradient**를 넣어 행동을 steer한다.

구조는 대략

```text
current observation
      +
candidate action chunk
      ↓
latent dynamics model
      ↓
predicted future outcome
      ↓
distance to desired / undesired outcomes
      ↓
gradient wrt action
      ↓
diffusion guidance
```

Base policy를 frozen 상태로 둘 수 있기 때문에 현재 연구 방향과 매우 가깝다.

### Preference와의 관계

DynaGuide 자체는 human pairwise preference learning 방법은 아니다.<br>
Guidance condition으로 desired / undesired future observations를 사용한다.

하지만 이 score를

\[
d(z_{\mathrm{future}},z_{\mathrm{goal}})
\]

대신

\[
U_u(\tau)
\]

같은 user-specific trajectory utility로 교체하면 preference guidance로 확장할 수 있다.

### 코드에서 볼 부분

Repo:
https://github.com/MaxDu17/DynaGuide

특히 diffusion denoising 과정에서

```python
scaled_grad = guidance_function(state, naction)
noise_pred = noise_pred - scale * scaled_grad
```

형태로 guidance gradient를 noise prediction에 직접 적용한다.

**현재 연구에서 가장 유용한 부분:**<br>
`pretrained policy + differentiable trajectory score + inference-time gradient guidance`

---

# 2. UF-OPS

**Update-Free On-Policy Steering via Verifiers**<br>
Maria Attarian et al., 2026.

- Paper: https://arxiv.org/abs/2603.10282
- GitHub: https://github.com/uoft-isl/uf-ops

### 핵심

Diffusion Policy를 고정하고 별도의 verifier를 학습한 뒤 inference 시 action sampling을 steer한다.

두 가지 방법을 제공한다.

1. **Best-of-N**
   - 여러 action chunk를 sample
   - verifier score가 가장 높은 것을 선택

2. **Classifier Guidance**
   - predicted \(\hat{x}_0\)에 대해 verifier gradient 계산
   - diffusion scheduler 내부에 gradient guidance 적용

### 구현상 중요한 파일

GitHub:
https://github.com/uoft-isl/uf-ops

주요 구현:

```text
diffusion_policy/uf_ops/networks/models.py
diffusion_policy/uf_ops/eval/tweedie_guided_ddpm.py
diffusion_policy/uf_ops/dataset/guided_classifier_dataset.py
```

Scoring model:

- `ContrastiveClassifier`
- `Time2Success`

### 현재 연구와의 관계

UF-OPS의 verifier를

```text
task-success verifier
```

대신

```text
user preference verifier / utility model
```

로 교체하는 것이 가장 직접적인 구현 경로 중 하나이다.

즉,

```text
Diffusion Policy
       ↓
candidate 16-step action chunk
       ↓
preference score
       ↓
∂ score / ∂ action
       ↓
Tweedie / classifier guidance
```

형태로 사용할 수 있다.

**현재 연구에서 가장 유용한 부분:**<br>
Diffusion Policy에 실제 classifier guidance를 넣은 공개 robotics implementation.

---

# 3. Freeform Preference Learning (FPL)

**Freeform Preference Learning for Robotic Manipulation**<br>
Marcel Torne, Anubha Mahajan, Abhijnya Bhat, Chelsea Finn. arXiv preprint, 2026. 공식 학회 게재 기록은 이번 조사에서 확인되지 않았다.

- Paper: https://arxiv.org/abs/2606.32027
- Project: https://freeform-pl.github.io/fpl.website/
- Main code: https://github.com/freeform-pl/fpl
- Real-world preference collection: https://github.com/freeform-pl/fpl_real

### 핵심

두 full trajectory를 비교할 때 단순히

```text
Which trajectory is better overall?
```

이라고 묻는 대신 사용자가 여러 preference axis를 정의한다.

예:

```text
speed
smoothness
safety
placement quality
carefulness
```

그리고 각 axis별로 A/B preference를 제공한다.

Reward model은

\[
r_\phi(\tau,l)
\]

형태로 **trajectory \(\tau\)** 와 natural-language preference axis \(l\)을 함께 입력받는다.

### 현재 연구와의 관계

현재 고려 중인

```text
G1: grasp / comfort
G2: presentation
G3: motion
```

같은 grouped preference와 매우 유사하다.

특히 full-trajectory-level preference annotation을 사용한다는 점이 중요하다.

### 중요한 차이

FPL은 few-shot personalized inference-time gradient guidance가 아니다.

다만 reward-conditioned policy를 학습한 후에는 추론 시 reward condition을 바꾸어 steering한다. 따라서 inference-time steering 자체가 없는 방법으로 분류하면 부정확하다. 기존의 arbitrary frozen BC policy에 사후 gradient guidance를 붙이는 방식과 구별한다. [논문](https://arxiv.org/html/2606.32027v3)

전체 pipeline은

```text
trajectory comparison
      ↓
multi-axis reward model
      ↓
reward score
      ↓
reward-conditioned flow-matching policy training
```

이다.

따라서 **preference formulation / annotation interface**를 참고하기에는 매우 좋지만,
base policy를 frozen하고 몇 개 preference로 바로 steer하는 방식과는 다르다.

### 코드

Main:
https://github.com/freeform-pl/fpl

Real-world preference collector:
https://github.com/freeform-pl/fpl_real

`fpl_real`은 실제 policy rollout 두 개를 보여주고 사용자가

```text
smoothness
neatness
...
```

같은 free-form comparison axis를 만든 뒤

```text
A / Equal / B
```

로 평가하는 UI를 제공한다.

**현재 연구에서 가장 유용한 부분:**<br>
full trajectory preference acquisition 및 multi-axis preference formulation.

---

# 4. PC-Flow

**PC-Flow: Preference Alignment in Flow Matching via Classifier**<br>
Shaomeng Wang, He Wang, Longquan Dai, Jinhui Tang. AAAI 2026.

- Paper / AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/37971
- PDF: https://ojs.aaai.org/index.php/AAAI/article/download/37971/41933

### 핵심

Flow Matching base model 전체를 fine-tune하지 않고 **lightweight preference classifier**를 별도로 학습한다.

핵심 아이디어:

```text
pretrained flow model
       +
preference classifier
       ↓
preference-guided flow sampling
```

즉 preference modeling과 generative model을 분리한다.

### 현재 연구와의 관계

Flow Matching policy를 쓰는 경우 이론적으로 매우 가깝다.

원하는 구조는 개념적으로

\[
v_{\mathrm{guided}}
=
v_\theta
+
\gamma(t)\nabla_x \log S_u(x)
\]

처럼 생각할 수 있다.

여기서 \(S_u\)를 user-specific trajectory preference classifier로 만들면 된다.

### 제한점

- robotics 논문이 아니다.
- 주 실험은 generative modeling / image preference alignment.
- 2026-09-11 기준 공식 GitHub 구현은 확인하지 못했다.

**현재 연구에서 가장 유용한 부분:**<br>
frozen flow model + lightweight preference classifier라는 구조와 이론.

---

# 5. Preference Aligned Visuomotor Diffusion Policies / RKO

**Preference Aligned Visuomotor Diffusion Policies for Deformable Object Manipulation**<br>
Marco Moletta, Michael C. Welle, Danica Kragic. IEEE RA-L, 2026.

- Paper: https://arxiv.org/abs/2602.09583
- DOI: https://doi.org/10.1109/LRA.2026.3665075

### 핵심

Pretrained visuomotor Diffusion Policy를 사람의 preferred behavior에 맞게 adaptation한다.

RKO는 preference optimization을 통해 preferred / non-preferred demonstrations를 활용한다.

### 특징

- 실제 robotic manipulation
- cloth / deformable object manipulation
- personalized preference alignment
- limited preferred demonstrations의 sample efficiency를 고려

### 현재 연구와의 차이

RKO는

```text
preference examples
      ↓
policy optimization / fine-tuning
```

방식이다.

즉 inference-time guidance보다는 post-training preference alignment에 가깝다.

### 코드

2026-09-11 기준 검색에서 공식 공개 GitHub repository는 확인하지 못했다.

**현재 연구에서 가장 유용한 부분:**<br>
robot manipulation에서 limited preference demonstrations로 Diffusion Policy를 personalize할 수 있다는 empirical reference.

---

# 6. FlowPRO

**FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization**<br>
Yihao Wu, He Zhang, Junbo Tan, Xueqian Wang, Zhengyou Zhang. 2026.

- Paper: https://arxiv.org/abs/2606.05468
- Project: https://wuyeyexvnainai.github.io/flowpro/

### 핵심

Flow-matching VLA를 preference optimization으로 fine-tune한다.

Human intervention을 이용해 자연스럽게

\[
(\tau^{w},\tau^{l})
\]

형태의 preferred / rejected trajectory pair를 만든다.

특히 중요한 부분은 trajectory-level correction을 **action-chunk granularity의 preference supervision**으로 변환한다는 것이다.

Project page의 설명:

```text
trajectory-level intervention
        ↓
preferred / rejected trajectory pair
        ↓
smooth interpolation
        ↓
dense per-state / action-chunk preference tuples
        ↓
flow preference optimization
```

### 현재 연구와의 관계

현재 설계의 핵심 문제:

> preference는 전체 trajectory에 대해 평가하지만 policy는 16-step action chunk를 생성한다.

와 직접적으로 관련 있다.

FlowPRO는 sparse trajectory-level feedback을 action-chunk-level supervision으로 변환하는 방법을 제공한다.

이 변환에는 intervention, rollback, scene restoration과 corrective teleoperation으로 얻은 대응 관계가 필요하다. 임의의 두 full-trajectory A/B preference만으로 모든 chunk의 선호를 식별해 주는 방법은 아니다. [논문](https://arxiv.org/html/2606.05468v1)

### 차이점

FlowPRO는 inference-time guidance가 아니라 **flow policy 자체를 fine-tune**한다.

### 코드

Project page:
https://wuyeyexvnainai.github.io/flowpro/

2026-09-11 기준 project page의 Code 항목은 **Coming soon** 상태이다.

**현재 연구에서 가장 유용한 부분:**<br>
full trajectory feedback ↔ action chunk supervision 연결 방법.

---

# 7. FDPP

**FDPP: Fine-tune Diffusion Policy with Human Preference**<br>
Yuxin Chen, Devesh K. Jha, Masayoshi Tomizuka, Diego Romeres. ICRA 2025.

- Paper: https://arxiv.org/abs/2501.08259
- MERL publication: https://www.merl.com/publications/TR2025-053
- DOI: https://doi.org/10.1109/ICRA55743.2025.11128127

### 핵심

Human preference로 reward를 학습한 뒤 pretrained Diffusion Policy를 RL로 fine-tune한다.

```text
human preference
      ↓
reward learning
      ↓
RL
      ↓
Diffusion Policy fine-tuning
```

KL regularization으로 original policy capability가 너무 많이 변하는 것을 방지한다.

### 현재 연구와의 차이

- explicit human preference를 사용한다는 점에서는 중요
- 하지만 inference-time guidance가 아니다.
- reward model과 policy fine-tuning에 상대적으로 많은 preference data가 필요하다.

### 코드

2026-09-11 기준 공식 공개 GitHub repository는 확인하지 못했다.

**현재 연구에서 가장 유용한 부분:**<br>
Diffusion Policy + explicit human preference의 직접적인 robotics baseline.

---

# 8. Preference Alignment with Flow Matching (PFM)

**Preference Alignment with Flow Matching**<br>
Minu Kim, Yongsik Lee, Sehyeok Kang, Jihwan Oh, Song Chong, Se-Young Yun. NeurIPS 2024.

- NeurIPS paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html
- GitHub: https://github.com/jadehaus/preference-flow-matching

### 핵심

Preference data로 별도의 flow를 학습하여 less-preferred samples를 preferred outcome 쪽으로 변환한다.

기존 모델 전체를 직접 fine-tune하는 것과 다른 alignment mechanism을 제공한다.

Repo에는

- MNIST preference
- IMDB preference
- preference-based RL toy environment

예제가 포함되어 있다.

### 현재 연구와의 관계

robot action policy를 직접 다루지는 않지만,

```text
pretrained model
+
separate preference flow
```

라는 separation 관점에서 참고할 수 있다.

PC-Flow와 함께 flow-based preference alignment의 이론적 related work로 적합하다.

---

# 9. Base Policy References

## Diffusion Policy

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**

- Project: https://diffusion-policy.cs.columbia.edu/
- GitHub: https://github.com/real-stanford/diffusion_policy
- Paper: https://arxiv.org/abs/2303.04137

Diffusion action chunk를 생성하는 기본 robotics policy reference.

---

## Streaming Flow Policy

**Streaming Flow Policy**

- GitHub: https://github.com/siddancha/streaming-flow-policy
- Paper: https://arxiv.org/abs/2505.21851

현재 toy environment에서 고려 중인 action chunk 기반 flow policy reference.

---

# 연구 방향별 분류

## A. Frozen policy + inference-time guidance

가장 직접적으로 참고할 논문:

### DynaGuide

```text
Frozen Diffusion Policy
        +
Dynamics-based differentiable score
        ↓
gradient during denoising
```

https://github.com/MaxDu17/DynaGuide

### UF-OPS

```text
Frozen Diffusion Policy
        +
Learned verifier
        ↓
classifier guidance / Best-of-N
```

https://github.com/uoft-isl/uf-ops

### PC-Flow

```text
Frozen Flow Model
        +
Preference classifier
        ↓
preference-guided flow
```

Paper:
https://ojs.aaai.org/index.php/AAAI/article/view/37971

---

## B. Human preference를 robot policy에 직접 사용하는 연구

### FPL

```text
full trajectory pair
        ↓
multi-axis human preference
        ↓
reward model
        ↓
reward-conditioned flow policy
```

https://github.com/freeform-pl/fpl

### RKO

```text
limited preferred demonstrations
        ↓
preference optimization
        ↓
Diffusion Policy adaptation
```

https://arxiv.org/abs/2602.09583

### FDPP

```text
human preference
        ↓
reward model
        ↓
RL fine-tuning
        ↓
Diffusion Policy
```

https://arxiv.org/abs/2501.08259

### FlowPRO

```text
human intervention
        ↓
preferred/rejected trajectory
        ↓
chunk-level preference tuples
        ↓
flow policy fine-tuning
```

https://arxiv.org/abs/2606.05468

---

# 현재 연구에 가장 적합한 조합

현재 목표에는 한 논문의 방법을 그대로 사용하는 것보다 아래 세 계열을 조합하는 것이 자연스럽다.

## 1. Preference acquisition — FPL

사용자에게 전체 trajectory를 보여주고

```text
Grasp
Presentation
Motion
```

등 각 dimension에 대해 pairwise preference를 받는다.

참고:
https://github.com/freeform-pl/fpl_real

---

## 2. Few-shot user model

큰 reward network를 새로 학습하기보다는 소수 trajectory preference로

\[
p(\tau_i \succ_u \tau_j)
=
\sigma
\left(
w_u^\top[f(\tau_i)-f(\tau_j)]
\right)
\]

와 같은 Bradley-Terry / Bayesian linear preference model을 사용한다.

예:

\[
f(\tau)=
[
f_G(\tau),
f_P(\tau),
f_M(\tau)
].
\]

이 경우 사용자마다 학습해야 할 것은 작은 \(w_u\)뿐이므로 few-shot adaptation에 유리하다.

---

## 3. Policy guidance — DynaGuide / UF-OPS / PC-Flow

Base policy는 유지한다.

### Diffusion

\[
\epsilon^{u}
=
\epsilon_\theta
-
\gamma_k
\nabla_{x_k}
U_u(\hat{x}_0,c).
\]

Implementation reference:

- DynaGuide: https://github.com/MaxDu17/DynaGuide
- UF-OPS: https://github.com/uoft-isl/uf-ops

### Flow Matching

\[
v^{u}
=
v_\theta
+
\gamma_t
\nabla_{x_t}
U_u(\hat{\tau},c).
\]

Theory reference:

- PC-Flow: https://ojs.aaai.org/index.php/AAAI/article/view/37971

---

# 현재 연구와의 gap

관련 연구는 대략 두 종류로 나뉜다.

### Preference learning은 직접 하지만 policy를 다시 학습

```text
FDPP
RKO
FPL
FlowPRO
```

### Policy를 frozen하고 inference-time에 steer

```text
DynaGuide
UF-OPS
PC-Flow
```

하지만 다음 조합은 상대적으로 비어 있다.

```text
few-shot personalized full-trajectory preference
        +
frozen robot diffusion / flow policy
        +
action-chunk-level inference-time gradient guidance
```

따라서 연구 방향을 다음처럼 정리할 수 있다.

> **Learn a lightweight user-specific utility from a few full-trajectory comparisons and use its gradient to steer a frozen diffusion/flow action policy at inference time.**

---

# 구현 우선순위

현재 toy example에서는 다음 순서가 현실적이다.

1. **UF-OPS**
   - classifier-guided Diffusion Policy implementation 확인
   - verifier → preference utility로 교체

2. **DynaGuide**
   - arbitrary differentiable guidance를 action diffusion에 넣는 구조 확인

3. **FPL**
   - trajectory comparison UI / multi-axis preference 구조 참고

4. **PC-Flow**
   - flow matching으로 옮길 때 preference classifier guidance formulation 참고

5. **FlowPRO**
   - full trajectory preference와 16-step action chunk 사이의 supervision mismatch를 다루는 reference로 활용

---

# 빠른 링크 모음

| Work | Paper | Code / Project |
|---|---|---|
| DynaGuide | https://arxiv.org/abs/2506.13922 | https://github.com/MaxDu17/DynaGuide |
| UF-OPS | https://arxiv.org/abs/2603.10282 | https://github.com/uoft-isl/uf-ops |
| FPL | https://arxiv.org/abs/2606.32027 | https://github.com/freeform-pl/fpl |
| FPL Real-world | — | https://github.com/freeform-pl/fpl_real |
| PC-Flow | https://ojs.aaai.org/index.php/AAAI/article/view/37971 | 공식 코드 확인 못함 |
| RKO | https://arxiv.org/abs/2602.09583 | 공식 코드 확인 못함 |
| FlowPRO | https://arxiv.org/abs/2606.05468 | https://wuyeyexvnainai.github.io/flowpro/ (Code coming soon) |
| FDPP | https://arxiv.org/abs/2501.08259 | 공식 코드 확인 못함 |
| Preference Alignment with Flow Matching | https://proceedings.neurips.cc/paper_files/paper/2024/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html | https://github.com/jadehaus/preference-flow-matching |
| Diffusion Policy | https://arxiv.org/abs/2303.04137 | https://github.com/real-stanford/diffusion_policy |
| Streaming Flow Policy | https://arxiv.org/abs/2505.21851 | https://github.com/siddancha/streaming-flow-policy |
