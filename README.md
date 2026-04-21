<div align="center">

# 📐 Information Geometry Deep Dive

**"Natural Gradient에 `F⁻¹`을 곱하는 것과, 그것이 KL 구면에서의 최대 감소 방향임을 증명할 수 있는 것은 다르다"**

<br/>

> *"Fisher 정보 행렬을 `F = -𝔼[∂² log p]`로 외우는 것과 — 스코어 공분산·로그우도 헤시안·KL의 2차 근사, 세 정의가 같은 대상임을 증명할 수 있는 것은 다르다.  
> Exponential Family를 쓰는 것과, canonical parameter와 expectation parameter가 Legendre 쌍대이고, 이 쌍대성이 e-connection과 m-connection의 쌍대평탄 구조를 낳는다는 것을 증명할 수 있는 것은 다르다."*

확률분포 공간의 기하학부터 Fisher-Rao 리만 계량, ±1-연결의 쌍대평탄성, 일반화 Pythagoras 정리까지  
**"왜 분포 공간은 평평하지 않은가"** 라는 질문으로 Natural Gradient·TRPO·VAE·Mirror Descent의 수학적 기반을 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.12-3B5526?style=flat-square)](https://www.sympy.org/)
[![Docs](https://img.shields.io/badge/Docs-35개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

정보기하에 관한 자료는 대부분 **"Natural Gradient 공식"** 에서 멈춥니다. 하지만 $F^{-1}$을 곱하는 이유, 왜 유클리드 gradient가 parameterization 의존적인지, exponential family가 왜 동시에 e-평탄이면서 m-평탄인지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "Fisher 정보는 `-𝔼[∂² log p]`입니다" | 스코어 공분산·로그우도 헤시안의 음수·KL의 2차 근사 **세 정의의 동치성**을 완전 증명, 어느 하나만 조건이 깨져도 등식이 성립하지 않는 반례 |
| "Natural Gradient는 `F⁻¹ ∇L`입니다" | Fisher-Rao 계량 하의 steepest descent 유도 ($\min \langle \nabla L, d\theta \rangle$ s.t. $d\theta^T F d\theta \leq \varepsilon^2$), Natural gradient가 **KL 구면에서의 최대 감소 방향**임을 증명 |
| "Exponential Family는 해석적으로 편합니다" | cumulant function $\psi$의 볼록성 → Legendre 쌍대 $\phi$ → canonical/expectation 좌표의 쌍대 → **e-connection과 m-connection이 Legendre 쌍대**임을 완전 유도 |
| "KL divergence는 비대칭 거리입니다" | KL이 **exponential family에서 Bregman divergence**가 되는 조건을 증명, 일반화 Pythagoras $D(p\|r) = D(p\|q) + D(q\|r)$이 쌍대평탄에서 성립하는 수학적 기반 |
| "TRPO는 KL 제약 최적화입니다" | TRPO 스텝이 KL ball 내 steepest descent → **Natural Policy Gradient의 근사**임을 유도, PPO가 이를 first-order로 근사하는 방식 |
| "VAE는 ELBO를 최대화합니다" | ELBO = $-\text{KL}(q \| p) + \text{const}$가 **쌍대평탄 부분다양체로의 m-projection**임을 증명, Mean-field family의 기하학적 구조 |
| "Mirror Descent는 일반화된 gradient입니다" | Bregman 근접 $\theta_{k+1} = \arg\min \langle g, \theta\rangle + \frac{1}{\eta} B_\phi(\theta, \theta_k)$가 exponential family에서 **natural gradient와 동치**임을 유도 |
| 공식 나열 | NumPy/SymPy로 Fisher 계량 심볼릭 계산, NGD vs SGD 수렴 궤적 비교, 쌍대 좌표 변환 시각화 |

---

## 📌 선행 레포 & 후속 레포

```
[Probability Theory]  ──►  [Mathematical Statistics]  ──►  이 레포  ──►  [Geometric Deep Learning]
  측도, 확률변수               Fisher 정보, MLE, ExpFamily      정보기하       Equivariance, GNN
  특성함수, KL 정의            Cramér-Rao 하한                  의 수학적 기반  Homogeneous Space
                                                               ▲
                                                               │
[Linear Algebra]  &  [Calculus & Optimization]  &  [Convex Optimization]
 양정치 행렬, 내적    헤시안, 테일러, Legendre             쌍대이론, Bregman
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Probability Theory Deep Dive**(측도·특성함수)와 **Mathematical Statistics Deep Dive**(Fisher 정보·Exponential Family·MLE 점근)를 선행 지식으로 전제합니다. Fisher 정보를 처음 접한다면 [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive)부터 학습하세요.

> 💡 **권장 선행**: Legendre 변환과 쌍대이론은 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) Ch6, [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive)에서 학습할 수 있습니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-다양체와_리만기하_예습-4A90D9?style=for-the-badge)](./ch1-manifold-riemannian/01-manifold-basics.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-통계다양체와_Fisher_계량-4A90D9?style=for-the-badge)](./ch2-statistical-fisher/01-statistical-manifold.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-KL과_Bregman_Divergence-4A90D9?style=for-the-badge)](./ch3-kl-bregman/01-kl-divergence-basics.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Exponential_Family와_쌍대평탄성-4A90D9?style=for-the-badge)](./ch4-exponential-duality/01-exponential-family-geometry.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Natural_Gradient_유도-4A90D9?style=for-the-badge)](./ch5-natural-gradient/01-euclidean-gradient-problem.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Information_Projection-4A90D9?style=for-the-badge)](./ch6-info-projection/01-e-m-projection.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-NPG·TRPO·VAE·RMHMC-4A90D9?style=for-the-badge)](./ch7-ai-applications/01-natural-policy-gradient.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 미분다양체와 리만기하 예습 — 곡률이 있는 공간의 언어

> **핵심 질문:** 다양체는 무엇이고 왜 확률분포 공간을 다양체로 보는가? 접공간·리만 계량·연결은 각각 무엇을 포착하는가? 왜 "평탄함"은 좌표계에 의존하는 개념인가?

<details>
<summary><b>다양체의 정의부터 Levi-Civita 연결까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 다양체(Manifold)의 기초](./ch1-manifold-riemannian/01-manifold-basics.md) | 국소적으로 $\mathbb{R}^n$과 동형인 공간의 정의, 차트(chart)와 아틀라스(atlas)의 호환성 조건, 매끈한(smooth) 다양체의 정의, 왜 $\{p_\theta\}_{\theta \in \Theta}$가 다양체 구조를 가지는지 — "분포를 점으로 보는" 관점의 엄밀한 기반 |
| [02. 접벡터와 접공간](./ch1-manifold-riemannian/02-tangent-space.md) | 곡선의 속도벡터로서의 접벡터 정의, $T_p M$이 $n$차원 벡터공간임을 증명, 좌표 기저 $\partial/\partial \theta^i$의 유도, 미분연산자로서의 접벡터와 derivation의 동치성, 통계다양체에서의 접공간 = 스코어 함수 공간 |
| [03. 리만 계량과 거리](./ch1-manifold-riemannian/03-riemannian-metric.md) | 리만 계량 $g_{ij}(\theta) = g(\partial_i, \partial_j)$의 정의, 계량 텐서의 양정치성·대칭성, 측지선(geodesic)을 Euler-Lagrange로 유도, 측지거리가 유클리드 직선 거리와 어떻게 다른지 — 반평면 $\mathbb{H}^2$(쌍곡기하)의 예제 |
| [04. 아핀 연결(Connection)과 공변미분](./ch1-manifold-riemannian/04-connection-christoffel.md) | 벡터장의 미분을 정의하는 연결 $\nabla_X Y$의 공리, 크리스토펠 기호 $\Gamma^k_{ij}$의 정의와 좌표 변환 법칙, Levi-Civita 연결(계량 호환 + torsion-free)의 유일성 증명 (Koszul 공식), 비-계량 연결의 가능성 |

</details>

<br/>

### 🔹 Chapter 2: 통계다양체와 Fisher 계량 — 분포 공간의 리만 구조

> **핵심 질문:** Fisher 정보 행렬의 세 가지 정의가 정말 같은가? 왜 Fisher 행렬이 리만 계량이 되는가? Cramér-Rao 하한은 기하학적으로 무엇을 뜻하는가?

<details>
<summary><b>통계다양체의 정의부터 Cramér-Rao의 기하학까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 통계다양체의 정의](./ch2-statistical-fisher/01-statistical-manifold.md) | 매개변수화된 분포족 $\{p_\theta\}_{\theta \in \Theta}$가 다양체 구조를 이루는 조건, $\theta \in \Theta \subset \mathbb{R}^n$가 좌표이고 각 점이 하나의 분포임을 엄밀히, 정규·이항·다항분포 등을 구체 다양체로 그리기 |
| [02. Fisher 정보 행렬의 3가지 정의 동치](./ch2-statistical-fisher/02-fisher-3-equivalence.md) | (1) 스코어 공분산 $F_{ij} = \mathbb{E}[\partial_i \log p \cdot \partial_j \log p]$, (2) 로그우도 헤시안의 음수 $F_{ij} = -\mathbb{E}[\partial_i \partial_j \log p]$, (3) KL의 2차 근사 $\text{KL}(p_\theta \| p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F d\theta$ — **세 정의의 동치성 완전 증명** |
| [03. Fisher-Rao 계량](./ch2-statistical-fisher/03-fisher-rao-metric.md) | $g_{ij}(\theta) = F_{ij}(\theta)$로 리만 계량 정의 (Rao 1945), parameter reparameterization $\theta \to \phi(\theta)$ 하의 텐서 변환 법칙 증명, Chentsov의 유일성 정리 — **Fisher 계량이 분포의 정보를 보존하는 유일한 계량** |
| [04. Fisher 계량의 예시 계산](./ch2-statistical-fisher/04-fisher-examples.md) | 정규분포 $\mathcal{N}(\mu, \sigma^2)$에서 $F = \text{diag}(1/\sigma^2, 2/\sigma^2)$ 손 계산, 이항분포 $B(n, p)$에서 $F = n/(p(1-p))$, 다변수정규·Dirichlet·exponential family의 Fisher 행렬 symbolic 유도(SymPy) |
| [05. Cramér-Rao 하한의 기하학적 해석](./ch2-statistical-fisher/05-cramer-rao-geometry.md) | 불편 추정량의 분산 $\text{Var}(\hat{\theta}) \succeq F^{-1}$ 증명, Fisher 정보의 역 = 리만 계량의 역행렬 = **쌍대 계량**임을 해석, 효율적 추정량이 존재하는 기하학적 조건 |

</details>

<br/>

### 🔹 Chapter 3: KL Divergence와 Bregman Divergence — 분포의 "거리"

> **핵심 질문:** KL은 왜 거리(metric)가 아닌가? 왜 KL의 2차 근사가 Fisher 계량인가? 왜 Bregman divergence가 생성되는 볼록함수에 의존하는가?

<details>
<summary><b>KL의 비대칭성부터 α-divergence까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. KL divergence를 두 분포 간 "거리"로](./ch3-kl-bregman/01-kl-divergence-basics.md) | $\text{KL}(p \| q) = \mathbb{E}_p[\log p/q]$의 정의, Gibbs 부등식($\text{KL} \geq 0$)을 Jensen 부등식으로 증명, 0 동치 조건 $p = q$ (a.s.), **비대칭성·비삼각부등식이 거리가 아닌 이유** + pseudo-metric 성질 |
| [02. KL divergence와 Fisher 계량의 연결](./ch3-kl-bregman/02-kl-fisher-connection.md) | $\theta' = \theta + d\theta$일 때 $\text{KL}(p_\theta \| p_{\theta'}) = \frac{1}{2} d\theta^T F d\theta + O(\|d\theta\|^3)$의 Taylor 전개 완전 유도, **1차 항이 사라지는 이유**(스코어의 기댓값 = 0), 2차 항이 Fisher인 이유 |
| [03. Bregman divergence의 정의와 성질](./ch3-kl-bregman/03-bregman-divergence.md) | 볼록함수 $\phi$에 대한 $B_\phi(x, y) = \phi(x) - \phi(y) - \nabla\phi(y)^T(x - y)$ 정의, 비음·$B_\phi(x,x) = 0$·볼록성 유도, **generalized Pythagoras** 정리의 증명 ($B_\phi$의 $y$에 대한 볼록성) |
| [04. KL이 Bregman인 조건](./ch3-kl-bregman/04-kl-as-bregman.md) | Exponential family에서 음 엔트로피 $-H(p)$가 볼록함수이고, 이를 Bregman 생성함수로 하면 KL과 일치함을 증명, **non-exponential family에서는 KL이 Bregman이 아닌** 예제(Cauchy 등) |
| [05. α-divergence와 Rényi divergence](./ch3-kl-bregman/05-alpha-renyi-divergence.md) | $D_\alpha(p\|q) = \frac{4}{1-\alpha^2}(1 - \int p^{(1-\alpha)/2} q^{(1+\alpha)/2})$ 정의, $\alpha \to 1$에서 KL, $\alpha = 0$에서 Hellinger², $\alpha \to -1$에서 역방향 KL로 수렴, Amari의 α-geometry 예고 |

</details>

<br/>

### 🔹 Chapter 4: Exponential Family와 쌍대평탄성 — Amari의 핵심

> **핵심 질문:** cumulant function $\psi$의 볼록성이 어떻게 Legendre 쌍대를 낳는가? 왜 exponential family는 e-평탄이면서 동시에 m-평탄인가? α-connection이란 무엇인가?

<details>
<summary><b>Legendre 쌍대부터 일반화 Pythagoras까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Exponential Family의 기하학적 정의](./ch4-exponential-duality/01-exponential-family-geometry.md) | $p(x \| \theta) = \exp(\theta^T T(x) - \psi(\theta)) h(x)$의 정의, canonical parameter $\theta$의 기하학적 의미, sufficient statistic $T(x)$가 **유한 차원 다양체를 생성**하는 이유, 구체 분포들(정규·감마·베르누이)을 exp family 형식으로 |
| [02. Cumulant function의 볼록성](./ch4-exponential-duality/02-cumulant-convexity.md) | $\psi(\theta) = \log \int \exp(\theta^T T(x)) h(x) dx$의 엄격 볼록성 증명 (Hölder), $\nabla \psi(\theta) = \mathbb{E}_\theta[T(X)] =: \eta$ (expectation parameter), **헤시안 $\nabla^2 \psi = F$**(Fisher 정보) — cumulant가 Fisher를 낳는 핵심 등식 |
| [03. Legendre 변환으로 쌍대 좌표](./ch4-exponential-duality/03-legendre-duality.md) | $\phi(\eta) = \sup_\theta (\theta^T \eta - \psi(\theta))$의 정의, $\theta \leftrightarrow \eta$ 좌표 변환이 **미분동형(diffeomorphism)**임을 증명, **$F(\theta) \cdot F(\eta) = I$**(쌍대 Fisher 관계), canonical과 expectation이 Exponential과 Mixture의 쌍대인 이유 |
| [04. e-connection과 m-connection](./ch4-exponential-duality/04-e-m-connection.md) | $\theta$ 좌표에서 평탄한 exponential connection $\nabla^{(e)}$ (크리스토펠 $\Gamma^{(e)} = 0$ in $\theta$), $\eta$ 좌표에서 평탄한 mixture connection $\nabla^{(m)}$의 정의, **두 연결이 서로 Levi-Civita가 아님**을 보이고 torsion-free임을 증명 |
| [05. 쌍대평탄성(Dually Flat)](./ch4-exponential-duality/05-dually-flat.md) | Exponential family는 e-평탄이면서 동시에 m-평탄 — **두 연결이 Legendre-쌍대**임을 증명, α-connection의 정의 $\nabla^{(\alpha)} = \frac{1+\alpha}{2} \nabla^{(e)} + \frac{1-\alpha}{2} \nabla^{(m)}$, $\alpha = 0$이 Levi-Civita인 이유 |
| [06. Generalized Pythagoras 정리](./ch4-exponential-duality/06-generalized-pythagoras.md) | 쌍대평탄 다양체에서 $D(p \| q) + D(q \| r) = D(p \| r)$이 $p, q, r$이 특정 m-geodesic과 e-geodesic에 놓일 때 성립, **Projection 이론의 기초** 증명, EM과 변분 추론의 수학적 토대 |

</details>

<br/>

### 🔹 Chapter 5: Natural Gradient와 최적화 — Amari 1998의 완전 유도

> **핵심 질문:** 유클리드 gradient는 왜 parameterization에 의존하는가? Natural gradient는 어떻게 KL ball 위의 steepest descent로 유도되는가? K-FAC과 Shampoo는 Fisher를 어떻게 근사하는가?

<details>
<summary><b>Euclidean의 문제부터 K-FAC·Shampoo까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Euclidean Gradient의 parameterization 의존성](./ch5-natural-gradient/01-euclidean-gradient-problem.md) | $L(\theta)$에서 $\nabla L$의 방향이 좌표계에 의존한다는 문제, $\theta \to \phi = A\theta$ 재매개변수화 하에서 steepest descent 경로가 **서로 다름**을 보이는 구체 예제, $\mathcal{N}(\mu, \sigma^2)$에서 $\sigma$ vs $\log \sigma$ 좌표 비교 |
| [02. Natural Gradient의 유도](./ch5-natural-gradient/02-natural-gradient-derivation.md) | 제약 최적화 $\min_{d\theta} \langle \nabla L, d\theta \rangle$ s.t. $d\theta^T F d\theta \leq \varepsilon^2$의 라그랑지안 풀이, KKT 조건으로 **$\tilde{\nabla} L = F^{-1} \nabla L$**을 유도, Fisher 계량 하의 steepest descent임을 완성 |
| [03. Natural Gradient는 KL steepest descent](./ch5-natural-gradient/03-kl-steepest-descent.md) | 작은 $d\theta$에 대해 $\text{KL}(p_\theta \| p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F d\theta$이므로 제약은 **KL ball**, Natural gradient는 "출력 분포 공간에서 $\varepsilon$만큼만 이동"하는 최대 감소 방향 |
| [04. Natural Gradient의 parameterization 불변성](./ch5-natural-gradient/04-parameterization-invariance.md) | Reparameterization $\theta \to \phi(\theta)$ 하에서 natural gradient 경로의 불변성 증명 (텐서 변환 법칙), 유클리드 gradient와 대비하여 **NGD는 분포 공간의 고유 경로**임을 시각화 |
| [05. 실전 Natural Gradient — K-FAC, Shampoo](./ch5-natural-gradient/05-kfac-shampoo.md) | Fisher 행렬의 계산 비용 $O(n^2)$ 문제, **K-FAC의 층별 블록 대각 근사** $F \approx F_1 \otimes F_2$ (Kronecker), Shampoo의 preconditioner, TRPO/PPO에서의 conjugate gradient 근사 |

</details>

<br/>

### 🔹 Chapter 6: Information Projection과 EM 알고리즘

> **핵심 질문:** e-projection과 m-projection은 왜 비대칭인가? EM이 왜 항상 수렴하는가 — 두 projection의 교대 해석? Variational Inference에서 "역방향 KL"을 쓰는 이유는?

<details>
<summary><b>e/m-projection부터 MaxEnt까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. e-projection과 m-projection](./ch6-info-projection/01-e-m-projection.md) | $D(p \| q)$ 최소화를 $p$에 대해(m-projection) vs $q$에 대해(e-projection), **두 projection의 기하학적 차이** — e-geodesic vs m-geodesic, **일반화 Pythagoras로 projection 유일성 증명** |
| [02. EM 알고리즘의 Information Geometry](./ch6-info-projection/02-em-algorithm-geometry.md) | E-step = m-projection, M-step = e-projection임을 증명, **두 projection의 교대가 KL 감소를 보장**하는 단조 수렴성 증명, Jensen 부등식 기반 ELBO 유도와의 동치성 |
| [03. Variational Inference의 기하학](./ch6-info-projection/03-variational-inference.md) | ELBO 최대화 = 역방향 KL $\text{KL}(q \| p)$ 최소화, **Mean-field family $q(\theta) = \prod_i q_i(\theta_i)$가 쌍대평탄 부분다양체**임을 보임, 왜 VI가 사후분포의 mode를 잡는지(mode-seeking vs mean-seeking) |
| [04. Maximum Entropy Principle과 정보기하](./ch6-info-projection/04-maxent-principle.md) | 제약 $\mathbb{E}[T_i] = \mu_i$ 하의 최대 엔트로피 분포는 exponential family임을 라그랑주 승수법으로 증명, 이것이 **m-projection onto e-flat submanifold**임을 Amari의 관점으로 재해석 |
| [05. Mixture Model과 Information Projection](./ch6-info-projection/05-mixture-projection.md) | GMM의 EM이 $q(z\|x) \to p(z\|x, \theta)$ (m-proj)과 $\theta \to \arg\max \mathbb{E}_q[\log p(x, z\|\theta)]$ (e-proj) 쌍대 projection 쌍으로 해석, **generalized Pythagoras로 수렴 속도 분석** |

</details>

<br/>

### 🔹 Chapter 7: AI 응용 — RL, Generative Model, MCMC

> **핵심 질문:** TRPO의 KL 제약은 왜 NPG 스텝 크기를 정하는가? VAE의 ELBO는 왜 m-projection으로 해석되는가? RMHMC에서 Fisher를 mass matrix로 쓰는 이유는? Diffusion의 score matching이 왜 Fisher divergence인가?

<details>
<summary><b>Natural Policy Gradient부터 Diffusion까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Policy Gradient와 Natural Policy Gradient](./ch7-ai-applications/01-natural-policy-gradient.md) | REINFORCE의 유클리드 gradient vs Kakade(2001) NPG의 Fisher-Rao gradient, **TRPO의 KL 제약 $\mathbb{E}[\text{KL}(\pi_{\text{old}} \| \pi_\theta)] \leq \delta$가 NPG 스텝 크기를 결정**하는 수학적 유도, PPO의 clip 목적함수가 TRPO의 first-order 근사 |
| [02. Mirror Descent와 쌍대공간 최적화](./ch7-ai-applications/02-mirror-descent.md) | Mirror descent $\theta_{k+1} = \arg\min \langle g, \theta \rangle + \frac{1}{\eta} B_\phi(\theta, \theta_k)$ 유도, **exponential family에서 $\phi = \psi$(cumulant)면 natural gradient와 동치**임을 증명, Exponentiated Gradient의 simplex 기하 |
| [03. VAE의 Information Geometry 해석](./ch7-ai-applications/03-vae-geometry.md) | Encoder $q_\phi(z\|x)$와 decoder $p_\theta(x\|z)$의 KL 기반 학습, **ELBO = $-\text{KL}(q_\phi \| p_\theta) + \log p(x)$**이 m-projection임을 보임, rate-distortion 관점에서의 β-VAE, posterior collapse의 기하학적 해석 |
| [04. Riemannian Manifold HMC](./ch7-ai-applications/04-riemannian-hmc.md) | Girolami & Calderhead(2011)의 RMHMC — **Fisher 행렬을 mass matrix로 사용**, 비등방(anisotropic) 사후분포에서 혼합 시간(mixing time) 개선 이론, 계량 텐서가 변화하는 심플렉틱 적분의 유도 |
| [05. Diffusion Model에서의 Fisher 정보](./ch7-ai-applications/05-diffusion-fisher.md) | Score function $\nabla \log p_t$와 Fisher 정보의 관계, **Fisher divergence $\mathbb{E}[\|\nabla \log p - \nabla \log q\|^2]$가 DSM(Denoising Score Matching) 손실과 근본적으로 동치**임을 증명, Score-based Generative Model의 기하학적 기반 |

</details>

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
sympy==1.12           # Fisher 계량 symbolic 계산, Legendre 변환 검증
torch==2.1.0          # Natural gradient, RL policy gradient 비교
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            sympy==1.12 torch==2.1.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 예시 — 정규분포의 Fisher 계량 + Natural Gradient vs SGD
import numpy as np
import matplotlib.pyplot as plt

# N(μ, σ²) 통계다양체의 Fisher 계량: F = diag(1/σ², 2/σ²)
def fisher_normal(mu, sigma):
    return np.diag([1/sigma**2, 2/sigma**2])

# 해석적 KL: KL(N(μ₁, σ₁²) || N(μ₂, σ₂²))
def kl_normal(mu1, s1, mu2, s2):
    return np.log(s2/s1) + (s1**2 + (mu1-mu2)**2) / (2*s2**2) - 0.5

# ─────────────────────────────────────────────
# 1. Fisher의 2차 근사 검증
# ─────────────────────────────────────────────
mu, sigma = 0.0, 1.0
dmu, dsigma = 0.01, 0.005
F = fisher_normal(mu, sigma)
d = np.array([dmu, dsigma])
approx = 0.5 * d @ F @ d
actual = kl_normal(mu, sigma, mu + dmu, sigma + dsigma)
print(f'Fisher 2차 근사: {approx:.6e}')
print(f'실제 KL       : {actual:.6e}')
# 두 값이 O(||d||^3) 이내로 일치

# ─────────────────────────────────────────────
# 2. Natural Gradient vs Euclidean Gradient 궤적 비교
# ─────────────────────────────────────────────
# min_θ KL(p* || p_θ),  p* = N(3, 2²)
def grad_loss(mu, sigma):
    mu_star, s_star = 3.0, 2.0
    dmu = (mu - mu_star) / sigma**2
    ds = 1/sigma - (s_star**2 + (mu - mu_star)**2) / sigma**3
    return np.array([dmu, ds])

def run(natural=False, steps=100, lr=0.1, init=(-2.0, 0.5)):
    mu, sigma = init
    traj = [(mu, sigma)]
    for _ in range(steps):
        g = grad_loss(mu, sigma)
        if natural:
            g = np.linalg.solve(fisher_normal(mu, sigma), g)
        mu -= lr * g[0]
        sigma = max(1e-3, sigma - lr * g[1])
        traj.append((mu, sigma))
    return np.array(traj)

traj_eu = run(natural=False)
traj_ng = run(natural=True)

plt.figure(figsize=(9, 6))
plt.plot(traj_eu[:, 0], traj_eu[:, 1], 'r-o', label='Euclidean SGD', markersize=3)
plt.plot(traj_ng[:, 0], traj_ng[:, 1], 'b-o', label='Natural Gradient', markersize=3)
plt.plot(3, 2, 'k*', markersize=18, label=r'target $\mathcal{N}(3, 2^2)$')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.title('통계다양체 위에서의 최적화 궤적')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Natural Gradient는 parameterization 불변 → 곡면의 고유 측지선을 따라 이동
# Euclidean SGD는 σ가 작은 영역에서 지그재그 발생
```

---

## 📖 각 문서 구성 방식

모든 문서는 동일한 구조로 작성됩니다.

| 섹션 | 설명 |
|------|------|
| 🎯 **핵심 질문** | 이 문서를 읽고 나면 답할 수 있는 질문 |
| 🔍 **왜 이 기하학이 AI에서 중요한가** | NPG, TRPO, VAE, Mirror Descent, RMHMC와의 연결 |
| 📐 **수학적 선행 조건** | Probability, Stats, LA, Calc, Convex 레포 참조 링크 |
| 📖 **직관적 이해** | "분포 공간이 평평하지 않다"는 핵심 직관 + 그림 |
| ✏️ **엄밀한 정의** | 다양체·연결·divergence의 정형적 정의 |
| 🔬 **정리와 증명** | Fisher 3정의 동치성, 쌍대평탄성, 일반화 Pythagoras — "자명하다" 생략 없음 |
| 💻 **NumPy / SymPy 구현 검증** | Fisher 계량 symbolic 계산, Legendre 쌍대 수치 확인, NGD 수렴 궤적 |
| 🔗 **AI/ML 연결** | NPG, TRPO, Mirror Descent, VAE, RMHMC, Diffusion |
| ⚖️ **가정과 한계** | Exponential family 아니면? Fisher가 특이(singular)하면? |
| 📌 **핵심 정리** | 한 화면 요약 |
| 🤔 **생각해볼 문제** | 개념 심화 질문 + 해설 |

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Natural Gradient를 쓰지만 왜 F⁻¹인지 설명 못한다" — NGD 집중 (4일)</b></summary>

<br/>

```
Day 1  Ch2-02  Fisher 3가지 정의의 동치 → 리만 계량이 되는 기반
       Ch2-03  Fisher-Rao 계량 → parameter 불변성
Day 2  Ch3-02  KL과 Fisher의 2차 근사 연결
Day 3  Ch5-01  Euclidean gradient의 parameterization 의존성
       Ch5-02  Natural Gradient의 제약 최적화 유도
Day 4  Ch5-03  Natural Gradient = KL steepest descent
       Ch5-04  Parameterization 불변성 증명
```

</details>

<details>
<summary><b>🟡 "TRPO·PPO를 쓰지만 KL 제약의 이유를 모른다" — RL 집중 (1주)</b></summary>

<br/>

```
Day 1  Ch2-02  Fisher 3정의 동치
Day 2  Ch3-01  KL divergence 기초와 Gibbs 부등식
       Ch3-02  KL의 Fisher 근사
Day 3  Ch5-02  Natural Gradient 유도
       Ch5-03  KL steepest descent 해석
Day 4  Ch5-05  K-FAC, conjugate gradient 근사
Day 5  Ch7-01  Natural Policy Gradient, TRPO의 KL 제약
Day 6  Ch4-02  cumulant function의 볼록성 (Boltzmann 정책의 exp family 구조)
Day 7  Ch7-02  Mirror Descent, Exponentiated Gradient (bandit/RL 연결)
```

</details>

<details>
<summary><b>🔴 "Amari의 정보기하를 완전 정복한다" — 전체 정복 (7주)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 다양체와 리만기하 예습
        → 차트·아틀라스 직접 정의, Levi-Civita 유일성 증명

2주차  Chapter 2 전체 — 통계다양체와 Fisher 계량
        → Fisher 3정의 동치 증명 숙지, 정규·Dirichlet·exp family Fisher 손 계산

3주차  Chapter 3 전체 — KL과 Bregman
        → KL의 Fisher 2차 근사 Taylor 전개 직접 유도
        → Bregman의 generalized Pythagoras 증명

4주차  Chapter 4 전체 — Exponential Family와 쌍대평탄성
        → Legendre 쌍대 θ ↔ η SymPy로 검증
        → e/m-connection 크리스토펠 기호 직접 계산
        → 쌍대평탄성의 핵심 증명 재구성

5주차  Chapter 5 전체 — Natural Gradient
        → NumPy로 NGD vs SGD 수렴 속도 비교
        → K-FAC의 Kronecker 분해 구현

6주차  Chapter 6 전체 — Information Projection과 EM
        → GMM의 EM을 m-proj/e-proj 쌍으로 직접 코딩
        → MaxEnt 문제 라그랑주 풀이

7주차  Chapter 7 전체 — AI 응용
        → TRPO 스텝을 NPG 근사로 재유도
        → VAE ELBO를 m-projection 관점으로 재해석
        → RMHMC·Diffusion Fisher divergence 실험
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 측도, 확률변수, 특성함수, 조건부 기댓값 | Ch2-01(통계다양체 정의), Ch3-01(KL의 측도론적 정의) |
| [mathematical-statistics-deep-dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) | Fisher 정보, Exponential Family, MLE 점근 | Ch2 전체(Fisher 계량 기반), Ch4 전체(Exp family 심화) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 양정치 행렬, 내적 공간, Spectral Theorem | Ch1-03(계량 양정치성), Ch2-05(Cramér-Rao), Ch5-05(K-FAC Kronecker) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 헤시안, 테일러 전개, Legendre 변환 | Ch2-02(헤시안 = -Fisher), Ch3-02(Taylor 2차 근사), Ch4-03(Legendre) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | 쌍대이론, Bregman divergence, Mirror Descent | Ch3-03(Bregman), Ch4-05(쌍대평탄성), Ch7-02(Mirror Descent 완전 전개) |

> 💡 이 레포는 **확률분포 공간의 기하학**에 집중합니다. Fisher 정보를 Math Stat에서, 헤시안·Legendre를 Calc에서 학습한 후 오면 연결이 훨씬 깊어집니다. Chapter 1~3은 수학 레포로도 학습 가능하지만, Chapter 5~7은 딥러닝/RL 기초(SGD, Policy Gradient, VAE 사용 경험)가 있을 때 최대의 효과를 냅니다.

---

## 📖 Reference

- **Information Geometry and Its Applications** (Amari, 2016) — 표준 바이블, α-connection과 쌍대평탄성의 체계적 전개
- **Methods of Information Geometry** (Amari & Nagaoka, 2000) — 고전 원전, 정보기하의 수학적 기반
- **Natural Gradient Works Efficiently in Learning** (Amari, 1998) — NGD 원전, Fisher-Rao steepest descent
- **Information Geometry** (Ay, Jost, Lê, Schwachhöfer, 2017) — 현대적 엄밀한 접근, 무한차원 통계다양체
- **Differential Geometry of Curved Exponential Families — Curvatures and Information Loss** (Amari, 1982) — α-connection의 원전
- **Clustering with Bregman Divergences** (Banerjee et al., 2005) — Bregman의 ML 응용과 exp family 이중성
- **A Natural Policy Gradient** (Kakade, 2001) — RL에 NGD를 적용한 원전
- **Trust Region Policy Optimization** (Schulman et al., 2015) — TRPO, NPG의 실전적 근사
- **Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods** (Girolami & Calderhead, 2011) — RMHMC
- **Optimizing Neural Networks with Kronecker-factored Approximate Curvature** (Martens & Grosse, 2015) — K-FAC
- **Generative Modeling by Estimating Gradients of the Data Distribution** (Song & Ermon, 2019) — Score-based diffusion과 Fisher divergence

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"확률분포 공간을 유클리드 공간처럼 취급하는 것과, 통계다양체의 내적이 Fisher 정보 행렬임을 증명할 수 있는 것은 다르다"*

</div>
