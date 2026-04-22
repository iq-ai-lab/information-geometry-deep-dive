# 04. 지수족에서 KL = Bregman — 쌍대 좌표와 잠재함수

> **"지수족의 KL은 cumulant 함수의 Bregman이다. 이것이 정보기하의 핵심 identity."**

---

## 🎯 핵심 질문

**지수족 $p_\theta(x) = h(x)\exp(\theta^\top T - \psi(\theta))$ 에서 KL과 Bregman의 정확한 관계는 무엇이며, 이것이 Ch4 dually flat 구조를 어떻게 예고하는가?**

$$
\boxed{
\begin{aligned}
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) &= D_\psi(\theta_2, \theta_1) \\
&= D_{\psi^*}(\eta_1, \eta_2) \\
&= \psi(\theta_2) + \psi^*(\eta_1) - \langle\theta_2, \eta_1\rangle
\end{aligned}}
$$

- $\psi(\theta)$: **Natural parameter 좌표에서의 cumulant 함수** (e-potential)
- $\psi^*(\eta)$: **Expectation parameter 좌표에서의 entropy 관련 함수** (m-potential)
- $\eta = \nabla\psi(\theta) = \mathbb{E}_\theta[T(X)]$: **expectation parameter**
- $\theta = \nabla\psi^*(\eta)$: **natural parameter** — 역 대응

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 통찰 | 구체적 결과 |
|---|---|
| **KL의 Bregman 구조** | KL의 closed-form 계산 — cumulant $\psi$만 알면 됨 |
| **쌍대 좌표** | $\theta$와 $\eta$는 "같은 분포의 두 표현" — NGD가 $\theta$-Bregman → $\eta$-Bregman flow |
| **Pythagoras theorem** | e-geodesic 과 m-geodesic 의 **직교성** → EM, Info projection 알고리즘 |
| **Fisher = cumulant Hessian** | $F(\theta) = \nabla^2\psi$ — 가장 계산하기 쉬운 Fisher |
| **Potential function** | $\psi, \psi^*$가 정보기하 전체를 encoding |

**Dually flat structure** (Ch4) 는 정보기하의 최고 성과 — 지수족이 KL 관점에서 "평면" 인 두 좌표계를 가짐. 그 기반이 바로 이 identity.

---

## 📐 수학적 선행 조건

- Ch2 [04. Fisher 예제들](../ch2-statistical-fisher/04-fisher-examples.md) — 지수족 $F = \operatorname{Hess}\psi$
- [01. KL 기초](./01-kl-divergence-basics.md)
- [03. Bregman 발산](./03-bregman-divergence.md)
- Legendre 변환, Fenchel 쌍대
- 지수족 정의 및 natural/expectation parameter

---

## 📖 직관적 이해

### 두 좌표계의 쌍대성

지수족 모델은 두 가지 "자연스러운" 좌표계가 있다:

**1. Natural parameter $\theta$** (canonical parameter)
- $p_\theta(x) = h(x)\exp(\theta^\top T(x) - \psi(\theta))$
- "linear combination of sufficient statistics"
- $\psi(\theta) = \log \int h(x)\exp(\theta^\top T)\, d\mu$: **log-partition function** (cumulant)

**2. Expectation parameter $\eta$** (mean parameter)
- $\eta := \mathbb{E}_\theta[T(X)]$
- 직관적 의미 있음 — 예: Gaussian mean, Bernoulli probability
- $\eta = \nabla\psi(\theta)$ (Legendre 대응)

**쌍대 관계**: $\theta$ 공간에서는 "분포의 추상적 표현", $\eta$ 공간에서는 "관측 가능한 통계값". 둘이 Legendre transform으로 정확히 대응.

### KL과 Bregman의 두 얼굴

같은 KL을 두 가지 Bregman으로 쓸 수 있음:

- $\theta$-좌표: $\operatorname{KL}(p_1\|p_2) = D_\psi(\theta_2, \theta_1)$ — cumulant의 Bregman, **argument 순서 뒤집힘**.
- $\eta$-좌표: $\operatorname{KL}(p_1\|p_2) = D_{\psi^*}(\eta_1, \eta_2)$ — dual의 Bregman, **순서 정상**.

**이것이 단순한 변환이 아니라 심오한 대칭성**: forward KL과 reverse KL이 두 좌표계에서 같은 모양.

### 세 번째 표현 — canonical divergence

$$
\operatorname{KL}(p_1 \| p_2) = \psi(\theta_2) + \psi^*(\eta_1) - \langle \theta_2, \eta_1\rangle.
$$

이 식은 **양쪽 좌표를 섞어** 쓴다: $\theta_2$와 $\eta_1$. 이것이 Amari의 **canonical divergence** — 정보기하의 "마스터 공식".

이 form이 가장 유용한 이유:
- $\theta_2$와 $\eta_1$은 independent 변수 → 대칭적.
- Pythagoras 정리와 깔끔한 연결 (Ch6).

### Pythagoras 정리 예고

$\theta_1, \eta_1$이 "$p_1$의 두 좌표", $\theta_2, \eta_2$가 "$p_2$의 두 좌표". 삼각형:

$$
\operatorname{KL}(p_1 \| p_3) = \operatorname{KL}(p_1 \| p_2) + \operatorname{KL}(p_2 \| p_3)
$$

성립하려면 **$\theta_2 - \theta_3$가 $\eta_1 - \eta_2$에 직교** (적절한 inner product 하에서). 이것이 **e/m-duality의 generalized Pythagoras** (Ch6).

---

## ✏️ 엄밀한 정의

### 정의 10.1 (지수족 재정리)

**Canonical form** of exponential family:

$$
p_\theta(x) = h(x) \exp\bigl(\theta^\top T(x) - \psi(\theta)\bigr), \qquad \theta \in \Theta := \{\theta: \psi(\theta) < \infty\}.
$$

- $\theta \in \mathbb{R}^k$: **natural parameter**
- $T: \mathcal{X} \to \mathbb{R}^k$: **sufficient statistic**
- $h: \mathcal{X} \to \mathbb{R}_{\ge 0}$: **base measure / carrier**
- $\psi: \Theta \to \mathbb{R}$: **cumulant / log-partition**

$\Theta$는 볼록 집합, $\psi$는 $\Theta$ 위에서 $C^\infty$-smooth and **strictly convex** (minimal 표현 아래).

### 정의 10.2 (Dual Coordinate $\eta$)

$$
\eta := \nabla \psi(\theta) = \mathbb{E}_\theta[T(X)].
$$

이것을 **expectation parameter** (또는 mean parameter). $\eta$ 좌표 공간을 $\mathrm{H} := \nabla\psi(\Theta)$.

$\psi$의 엄격 볼록성 + smoothness ⟹ $\nabla\psi: \Theta \to \mathrm{H}$은 $C^\infty$-diffeomorphism.

### 정의 10.3 (Dual Potential $\psi^*$)

Legendre 변환:

$$
\psi^*(\eta) := \sup_\theta \bigl[\langle \theta, \eta\rangle - \psi(\theta)\bigr] = \langle \theta(\eta), \eta\rangle - \psi(\theta(\eta)),
$$

여기서 $\theta(\eta) = (\nabla\psi)^{-1}(\eta) = \nabla\psi^*(\eta)$.

**통계적 해석**: $-\psi^*(\eta) = H(p_\theta) - \mathbb{E}_\theta[\log h(X)]$ (엔트로피 minus base-measure term). 특히 $h(x) = $ constant 경우 $-\psi^*$ = 엔트로피.

### 정의 10.4 (Canonical Divergence)

Amari의 canonical divergence:

$$
D(\theta_1 \| \theta_2) := \psi(\theta_2) + \psi^*(\eta_1) - \langle \theta_2, \eta_1\rangle = D_\psi(\theta_2, \theta_1).
$$

**Triple representation:**
$$
D(\theta_1 \| \theta_2) = D_\psi(\theta_2, \theta_1) = D_{\psi^*}(\eta_1, \eta_2) = \psi(\theta_2) + \psi^*(\eta_1) - \langle\theta_2, \eta_1\rangle.
$$

세 형태는 수학적으로 동일.

---

## 🔬 정리와 증명

### 정리 10.1 (지수족의 기본 identity)

지수족 $p_\theta$에서 다음 성립:

(a) $\nabla \psi(\theta) = \mathbb{E}_\theta[T(X)] = \eta(\theta)$.

(b) $\operatorname{Hess} \psi(\theta) = \operatorname{Cov}_\theta[T(X)] = F(\theta)$ (Fisher 정보행렬).

**증명.**

(a) $\psi(\theta) = \log Z(\theta)$, $Z(\theta) = \int h(x)\exp(\theta^\top T(x))\, d\mu$.

$\partial_i \psi = \frac{\partial_i Z}{Z} = \frac{\int T_i h \exp(\theta^\top T)\, d\mu}{Z} = \mathbb{E}_\theta[T_i]$. ✓

(b) $\partial_i \partial_j \psi = \partial_j \mathbb{E}_\theta[T_i] = \frac{\partial}{\partial \theta_j}\int T_i(x) p_\theta(x)\, d\mu$.

$\partial_j p_\theta = p_\theta (T_j - \partial_j \psi) = p_\theta (T_j - \eta_j)$.

$\partial_j \mathbb{E}_\theta[T_i] = \int T_i p_\theta (T_j - \eta_j)\, d\mu = \mathbb{E}[T_i T_j] - \eta_i \eta_j = \operatorname{Cov}(T_i, T_j)$. ✓

**Q.E.D.**

---

### 정리 10.2 (지수족 KL = cumulant Bregman)

$$
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) = D_\psi(\theta_2, \theta_1).
$$

(정리 9.5의 재방문 — 여기서는 구조적 통찰 강조.)

**증명.** 이미 정리 9.5에서 증명. 핵심 단계 요약:

$$
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) = \mathbb{E}_{\theta_1}[\log p_{\theta_1} - \log p_{\theta_2}]
= (\theta_1 - \theta_2)^\top \eta_1 - (\psi(\theta_1) - \psi(\theta_2)).
$$

정리:
$= \psi(\theta_2) - \psi(\theta_1) - \langle \theta_2 - \theta_1, \eta_1\rangle = \psi(\theta_2) - \psi(\theta_1) - \langle \nabla\psi(\theta_1), \theta_2 - \theta_1\rangle = D_\psi(\theta_2, \theta_1)$.

**의의.** KL이 단순히 "두 분포의 차이"가 아니라 **cumulant $\psi$의 tangent-gap**. $\psi$가 정보기하의 **potential function**.

---

### 정리 10.3 (Dual 좌표에서 Bregman)

Legendre 대응 $\theta \leftrightarrow \eta$, $\psi \leftrightarrow \psi^*$에서

$$
D_\psi(\theta_2, \theta_1) = D_{\psi^*}(\eta_1, \eta_2).
$$

(정리 9.6의 재확인.)

**따름 10.3.1 (KL in expectation coord).**

$$
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) = D_{\psi^*}(\eta_1, \eta_2) = \psi^*(\eta_1) - \psi^*(\eta_2) - \langle \nabla\psi^*(\eta_2), \eta_1 - \eta_2\rangle.
$$

**의의.** **$\theta$-좌표 Bregman**은 argument 순서가 뒤집힌 반면, **$\eta$-좌표 Bregman**은 정상. 이것이 두 좌표계의 "dual role".

---

### 정리 10.4 (Canonical divergence의 통합 형태)

$$
D(\theta_1 \| \theta_2) = \psi(\theta_2) + \psi^*(\eta_1) - \langle \theta_2, \eta_1\rangle.
$$

**증명.** 정리 10.2 적용:

$D_\psi(\theta_2, \theta_1) = \psi(\theta_2) - \psi(\theta_1) - \langle\eta_1, \theta_2 - \theta_1\rangle$
$= \psi(\theta_2) - \psi(\theta_1) - \langle\eta_1, \theta_2\rangle + \langle\eta_1, \theta_1\rangle$.

한편 Legendre identity: $\psi(\theta_1) + \psi^*(\eta_1) = \langle \theta_1, \eta_1\rangle$. 즉 $-\psi(\theta_1) + \langle\eta_1, \theta_1\rangle = \psi^*(\eta_1)$. 대입:

$= \psi(\theta_2) + \psi^*(\eta_1) - \langle \theta_2, \eta_1\rangle$. **Q.E.D.**

**의의.** **Canonical divergence는 두 좌표 mixing** — $\theta_2$와 $\eta_1$이 같은 식에 등장. 이것이 Ch6 Pythagoras 정리의 기반.

---

### 정리 10.5 (Fisher의 Cumulant 표현)

지수족에서 natural 좌표 $\theta$의 Fisher 행렬 = $\psi$의 Hessian:

$$
F_{ij}(\theta) = \frac{\partial^2 \psi}{\partial \theta^i \partial \theta^j}.
$$

Expectation 좌표 $\eta$의 Fisher $F^*$:

$$
F^*_{ij}(\eta) = \frac{\partial^2 \psi^*}{\partial \eta^i \partial \eta^j} = (F(\theta))^{-1}_{ij} \qquad (\eta = \nabla\psi(\theta)).
$$

즉 **두 좌표계의 Fisher가 서로 역행렬**.

**증명.** 정리 10.1(b) 로 첫 식. 둘째 식: Legendre 쌍대 공식.

$\psi^*(\eta) = \langle\theta(\eta), \eta\rangle - \psi(\theta(\eta))$, $\nabla\psi^*(\eta) = \theta(\eta)$.

$\nabla^2 \psi^*(\eta) = \nabla_\eta \theta(\eta) = (\nabla\psi)^{-1}$ 의 derivative = $(\nabla^2\psi(\theta))^{-1} = F(\theta)^{-1}$. **Q.E.D.**

**의의.** 두 좌표가 "서로의 역" — NGD가 $\theta$ 좌표에서 $F^{-1}g$ vs $\eta$ 좌표에서 $Fg$. **같은 분포 이동**이지만 **pre-conditioner 부호가 반대**.

---

### 정리 10.6 (KL의 세 가지 equivalent 형태)

임의 지수족 $p_{\theta_1}, p_{\theta_2}$에 대해

$$
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) = D_\psi(\theta_2, \theta_1) = D_{\psi^*}(\eta_1, \eta_2) = \psi(\theta_2) + \psi^*(\eta_1) - \langle\theta_2, \eta_1\rangle.
$$

즉 **같은 scalar 가 세 가지 방식으로 표현** — argument 순서·좌표계·potential 선택이 모두 self-consistent.

---

### 정리 10.7 (Pythagoras 정리: 예고판)

$p_1, p_2, p_3$이 지수족 분포, $\theta_i, \eta_i$ 좌표. 만약

$$
\langle \theta_2 - \theta_3, \eta_1 - \eta_2\rangle = 0
$$

이면,

$$
\operatorname{KL}(p_1 \| p_3) = \operatorname{KL}(p_1 \| p_2) + \operatorname{KL}(p_2 \| p_3).
$$

**직교 조건**: $\theta$-증분과 $\eta$-증분의 내적 0. 이것이 **e-geodesic (θ 선)** 과 **m-geodesic (η 선)** 의 직교성 (Ch4).

**증명.** Three-point identity (정리 9.2) 를 $D_\psi$에 적용:

$D_\psi(\theta_3, \theta_1) = D_\psi(\theta_2, \theta_1) + D_\psi(\theta_3, \theta_2) + \langle\nabla\psi(\theta_1) - \nabla\psi(\theta_2), \theta_2 - \theta_3\rangle$
$= D_\psi(\theta_2, \theta_1) + D_\psi(\theta_3, \theta_2) + \langle\eta_1 - \eta_2, \theta_2 - \theta_3\rangle$.

마지막 항 0 ⟺ 직교 조건. **Q.E.D.**

**의의.** 이것이 **Amari-Chentsov 정리**의 특별한 경우 — Ch6에서 일반화.

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: Gaussian Exp Family — 정리 10.6 세 공식 검증

```python
import numpy as np

# N(μ, σ² = 1)을 exp family로: η = μ, T = x, ψ(η) = η²/2
# η = ∇ψ = η, ψ*(η) = η²/2 (self-dual!)

def psi(theta):
    return 0.5 * theta**2

def psi_star(eta):
    return 0.5 * eta**2   # Legendre self-dual for quadratic

def kl_normal_unit(mu1, mu2):
    return 0.5 * (mu1 - mu2)**2

def D_psi(theta1, theta2):
    # D_ψ(θ1, θ2) = ψ(θ1) - ψ(θ2) - ∇ψ(θ2)(θ1 - θ2) = ψ(θ1) - ψ(θ2) - θ2(θ1-θ2)
    return psi(theta1) - psi(theta2) - theta2 * (theta1 - theta2)

def D_psi_star(eta1, eta2):
    return psi_star(eta1) - psi_star(eta2) - eta2 * (eta1 - eta2)

def canonical(theta1, theta2):
    eta1 = theta1  # ∇ψ(η) = η
    return psi(theta2) + psi_star(eta1) - theta2 * eta1

theta1, theta2 = 0.5, 2.0
eta1, eta2 = theta1, theta2  # identity

kl = kl_normal_unit(theta1, theta2)
form1 = D_psi(theta2, theta1)       # D_ψ(θ2, θ1)
form2 = D_psi_star(eta1, eta2)      # D_ψ*(η1, η2)
form3 = canonical(theta1, theta2)   # canonical form

print(f"KL(N({theta1}, 1) || N({theta2}, 1)) = {kl:.6f}")
print(f"  Form 1: D_ψ(θ2, θ1)           = {form1:.6f}")
print(f"  Form 2: D_ψ*(η1, η2)          = {form2:.6f}")
print(f"  Form 3: ψ(θ2)+ψ*(η1)-θ2·η1    = {form3:.6f}")
print(f"  All equal? {np.allclose([kl, form1, form2, form3], kl)}")
```

---

### 예제 2: Bernoulli Exp Family — non-trivial Legendre

```python
import numpy as np
import sympy as sp

# Bernoulli: P(X=1) = p, exp family:
# natural param η = log(p/(1-p))
# T(x) = x
# ψ(η) = log(1 + e^η)   (cumulant)
# p = σ(η) = e^η / (1 + e^η)

# ∇ψ(η) = σ(η) = p = μ (expectation)

# Legendre dual: ψ*(μ) = sup_η [μη - ψ(η)]
# At optimum: μ = σ(η), i.e., η = log(μ/(1-μ))
# ψ*(μ) = μ·log(μ/(1-μ)) - log(1 + μ/(1-μ)) = μ log μ + (1-μ) log(1-μ) + log(1-μ) + log((1-μ)/(1-μ))
# 실제 계산:
# ψ*(μ) = μ log(μ/(1-μ)) - log(1/(1-μ)) = μ log μ - μ log(1-μ) + log(1-μ)
#        = μ log μ + (1-μ) log(1-μ)
# 즉 negative binary entropy!

eta_sym, mu_sym = sp.symbols('eta mu', real=True)
psi_sym = sp.log(1 + sp.exp(eta_sym))
psi_star_sym = mu_sym * sp.log(mu_sym / (1 - mu_sym)) - sp.log(1 + sp.exp(sp.log(mu_sym/(1-mu_sym))))
psi_star_simplified = sp.simplify(psi_star_sym)
print(f"ψ*(μ) simplified: {psi_star_simplified}")
# 실제: μ log μ + (1-μ) log(1-μ) = -H(μ)
psi_star_expected = mu_sym*sp.log(mu_sym) + (1-mu_sym)*sp.log(1-mu_sym)
print(f"Expected form:    {sp.simplify(psi_star_expected)}")
print(f"Match? {sp.simplify(psi_star_simplified - psi_star_expected) == 0}")

# 수치 검증
def psi(eta):
    return np.log(1 + np.exp(eta))

def psi_star(mu):
    return mu * np.log(mu) + (1 - mu) * np.log(1 - mu)

def grad_psi(eta):
    return 1 / (1 + np.exp(-eta))  # sigmoid

def grad_psi_star(mu):
    return np.log(mu / (1 - mu))

# KL(Bern(p1) || Bern(p2))
def kl_bernoulli(p1, p2):
    return p1 * np.log(p1/p2) + (1-p1) * np.log((1-p1)/(1-p2))

p1, p2 = 0.3, 0.7
eta1, eta2 = grad_psi_star(p1), grad_psi_star(p2)

kl = kl_bernoulli(p1, p2)
D1 = psi(eta2) - psi(eta1) - grad_psi(eta1) * (eta2 - eta1)  # D_ψ(η2, η1)
D2 = psi_star(p1) - psi_star(p2) - grad_psi_star(p2) * (p1 - p2)  # D_ψ*(μ1, μ2)
D3 = psi(eta2) + psi_star(p1) - eta2 * p1  # canonical

print(f"\np1={p1}, p2={p2}, η1={eta1:.4f}, η2={eta2:.4f}")
print(f"KL(Bern(p1) || Bern(p2)) = {kl:.6f}")
print(f"  D_ψ(η2, η1)     = {D1:.6f}")
print(f"  D_ψ*(μ1, μ2)    = {D2:.6f}")
print(f"  canonical       = {D3:.6f}")
```

---

### 예제 3: Pythagoras 정리 검증 (지수족)

```python
import numpy as np

# 지수족: Bernoulli, exp family
def grad_psi_star(mu):
    return np.log(mu / (1 - mu))  # logit

def kl_bern(p1, p2):
    return p1*np.log(p1/p2) + (1-p1)*np.log((1-p1)/(1-p2))

# 세 점 p1, p2, p3 (Bernoulli probabilities)
# 직교성 조건 테스트: <θ2 - θ3, η1 - η2> = 0
# 즉 (η2 - η3)(μ1 - μ2) = 0 (Bernoulli 에서 scalar)

# p1 = 0.5 (η1 = 0), p3 = 0.5 (η3 = 0), p2 = 0.3 (η2 < 0)
# η2 - η3 = η2, μ1 - μ2 = 0.5 - 0.3 = 0.2
# 이 조건에서는 직교 안 됨. 다른 예:

# Set p1, p2, p3 with η2 = η3 (즉 p2 = p3), but then trivial
# 다른 예: simplex within 2D categorical (이것은 scalar case라 제한적)

# 2D 지수족 예: Multinomial(3 categories) — 2차원
# T(x) = (1[x=1], 1[x=2])  (x=3 reference)
# η = (log(p1/p3), log(p2/p3))
# ψ(η) = log(1 + e^η1 + e^η2)
# μ = (p1, p2)
# ψ*(μ) = μ1 log μ1 + μ2 log μ2 + (1-μ1-μ2) log(1-μ1-μ2)

def kl_multinoulli(p, q):
    p = np.asarray(p); q = np.asarray(q)
    return np.sum(p * np.log(p / q))

def eta_from_mu(mu):
    """μ → η for multinoulli with 3 cats"""
    p3 = 1 - mu.sum()
    return np.log(mu / p3)

# p1, p2, p3 (three multinoulli dists)
p1 = np.array([0.5, 0.3, 0.2])
p2 = np.array([0.4, 0.4, 0.2])
p3 = np.array([0.3, 0.5, 0.2])  # μ2, μ3 only first two

# Test Pythagoras at p2: (η2 - η3) · (μ1 - μ2) = 0 가 필요
mu1 = p1[:2]; mu2 = p2[:2]; mu3 = p3[:2]
eta1 = eta_from_mu(mu1)
eta2 = eta_from_mu(mu2)
eta3 = eta_from_mu(mu3)

inner = np.dot(eta2 - eta3, mu1 - mu2)
print(f"<η2 - η3, μ1 - μ2> = {inner:.4f} (should be 0 for Pythagoras)")

# 이 예는 직교 아님. 인위적으로 직교한 예 구성:
# p1과 p2가 같은 η를 공유하거나, specific construct
# Let's just show Pythagoras doesn't hold when not orthogonal
kl13 = kl_multinoulli(p1, p3)
kl12 = kl_multinoulli(p1, p2)
kl23 = kl_multinoulli(p2, p3)

print(f"\nKL(p1||p3) = {kl13:.6f}")
print(f"KL(p1||p2) + KL(p2||p3) = {kl12 + kl23:.6f}")
print(f"Difference (=inner product term): {kl13 - kl12 - kl23:.6f}")
print(f"\nPythagoras holds? {np.isclose(kl13, kl12+kl23)}")
```

---

### 예제 4: $F$ 와 $F^*$ 역행렬 관계

```python
import numpy as np

# Gaussian N(μ, σ²) as exp family:
# T(x) = (x, x²)
# η = (μ, μ² + σ²)  [expectation params]
# 그러나 natural param θ = (μ/σ², -1/(2σ²))
# ψ(θ) = -θ1²/(4θ2) - 0.5 log(-2θ2) + 0.5 log 2π  (symbolic)

# 대신 간단한 예: Poisson
# T(x) = x, ψ(θ) = e^θ, η = e^θ = λ

# Fisher at θ: F(θ) = ψ''(θ) = e^θ = λ
# In η coord: η = λ, ψ*(η) = η log η - η
# Fisher at η: F*(η) = ψ*''(η) = 1/η = 1/λ

# 확인: F(θ) · F*(η) = λ · 1/λ = 1 (역행렬 관계, 스칼라)

theta_vals = np.array([0.0, 0.5, 1.0, 1.5])
eta_vals = np.exp(theta_vals)

F_theta = np.exp(theta_vals)
F_eta = 1 / eta_vals

print(f"{'θ':>8} {'η':>8} {'F(θ)':>10} {'F*(η)':>10} {'product':>10}")
for t, e, Ft, Fe in zip(theta_vals, eta_vals, F_theta, F_eta):
    print(f"{t:>8.3f} {e:>8.3f} {Ft:>10.4f} {Fe:>10.4f} {Ft*Fe:>10.4f}")

print("\n✓ Product = 1 (F, F* are inverses — 정리 10.5)")
```

---

### 예제 5: Natural Gradient in θ vs Vanilla Gradient in η

```python
import numpy as np
import matplotlib.pyplot as plt

# 목적: maximize log-likelihood of Poisson data
# p_θ(x) = exp(θ x - e^θ) / x! (base meas h = 1/x!)
# Data: X ~ Poi(λ*=3.0)
# MLE: θ* = log λ* = log 3 ≈ 1.0986

np.random.seed(0)
lam_true = 3.0
N = 1000
X = np.random.poisson(lam_true, N)

# NLL in θ: -E[log p_θ] = -θ x̄ + e^θ
x_bar = np.mean(X)

def nll_theta(theta):
    return -theta * x_bar + np.exp(theta)

def grad_nll_theta(theta):
    return -x_bar + np.exp(theta)  # = -x_bar + η

def fisher_theta(theta):
    return np.exp(theta)  # = η

# η coord: NLL in η = -log λ · x̄ + λ
def nll_eta(eta):
    return -np.log(eta) * x_bar + eta

def grad_nll_eta(eta):
    return -x_bar / eta + 1

def fisher_eta(eta):
    return 1 / eta

theta_trajectory_gd = []
theta_trajectory_ngd = []
eta_trajectory_gd = []
eta_trajectory_ngd = []

# GD in θ
theta = 0.0
for _ in range(100):
    theta_trajectory_gd.append(theta)
    theta = theta - 0.1 * grad_nll_theta(theta)
theta_trajectory_gd.append(theta)

# NGD in θ (= F^{-1} g)
theta = 0.0
for _ in range(100):
    theta_trajectory_ngd.append(theta)
    F = fisher_theta(theta)
    theta = theta - 0.5 * grad_nll_theta(theta) / F
theta_trajectory_ngd.append(theta)

# GD in η 
eta = 1.0  # start at η=1
for _ in range(100):
    eta_trajectory_gd.append(eta)
    eta = max(eta - 0.1 * grad_nll_eta(eta), 0.01)  # positivity
eta_trajectory_gd.append(eta)

# NGD in η
eta = 1.0
for _ in range(100):
    eta_trajectory_ngd.append(eta)
    F = fisher_eta(eta)
    eta = max(eta - 0.5 * grad_nll_eta(eta) / F, 0.01)
eta_trajectory_ngd.append(eta)

print(f"True:        λ* = {lam_true}, θ* = {np.log(lam_true):.4f}")
print(f"MLE estimate: x̄ = {x_bar:.4f}")
print(f"\nGD in θ:  final θ = {theta_trajectory_gd[-1]:.4f}, λ = {np.exp(theta_trajectory_gd[-1]):.4f}")
print(f"NGD in θ: final θ = {theta_trajectory_ngd[-1]:.4f}, λ = {np.exp(theta_trajectory_ngd[-1]):.4f}")
print(f"GD in η:  final η = {eta_trajectory_gd[-1]:.4f}")
print(f"NGD in η: final η = {eta_trajectory_ngd[-1]:.4f}")

# NGD in θ = ??? (vs GD in η); both should go to MLE quickly
# Key insight: NGD in θ traverse η-space vanilla, and vice versa
```

---

## 🔗 AI/ML 연결

### 1. Natural Gradient의 Coordinate 해석

지수족 parameter:
- **GD in θ**: Euclidean gradient descent in natural coordinates.
- **NGD in θ**: $F^{-1}g$ — effectively moves in η-space uniformly.
- **GD in η**: "Dual gradient descent" — MLE-natural updates (e.g., Gaussian MLE updates $\mu \leftarrow \mu - \eta (\mu - x)$).

**Softmax output layer**: η = probabilities, θ = logits. Standard gradient on θ = natural gradient on η (up to Fisher scaling).

### 2. Softmax Cross-Entropy의 특별함

Softmax 분류의 CE loss는 지수족 (Multinomial) 에서 NLL. 그래서

$$
\operatorname{Hess}(\operatorname{CE}) = F = \operatorname{diag}(\pi) - \pi\pi^\top,
$$

즉 **Hessian = Fisher** (정리 10.5). Newton method와 NGD가 정확히 일치. 이것이 softmax의 빠른 수렴성의 이유.

### 3. Exponential Family Variational Inference

Mean-field VI + exp family:

$$
q_\phi(z) = \prod_i q_{\phi_i}(z_i), \quad q_{\phi_i} \in \text{ExpFam}.
$$

각 $q_{\phi_i}$의 natural parameter $\theta_i$ 업데이트 공식이 ELBO의 gradient + cumulant의 Hessian → **natural gradient VI**.

### 4. Reinforcement Learning의 Policy Parameterization

Softmax policy $\pi(a|s) = \operatorname{softmax}(\theta^\top \phi(s, a))$. 이것이 지수족. Policy gradient = $\nabla_\theta \log \pi$ = score. Fisher = $\operatorname{Cov}[\nabla_\theta \log \pi]$. TRPO/NGD가 softmax policy에서 특히 잘 작동.

### 5. ELBO Decomposition

ELBO:

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi}[\log p_\theta(x, z)] - \mathbb{E}_{q_\phi}[\log q_\phi(z)].
$$

지수족 $q_\phi$에서 두 번째 항 = $-\psi^*(\eta_\phi)$ (dual potential). 첫 번째 항 + dual potential = **Fenchel-Young loss**.

### 6. Dually Flat Structure의 AI 의미 (Ch4 예고)

지수족은 **두 좌표계 모두에서 flat** — $\theta$-선 (e-geodesic) 과 $\eta$-선 (m-geodesic) 모두 최단 경로. 이 구조 덕에:

- **EM 알고리즘** (M-projection + E-projection의 반복) 이 잘 수렴.
- **Free energy** 최소화가 **KL 최소화**와 동등.
- **Information projection** 이 명확히 정의.

### 7. Softmax vs Log-Softmax의 수치 안정성

$\eta \to \theta$ 변환: $\operatorname{softmax}$. $\theta \to \eta$ 변환의 inverse는 없음 (non-unique without normalization). 대신 **logit** = θ, **probability** = η. Logit 공간에서 loss 계산이 수치적으로 안정 (log-sum-exp trick).

---

## ⚖️ 가정과 한계

### 지수족만의 identity

정리 10.2는 **지수족에서만** 정확. 비지수족 (Cauchy, Student-t, mixtures 등) 에서는 KL이 Bregman 형태로 표현 안 됨.

**부분 해법**: 지수족 근사 (moment matching, Laplace approximation) 으로 Bregman 적용.

### Minimal Representation 요구

$\psi$의 엄격 볼록성이 minimal representation (sufficient statistic이 affine independent) 필요. Over-complete 표현 (e.g., one-hot with all-zeros 포함) 에서는 $\psi$ flat 방향 존재 → **Pseudo-inverse Fisher** 필요.

### Legendre Conjugate 계산의 어려움

$\psi^*$의 closed form 이 쉬운 경우 (Gaussian, Bernoulli, Poisson) 와 어려운 경우 (복잡한 모델) 분화. Numerical Legendre transform은 $\theta \leftrightarrow \eta$ bijection을 iterative하게 구함.

### Infinite-dimensional Exponential Families

Gaussian process, non-parametric exp family 에서 $\theta$는 무한차원. Fisher, cumulant 모두 operator. 이론은 복잡.

### 실무에서의 $\eta$ 좌표의 장점

$\eta = \mathbb{E}[T]$ 이므로 **직접 관측 가능** (충분통계량의 기대값). MLE = **matching η** (empirical $\bar T$ = model $\eta$). 반면 $\theta$는 가끔 해석 어려움 (자연 parameter가 abstract).

---

## 📌 핵심 정리

| 관계 | 수식 |
|---|---|
| **KL = Cumulant Bregman** | $\operatorname{KL}(p_1\|p_2) = D_\psi(\theta_2, \theta_1)$ |
| **Dual Bregman** | $= D_{\psi^*}(\eta_1, \eta_2)$ |
| **Canonical form** | $= \psi(\theta_2) + \psi^*(\eta_1) - \langle\theta_2, \eta_1\rangle$ |
| **Fisher = Hess ψ** | $F(\theta) = \operatorname{Hess}\psi(\theta)$, $F^*(\eta) = \operatorname{Hess}\psi^*(\eta) = F^{-1}$ |
| **Legendre** | $\theta = \nabla\psi^*(\eta), \eta = \nabla\psi(\theta)$ |
| **Pythagoras (special)** | $\langle\theta_2 - \theta_3, \eta_1 - \eta_2\rangle = 0 \Rightarrow$ KL 가법성 |

**지수족의 삼각 구조:**
```
        θ-space (natural)                 η-space (expectation)
      ────────────────                   ───────────────────
         ψ(θ) convex          ←→         ψ*(η) convex
         F = Hess ψ           ←→         F⁻¹ = Hess ψ*
         e-geodesic (linear)   ⟂       m-geodesic (linear)
```

**Takeaway:**

1. **정보기하의 "원자" = 지수족의 Bregman 구조**.
2. **두 좌표계의 쌍대성** — NGD의 근본 이유, EM 알고리즘의 근원.
3. **Canonical divergence**는 Pythagoras 정리의 기반 (Ch6).
4. **Softmax, sigmoid, Gaussian MLE** — 모두 이 framework의 특수 경우.

---

## 🤔 생각해볼 문제

1. **비지수족의 Bregman 근사**. Cauchy 분포 $p_\theta(x) = 1/(\pi(1 + (x-\theta)^2))$의 KL은 Bregman으로 정확히 표현 안 됨. 왜? Laplace approximation 하면 지수족 근사되는데, 이때 생기는 error는?

2. **다양한 지수족의 $\psi^*$**. 아래 분포들의 $\psi(\theta)$와 $\psi^*(\eta)$를 모두 closed form으로 유도:
   - Gamma distribution
   - Beta distribution
   - Categorical (multinoulli)
   - Multivariate Gaussian with fixed Σ

3. **NGD와 Mirror Descent**. Mirror Descent with potential $\psi$가 natural parameter 공간에서의 vanilla GD와 같음을 보여라. 어떤 조건에서 정확히 equivalent?

4. **$\theta$-좌표의 "자연스러움"**. $\theta$가 natural coord라 불리는 이유는 "exp 안에서 linear"이기 때문. 그러면 non-exp 모델에서 "natural" 한 좌표는 무엇인가? 예: Cauchy에서.

5. **Multivariate Gaussian의 전체 Fisher**. $\mathcal{N}(\mu, \Sigma)$에서 natural param $\theta$는 $k + k(k+1)/2$ 차원. $\psi(\theta)$의 정확한 형태 유도. 결과에서 block 구조 확인.

6. **EM 알고리즘과 쌍대성**. EM: E-step ($\eta$ 업데이트), M-step ($\theta$ 업데이트). 이것이 $\theta \leftrightarrow \eta$ oscillation인 이유. Convergence가 each step에서 ELBO 증가로 보장되는 기하학적 해석?

7. **Entropy로서의 $\psi^*$**. Bernoulli에서 $\psi^*(\mu) = -H(\mu)$ (엔트로피). 일반 지수족에서 $-\psi^*(\eta) = H(p_\eta) - \mathbb{E}_{p_\eta}[\log h(X)]$. 이것이 Gibbs distribution의 free energy 해석과 일치하는지.

---

<div align="center">

| [◀ 03. Bregman 발산](./03-bregman-divergence.md) | [📚 메인 README](../README.md) | [05. α-divergence와 Rényi ▶](./05-alpha-renyi-divergence.md) |
|:---:|:---:|:---:|

</div>
