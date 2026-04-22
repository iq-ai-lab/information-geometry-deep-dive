# 03. Legendre 변환으로 쌍대 좌표 — $\theta \leftrightarrow \eta$ 미분동형

<div align="center">

> *"canonical parameter $\theta$와 expectation parameter $\eta = \mathbb{E}[T]$는 '같은 분포를 가리키는 두 이름'이다.*  
> *하나는 지수의 지수에 들어가는 좌표, 하나는 관측 가능한 기댓값 좌표.*
>
> *그리고 이 두 좌표 사이에는 볼록 해석학의 가장 아름다운 대응이 숨어있다 — Legendre 변환.  
> 이것이 Amari가 정보기하의 '쌍대 구조'를 발견한 출발점이다."*

</div>

---

## 🎯 핵심 질문

1. **Legendre 변환** $\psi^*(\eta) = \sup_\theta(\theta^T \eta - \psi(\theta))$의 정의와 볼록해석적 성질은 무엇인가?
2. regular 지수족에서 $\theta \leftrightarrow \eta = \nabla\psi(\theta)$는 **미분동형(diffeomorphism)**임을 어떻게 증명하는가?
3. 쌍대 좌표 $\eta$에서의 Fisher 정보 $F^*(\eta) = \nabla^2\psi^*(\eta)$는 $F(\theta)$와 **$F \cdot F^* = I$**(역행렬 관계)임을 증명.
4. $\psi^*$의 통계적 의미는? → **음 엔트로피** $\psi^*(\eta) = -H(p_\theta) + $ base. 이것이 Ch3-04의 KL = Bregman 공식의 원천.
5. **canonical = Exponential affine, expectation = Mixture affine** — 왜 두 좌표는 각각 서로 다른 아핀 구조를 정의하는가? (Ch4-04 e/m-connection의 예고)

---

## 🔍 왜 이 기하학이 AI에서 중요한가

| AI/ML 기법 | Legendre 쌍대가 하는 일 |
|-----------|----------------|
| **Mirror Descent** | $\theta_{k+1} = \nabla\psi^*(\nabla\psi(\theta_k) - \eta g)$ — dual space에서 gradient step (Ch7-02) |
| **Logistic/Softmax** | $\theta = $ logit, $\eta = $ probability — 두 좌표가 같은 분포를 표현 |
| **Variational Inference** | ELBO를 $\eta$ 좌표에서 쓰면 closed-form; $\theta$에서 쓰면 gradient 필요 |
| **Exponentiated Gradient** | Simplex 위의 EG는 mirror descent 특수 사례 = natural gradient in multinomial 지수족 |
| **Information Projection** | KL 최소화가 두 좌표에서 서로 다른 기하학적 해석 (Ch6-01) |
| **Bregman Clustering** | K-means의 일반화 — Bregman divergence의 center가 $\eta$-평균 |

---

## 📐 수학적 선행 조건

| 개념 | 참조 |
|------|------|
| **Legendre-Fenchel 변환**과 볼록 켤레 함수 | Convex Optimization Deep Dive Ch3 |
| **역함수 정리**와 미분동형 | Calculus & Optimization Deep Dive Ch5 |
| **엄격 볼록함수** ↔ 미분 가능의 단사성 | Convex Opt Deep Dive Ch2 |
| $\psi$의 볼록성, $\nabla\psi = \mathbb{E}[T]$, $\nabla^2\psi = F$ | **Ch4-02** |
| Bregman divergence와 삼점 정체성 | **Ch3-03** |

---

## 📖 직관적 이해

### 1. 볼록함수의 두 얼굴: $\psi(\theta)$와 $\psi^*(\eta)$

Legendre 변환은 볼록함수의 "기울기에 의한 parameterization":

$$
\psi^*(\eta) := \sup_\theta \big(\theta^T \eta - \psi(\theta)\big)
$$

직관:
- $\psi(\theta)$는 **값**(value) 좌표 — $\theta$를 넣으면 $\psi$ 나옴.
- $\psi^*(\eta)$는 **기울기**(slope) 좌표 — $\eta$는 $\psi$의 기울기가 $\eta$인 지점에서의 "잘라낸 절편".

**그림으로:** $y = \psi(\theta)$ 곡선에 $\theta = \theta_0$에서 접선을 그리면 기울기 $\eta_0 = \psi'(\theta_0)$. 그 접선의 $y$-절편은 $\psi(\theta_0) - \theta_0 \eta_0 = -\psi^*(\eta_0)$.

$$
\theta \xrightarrow{\nabla\psi} \eta, \quad \eta \xrightarrow{\nabla\psi^*} \theta
$$

### 2. 두 좌표는 "같은 분포"

지수족 점 $p_\theta$에 대해:
- **Canonical $\theta$**: logit, log-rate, 자연 파라미터 — 최적화에서 로그우도가 볼록.
- **Expectation $\eta = \mathbb{E}_\theta[T]$**: 평균, 확률, 기댓값 — 관측·추정과 직결.

Bernoulli: $\theta = \log\frac{p}{1-p} \in \mathbb{R}$, $\eta = p \in (0,1)$.

두 좌표 간 변환:
$$
\theta = \text{logit}(\eta), \quad \eta = \sigma(\theta) = 1/(1+e^{-\theta})
$$

어떤 좌표가 "옳은가"? 답: **둘 다 옳다**. 문제마다 유리한 좌표가 다를 뿐이며, 이 대칭성이 정보기하의 쌍대 구조의 출발점이다.

### 3. Fisher의 쌍대성: $F \cdot F^* = I$

핵심 공식:

$$
F(\theta) = \nabla^2\psi(\theta), \quad F^*(\eta) = \nabla^2\psi^*(\eta)
$$

그리고 $\eta = \nabla\psi(\theta)$의 야코비를 미분하면:

$$
\frac{\partial \eta}{\partial \theta} = \nabla^2\psi(\theta) = F(\theta)
$$

역함수 정리에서:

$$
\frac{\partial \theta}{\partial \eta} = (\nabla^2\psi)^{-1} = F(\theta)^{-1}
$$

그런데 $\theta = \nabla\psi^*(\eta)$이므로 $\partial\theta/\partial\eta = \nabla^2\psi^*(\eta) = F^*(\eta)$.

$$
\boxed{\;F^*(\eta) = F(\theta)^{-1}\;}
$$

즉 두 좌표계의 Fisher는 **서로 역행렬**.

### 4. $\psi^*$는 음 엔트로피

핵심 관찰 (Ch3-04과 연결):

$$
\psi^*(\eta) = -H(p_\theta) + \int p_\theta \log h \, d\nu
$$

즉 **Legendre dual = 음 엔트로피**(plus base term). 이것이 Ch3-04에서 "KL이 Bregman"이 되는 이유.

**직관:** 엔트로피는 "얼마나 퍼져 있는가", expectation parameter는 "평균이 어디에 있는가". 둘은 같은 분포의 다른 측면이고, Legendre 변환이 이들을 엮는다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 (Legendre-Fenchel 변환)

볼록함수 $\psi: \Theta \to \mathbb{R}$ ($\Theta \subseteq \mathbb{R}^d$ 볼록)에 대해 그 **볼록 켤레(convex conjugate)**는

$$
\psi^*(\eta) := \sup_{\theta \in \Theta} \big(\theta^T \eta - \psi(\theta)\big), \quad \eta \in \mathbb{R}^d
$$

정의역: $\text{dom}\,\psi^* := \{\eta : \psi^*(\eta) < \infty\}$.

$\psi$가 **proper, 하반연속(l.s.c.), 볼록**이면 $\psi^*$도 그렇고 $\psi^{**} = \psi$ (Fenchel-Moreau 정리, Convex Opt Ch3).

### 정의 3.2 (지수족의 expectation parameter 공간)

regular 지수족의 자연 파라미터 공간 $\Theta$ 내부에서 $\nabla\psi: \Theta^\circ \to \mathbb{R}^d$의 상(image)을

$$
\mathcal{E} := \nabla\psi(\Theta^\circ) = \{\mathbb{E}_\theta[T(X)] : \theta \in \Theta^\circ\}
$$

로 정의. 이것은 $\{T(x) : x \in \mathcal{X}\}$의 convex hull의 내부와 일치한다 (Brown 1986).

### 정의 3.3 (쌍대 좌표, dual coordinates)

지수족에서

$$
\theta \in \Theta^\circ : \text{canonical (natural) parameter}
$$
$$
\eta = \nabla\psi(\theta) \in \mathcal{E}^\circ : \text{expectation (dual, mean-value) parameter}
$$

두 좌표가 같은 분포 $p_\theta = p_\eta$를 표현하지만 서로 다른 기하학적 역할(aff의 방향, affine 구조)을 한다.

---

## 🔬 정리와 증명

### 정리 3.4 (Legendre 대응 관계, Fenchel-Young 부등식)

볼록함수 $\psi$와 켤레 $\psi^*$에 대해

$$
\theta^T \eta \le \psi(\theta) + \psi^*(\eta)
$$

모든 $\theta, \eta$에 대해. 등호는 $\eta \in \partial\psi(\theta)$ ($\psi$의 $\theta$에서의 subgradient), 즉 $\psi$ 미분가능한 경우 $\eta = \nabla\psi(\theta)$일 때.

**증명.** 정의에서

$$
\psi^*(\eta) = \sup_{\theta'}(\theta'^T \eta - \psi(\theta')) \ge \theta^T \eta - \psi(\theta)
$$

즉 $\psi(\theta) + \psi^*(\eta) \ge \theta^T \eta$. 등호 조건은 sup가 $\theta$에서 달성되어야 하며, 이는 $\nabla_\theta[\theta^T \eta - \psi(\theta)] = 0 \iff \eta = \nabla\psi(\theta)$. $\blacksquare$

### 정리 3.5 (엄격 볼록성의 유지)

$\psi$가 $\Theta^\circ$에서 엄격 볼록이고 $C^2$이면 $\psi^*$는 $\mathcal{E}^\circ$에서 엄격 볼록이고 $C^2$.

**증명.** $\psi$ 엄격 볼록 $\Rightarrow$ $\nabla\psi$가 단사 (Convex Opt Ch2). $\nabla^2\psi \succ 0$이므로 역함수 정리에서 $(\nabla\psi)^{-1} = \nabla\psi^*$가 $C^1$, 그리고 $\nabla^2\psi^* = (\nabla^2\psi)^{-1} \succ 0$. 따라서 $\psi^*$ 엄격 볼록, $C^2$. $\blacksquare$

### 정리 3.6 ($\theta \leftrightarrow \eta$ 미분동형) — 핵심

regular minimal 지수족에서 $\nabla\psi: \Theta^\circ \to \mathcal{E}^\circ$는 $C^\infty$-미분동형(diffeomorphism)이다.

**증명.**
1. **전사(surjective)**: $\mathcal{E}^\circ$의 정의에서 자동. (더 일반적으로는 Brown 1986의 결과.)
2. **단사(injective)**: 엄격 볼록성 (Ch4-02 정리 2.3). $\theta_1 \neq \theta_2 \Rightarrow \psi$ 엄격 볼록 $\Rightarrow$ $\nabla\psi(\theta_1) \neq \nabla\psi(\theta_2)$ (볼록 해석 표준 사실).
3. **미분동형**: 야코비 $\partial\eta/\partial\theta = \nabla^2\psi(\theta) = F(\theta) \succ 0$ (엄격 볼록 + minimal $\Rightarrow$ 양정치). 역함수 정리에서 $\nabla\psi$가 $C^\infty$-미분동형.

$\blacksquare$

### 정리 3.7 ($\nabla\psi^* = (\nabla\psi)^{-1}$, Legendre 역관계)

$\nabla\psi^*(\eta) = \theta$ where $\eta = \nabla\psi(\theta)$.

**증명.** $\psi$ 엄격 볼록 미분가능이므로 정리 3.4에서 $\psi^*(\eta) = \theta^T\eta - \psi(\theta)$ (sup 달성).

이제 $\eta$에 대해 $\psi^*$를 미분하려면 체인룰이 필요. $\eta$에 대해 미세 변화:

$$
\psi^*(\eta + d\eta) = (\theta + d\theta)^T(\eta + d\eta) - \psi(\theta + d\theta)
$$

$$
= \theta^T\eta + \theta^T d\eta + d\theta^T \eta + O(|d|^2) - \psi(\theta) - \nabla\psi(\theta)^T d\theta + O(|d\theta|^2)
$$

$\nabla\psi(\theta) = \eta$이므로 $d\theta^T \eta - \nabla\psi^T d\theta$ 소거:

$$
\psi^*(\eta + d\eta) - \psi^*(\eta) = \theta^T d\eta + O(|d|^2)
$$

즉 $\nabla\psi^*(\eta) = \theta$. $\blacksquare$

### 정리 3.8 (Fisher의 쌍대 역행렬 관계)

$$
\boxed{\;F^*(\eta) := \nabla^2\psi^*(\eta) = F(\theta)^{-1} \quad \text{at } \eta = \nabla\psi(\theta)\;}
$$

**증명.** $\eta = \nabla\psi(\theta)$ 양변 $\theta$에 대해 미분하면

$$
d\eta = \nabla^2\psi(\theta) d\theta = F(\theta) d\theta
$$

반대로 $\theta = \nabla\psi^*(\eta)$ 양변 $\eta$에 대해 미분:

$$
d\theta = \nabla^2\psi^*(\eta) d\eta = F^*(\eta) d\eta
$$

두 식에서 $d\theta = F^*(\eta) F(\theta) d\theta$, 임의 $d\theta$에 대해 성립하므로 $F^*(\eta) F(\theta) = I$. $\blacksquare$

**귀결.** Fisher 행렬의 두 좌표계 표현은 서로 역행렬. 이것이 natural gradient $F^{-1}\nabla L$가 $\eta$ 좌표계에서 **유클리드 gradient**와 같아지는 이유 (Ch5-04).

### 정리 3.9 ($\psi^*$ = 음 엔트로피)

지수족 $p_\theta(x) = \exp(\theta^T T(x) - \psi(\theta))h(x)$에서

$$
\psi^*(\eta) = \mathbb{E}_{p_\theta}[\log p_\theta(X)/h(X)] = \theta^T\eta - \psi(\theta)
$$

이것은 **$h$-relative entropy의 음수**이다. 특히 $h = $ 상수이면 $\psi^*(\eta) = -H(p_\theta) + $ const, 즉 음 엔트로피.

**증명.**

$$
\mathbb{E}[\log p_\theta - \log h] = \mathbb{E}[\theta^T T - \psi] = \theta^T \mathbb{E}[T] - \psi(\theta) = \theta^T \eta - \psi(\theta) = \psi^*(\eta)
$$

정의 3.1 Legendre 등호 조건. $\blacksquare$

### 정리 3.10 (KL = Bregman = Legendre) — **정보기하의 "삼중 식별"**

지수족에서 두 분포 $p_{\theta_1}, p_{\theta_2}$, $\eta_i = \nabla\psi(\theta_i)$:

$$
\boxed{\;\text{KL}(p_{\theta_1} \| p_{\theta_2}) = D_\psi(\theta_2, \theta_1) = D_{\psi^*}(\eta_1, \eta_2) = \psi(\theta_2) + \psi^*(\eta_1) - \theta_2^T \eta_1\;}
$$

**증명.** Ch3-04 정리에서 이미 증명. 여기서는 마지막 "**canonical** 형태" 한 줄 재유도:

$$
D_\psi(\theta_2, \theta_1) = \psi(\theta_2) - \psi(\theta_1) - \nabla\psi(\theta_1)^T(\theta_2 - \theta_1)
$$

$= \psi(\theta_2) - \psi(\theta_1) - \eta_1^T \theta_2 + \eta_1^T \theta_1$

$= \psi(\theta_2) + (\theta_1^T \eta_1 - \psi(\theta_1)) - \eta_1^T \theta_2$

$= \psi(\theta_2) + \psi^*(\eta_1) - \theta_2^T \eta_1$  (∵ $\theta_1^T\eta_1 - \psi(\theta_1) = \psi^*(\eta_1)$)

$\blacksquare$

**아름다움:** KL은 $\theta_2, \eta_1$ 두 좌표가 **혼합된 표현**을 갖는다 — "한 분포는 $\theta$, 다른 한 분포는 $\eta$로". 이 "혼합 좌표 표현"이 Ch6-01 Pythagoras 정리의 정확한 출처이다.

### 정리 3.11 (Exponential-Affine vs Mixture-Affine 구조)

지수족 $\mathcal{E}$는 두 아핀 구조를 가진다:

- **e-affine (exponential affine)**: $\theta$ 좌표에서 직선 $\theta(t) = (1-t)\theta_1 + t\theta_2$는 $p_{\theta(t)}(x) \propto p_{\theta_1}(x)^{1-t} p_{\theta_2}(x)^t$. 로그 공간에서 직선 = exponential family "curve".
- **m-affine (mixture affine)**: $\eta$ 좌표에서 직선 $\eta(t) = (1-t)\eta_1 + t\eta_2$는 $p$ 자체의 convex combination에 대응하지 **않지만**, $\mathbb{E}_{p_{\eta(t)}}[T] = (1-t)\eta_1 + t\eta_2$인 mixture의 "대표" 지수족 분포를 가리킨다.

**핵심 관찰 (Ch4-04 예고).** 두 아핀 구조는 **서로 다른 affine connection** $\nabla^{(e)}, \nabla^{(m)}$을 정의하며, 각 좌표계에서 크리스토펠 기호가 0이다.

---

## 💻 NumPy / SymPy 구현으로 검증

### 코드 1: Bernoulli에서 Legendre 변환 직접 계산

```python
import numpy as np

# ψ(θ) = log(1 + e^θ)
def psi(theta):
    return np.log1p(np.exp(theta))

def grad_psi(theta):
    return 1.0 / (1 + np.exp(-theta))  # sigmoid

# Legendre 쌍대: η ∈ (0, 1)
# ψ*(η) = sup_θ (θη - ψ(θ))
# 1계 조건: η = ψ'(θ) = σ(θ)  →  θ = logit(η) = log(η/(1-η))
# ψ*(η) = logit(η)·η - log(1 + e^{logit(η)})
#       = η log(η/(1-η)) - log(1/(1-η))
#       = η log η + (1-η) log(1-η)  = negative entropy (unit base)

def psi_star(eta):
    return eta * np.log(eta) + (1 - eta) * np.log(1 - eta)

# 수치적으로 Legendre sup 확인
etas = np.linspace(0.05, 0.95, 5)
for eta in etas:
    # sup 수치 계산
    theta_grid = np.linspace(-10, 10, 2001)
    val = theta_grid * eta - psi(theta_grid)
    num_sup = val.max()
    # 이론값
    anal = psi_star(eta)
    print(f"η={eta:.2f}: sup(θη-ψ)={num_sup:.6f}, ψ*(η)={anal:.6f}, diff={abs(num_sup-anal):.2e}")
# 머신정밀도로 일치
```

### 코드 2: $F \cdot F^* = I$ 가우스에서 검증

```python
import numpy as np
import sympy as sp

# N(μ, σ²), θ₁=μ/σ², θ₂=-1/(2σ²), η₁=μ, η₂=μ²+σ²
theta1, theta2 = sp.symbols('theta1 theta2', real=True)

psi = -theta1**2 / (4 * theta2) - sp.Rational(1, 2) * sp.log(-2 * theta2)
# F(θ) = Hessian of ψ
F = sp.hessian(psi, (theta1, theta2))
F_inv = F.inv()
print("F(θ) =")
sp.pprint(sp.simplify(F))
print("F(θ)⁻¹ =")
sp.pprint(sp.simplify(F_inv))

# ψ*(η)를 직접 구성: η₁=μ, η₂=μ²+σ² → μ=η₁, σ²=η₂-η₁²
eta1, eta2 = sp.symbols('eta1 eta2', real=True)
mu_in = eta1
sigma2_in = eta2 - eta1**2
# ψ*(η) = θᵀη - ψ(θ)|_{θ=θ(η)}
theta1_of_eta = mu_in / sigma2_in
theta2_of_eta = -1 / (2 * sigma2_in)
psi_val = (-theta1_of_eta**2 / (4 * theta2_of_eta)
           - sp.Rational(1, 2) * sp.log(-2 * theta2_of_eta))
psi_star = sp.simplify(theta1_of_eta * eta1 + theta2_of_eta * eta2 - psi_val)
print("ψ*(η) =")
sp.pprint(sp.simplify(psi_star))

# F*(η) = Hessian of ψ*
F_star = sp.hessian(psi_star, (eta1, eta2))
F_star_simpl = sp.simplify(F_star)
print("F*(η) =")
sp.pprint(F_star_simpl)

# θ, η을 μ, σ² 좌표로 맞춰 F·F* = I 확인
mu_sym, sigma2_sym = sp.symbols('mu sigma2', real=True, positive=True)
F_in_muSigma = sp.simplify(F.subs({theta1: mu_sym/sigma2_sym, theta2: -1/(2*sigma2_sym)}))
F_star_in_muSigma = sp.simplify(F_star_simpl.subs({eta1: mu_sym, eta2: mu_sym**2 + sigma2_sym}))
product = sp.simplify(F_in_muSigma * F_star_in_muSigma)
print("F(θ)·F*(η) =")
sp.pprint(product)
# [[1, 0], [0, 1]]
```

### 코드 3: KL = triple form 검증

```python
import numpy as np
from scipy.stats import norm

# 두 정규분포 p1 = N(μ₁, σ₁²), p2 = N(μ₂, σ₂²)
def kl_normal(mu1, s1, mu2, s2):
    return np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5

def psi_normal(mu, s2):
    # θ₁=μ/σ², θ₂=-1/(2σ²)
    t1 = mu / s2
    t2 = -1 / (2 * s2)
    return -t1**2/(4*t2) - 0.5 * np.log(-2*t2)

def psi_star_normal(mu, s2):
    # ψ* in expectation params  η₁=μ, η₂=μ²+σ²
    # Legendre: ψ* = θᵀη - ψ
    t1 = mu / s2; t2 = -1 / (2*s2)
    e1 = mu;      e2 = mu**2 + s2
    return t1*e1 + t2*e2 - psi_normal(mu, s2)

# KL(p1 || p2) = ψ(θ₂) + ψ*(η₁) - θ₂ᵀη₁
def kl_via_legendre(mu1, s1, mu2, s2):
    psi_th2 = psi_normal(mu2, s2**2)
    psi_star_e1 = psi_star_normal(mu1, s1**2)
    t1_2 = mu2 / s2**2; t2_2 = -1 / (2 * s2**2)
    e1_1 = mu1; e2_1 = mu1**2 + s1**2
    return psi_th2 + psi_star_e1 - (t1_2*e1_1 + t2_2*e2_1)

cases = [(0, 1, 0, 2), (1, 1, 0, 1), (0.5, 2, 2, 1)]
print(f"{'params':>25} {'KL direct':>12} {'KL Legendre':>14} {'diff':>10}")
for mu1, s1, mu2, s2 in cases:
    direct = kl_normal(mu1, s1, mu2, s2)
    via = kl_via_legendre(mu1, s1, mu2, s2)
    print(f"({mu1},{s1},{mu2},{s2})".rjust(25) + f" {direct:12.6f} {via:14.6f} {abs(direct-via):10.2e}")
# 세 좌표 표현이 모두 일치
```

### 코드 4: Fenchel-Young 부등식 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(-4, 4, 200)
psi_v = np.log1p(np.exp(theta))

# η=0.7에서 접선 (θ₀=logit(0.7))
eta0 = 0.7
theta0 = np.log(eta0 / (1 - eta0))
psi0 = np.log1p(np.exp(theta0))
# 접선: y = eta0·(θ - θ₀) + psi0 = eta0·θ + (psi0 - eta0·θ₀)
tan_line = eta0 * theta + (psi0 - eta0 * theta0)
psi_star_eta0 = theta0 * eta0 - psi0

plt.figure(figsize=(7, 5))
plt.plot(theta, psi_v, 'b', label=r'$\psi(\theta) = \log(1+e^\theta)$')
plt.plot(theta, tan_line, 'r--', label=rf'접선: slope={eta0}, y-intercept={-psi_star_eta0:.3f}')
plt.scatter([theta0], [psi0], c='k', s=50, zorder=5)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\psi(\theta)$')
plt.title(r'Legendre 변환: 기울기 $\eta_0$를 갖는 접선의 y절편 = $-\psi^*(\eta_0)$')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('legendre_tangent.png', dpi=120)
```

### 코드 5: $\theta \leftrightarrow \eta$ 미분동형 시각화 (베르누이)

```python
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(-5, 5, 200)
eta = 1/(1 + np.exp(-theta))

# 두 좌표에서의 Fisher
F_theta = eta * (1 - eta)       # = p(1-p)
F_star_eta = 1 / F_theta         # = 1/(p(1-p))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(theta, eta, 'b', lw=2)
axes[0].set_xlabel(r'$\theta$'); axes[0].set_ylabel(r'$\eta = \nabla\psi(\theta)$')
axes[0].set_title(r'$\theta \to \eta$ 미분동형')
axes[0].grid(alpha=0.3)

axes[1].plot(theta, F_theta, 'g', lw=2)
axes[1].set_xlabel(r'$\theta$'); axes[1].set_ylabel(r'$F(\theta)$')
axes[1].set_title(r'Fisher in $\theta$ coord')
axes[1].grid(alpha=0.3)

axes[2].plot(eta, F_star_eta, 'r', lw=2)
axes[2].set_xlabel(r'$\eta$'); axes[2].set_ylabel(r'$F^*(\eta)$')
axes[2].set_title(r'Fisher in $\eta$ coord ($F^*=1/F$)')
axes[2].set_yscale('log')
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dual_coords.png', dpi=120)
```

---

## 🔗 AI/ML 연결

### 1. Mirror Descent (Nemirovski 1983)

$$
\theta_{k+1} = \nabla\psi^*(\nabla\psi(\theta_k) - \eta_k g_k)
$$

즉 "$\theta$를 $\eta$로 옮겨 gradient step, 다시 $\theta$로 돌림". 이것은 **$\psi$가 cumulant인 지수족에서 natural gradient와 정확히 동치** (Ch7-02).

### 2. Exponentiated Gradient (Kivinen-Warmuth 1997)

심플렉스 $\Delta^K$에서 $\psi = \sum p_i \log p_i$ (음 엔트로피). Mirror descent는:

$$
p_{k+1,i} \propto p_{k,i} \exp(-\eta g_i)
$$

**Multiplicative update** — bandit, boosting에서의 Hedge·AdaBoost 알고리즘의 수학적 기반.

### 3. Softmax Layer의 기하학적 의미

Neural network의 softmax output을 **expectation parameter** $\eta = (p_1, \dots, p_K)$, 그 직전의 logit $z_i$를 **canonical parameter**로 보면 softmax = $\nabla\psi$. Cross-entropy loss = KL의 Legendre 표현 정리 3.10.

### 4. Variational Inference의 Closed-Form Updates

Mean-field VI에서 $q_i(\theta_i) = \exp(\phi_i^T T_i(\theta_i) - \psi_i(\phi_i))$로 지수족 가정. ELBO 최적화에서 coordinate ascent:

$$
\phi_i^{\text{new}} = \mathbb{E}_{q_{-i}}[\nabla_{\theta_i} \log p(\theta_i, \theta_{-i}, x)]
$$

$\phi$(canonical)에서 업데이트가 closed-form, 반면 $q_i$ 자체(expectation coord)에서는 적분 필요. Legendre 쌍대가 계산의 용이함을 결정.

### 5. Wasserstein GAN의 Dual

WGAN에서 $\sup_{\|f\|_L \le 1} \mathbb{E}_p[f] - \mathbb{E}_q[f]$ (Kantorovich-Rubinstein duality)는 Legendre 변환의 일반화. 비 지수족까지 확장된 쌍대성.

### 6. Logistic Regression = Bernoulli 지수족의 conditional

$\theta(x) = w^T x$로 매개변수화 → 조건부 지수족. cross-entropy loss는 $-\log p(y | x) = -y w^T x + \log(1 + e^{w^Tx}) = -y\theta(x) + \psi(\theta(x))$. $w$에 대해 볼록.

---

## ⚖️ 가정과 한계

### 가정

1. **regular minimal 지수족**: Legendre가 well-defined + $\nabla\psi$가 미분동형.
2. **$\Theta^\circ \neq \emptyset$**: $\Theta$의 내부 존재.
3. **엄격 볼록성**: $\nabla\psi$ 단사의 충분조건.

### 한계

1. **경계에서의 문제**: Bernoulli에서 $\eta = 0$ or $1$에 해당하는 $\theta = \pm\infty$는 $\mathcal{E}^\circ$의 경계. Legendre가 잘 정의되지 않음.

2. **curved exp family에서 실패**: $\mathcal{N}(\theta, \theta^2)$ 같은 곡선 부분다양체에서는 원래 $\theta$가 full 지수족의 자연 파라미터가 아니므로 위 공식들이 직접 적용되지 않음 — Amari의 α-connection 이론으로 확장 필요 (Ch4-04, 4-05).

3. **계산 어려움**: $\psi^*$를 analytic하게 구하려면 $\nabla\psi$의 역함수를 풀어야 함. 복잡한 multivariate 지수족(예: Wishart)에서 closed-form이 없을 수 있음.

4. **무한차원 RKHS 지수족**: $\psi$가 functional이면 Legendre 변환도 functional duality — 유한차원 이론이 그대로 성립하지 않음.

5. **비 지수족**: Cauchy, mixture, Laplace는 Legendre 구조 없음. 이 쌍대성은 **지수족의 특권**이다.

---

## 📌 핵심 정리

| 대상 | 공식 |
|------|--------|
| Legendre 정의 | $\psi^*(\eta) = \sup_\theta(\theta^T\eta - \psi(\theta))$ |
| Fenchel-Young | $\theta^T\eta \le \psi(\theta) + \psi^*(\eta)$, 등호 $\iff \eta = \nabla\psi(\theta)$ |
| 쌍대 좌표 미분동형 | $\nabla\psi: \Theta^\circ \to \mathcal{E}^\circ$ $C^\infty$-diffeomorphism |
| 역관계 | $\nabla\psi^* = (\nabla\psi)^{-1}$ |
| Fisher 쌍대 | $F^*(\eta) = \nabla^2\psi^*(\eta) = F(\theta)^{-1}$ |
| ψ* 통계적 의미 | $\psi^*(\eta) = $ 음 엔트로피 + base |
| KL 삼중 표현 | $\text{KL}(p_1\|p_2) = D_\psi(\theta_2, \theta_1) = D_{\psi^*}(\eta_1, \eta_2) = \psi(\theta_2) + \psi^*(\eta_1) - \theta_2^T\eta_1$ |
| 아핀 구조 | $\theta$ = e-affine, $\eta$ = m-affine (Ch4-04 예고) |

**한 줄 요약:** Legendre 변환이 $\theta \leftrightarrow \eta$를 전단사 대응시키고, **Fisher 행렬은 두 좌표에서 서로 역행렬**이며, **KL은 이 두 좌표의 혼합으로 표현**된다. 이것이 정보기하에서 "쌍대성"이 의미하는 전부이다.

---

## 🤔 생각해볼 문제

1. **(경계의 Legendre)** Bernoulli에서 $\eta = 0$일 때 $\psi^*(0) = ?$ $\theta \to -\infty$에서 $\theta \cdot 0 - \psi(\theta) \to -\psi(\theta) \to 0$. 따라서 $\psi^*(0) = 0$. 이것이 엔트로피 $-H(p) = p\log p + (1-p)\log(1-p)$의 $p=0$에서의 한계값과 일치함을 확인.

2. **(Legendre가 involution)** $\psi^{**} = \psi$. 지수족 $\psi$에 대해 $\psi^{**} = \psi$임을 증명하고, 이것이 **지수족의 쌍대 역시 지수족**(mixture family)임과 어떻게 연결되는가?

3. **(Poisson Legendre)** Poisson $\psi(\theta) = e^\theta$의 Legendre 쌍대를 계산. $\eta = e^\theta = \lambda$, $\psi^*(\eta) = \theta\eta - e^\theta = \eta\log\eta - \eta$. 이것이 Poisson의 음 엔트로피와 일치함을 확인 ($H = -\sum p_k\log p_k$ 계산).

4. **(Multinomial vs Dirichlet Legendre)** Multinomial의 $\psi(\theta) = \log(1 + \sum e^{\theta_i})$의 Legendre는 $\psi^*(\eta) = \sum \eta_i \log\eta_i + (1-\sum\eta_i)\log(1-\sum\eta_i)$ 심플렉스 음 엔트로피. Dirichlet과 어떤 관계?

5. **(Dual Fisher 직관)** Bernoulli에서 $F(\theta) = p(1-p)$, $F^*(\eta) = 1/(p(1-p))$. $p \to 0.5$에서 $F$ 최대, $F^*$ 최소. 이것이 정보이론적으로 "분류 엔트로피가 최대일 때 Fisher가 최대"와 어떻게 연결되는가?

6. **(Mirror Descent = Natural Gradient)** Mirror descent 업데이트 $\theta_{k+1} = \nabla\psi^*(\nabla\psi(\theta_k) - \eta g_k)$가 Natural gradient $\theta_{k+1} = \theta_k - \eta F^{-1} g_k$와 **first-order 동치**임을 증명. (힌트: $\nabla\psi(\theta_k) - \eta g_k$를 $\theta_k$ 근방에서 Taylor 전개.)

7. **(Legendre와 Fisher Information Inequality)** Cramér-Rao (Ch2-05)를 Legendre 쌍대 관점에서 재서술: expectation parameter 추정량 $\hat\eta$의 $\eta$ 공간에서 CR 하한은? $\text{Var}(\hat\eta) \succeq F^*(\eta)^{-1} = F(\theta)$ — 원래 $F$ 자체가 상한이 됨.

8. **(KL은 왜 비대칭인가?)** 정리 3.10의 $\text{KL}(p_1\|p_2) = \psi(\theta_2) + \psi^*(\eta_1) - \theta_2^T\eta_1$에서 $p_1 \leftrightarrow p_2$ 교환 시 **$\theta$-정보는 $p_2$, $\eta$-정보는 $p_1$**로 혼합되어 나오는 점을 관찰. 비대칭성의 기하학적 원인은?

---

<div align="center">

| [◀ 02. Cumulant의 볼록성](./02-cumulant-convexity.md) | [📚 메인 README](../README.md) | [04. e/m-connection ▶](./04-e-m-connection.md) |
|:---:|:---:|:---:|

</div>
