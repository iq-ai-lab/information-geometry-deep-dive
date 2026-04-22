# 01. Euclidean Gradient의 Parameterization 의존성 — 왜 $\nabla L$은 "방향"이 아닌가

<div align="center">

> *"Gradient descent는 모두가 쓴다.*  
> *하지만 $\nabla L$이 어떤 방향을 가리키는지 정확히 답할 수 있는 사람은 드물다.*
>
> *사실 $\nabla L$은 방향이 아니다 — 그것은 좌표 선택에 의존하는, parameterization의 유물(artifact)이다.*  
> $\sigma$로 파라미터화하느냐 $\log\sigma$로 하느냐에 따라 gradient 방향이 **서로 다르다**.*
>
> *이것이 Natural Gradient의 필요성 — 좌표계에 독립적인 gradient를 만들기 위한 여정의 출발점이다."*

</div>

---

## 🎯 핵심 질문

1. **Gradient는 벡터인가, 공벡터(covector)인가?** 유클리드 공간에서는 혼동되지만 일반 리만 다양체에서는 서로 다른 대상. 이 차이는 무엇이고 왜 중요한가?
2. 같은 목적함수를 두 좌표계로 표현했을 때 gradient descent 경로가 **다른 이유**: 구체적 예로 $\mathcal{N}(\mu, \sigma^2)$를 $\sigma$ 좌표 vs $\log\sigma$ 좌표로 최적화하면 무엇이 달라지는가?
3. **"Riemannian gradient"**의 정의: 리만 계량 $g$ 하에서 $\text{grad}\,L$이 어떻게 $\nabla L$(편미분 벡터)과 다른가?
4. **"Euclidean"에서만 위 두 개가 일치**하는 이유와 $g = I$ 가정이 깨지는 통계 다양체에서 무엇이 일어나는가?
5. **Natural Gradient의 정의**: $\tilde\nabla L = F^{-1}\nabla L$이 왜 "옳은" gradient인가?

---

## 🔍 왜 이 문제가 AI에서 중요한가

| AI/ML 기법 | Gradient 문제가 하는 일 |
|-----------|-----------------|
| **Learning Rate 튜닝** | Parameterization에 따라 optimal learning rate가 달라짐 — hyperparameter 튜닝의 근본 원인 |
| **Adam/RMSProp의 효과** | Adam의 per-parameter scaling은 Fisher의 대각 근사 — 정당한 수학적 이유 |
| **LoRA, Adapter 튜닝** | 재매개변수화가 수렴 속도에 극적 영향 — Natural gradient 관점에서 왜 그런지 설명 |
| **BatchNorm, LayerNorm** | Normalization은 암묵적으로 Fisher를 대각화하는 재매개변수화 |
| **Softmax 경계의 saturation** | $\sigma$ 좌표에서 Euclidean GD가 saturating region에서 stuck — Natural gradient는 해결 |
| **VAE reparameterization trick** | $\sigma$ vs $\log\sigma$ parameterization 차이 — variance 학습 안정성에 영향 |

---

## 📐 수학적 선행 조건

| 개념 | 참조 |
|------|------|
| **Gradient의 이중 성격** (co- vs contra-variant) | Ch1-02, Calculus & Optimization Deep Dive Ch5 |
| **리만 계량**과 raise/lower index | Ch1-03 |
| **Chain rule**과 Jacobian의 covariance | Calculus Deep Dive Ch3 |
| Fisher 정보 행렬 $F = \nabla^2\psi$ | **Ch2-02, Ch4-02** |
| 쌍대평탄 구조와 $\theta \leftrightarrow \eta$ | **Ch4-03, Ch4-05** |

---

## 📖 직관적 이해

### 1. "Gradient가 방향이다"는 유클리드 특권

유클리드 $\mathbb{R}^n$에서 standard basis와 standard inner product $\langle u, v\rangle = u^T v$를 쓰면, 함수 $L(\theta)$의 편미분 벡터 $\nabla L = (\partial_1 L, \dots, \partial_n L)$이 **steepest ascent 방향** = "gradient".

하지만 이것은 두 가지 특수 가정에 의존:
- **좌표계가 직교-정규**: basis vectors가 orthonormal w.r.t. $\langle,\rangle$.
- **계량이 $I$**: $g_{ij} = \delta_{ij}$.

리만 다양체에서 두 가정이 모두 무너진다.

### 2. Covector vs Vector

미분기하에서 ($L: M \to \mathbb{R}$ 매끈 함수):
- **Differential** $dL = \partial_i L \, d\theta^i$는 **공벡터(covector, 1-form)**: $T^*_p M$의 원소, 방향에 "미분값"을 주는 함수.
- **Gradient** $\text{grad}\,L$는 **벡터**: $T_p M$의 원소, 방향을 가리킨다.

두 대상의 관계는 계량 $g$에 의해:

$$
g(\text{grad}\,L, V) = dL(V) \quad \forall V \in T_p M
$$

좌표로:

$$
(\text{grad}\,L)^i = g^{ij} \partial_j L
$$

$g^{ij}$는 $g_{ij}$의 역행렬. 유클리드에서 $g^{ij} = \delta^{ij}$이므로 $(\text{grad}\,L)^i = \partial_i L$, 즉 두 개가 같다. **이것은 특수한 일치**일 뿐이다.

### 3. Parameterization 바꾸기

$\theta$ 좌표에서 $\phi = A\theta$ 선형 재매개변수화 (가역 $A$). Chain rule:

$$
\tilde\partial_i L = \frac{\partial L}{\partial \phi^i} = \frac{\partial \theta^j}{\partial \phi^i}\partial_j L = (A^{-T})_{ij}\partial_j L
$$

즉 **covector로 변환** (역전치 Jacobian).

반면 steepest descent vector (원래 공간에서 실제 방향)은 **vector**이고 변환:

$$
\tilde v^i = (A)_{ij} v^j
$$

만약 우리가 $\tilde\partial L$을 vector처럼 취급하여 $\phi \to \phi - \eta \tilde\partial L$로 업데이트한다면, 이것은 **잘못된 방향을 가리키는** 벡터를 방향 업데이트로 쓰는 것이다. 두 방향 $\nabla L$과 $\tilde\nabla L$는 $A^{-T}$로 관계, **gradient 경로 전체가 달라진다**.

### 4. 정규분포 예시 — 핵 직관

$\mathcal{N}(\mu, \sigma^2)$에서 $L = \text{KL}(N(\mu, \sigma^2) \| N(0, 1))$를 최소화. 두 좌표계:

**좌표 A ($\mu, \sigma$)**: $L_A(\mu, \sigma) = \log(1/\sigma) + (\sigma^2 + \mu^2)/2 - 1/2$. $\nabla L_A = (\mu, \sigma - 1/\sigma)$.

**좌표 B ($\mu, \log\sigma$)**: $\tau = \log\sigma$. $L_B(\mu, \tau) = -\tau + (e^{2\tau} + \mu^2)/2 - 1/2$. $\nabla L_B = (\mu, -1 + e^{2\tau})$.

두 gradient를 "방향"으로 쓰면 경로가 다름:
- 좌표 A: $\sigma$ 근처가 작을 때 $\sigma - 1/\sigma \approx -1/\sigma$ → 큰 negative step (divergent)
- 좌표 B: $-1 + e^{2\tau} \approx -1$ → bounded step (안정)

$\log\sigma$ 좌표가 **더 잘 작동**한다. 이 차이가 **parameterization에 의한 gradient의 imprinted bias**이다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 (Differential, 1-form)

매끈 함수 $L: M \to \mathbb{R}$의 $p \in M$에서 **differential** $dL_p: T_p M \to \mathbb{R}$:

$$
dL_p(V) := V(L) = V^i \partial_i L(p)
$$

좌표 표현 $dL = \partial_i L \, d\theta^i$ (covector).

### 정의 1.2 (Gradient w.r.t. Metric)

리만 $(M, g)$에서 $L$의 **gradient** $\text{grad}\,L \in T_p M$:

$$
g_p(\text{grad}\,L, V) = dL_p(V) \quad \forall V \in T_p M
$$

좌표: $(\text{grad}\,L)^i = g^{ij}\partial_j L$.

### 정의 1.3 (Euclidean Gradient)

$\theta \in \mathbb{R}^n$에서 $g = I$로 가정한 경우:

$$
\nabla L(\theta) := (\partial_1 L, \dots, \partial_n L)^T
$$

이것은 differential의 계산(편미분)일 뿐, **계량-의존적 vector는 아니다**. 유클리드에서만 gradient과 일치.

### 정의 1.4 (Natural Gradient)

통계다양체에서 Fisher 계량 $g = F$를 쓴 gradient:

$$
\boxed{\;\tilde\nabla L(\theta) := F^{-1}(\theta) \nabla L(\theta) = \text{grad}^{F}\,L\;}
$$

**Amari 1998**의 용어. 이것이 "parameterization 불변"인 gradient (Ch5-04에서 완전 증명).

### 정의 1.5 (Steepest Descent Direction w.r.t. Metric)

$L$을 $p$에서 단위 ball $\{V : g(V, V) \le 1\}$ 안에서 가장 크게 감소시키는 방향:

$$
V^* = -\frac{\text{grad}\,L}{\|\text{grad}\,L\|_g}
$$

유클리드에서 $V^* = -\nabla L / \|\nabla L\|$; 리만에서 $V^* = -F^{-1}\nabla L / \|F^{-1}\nabla L\|_F$.

---

## 🔬 정리와 증명

### 정리 1.6 (Gradient의 좌표 변환 — Covector 규칙)

$\phi = \phi(\theta)$ 재매개변수화 ($\phi: \Theta \to \Phi$ 미분동형). $L(\theta) = \tilde L(\phi(\theta))$. 그러면

$$
\partial^\theta_i L = \frac{\partial\phi^j}{\partial\theta^i}\partial^\phi_j \tilde L
$$

즉 **covector 변환**.

**증명.** 체인룰. $\blacksquare$

**귀결.** 두 좌표계의 Euclidean gradient $\nabla^\theta L$과 $\nabla^\phi \tilde L$은 $\nabla^\theta L = J^T \nabla^\phi \tilde L$ where $J = \partial\phi/\partial\theta$. **서로 다른 벡터**.

### 정리 1.7 (Natural Gradient의 좌표 변환 — Vector 규칙)

$\phi = \phi(\theta)$ 재매개변수화에서 Fisher $F^\theta = J^T F^\phi J$ (Ch2-03 정리 4.3). Natural gradient:

$$
\tilde\nabla^\theta L = (F^\theta)^{-1}\nabla^\theta L = (J^T F^\phi J)^{-1} J^T \nabla^\phi \tilde L = J^{-1}(F^\phi)^{-1}(J^T)^{-1} J^T \nabla^\phi \tilde L = J^{-1}(F^\phi)^{-1}\nabla^\phi\tilde L = J^{-1}\tilde\nabla^\phi\tilde L
$$

즉 **vector 변환** ($V^i$ 변환 규칙).

**귀결** (Ch5-04에서 확장): Natural gradient의 **방향**은 두 좌표계에서 동일한 물리적 방향을 가리킨다.

### 정리 1.8 (Euclidean GD 경로는 좌표 의존)

$L, g$가 주어졌을 때, 두 좌표계 $\theta, \phi$의 Euclidean GD $\theta_{k+1} = \theta_k - \eta \nabla^\theta L$과 $\phi_{k+1} = \phi_k - \eta \nabla^\phi \tilde L$은 **일반적으로 다른 궤적**을 만든다 (즉 $\phi(\theta_{k+1}) \neq \phi_{k+1}$).

**증명 (스케치).** $\phi_{k+1} \stackrel{?}{=} \phi(\theta_{k+1}) = \phi(\theta_k - \eta\nabla^\theta L) \approx \phi(\theta_k) - \eta J \nabla^\theta L + O(\eta^2)$. $J \nabla^\theta L = J J^T \nabla^\phi \tilde L = (J J^T)\nabla^\phi\tilde L$. 만약 $\phi_{k+1} = \phi_k - \eta\nabla^\phi\tilde L$을 원한다면 $JJ^T = I$ (즉 $J$ 직교)가 필요. 일반 재매개변수화에서 $JJ^T \neq I$이므로 두 경로 다름. $\blacksquare$

**예 (Normal)**: $\sigma \to \log\sigma$. Jacobian $J = d\log\sigma/d\sigma = 1/\sigma$. $JJ^T = 1/\sigma^2 \neq 1$. 경로 다름.

### 정리 1.9 (Natural GD 경로의 불변성)

Natural gradient update $\theta_{k+1} = \theta_k - \eta F^{-1}(\theta_k)\nabla L(\theta_k)$는 (first-order infinitesimal) 좌표 변환 $\phi = \phi(\theta)$ 하에서 불변: $\phi_{k+1} \approx \phi(\theta_{k+1})$.

**증명 (first-order).** $\phi_{k+1} = \phi_k - \eta (F^\phi)^{-1}\nabla^\phi \tilde L$. $(F^\phi)^{-1} = J(F^\theta)^{-1}J^T$, $\nabla^\phi \tilde L = J^{-T}\nabla^\theta L$. 따라서 $-\eta (F^\phi)^{-1}\nabla^\phi\tilde L = -\eta J(F^\theta)^{-1}J^T J^{-T}\nabla^\theta L = -\eta J(F^\theta)^{-1}\nabla^\theta L = J \cdot \tilde\nabla^\theta L \cdot (-\eta) = J (\theta_{k+1} - \theta_k)$. 정확한 first-order로 $\phi(\theta_{k+1}) - \phi_k \approx J(\theta_{k+1} - \theta_k) = -\eta J(F^\theta)^{-1}\nabla^\theta L = $ 위 계산과 일치. $\blacksquare$

(전체 경로의 정확한 불변성은 geodesic flow에서 보장 — Ch5-04에서 확장.)

### 정리 1.10 ($L$의 2차 근사 — quadratic bound)

유클리드 GD $\theta_{k+1} = \theta_k - \eta \nabla L$의 loss decrease:

$$
L(\theta_{k+1}) \approx L(\theta_k) - \eta \|\nabla L\|^2 + \frac{\eta^2}{2}\nabla L^T H \nabla L
$$

where $H = \nabla^2 L$는 Hessian. 최적 learning rate $\eta^* = \|\nabla L\|^2 / (\nabla L^T H \nabla L)$, decrease $\approx \frac{1}{2}\|\nabla L\|^2/\lambda_{\max}(H)$.

**Natural GD** $\theta_{k+1} = \theta_k - \eta F^{-1}\nabla L$:

$$
L(\theta_{k+1}) \approx L(\theta_k) - \eta \nabla L^T F^{-1}\nabla L + \frac{\eta^2}{2}\nabla L^T F^{-1} H F^{-1}\nabla L
$$

지수족 MLE에서 $H = F$ (Fisher identity)면 decrease $= -\eta \nabla L^T F^{-1}\nabla L + \frac{\eta^2}{2}\nabla L^T F^{-1}\nabla L$. 최적 $\eta^* = 1$, decrease $= \frac{1}{2}\nabla L^T F^{-1}\nabla L$. **한 스텝으로 수렴**! (Newton's method 재발견.)

### 정리 1.11 (NGD = Newton for MLE in Exp Family)

지수족 MLE에서 $L(\theta) = -\log p(x|\theta)$의 헤시안 $H = \nabla^2 L = -\nabla^2 \log p = \nabla^2 \psi = F$. 따라서

$$
\theta - F^{-1}\nabla L = \theta - H^{-1}\nabla L = \text{Newton step}
$$

즉 **NGD = Newton's method in exponential family MLE**.

**증명.** 위 전개에서 지수족 $\log p = \theta^T T - \psi$, $\nabla^2_\theta \log p = -\nabla^2\psi = -F$. $L = -\log p$, $\nabla^2 L = F$. $\blacksquare$

---

## 💻 NumPy / SymPy 구현으로 검증

### 코드 1: 정규 분포에서 $\sigma$ vs $\log\sigma$ 비교

```python
import numpy as np
import matplotlib.pyplot as plt

# Target: N(μ*, σ²*) = N(2, 2²), minimize KL(N(μ,σ²) || N(μ*, σ*²))
mu_star, s_star = 2.0, 2.0

def KL(mu, s, mu2, s2):
    return np.log(s2/s) + (s**2 + (mu-mu2)**2)/(2*s2**2) - 0.5

# 좌표 A: (μ, σ)
def grad_A(mu, s):
    # d/dμ KL = (μ - μ*) / σ*²
    # d/dσ KL = -1/σ + σ/σ*² - 0 (∵ (σ²)/(2σ*²) 미분 = σ/σ*²)
    dmu = (mu - mu_star) / s_star**2
    ds  = -1/s + s/s_star**2
    return np.array([dmu, ds])

# 좌표 B: (μ, τ=logσ)
def grad_B(mu, tau):
    s = np.exp(tau)
    # d/dμ = (μ - μ*) / σ*²
    # d/dτ = d(-τ + (σ²+...)/(2σ*²) - 0.5)/dτ = -1 + (2σ·σ)/(2σ*²) · (dσ/dτ ... but σ=e^τ, so dσ²/dτ = 2σ²)
    # 실제: KL_B(μ, τ) = -τ + (e^{2τ} + (μ-μ*)²)/(2σ*²) - 0.5 + logσ*
    dmu = (mu - mu_star) / s_star**2
    dtau = -1 + np.exp(2*tau)/s_star**2
    return np.array([dmu, dtau])

# 실행
def gd_A(mu0, s0, lr=0.1, steps=100):
    traj = [(mu0, s0)]
    mu, s = mu0, s0
    for _ in range(steps):
        g = grad_A(mu, s)
        mu -= lr * g[0]
        s  = max(1e-3, s - lr * g[1])
        traj.append((mu, s))
    return np.array(traj)

def gd_B(mu0, s0, lr=0.1, steps=100):
    tau = np.log(s0)
    traj = [(mu0, np.exp(tau))]
    mu = mu0
    for _ in range(steps):
        g = grad_B(mu, tau)
        mu -= lr * g[0]
        tau -= lr * g[1]
        traj.append((mu, np.exp(tau)))
    return np.array(traj)

trajA = gd_A(0.5, 0.3)
trajB = gd_B(0.5, 0.3)

plt.figure(figsize=(9, 6))
plt.plot(trajA[:, 0], trajA[:, 1], 'r-o', markersize=3, label=r'GD in $(\mu, \sigma)$')
plt.plot(trajB[:, 0], trajB[:, 1], 'b-o', markersize=3, label=r'GD in $(\mu, \log\sigma)$')
plt.plot(mu_star, s_star, 'k*', markersize=20, label='target')
plt.xlabel(r'$\mu$'); plt.ylabel(r'$\sigma$')
plt.title(r'같은 KL objective, 서로 다른 좌표 → 서로 다른 경로')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gd_paths_sigma_vs_logsigma.png', dpi=120)
# 두 경로가 완전히 다름. $\log\sigma$가 보통 더 안정적.
```

### 코드 2: Gradient의 공벡터 성질 — $A\theta$ 변환

```python
import numpy as np

# L(θ) = θᵀ Q θ, θ = (θ₁, θ₂)
Q = np.array([[1.0, 0.5], [0.5, 2.0]])
theta = np.array([1.0, 1.0])
grad_theta = 2 * Q @ theta  # ∇L in θ coord

# 재매개변수화: φ = A θ
A = np.array([[1, 2], [0, 3]])
phi = A @ theta
# L(θ(φ)) = (A⁻¹ φ)ᵀ Q (A⁻¹ φ) = φᵀ (A⁻ᵀ Q A⁻¹) φ
Q_phi = np.linalg.inv(A).T @ Q @ np.linalg.inv(A)
grad_phi = 2 * Q_phi @ phi  # ∇L in φ coord

# Covector 관계: grad_phi = A⁻ᵀ grad_θ
expected = np.linalg.inv(A).T @ grad_theta
print(f"grad_φ (직접)   = {grad_phi}")
print(f"A⁻ᵀ grad_θ     = {expected}")
print(f"diff          = {np.abs(grad_phi - expected).max():.2e}")  # 0

# "Vector"로 변환했다면 (잘못된 취급):
vector_transformed = A @ grad_theta
print(f"A grad_θ (vector 변환 — 틀림) = {vector_transformed}")  # 다름
```

### 코드 3: Natural Gradient의 vector 변환

```python
import numpy as np

# 지수족 Gaussian의 Fisher (θ coord, full): F = Hessian ψ
# 간단화: 1D Gaussian σ² fixed, only μ varies (θ=μ, F=1/σ²=1)
# 2D 예시를 위해: Bernoulli 2개의 곱 → 2D exp family
theta = np.array([0.3, 0.7])  # μ₁, μ₂ params (Gaussian fixed sigma=1)
F_theta = np.diag([1.0, 1.0])  # = I  (σ²=1 Gaussian, F=1/σ²)
grad_theta_L = np.array([0.1, -0.2])  # 임의 예시
nat_grad_theta = np.linalg.solve(F_theta, grad_theta_L)

# 재매개변수화: φ = 2θ (선형)
J = 2 * np.eye(2)  # Jacobian ∂φ/∂θ = 2I
# F_φ = J⁻ᵀ F_θ J⁻¹ = (1/4) F_θ (유도)
# 실제: tensor law - for covariant F, F_ij^φ = (∂θ/∂φ)^i (∂θ/∂φ)^j F_kl^θ = (1/4)F_θ
F_phi = (1/4) * F_theta  
# ∇^φ L = J⁻ᵀ ∇^θ L = (1/2) ∇^θ L
grad_phi_L = (1/2) * grad_theta_L
# Natural gradient in φ coord
nat_grad_phi = np.linalg.solve(F_phi, grad_phi_L)

# Vector 변환 확인: ~∇^φ = J ~∇^θ = 2 * nat_grad_theta
print(f"Natural gradient in θ coord: {nat_grad_theta}")
print(f"Natural gradient in φ coord: {nat_grad_phi}")
print(f"J * ~∇^θ                    : {J @ nat_grad_theta}")
print(f"diff                         : {np.abs(nat_grad_phi - J @ nat_grad_theta).max():.2e}")  # 0
```

### 코드 4: NGD = Newton in exp family

```python
import numpy as np

# Bernoulli likelihood: -log p(x|θ) = -θx + log(1 + e^θ)
# 데이터: x̄ = 관측된 평균
x_bar = 0.7
def L(theta):
    return -theta * x_bar + np.log1p(np.exp(theta))
def dL(theta):
    return -x_bar + 1/(1 + np.exp(-theta))
def F(theta):
    p = 1/(1 + np.exp(-theta))
    return p * (1 - p)

# Hessian of L = F (exp family)
def ddL(theta):
    p = 1/(1 + np.exp(-theta))
    return p * (1 - p)

# NGD step = Newton step
theta0 = 0.0
# Newton
theta_newton = theta0 - dL(theta0) / ddL(theta0)
# NGD
theta_ngd = theta0 - dL(theta0) / F(theta0)
print(f"Newton step: {theta_newton}")
print(f"NGD step   : {theta_ngd}")
# 두 값이 정확히 같음 — NGD = Newton in exp family MLE

# 수렴점?
# dL = 0 → 1/(1+e^{-θ}) = x̄ → θ* = logit(0.7) = log(7/3)
print(f"Closed-form MLE: {np.log(x_bar/(1-x_bar)):.4f}")
print(f"One NGD step from θ₀=0: {theta_ngd:.4f}")
# 한 스텝으로 근사
```

### 코드 5: 좌표-의존적 saturation

```python
import numpy as np
import matplotlib.pyplot as plt

# softmax output p = σ(z). $L = -\log p$ = softplus(-z)
# z 좌표에서는 gradient = -σ(-z) = -1/(1+e^z) → well-bounded
# p 좌표에서는 L = -log p, dL/dp = -1/p → saturated at p=0 (divergent)

ps = np.linspace(0.01, 0.99, 100)
zs = np.log(ps/(1-ps))  # logit

grad_p = -1/ps  # in p coord (잘못된 사용 시)
grad_z = -1/(1 + np.exp(zs))  # in z coord

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ps, grad_p, 'r-', lw=2)
axes[0].set_xlabel(r'$p$ (probability)'); axes[0].set_ylabel(r'$\nabla L$')
axes[0].set_title('p 좌표: saturating at $p \\to 0$')
axes[0].grid(alpha=0.3); axes[0].set_ylim(-100, 0)
axes[1].plot(zs, grad_z, 'b-', lw=2)
axes[1].set_xlabel(r'$z = \text{logit}(p)$'); axes[1].set_ylabel(r'$\nabla L$')
axes[1].set_title(r'$z$ 좌표: bounded in $[-1, 0]$')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('saturation_comparison.png', dpi=120)
# logit 좌표가 훨씬 안정 — 이것이 cross-entropy loss를 logit에 쓰는 이유
```

---

## 🔗 AI/ML 연결

### 1. Softmax + Cross-Entropy Loss의 수학적 정당성

분류 문제에서 softmax output $p = \text{softmax}(z)$ 대신 **logit $z$ 좌표**에서 Cross-Entropy를 계산하는 것이 수치적으로 안정. 이것은 지수족 canonical parameter $z$가 natural scale임을 reflectly 이용. Euclidean GD in $z$ 좌표 ≈ Natural gradient in $p$ 좌표.

### 2. BatchNorm의 기하학적 효과

BatchNorm이 activation을 $(mean=0, std=1)$로 정규화하는 것은 **각 레이어에서 Fisher를 $I$에 가깝게 만드는 재매개변수화**. 결과로 Euclidean GD가 natural gradient의 근사가 됨. He-initialization, LayerNorm 모두 비슷한 역할.

### 3. VAE의 log-variance trick

VAE encoder $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$에서 $\sigma$ 직접 출력 대신 $\log\sigma$ 출력 관습:

```python
mu, log_sigma = encoder(x)
sigma = exp(log_sigma)
```

이유: $\log\sigma$ 좌표에서 gradient flow가 안정 (위 코드 1 예제가 정확히 이 현상).

### 4. LoRA, Adapter 튜닝

Full parameter $\theta \in \mathbb{R}^{d \times k}$ 대신 low-rank $\theta = \theta_0 + AB^T$ ($A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{k \times r}$). 이것은 **재매개변수화**이며, 해당 subspace의 natural gradient가 full space gradient와 다르게 작동. LoRA의 효과 = 이 재매개변수화의 Fisher 구조를 활용.

### 5. Adam, RMSProp의 수학적 해석

Adam의 per-parameter adaptive learning rate $g / \sqrt{\mathbb{E}[g^2]}$는 **Fisher의 대각 근사** $\hat F_{ii} = \mathbb{E}[g_i^2]$에 의한 natural gradient의 대각 근사. 다만 moment matching이 정확하지 않아 완벽하지 않음. K-FAC, Shampoo는 이를 개선 (Ch5-05).

### 6. 학습 초기 failure modes

Training 초기 "gradient vanishing"이나 "exploding" 현상은 대개 **parameterization이 Fisher의 특성을 반영하지 않음**에서 비롯. Natural gradient는 이론적으로 이를 자동 해결.

---

## ⚖️ 가정과 한계

### 가정

1. **Smooth parameterization**: $\theta \to \phi$ $C^1$ 미분동형. 경계점, non-smooth에서 처리 복잡.
2. **Fisher well-defined**: 통계다양체 구조 (Ch2-01 R1-R7) 필요.
3. **First-order analysis**: 위 정리들이 infinitesimal (step $\eta \to 0$) 수준.

### 한계

1. **Fisher 계산 비용**: $\dim\theta = d$에 대해 $F$ 저장 $O(d^2)$, 역행렬 $O(d^3)$. Neural net ($d = 10^9$)에서 **직접 적용 불가**. K-FAC, Shampoo의 근사 필요 (Ch5-05).

2. **Global behavior 보장 안됨**: Natural gradient의 불변성은 first-order. 큰 step에서 geodesic을 정확히 따르지 않으면 경로 달라질 수 있음 (exponential map 필요, 보통 구현 안됨).

3. **Fisher singularity**: 어떤 $\theta$에서 $F$가 rank-deficient (예: overparameterized NN) → $F^{-1}$ 존재 안 함. Pseudoinverse, damping 등으로 우회.

4. **Noise in stochastic estimate**: 미니배치에서 $F$ estimate에 noise → NGD의 unbiased estimator가 어려움.

5. **Euclidean GD의 "좋은" 좌표**: 실무에서 적절한 parameterization(normalize, whiten)이면 Euclidean GD가 충분히 잘 작동 → natural gradient의 실용적 이점 작을 수 있음.

---

## 📌 핵심 정리

| 대상 | 공식 / 사실 |
|------|---------|
| Differential (covector) | $dL = \partial_i L \, d\theta^i$ |
| Gradient (vector) | $(\text{grad}\,L)^i = g^{ij}\partial_j L$ |
| Euclidean special | $g = I \Rightarrow \nabla L = \text{grad}\,L$ |
| 좌표 변환 (covector) | $\nabla^\phi = J^{-T}\nabla^\theta$ |
| 좌표 변환 (vector) | $V^\phi = J V^\theta$ |
| Euclidean GD 경로 | 좌표 의존 (정리 1.8) |
| Natural gradient | $\tilde\nabla L = F^{-1}\nabla L = \text{grad}^F L$ |
| Natural GD 경로 | 좌표 불변 (first-order, 정리 1.9) |
| NGD = Newton (exp family MLE) | $F = H$ for MLE → NGD = Newton |
| 2차 수렴 | NGD 지수족 MLE에서 한 스텝으로 수렴 |

**한 줄 요약:** Euclidean gradient $\nabla L$은 **covector**이고 좌표에 의존한다. "올바른" gradient는 Fisher 계량 하의 $\text{grad}^F L = F^{-1}\nabla L$ = **natural gradient**이며, 이것이 parameterization 불변이다.

---

## 🤔 생각해볼 문제

1. **(PyTorch 실험)** `torch.optim.SGD`를 $\sigma$와 $\log\sigma$ 좌표에서 각각 돌려 수렴 속도·안정성 비교. 차이의 근본 원인 설명.

2. **(Whitening과 Natural Gradient)** Data $X$를 whiten ($X \to \Sigma^{-1/2}X$)하면 linear regression의 Fisher가 $I$가 됨을 증명. 이것이 whitening preprocessing의 "natural gradient" 설명.

3. **(K-FAC 원리)** Neural net layer에서 weight $W$와 activation $a$의 Kronecker 구조 $F \approx \mathbb{E}[aa^T] \otimes \mathbb{E}[gg^T]$. 이 근사가 정확한 조건?

4. **(Mirror descent의 parameterization 불변성)** Mirror descent step $\theta_{k+1} = \nabla\psi^*(\nabla\psi(\theta_k) - \eta g_k)$이 (Bregman $\psi$가 고정되면) coordinate change에 어떻게 반응하는가?

5. **(Adam의 bias)** Adam의 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2$는 $\mathbb{E}[g^2]$의 추정. 그러나 $\mathbb{E}[g^2] \neq F_{ii}$ (diagonal). 이 **bias**의 크기와 NGD와의 괴리를 분석.

6. **(Softmax와 logit의 정확한 관계)** cross-entropy loss $L(z) = -\log\text{softmax}(z)_{y}$의 $z$-gradient $\partial L/\partial z_i = p_i - \mathbb{1}[i=y]$. 이것이 **왜** natural gradient의 모습인가? 지수족 canonical coord에서의 MLE 관점으로 설명.

7. **(LoRA의 subspace geometry)** $\theta = \theta_0 + AB^T$의 **tangent space**에서 Fisher의 restriction이 어떻게 계산되는가? Full-space NGD를 LoRA subspace로 project하는 게 LoRA's natural gradient?

8. **(Geodesic NGD)** 실제 natural gradient의 "완전한" 구현은 $\theta_{k+1} = \text{Exp}_{\theta_k}(-\eta F^{-1}\nabla L)$ (geodesic exponential map 사용). 선형 $\theta - \eta F^{-1}g$의 근사가 언제 실패하는가?

---

<div align="center">

| [◀ Ch4-06. 일반화 Pythagoras](../ch4-exponential-duality/06-generalized-pythagoras.md) | [📚 메인 README](../README.md) | [02. Natural Gradient 유도 ▶](./02-natural-gradient-derivation.md) |
|:---:|:---:|:---:|

</div>
