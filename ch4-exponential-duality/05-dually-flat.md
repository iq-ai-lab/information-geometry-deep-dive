# 05. 쌍대평탄(Dually Flat) — Amari 정보기하의 핵심 구조

<div align="center">

> *"한 다양체가 두 좌표계 모두에서 평평한 경우가 있다.*  
> *하나는 $\theta$, 다른 하나는 $\eta$. 각각에서 크리스토펠이 0.*
>
> *이 구조를 '쌍대평탄(dually flat)'이라 부르며,  
> Amari의 모든 정보기하학은 이 한 단어로 요약된다.  
> 일반화 Pythagoras, projection 이론, α-divergence, EM의 기하학적 해석 —  
> 모두 쌍대평탄성의 직접적 귀결이다."*

</div>

---

## 🎯 핵심 질문

1. **쌍대평탄(dually flat) 다양체**의 엄밀한 정의는? 단순히 "두 flat connection"이 아닌, 추가 조건은?
2. 지수족이 쌍대평탄이라는 사실을 완전히 증명. 그 역(쌍대평탄 다양체가 지수족과 동형)도 어디까지 성립하는가?
3. **Canonical divergence** $D(p\|q) = \psi(\theta_q) + \psi^*(\eta_p) - \theta_q^T\eta_p$가 어떻게 쌍대평탄성에서 유일하게 튀어나오는가?
4. **α-connection이 flat**인 것은 오직 $\alpha = \pm 1$뿐. 이것의 기하학적 의미와 $\alpha=0$ Levi-Civita의 non-zero curvature의 존재 이유는?
5. 쌍대평탄성의 **non-statistical 응용**: 열역학, 양자 정보, 최적 수송(OT)에서 어떻게 나타나는가?

---

## 🔍 왜 이 기하학이 AI에서 중요한가

| AI/ML 기법 | 쌍대평탄성이 하는 일 |
|-----------|----------------|
| **Natural Gradient** | $\theta$에서 $F^{-1}g$는 $\eta$에서 euclidean gradient — 쌍대성이 parameterization 불변성을 준다 |
| **EM의 monotone convergence** | e-projection와 m-projection의 교대가 KL을 **단조 감소**시키는 이유 (쌍대성 + Pythagoras) |
| **Variational Inference** | Mean-field family가 쌍대평탄 부분다양체 → KL 최소화의 closed form |
| **Mirror Descent = NGD** | Exp family에서 두 알고리즘이 동치인 이유가 쌍대평탄성 |
| **Information Bottleneck** | $p(z\|x), p(x\|z)$의 쌍대 교대가 IB 라그랑주 해를 찾는 방법 |
| **Maximum Entropy** | e-flat 지수족 + m-flat 제약면의 교차 → unique MaxEnt 해 |
| **Bregman Clustering** | 쌍대평탄 공간에서 "centroid"가 Bregman 정의로 유일 |

---

## 📐 수학적 선행 조건

| 개념 | 참조 |
|------|------|
| $\nabla^{(e)}, \nabla^{(m)}$ 정의와 쌍대성 | **Ch4-04** |
| Bregman divergence와 삼점 정체성 | **Ch3-03** |
| KL = Bregman in 지수족, Legendre 쌍대 | **Ch4-03, Ch3-04** |
| Flat connection과 affine coordinates | Ch1-04 |
| Fisher 정보 $F = \nabla^2\psi$ | **Ch4-02** |

---

## 📖 직관적 이해

### 1. "두 좌표계 모두에서 평평함"

쌍대평탄 다양체 $M$에는 두 좌표계 $\theta, \eta$가 있어:
- $\theta$에서 $\nabla^{(e)}$의 크리스토펠 = 0 ($\theta$-직선이 e-geodesic)
- $\eta$에서 $\nabla^{(m)}$의 크리스토펠 = 0 ($\eta$-직선이 m-geodesic)

유클리드 공간은 **하나의** flat connection을 가진다. 정보기하에서는 **두 개의 서로 다른** flat connection을 가진 구조가 가능하다. 이것이 분포 공간의 고유한 기하학적 특징.

### 2. 두 좌표계가 Legendre-쌍대

단순히 "두 flat connection"이 있다고 쌍대평탄이 되지 않는다. 추가 조건:

$$
\eta = \nabla\psi(\theta), \quad \theta = \nabla\psi^*(\eta)
$$

두 좌표가 **Legendre 변환으로 연결**되어야 한다. 즉 potential function $\psi, \psi^*$가 존재하며 서로 Legendre 쌍대.

### 3. Canonical divergence의 자동 생성

쌍대평탄 다양체에서 **canonical divergence**는

$$
D(p \| q) = \psi(\theta_q) + \psi^*(\eta_p) - \theta_q^T \eta_p
$$

이 한 공식만으로 KL divergence와 Bregman divergence를 **동시에** 표현한다. 정보기하의 "통일된 거리 이론"의 출발점.

### 4. Pythagoras 정리의 예고

쌍대평탄 다양체에서 세 점 $P, Q, R$이 조건

- $P$와 $Q$가 **m-geodesic**으로 연결
- $Q$와 $R$이 **e-geodesic**으로 연결
- 두 geodesic이 $Q$에서 **직교** (Fisher 내적 기준)

을 만족하면 **generalized Pythagoras**:

$$
D(P \| R) = D(P \| Q) + D(Q \| R)
$$

이것이 Ch4-06의 주인공이자, EM/VI/projection 이론의 수학적 뿌리.

---

## ✏️ 엄밀한 정의

### 정의 5.1 (Dually Flat Manifold, Amari 2016)

매끈 다양체 $M$ 위의 리만 계량 $g$와 쌍대 연결 $(\nabla, \nabla^*)$에 대해, 다음이 성립하면 $(M, g, \nabla, \nabla^*)$를 **쌍대평탄 다양체(dually flat manifold)**라 한다:

1. $\nabla$가 flat ($R^\nabla = 0$): 좌표계 $\theta$에서 $\Gamma^{\nabla}(\theta) = 0$.
2. $\nabla^*$가 flat ($R^{\nabla^*} = 0$): 좌표계 $\eta$에서 $\Gamma^{\nabla^*}(\eta) = 0$.
3. $(\nabla, \nabla^*)$은 $g$에 대해 쌍대: $Xg(Y, Z) = g(\nabla_X Y, Z) + g(Y, \nabla^*_X Z)$.

추가로, 두 좌표계의 관계는 **Legendre 변환**에 의해 주어지며 potential $\psi, \psi^*$가 존재한다:

$$
\eta = \nabla\psi(\theta), \quad \theta = \nabla\psi^*(\eta), \quad \psi(\theta) + \psi^*(\eta) = \theta^T\eta
$$

### 정의 5.2 (Canonical Divergence)

쌍대평탄 다양체 $(M, g, \nabla, \nabla^*)$ 위의 **canonical divergence**는

$$
\boxed{\;D(P \| Q) := \psi(\theta_Q) + \psi^*(\eta_P) - \theta_Q^T \eta_P\;}
$$

이는 **오직 $(P, Q)$의 좌표**에만 의존하며, non-negative이고 $D(P\|P) = 0$을 만족 (정리 5.6).

### 정의 5.3 (α-Flat Family)

α-connection $\nabla^{(\alpha)}$이 $R^{(\alpha)} = 0$이면 **α-flat**. 지수족은 $\alpha = \pm 1$에서만 α-flat ($\alpha = 1$: e-flat, $\alpha = -1$: m-flat).

---

## 🔬 정리와 증명

### 정리 5.4 (지수족은 쌍대평탄) — **핵심**

regular minimal exponential family $\mathcal{E}$는 Fisher 계량 $g = F$와 $(\nabla^{(e)}, \nabla^{(m)})$에 대해 쌍대평탄이다.

**증명.** 정의 5.1의 세 조건 확인:

1. **$\nabla^{(e)}$ flat**: Ch4-04 정리 4.11에서 $R^{(e)} = 0$. $\theta$가 affine coordinate.
2. **$\nabla^{(m)}$ flat**: 정리 4.12에서 $R^{(m)} = 0$. $\eta$가 affine coordinate.
3. **쌍대**: 정리 4.10에서 $(\nabla^{(e)}, \nabla^{(m)})$이 Fisher에 대해 쌍대.

Legendre 변환으로서의 관계는 Ch4-03 정리 3.7에서 $\eta = \nabla\psi(\theta)$, $\theta = \nabla\psi^*(\eta)$. $\blacksquare$

**주의**: 이 정리의 "역"은 완전히 성립하지 않는다. 쌍대평탄 다양체 중에는 지수족과 isomorphic하지 않은 것도 있다 (예: $\mathbb{R}_+^n$을 positive orthant로 하는 쌍대평탄 구조는 exp family와 다를 수 있음). 그러나 유한차원, 단순연결, regular 하에서 많은 경우 지수족과 동치로 취급 가능.

### 정리 5.5 (Canonical Divergence = KL in Exp Family)

지수족에서 canonical divergence는 KL divergence:

$$
D(P_\theta \| P_{\theta'}) = \text{KL}(P_\theta \| P_{\theta'})
$$

**증명.** Ch3-04에서 이미 증명 (정리 6.2). 여기서는 canonical 형태:

$$
D(P_\theta \| P_{\theta'}) = \psi(\theta') + \psi^*(\eta) - \theta'^T \eta
$$

Ch3-04 정리에서 $\text{KL} = \psi(\theta') - \psi(\theta) - \theta'^T\eta + \theta^T\eta + \eta^T\theta - \theta^T\eta$ 재정리 (혹은 정리 3.10 직접).

$\blacksquare$

### 정리 5.6 (Canonical Divergence의 성질)

쌍대평탄 다양체에서 canonical divergence $D$는:

1. **Non-negative**: $D(P\|Q) \ge 0$.
2. **0-동치**: $D(P\|Q) = 0 \iff P = Q$.
3. **비대칭**: $D(P\|Q) \neq D(Q\|P)$ 일반적.
4. **2차 근사**: $D(P \| P + dP) \approx \frac{1}{2} d\theta^T F d\theta = \frac{1}{2}|dP|^2$ (Ch2-02, Ch3-02 Fisher 연결).

**증명.**
1. Fenchel-Young 부등식 (Ch4-03 정리 3.4): $\theta^T\eta \le \psi(\theta) + \psi^*(\eta)$, 등호 iff $\eta = \nabla\psi(\theta)$. 따라서
$$
D(P_\theta\|P_{\theta'}) = \psi(\theta') + \psi^*(\eta) - \theta'^T\eta \ge 0
$$
2. 등호 iff $\eta = \nabla\psi(\theta')$, 즉 $\theta' = \theta$, $P = Q$.
3. 정리 3.10의 "혼합 좌표" 표현에서 직접.
4. Taylor 전개 (Ch3-02 정리 8.3 방식).

$\blacksquare$

### 정리 5.7 (α-flat iff α = ±1)

지수족 위 α-connection $\nabla^{(\alpha)}$가 flat인 것은 **$\alpha = \pm 1$에 한한다**. $\alpha \in (-1, 1)$에서 $R^{(\alpha)} \neq 0$ (일반적으로 non-trivial 지수족).

**증명 (스케치).** $\nabla^{(\alpha)} = \frac{1+\alpha}{2}\nabla^{(e)} + \frac{1-\alpha}{2}\nabla^{(m)}$. Christoffel:

$$
\Gamma^{(\alpha)}_{ij,k} = \frac{1+\alpha}{2} \cdot 0 + \frac{1-\alpha}{2} \kappa_{ijk} = \frac{1-\alpha}{2}\kappa_{ijk}
$$

($\theta$ 좌표에서). 곡률:

$$
R^{(\alpha)l}_{kij} = \partial_i\Gamma^{(\alpha)l}_{jk} - \partial_j\Gamma^{(\alpha)l}_{ik} + \Gamma^{(\alpha)l}_{im}\Gamma^{(\alpha)m}_{jk} - \Gamma^{(\alpha)l}_{jm}\Gamma^{(\alpha)m}_{ik}
$$

Amari-Chentsov 텐서 $\kappa_{ijk}$의 non-trivial 조합으로 구성되며, $\alpha = \pm 1$에서 precoefficient 소거로 0. $\alpha \in (-1, 1)$에서 일반적으로 $\neq 0$.

상세 계산은 Amari (2016) Ch.6.6 참조. $\blacksquare$

### 정리 5.8 (쌍대평탄성의 대안적 정의 — Potential Function)

$(M, g)$ 위의 매끈 좌표계 $\theta = (\theta^1, \dots, \theta^n)$와 볼록함수 $\psi: \Theta \to \mathbb{R}$이 주어졌을 때, 다음은 $M$이 쌍대평탄하게 만드는 **충분조건**:

- $g_{ij}(\theta) = \partial_i\partial_j\psi(\theta)$

이때 $\eta = \nabla\psi(\theta)$, $\psi^*$가 Legendre 쌍대이며 $\nabla^{(e)}, \nabla^{(m)}$이 정리 5.4의 구조를 가진다.

**증명.** 지수족 구조를 직접 드러내지 않고도, $\psi$가 potential function이면 Christoffel을 명시적으로 계산하여 쌍대평탄 구조 확인. 이것을 **Hessian geometry**라 부른다 (Shima 2007).

$\blacksquare$

### 정리 5.9 (쌍대평탄 다양체의 부분구조)

쌍대평탄 $M$의 부분다양체 $S \subset M$이:

- **e-flat submanifold**: $\theta$ 좌표에서 affine (선형 제약) — $\mathcal{E}$ 내부의 exponential family 부분족.
- **m-flat submanifold**: $\eta$ 좌표에서 affine — moment 제약으로 정의된 분포족.

이들은 유도된 쌍대평탄 구조를 가진다.

**예시**:
- $\mathcal{N}(\mu, 1)$ ($\sigma$ 고정)은 full 가우스 내 **e-flat 1차원** (affine in $\theta = \mu$).
- 고정 평균 $\mathbb{E}[X] = \mu_0$ 제약 분포는 **m-flat**.

### 정리 5.10 (일반화 Pythagoras, 예고) — Ch4-06

쌍대평탄 $M$에서 세 점 $P, Q, R$이:

1. $Q$-$R$을 **e-geodesic**으로 연결
2. $P$-$Q$를 **m-geodesic**으로 연결
3. 두 geodesic이 $Q$에서 **직교**(Fisher 기준)

이면

$$
D(P \| R) = D(P \| Q) + D(Q \| R)
$$

**증명은 Ch4-06에서 완전 전개.** 여기서는 사실만.

### 정리 5.11 (Hessian Geometry와의 관계)

쌍대평탄 다양체 $(M, g, \nabla, \nabla^*)$는 국소적으로 **Hessian structure**를 가진다: potential $\psi$가 존재하여 $g_{ij} = \partial^2_{ij}\psi$. 역으로 Hessian geometry ($g = \text{Hess}\,\psi$)는 canonical 쌍대평탄 구조를 유도한다 (Shima's theorem).

이로써 정보기하는 **Hessian geometry의 특수 사례**로 볼 수 있다. 열역학, mirror descent 등의 비통계적 응용이 이 관점에서 나온다.

---

## 💻 NumPy / SymPy 구현으로 검증

### 코드 1: Canonical divergence = KL 검증 (가우스)

```python
import numpy as np
import sympy as sp

# 가우스 full family: ψ(θ) = -θ₁²/(4θ₂) - (1/2) log(-2θ₂)
theta1, theta2 = sp.symbols('theta1 theta2', real=True)
psi = -theta1**2/(4*theta2) - sp.Rational(1,2)*sp.log(-2*theta2)

# η = ∇ψ
eta1 = sp.diff(psi, theta1)  # = -θ₁/(2θ₂) = μ
eta2 = sp.diff(psi, theta2)  # = θ₁²/(4θ₂²) - 1/(2θ₂) = μ² + σ²

# ψ*(η) = θᵀη - ψ
psi_star = theta1*eta1 + theta2*eta2 - psi
psi_star = sp.simplify(psi_star)
print("ψ*(η) =", psi_star)

# Canonical D(P‖Q) = ψ(θ_Q) + ψ*(η_P) - θ_Q · η_P
# 예: P = N(0,1), Q = N(1,2)
# θ_P = (0/1, -1/2) = (0, -0.5), η_P = (0, 1)
# θ_Q = (1/4, -1/8),  η_Q = (1, 5)
def canonical_D(theta_Q, eta_P, psi_Q, psi_star_P):
    return psi_Q + psi_star_P - np.dot(theta_Q, eta_P)

def psi_num(th):
    t1, t2 = th
    return -t1**2/(4*t2) - 0.5*np.log(-2*t2)

def eta_num(th):
    t1, t2 = th
    return np.array([-t1/(2*t2), t1**2/(4*t2**2) - 1/(2*t2)])

def psi_star_num(th):
    return np.dot(th, eta_num(th)) - psi_num(th)

# P = N(0, 1), Q = N(1, sqrt(2))
th_P = np.array([0/1, -1/2])
th_Q = np.array([1/2, -1/4])

D = canonical_D(th_Q, eta_num(th_P), psi_num(th_Q), psi_star_num(th_P))
# KL(N(0,1) || N(1, 2))
# = log(sqrt(2)/1) + (1 + 1)/(2*2) - 0.5 = 0.5*log(2) + 0.5 - 0.5 = 0.5*log 2
KL_direct = 0.5*np.log(2) + (1 + 1)/(2*2) - 0.5
print(f"Canonical D: {D:.6f}")
print(f"KL direct : {KL_direct:.6f}")
print(f"diff: {abs(D-KL_direct):.2e}")
```

### 코드 2: 쌍대성 검증 — Christoffel 계산

```python
import numpy as np

# 1차원 지수족 (Bernoulli): ψ(θ) = log(1 + e^θ)
# F(θ) = ψ''(θ) = p(1-p) where p = σ(θ)
# e-connection: Γ^(e) = 0
# m-connection: Γ^(m)_{θθ,θ} = ψ'''(θ) = κ₃ = σ(θ)(1-σ(θ))(1-2σ(θ))

def sigmoid(x): return 1/(1+np.exp(-x))

def F(theta):
    p = sigmoid(theta)
    return p * (1 - p)

def kappa3(theta):
    p = sigmoid(theta)
    return p * (1-p) * (1 - 2*p)

# 쌍대 공식: ∂_k F_{ij} = Γ^(m)_{kj,i} (Ch4-04 정리 4.10)
# 1차원: ∂_θ F(θ) = κ₃(θ)
theta_test = 0.5
dF_num = (F(theta_test + 1e-6) - F(theta_test - 1e-6)) / 2e-6
k3 = kappa3(theta_test)
print(f"∂F/∂θ = {dF_num:.6f}, κ₃ = {k3:.6f}, diff: {abs(dF_num-k3):.2e}")
# 수치미분과 해석값이 일치 → 쌍대성 성립
```

### 코드 3: α-connection의 curvature — 2차원 가우스

```python
import sympy as sp

# 가우스 full (θ₁, θ₂) 공간에서 α-connection의 Christoffel 계산
theta1, theta2 = sp.symbols('theta1 theta2', real=True)
psi = -theta1**2/(4*theta2) - sp.Rational(1,2)*sp.log(-2*theta2)

vars = [theta1, theta2]

# Fisher
F = sp.Matrix([[sp.diff(psi, v, u) for u in vars] for v in vars])
F_inv = F.inv()

# Amari-Chentsov tensor κ_ijk = ∂³ψ
def kappa(i, j, k):
    return sp.diff(psi, vars[i], vars[j], vars[k])

# Γ^(α)_{ij,k} = ((1-α)/2) κ_ijk
alpha = sp.symbols('alpha', real=True)

def Gamma_alpha_lower(i, j, k):
    return (1 - alpha)/2 * kappa(i, j, k)

# Γ^(α)^l_{ij} = g^{lk} Γ^(α)_{ij,k}
def Gamma_alpha_upper(l, i, j):
    return sum(F_inv[l, k] * Gamma_alpha_lower(i, j, k) for k in range(2))

# R^(α)^l_{kij} = ∂_i Γ^l_{jk} - ∂_j Γ^l_{ik} + Γ^l_{im}Γ^m_{jk} - Γ^l_{jm}Γ^m_{ik}
def R_alpha(l, k, i, j):
    G_jk_l = Gamma_alpha_upper(l, j, k)
    G_ik_l = Gamma_alpha_upper(l, i, k)
    term1 = sp.diff(G_jk_l, vars[i])
    term2 = sp.diff(G_ik_l, vars[j])
    term3 = sum(Gamma_alpha_upper(l, i, m) * Gamma_alpha_upper(m, j, k) for m in range(2))
    term4 = sum(Gamma_alpha_upper(l, j, m) * Gamma_alpha_upper(m, i, k) for m in range(2))
    return sp.simplify(term1 - term2 + term3 - term4)

# 한 성분 샘플
R_sample = R_alpha(0, 1, 0, 1)
R_symb = sp.simplify(R_sample)
print("R^(α) 샘플 성분:", R_symb)
# α=1, α=-1 대입
print("  at α=1:", sp.simplify(R_symb.subs(alpha, 1)))   # 0
print("  at α=-1:", sp.simplify(R_symb.subs(alpha, -1))) # 0
print("  at α=0:", sp.simplify(R_symb.subs(alpha, 0)))   # generally != 0
```

### 코드 4: Dually flat 부분다양체 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# full 가우스 (θ₁, θ₂) 공간
# e-flat 부분다양체 (σ² 고정 = 1 → θ₂ = -0.5): θ₁ 자유
# m-flat 부분다양체 (μ 고정 = 0 → η₁ = 0): η₂ 자유 → σ² 변함

fig, ax = plt.subplots(figsize=(7, 6))

# Background: θ₁-θ₂ plane (θ₂ < 0)
ax.fill_betweenx([-3, 0], -3, 3, alpha=0.05, color='blue')

# e-flat (σ²=1): θ₂ = -0.5, θ₁ 자유
th1_range = np.linspace(-2, 2, 50)
ax.plot(th1_range, np.full_like(th1_range, -0.5), 'r-', lw=3, label=r'e-flat: $\sigma^2=1$ 고정')

# m-flat (μ=0): η₁ = μ = -θ₁/(2θ₂) = 0 → θ₁ = 0
th2_range = np.linspace(-3, -0.1, 50)
ax.plot(np.zeros_like(th2_range), th2_range, 'b-', lw=3, label=r'm-flat: $\mu=0$ 고정')

# 교차점
ax.scatter([0], [-0.5], s=100, color='purple', zorder=5, label=r'$\mathcal{N}(0,1)$')
ax.set_xlabel(r'$\theta_1 = \mu/\sigma^2$')
ax.set_ylabel(r'$\theta_2 = -1/(2\sigma^2)$')
ax.set_title('쌍대평탄 가우스 다양체 위의 e-flat / m-flat 부분다양체')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(-2, 2); ax.set_ylim(-3, 0)
plt.tight_layout()
plt.savefig('dually_flat_submanifolds.png', dpi=120)
```

### 코드 5: Mirror Descent = Natural Gradient (exp family) 확인

```python
import numpy as np

# 간단 문제: ϕ(θ) = 1/(1+e^{-θ})에 대한 MSE 최소화를 시뮬레이션
# exp family의 canonical param θ에서 NGD: θ_{k+1} = θ_k - η F⁻¹ g
# Mirror descent with ψ = log(1+e^θ): same update

def f(eta_target, theta):  # Loss: (η - η_target)²
    eta = 1/(1 + np.exp(-theta))
    return (eta - eta_target)**2

def grad_f(eta_target, theta):  # w.r.t. θ
    eta = 1/(1 + np.exp(-theta))
    return 2 * (eta - eta_target) * eta * (1 - eta)  # = 2(η-η_t) F

def Fisher(theta):
    eta = 1/(1 + np.exp(-theta))
    return eta * (1 - eta)

def NGD_step(theta, eta_t, lr):
    g = grad_f(eta_t, theta)
    F = Fisher(theta)
    return theta - lr * g / F

def Mirror_step(theta, eta_t, lr):
    # η_k = ∇ψ(θ_k) = sigmoid(θ_k)
    eta_k = 1/(1 + np.exp(-theta))
    # g (grad w.r.t. θ)
    g = grad_f(eta_t, theta)
    # η 공간에서 gradient: g_η = F⁻¹ g (dual space)
    F = Fisher(theta)
    g_eta = g / F  # = 2(η - η_target)
    # Mirror update: η_{k+1} in dual space
    eta_new_dual = eta_k - lr * g_eta
    # Actually mirror descent does: θ = (∇ψ*)(∇ψ(θ) - lr·g)
    # ∇ψ(θ) = η, ∇ψ*(η) = θ = logit(η)
    new_eta = eta_k - lr * g  # no, this is wrong; let me be careful
    # 정확한 mirror descent: new_θ = argmin <g, θ'> + (1/η) B_ψ(θ', θ)
    # 1차 최적: ∇_θ' stuff = 0 → g + (1/η)(∇ψ(θ') - ∇ψ(θ)) = 0
    # → ∇ψ(θ') = ∇ψ(θ) - η·g → θ' = (∇ψ)⁻¹(η_k - η·g)
    eta_new = eta_k - lr * g
    eta_new = np.clip(eta_new, 1e-6, 1-1e-6)
    theta_new = np.log(eta_new/(1-eta_new))
    return theta_new

# 두 방법 시뮬레이션
eta_target = 0.7
theta0 = np.log(0.1/0.9)  # 시작 p=0.1
lr = 0.5
theta_ngd = theta0
theta_md = theta0
for _ in range(50):
    theta_ngd = NGD_step(theta_ngd, eta_target, lr)
    theta_md = Mirror_step(theta_md, eta_target, lr)
print(f"NGD final θ = {theta_ngd:.4f}, p = {1/(1+np.exp(-theta_ngd)):.4f}")
print(f"MD  final θ = {theta_md:.4f}, p = {1/(1+np.exp(-theta_md)):.4f}")
print(f"Target: 0.7")
# 두 방법이 first-order 수준에서 동일한 수렴 경로
```

---

## 🔗 AI/ML 연결

### 1. EM의 단조 수렴 (Ch6-02)

쌍대평탄 구조 덕분에 EM 알고리즘:
- E-step: $q \leftarrow p(z|x, \theta^{(t)})$ ← **m-projection onto $p(z, x|\theta^{(t)})$-level set**
- M-step: $\theta \leftarrow \arg\max \mathbb{E}_q[\log p(x, z|\theta)]$ ← **e-projection onto $\mathcal{E}$**

이 두 projection이 각각 KL을 감소시키는 것이 쌍대평탄성의 직접적 귀결.

### 2. Variational Inference의 Mean-Field

Mean-field family $q(\theta) = \prod q_i(\theta_i)$는 full posterior 공간에서 **e-flat 부분다양체** (각 factor가 개별 지수족의 직접곱). 반면 "moment constraints"는 m-flat. ELBO 최적화가 두 구조의 교차를 찾는 것.

### 3. TRPO의 Trust Region = e-ball

TRPO 제약 $\mathbb{E}[\text{KL}(\pi_{\text{old}} \| \pi_\theta)] \le \delta$는 $\theta$ 공간(exp family canonical)에서의 **e-ball**. 쌍대 m-ball은 action distribution의 변화를 바운드 — 두 관점이 동일한 step을 정의.

### 4. Wasserstein vs KL — 두 기하학의 비교

- **KL**: 쌍대평탄 (두 연결이 Legendre 쌍대)
- **Wasserstein**: curved (Benamou-Brenier dynamic formulation에서 나오는 새로운 기하)

두 기하학은 "분포 사이의 거리"를 다르게 측정. Diffusion model에서 KL은 score matching, W는 OT 기반 (연구 진행 중).

### 5. Natural Policy Gradient의 Invariance

NGD $\theta \to \theta - \alpha F^{-1}g$는 쌍대평탄 구조에서 **$\eta$ 공간의 유클리드 gradient**로 해석:

$$
\eta \to \eta - \alpha g_\eta
$$

($g_\eta = F^{-1}g$는 $\eta$ 공간 gradient.) 이것이 NGD의 parameterization 불변성의 기하학적 증명 (Ch5-04).

### 6. Information Bottleneck

Tishby-Pereira의 IB: $\min_{p(z|x)} I(X; Z) - \beta I(Z; Y)$. Lagrangian 해가 exp family (Gibbs form)이고, 이것의 **쌍대평탄 구조**에서 closed-form update가 유도됨 (Blahut-Arimoto 알고리즘).

---

## ⚖️ 가정과 한계

### 가정

1. **Regular minimal 지수족**: 다른 경우 $\nabla^{(e)}, \nabla^{(m)}$ 정의 자체가 불가능하거나 Legendre 구조 실패.
2. **유한 차원**: 무한차원 RKHS family는 일반화 필요.
3. **단순연결**: 전역 좌표계 존재. complex topology (e.g., circular family) 처리 복잡.

### 한계

1. **역의 부분적 성립**: 쌍대평탄 다양체 ≠ 지수족. 역방향 (쌍대평탄 → 지수족)은 일반적으로 참이 아님.

2. **α = 0에서의 non-flatness**: Levi-Civita가 flat이 아니면서도 유일 — 이것이 "Fisher-Rao geodesic이 non-trivial"임의 본질. 실용에서는 α = ±1을 쓰는 것이 더 간단.

3. **Curvature의 통계적 의미**: $R^{(\alpha)} \neq 0$ for $\alpha \neq \pm 1$은 **Efron statistical curvature**와 연결 — curved exp family의 MLE 성능 저하를 측정.

4. **비 지수족의 확장**: Cauchy, mixture 등을 쌍대평탄 구조로 담는 방법 — Amari의 "q-exponential family" 확장이 있지만 여전히 제약됨.

5. **계산 비용**: $\psi^*$의 명시적 형태가 없으면 canonical divergence 계산이 어려움. 예: Wishart family.

---

## 📌 핵심 정리

| 대상 | 공식 / 사실 |
|------|---------|
| Dually flat 정의 | $\nabla$ flat + $\nabla^*$ flat + $(\nabla, \nabla^*)$ $g$-쌍대 + Legendre |
| 지수족 쌍대평탄성 | $(\mathcal{E}, F, \nabla^{(e)}, \nabla^{(m)})$이 canonical 쌍대평탄 예 |
| Canonical divergence | $D(P\|Q) = \psi(\theta_Q) + \psi^*(\eta_P) - \theta_Q^T\eta_P$ |
| Canonical = KL | 지수족에서 $D$ = $\text{KL}$ |
| α-flat iff | $\alpha = \pm 1$ |
| Levi-Civita | $\nabla^{(0)}$, non-flat (curvature 있음) |
| Hessian geometry | $g = \text{Hess}\psi$ → dually flat의 대안적 특성화 |
| Pythagoras (예고) | e/m-geodesic 직교 3점 조건 → $D(P\|R) = D(P\|Q) + D(Q\|R)$ |

**한 줄 요약:** 지수족은 **Fisher 계량 하 $(\nabla^{(e)}, \nabla^{(m)})$ 쌍대평탄** 다양체이고, 이 구조가 **canonical divergence = KL**을 자동 생성하며, **Pythagoras·projection·EM·NGD·Mirror Descent**의 기하학적 뿌리를 이룬다.

---

## 🤔 생각해볼 문제

1. **(α = 0에서 curvature 존재의 의미)** Gauss full family에서 Levi-Civita의 Ricci curvature를 계산 (참조: 쌍곡면 $\mathbb{H}^2$). 상수 음수 곡률 $-1/2$가 나오는데, 이것의 통계적 해석은?

2. **(Pythagoras의 사전 맛보기)** 2차원 가우스에서 $P = \mathcal{N}(0, 1), Q = \mathcal{N}(0, 2), R = \mathcal{N}(1, 2)$. $P$-$Q$는 m-geodesic (η₁=0 고정, η₂ 이동), $Q$-$R$은 e-geodesic (θ₂=-0.25 고정, θ₁ 이동). 두 geodesic이 $Q$에서 직교함을 확인하고 $D(P\|R) = D(P\|Q) + D(Q\|R)$ 수치 검증.

3. **(Multinomial 쌍대평탄)** $K$-multinomial의 canonical $(θ_1, ..., θ_{K-1})$과 expectation $(p_1, ..., p_{K-1})$. $\psi, \psi^*$를 명시적으로 적고 쌍대평탄성을 확인.

4. **(Hessian vs non-Hessian)** 임의의 리만 계량이 Hessian form $g = \text{Hess}\psi$로 쓰일 수 있는가? 언제 실패하는가? (힌트: 2차원에서 Gaussian curvature가 0 아니면 실패.)

5. **(최적 수송의 기하학)** Wasserstein distance도 canonical divergence로 볼 수 있는가? $W_2^2$가 어떤 쌍대평탄 구조를 유도하지 않음을 확인. 그 대신 Otto calculus의 infinite-dimensional Riemannian 관점 필요.

6. **(EM의 쌍대평탄 증명)** E-step이 $\mathcal{M}$의 m-flat 부분다양체로의 m-projection임과 M-step이 $\mathcal{E}$의 e-flat 부분다양체로의 e-projection임을 정확히 보이시오 (Ch6-02 예고).

7. **(Hessian structure of thermodynamics)** Ising model의 thermodynamics를 쌍대평탄 구조로 재해석. 온도·magnetization·free energy가 각각 $\theta, \eta, \psi$에 대응됨을 보이기.

8. **(Q-exponential family)** Amari의 "q-exponential" $p(x;\theta) = \exp_q(\theta^T T(x) - \psi_q(\theta))$에서 쌍대평탄 구조가 어떻게 변형되는가? Renyi divergence와의 연결.

---

<div align="center">

| [◀ 04. e/m-connection](./04-e-m-connection.md) | [📚 메인 README](../README.md) | [06. Generalized Pythagoras ▶](./06-generalized-pythagoras.md) |
|:---:|:---:|:---:|

</div>
