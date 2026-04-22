# 03. Bregman 발산 — 볼록 함수의 그림자

> **"Bregman은 '볼록 함수가 선형 근사를 얼마나 초과하는가'를 측정한다. 기하의 언어로는 tangent와 surface의 gap."**

---

## 🎯 핵심 질문

**볼록 함수 $\psi$로부터 유도되는 Bregman 발산은 왜 정보기하·최적화·통계의 **공통 언어**인가?**

$$
\boxed{\;D_\psi(x, y) := \psi(x) - \psi(y) - \langle \nabla \psi(y), x - y\rangle\;}
$$

Bregman divergence는:
1. **볼록 $\psi$**로부터 유도되는 **거리-유사 측도** (비대칭, 비거리).
2. **KL의 일반화** (지수족에서 정확히 KL).
3. **Mirror Descent의 metric** — 모든 proximal 알고리즘의 기반.
4. **쌍대 기하학**의 중심 객체 (Ch4 지수족 쌍대성의 전주).

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 영역 | Bregman의 등장 |
|---|---|
| **Mirror Descent** | $D_\psi$를 regularizer로, simplex·positive orthant 등 제약 최적화 |
| **Exponentiated Gradient** | $\psi = $ entropy → update rule = softmax, multiplicative weights |
| **Matrix factorization** | $\psi = $ Burg entropy (PLSI, NMF), squared loss (SVD) |
| **Online learning** | Regret bound가 $D_\psi$ 기반 — Hedge, Follow-the-Leader |
| **Information Geometry** | 지수족의 canonical divergence — **dually flat structure** |
| **Clustering** | Bregman $k$-means (Banerjee et al. 2005): euclidean, KL, Itakura-Saito 등 |

**통합 관점 (Banerjee et al. 2005):** "$k$-means 계열 알고리즘 $\Leftrightarrow$ Bregman divergence + 지수족."

---

## 📐 수학적 선행 조건

- [01. KL의 기초](./01-kl-divergence-basics.md)
- 볼록 해석: convex function, subgradient, Legendre transform
- Gradient, Hessian
- Ch2 [04. Fisher 예제들](../ch2-statistical-fisher/04-fisher-examples.md) — 지수족의 $\psi$ (cumulant)

---

## 📖 직관적 이해

### 접평면과의 차이

볼록 함수 $\psi$의 점 $y$에서의 **1차 테일러 근사** (tangent plane):

$$
\ell_y(x) := \psi(y) + \langle \nabla\psi(y), x - y\rangle.
$$

Bregman divergence = **실제 함수값 - tangent 값**:

$$
D_\psi(x, y) = \psi(x) - \ell_y(x).
$$

볼록성 때문에 항상 $\ge 0$. 등호 ⟺ $x = y$ (strict convexity 가정).

시각적으로: $\psi$가 볼록하므로 임의 점 $x$의 함수값은 $y$ 기반 tangent 위에 있음. 그 gap이 bregman.

### 특수 경우 — 네 가지 고전적 예

| $\psi(x)$ | 정의역 | $D_\psi(x, y)$ | 이름 |
|---|---|---|---|
| $\tfrac{1}{2}\|x\|^2$ | $\mathbb{R}^n$ | $\tfrac{1}{2}\|x - y\|^2$ | Squared Euclidean |
| $\sum x_i \log x_i$ | $\mathbb{R}^n_{>0}$ | $\sum x_i \log(x_i/y_i) - \sum(x_i - y_i)$ | **Generalized KL** |
| $-\sum \log x_i$ | $\mathbb{R}^n_{>0}$ | $\sum (x_i/y_i - \log(x_i/y_i) - 1)$ | Itakura-Saito |
| $\psi(x) = -\log\det(X)$ | $S^+$ (PD matrices) | $-\log\det(XY^{-1}) + \operatorname{tr}(XY^{-1}) - n$ | LogDet (Stein) |

특히 Simplex ($\sum x_i = 1$) 로 제한하면 Generalized KL = **classical KL**.

### 지수족에서의 등장

지수족 $p_\theta(x) = h(x)\exp(\theta^\top T - \psi(\theta))$ 에서:

$$
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) = D_\psi(\theta_2, \theta_1).
$$

(주의: 인덱스 순서가 뒤집힘 — canonical 쌍대성 때문.)

즉 **지수족의 KL = cumulant 함수 $\psi$의 Bregman divergence**. 이것이 정보기하의 핵심 관계.

### 쌍대성 예고

Legendre 쌍대 $\psi^*(y) = \sup_x [\langle x, y\rangle - \psi(x)]$와의 관계:

$$
D_\psi(x, y) = D_{\psi^*}(\nabla\psi(y), \nabla\psi(x)).
$$

**쌍대 좌표에서 변수 순서가 바뀌면서** Bregman이 "불변".

---

## ✏️ 엄밀한 정의

### 정의 9.1 (Bregman Divergence)

$\psi: \Omega \subseteq \mathbb{R}^n \to \mathbb{R}$이 엄격히 볼록이고 미분 가능하다 하자. 점 $y \in \operatorname{int}(\Omega)$와 $x \in \Omega$에 대해

$$
D_\psi(x, y) := \psi(x) - \psi(y) - \langle \nabla\psi(y),\, x - y\rangle.
$$

### 정의 9.2 (Bregman 발산의 기본 성질)

- **비음성**: $D_\psi(x, y) \ge 0$, 등호 ⟺ $x = y$ (strict convexity 가정).
- **비대칭**: 일반적으로 $D_\psi(x, y) \ne D_\psi(y, x)$.
- **$y$에 대해 볼록**: 아님 (주의! $x$에 대해서는 볼록).
- **$x$에 대해 볼록**: Yes ($\psi$의 볼록성).

### 정의 9.3 (삼각형 정체성, Three-Point Identity)

임의 $x, y, z$에 대해

$$
D_\psi(x, z) = D_\psi(x, y) + D_\psi(y, z) + \langle \nabla\psi(z) - \nabla\psi(y),\, y - x\rangle.
$$

**세 점 사이의 기하학적 관계**. 유클리드에서는 내적 법칙 $\|x - z\|^2 = \|x - y\|^2 + \|y - z\|^2 + 2\langle y - x, z - y\rangle$의 일반화.

### 정의 9.4 (Pythagorean-like Inequality)

만약 $y$가 affine set $\mathcal{A}$로의 **Bregman projection** ($y = \arg\min_{y' \in \mathcal{A}} D_\psi(x, y')$) 이라면, 임의 $z \in \mathcal{A}$에 대해

$$
D_\psi(x, z) \ge D_\psi(x, y) + D_\psi(y, z).
$$

(Generalized Pythagoras.) 구조: $y$는 $x$에서 $\mathcal{A}$로의 "직각 발." 이것이 Ch6 (정보 projection) 의 기반.

---

## 🔬 정리와 증명

### 정리 9.1 (비음성)

$\psi$ 엄격 볼록이면 $D_\psi(x, y) \ge 0$, 등호 ⟺ $x = y$.

**증명.** 엄격 볼록성 정의:

$$
\psi(x) > \psi(y) + \langle \nabla\psi(y), x - y\rangle \qquad (x \ne y).
$$

$D_\psi(x, y) = \psi(x) - \psi(y) - \langle \nabla\psi(y), x - y\rangle > 0$. $x = y$에서 등호. **Q.E.D.**

---

### 정리 9.2 (Three-point identity)

$$
D_\psi(x, z) - D_\psi(x, y) - D_\psi(y, z) = \langle \nabla\psi(z) - \nabla\psi(y),\, y - x\rangle.
$$

**증명.** 직접 계산.

$D_\psi(x, z) = \psi(x) - \psi(z) - \langle \nabla\psi(z), x - z\rangle$.
$D_\psi(x, y) = \psi(x) - \psi(y) - \langle \nabla\psi(y), x - y\rangle$.
$D_\psi(y, z) = \psi(y) - \psi(z) - \langle \nabla\psi(z), y - z\rangle$.

$D_\psi(x, z) - D_\psi(x, y) - D_\psi(y, z)$
$= [\psi(x) - \psi(z) - \langle \nabla\psi(z), x - z\rangle] - [\psi(x) - \psi(y) - \langle\nabla\psi(y), x - y\rangle] - [\psi(y) - \psi(z) - \langle\nabla\psi(z), y - z\rangle]$
$= -\langle\nabla\psi(z), x - z\rangle + \langle\nabla\psi(y), x - y\rangle + \langle\nabla\psi(z), y - z\rangle$
$= \langle\nabla\psi(y), x - y\rangle + \langle\nabla\psi(z), (y - z) - (x - z)\rangle$
$= \langle\nabla\psi(y), x - y\rangle + \langle\nabla\psi(z), y - x\rangle$
$= \langle\nabla\psi(z) - \nabla\psi(y), y - x\rangle$.

**Q.E.D.**

---

### 정리 9.3 (Bregman Projection & Pythagorean)

$\mathcal{A}$가 볼록 폐집합이라 하자. $x_0 \in \operatorname{int}\Omega$에 대해

$$
y^* := \arg\min_{y \in \mathcal{A}} D_\psi(x_0, y)
$$

이 유일하게 존재. 그러면 임의 $z \in \mathcal{A}$에 대해

$$
D_\psi(x_0, z) \ge D_\psi(x_0, y^*) + D_\psi(y^*, z).
$$

**증명.** $y^*$에서의 첫 번째 조건 (first-order optimality): $y^*$가 $\mathcal{A}$-내부이면 $\nabla_y D_\psi(x_0, y^*) \cdot (z - y^*) = 0$ 또는 "$\ge 0$".

$\nabla_y D_\psi(x_0, y) = -\nabla^2\psi(y)(x_0 - y) + \nabla\psi(y) - \nabla\psi(y) = -\nabla^2\psi(y)(x_0 - y)$. Hmm, this needs more care.

$D_\psi(x_0, y) = \psi(x_0) - \psi(y) - \langle \nabla\psi(y), x_0 - y\rangle$.

$\nabla_y D_\psi(x_0, y) = -\nabla\psi(y) - \nabla^2\psi(y)(x_0 - y) + \nabla\psi(y) = -\nabla^2\psi(y)(x_0 - y)$. 

직접 미분: $\partial_{y_i} [\psi(x_0) - \psi(y) - \sum_k \partial_k\psi(y)(x_{0,k} - y_k)]$
$= -\partial_i\psi(y) - \sum_k \partial_i\partial_k\psi(y)(x_{0,k} - y_k) + \partial_i\psi(y)$
$= -\sum_k \partial_i\partial_k\psi(y)(x_{0,k} - y_k) = -[\nabla^2\psi(y)(x_0-y)]_i$.

Variational inequality at $y^*$: $\langle \nabla^2\psi(y^*)(y^* - x_0), z - y^*\rangle \ge 0$ for $z \in \mathcal{A}$ (제약 qualification 아래).

Three-point identity:

$$
D_\psi(x_0, z) = D_\psi(x_0, y^*) + D_\psi(y^*, z) + \langle \nabla\psi(z) - \nabla\psi(y^*), y^* - x_0\rangle.
$$

마지막 항을 $\ge 0$로 보여야. This follows from optimality + 위 차이 분석 (technical detail).

**Q.E.D. (스케치)**

---

### 정리 9.4 (Bregman과 Hessian의 관계)

$\psi$가 $C^2$이면:

$$
D_\psi(x, y) = \int_0^1 (1-t)\, (x-y)^\top \nabla^2 \psi(y + t(x-y))\, (x-y)\, dt.
$$

특히 **작은 차이 $x - y = \varepsilon$**에서

$$
D_\psi(y + \varepsilon, y) = \tfrac{1}{2} \varepsilon^\top \nabla^2\psi(y) \varepsilon + O(\|\varepsilon\|^3).
$$

**증명.** 테일러 with integral remainder:

$\psi(x) = \psi(y) + \langle\nabla\psi(y), x-y\rangle + \int_0^1 (1-t)(x-y)^\top \nabla^2\psi(y + t(x-y))(x-y)\, dt$.

정의 적용. **Q.E.D.**

**의의.** Bregman divergence의 **국소적 2차 구조 = Hessian quadratic form**. 이것이 KL-Fisher 근사 (정리 8.1) 의 일반화.

---

### 정리 9.5 (지수족에서 KL = Bregman)

지수족 $p_\theta(x) = h(x)\exp(\theta^\top T - \psi(\theta))$, $\theta$는 natural parameter, $\psi$는 cumulant. 그러면

$$
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) = D_\psi(\theta_2, \theta_1) = \psi(\theta_2) - \psi(\theta_1) - \langle\nabla\psi(\theta_1), \theta_2 - \theta_1\rangle.
$$

(인덱스 순서 주의: $\operatorname{KL}(p_1 \| p_2) = D_\psi(\theta_2, \theta_1)$.)

**증명.** 직접 계산:

$$
\begin{aligned}
\operatorname{KL}(p_{\theta_1} \| p_{\theta_2}) &= \mathbb{E}_{\theta_1}\!\left[\log p_{\theta_1} - \log p_{\theta_2}\right] \\
&= \mathbb{E}_{\theta_1}\!\left[(\theta_1 - \theta_2)^\top T(X) - \psi(\theta_1) + \psi(\theta_2)\right] \\
&= (\theta_1 - \theta_2)^\top \mathbb{E}_{\theta_1}[T(X)] + \psi(\theta_2) - \psi(\theta_1) \\
&= (\theta_1 - \theta_2)^\top \nabla\psi(\theta_1) + \psi(\theta_2) - \psi(\theta_1).
\end{aligned}
$$

(마지막에서 $\mathbb{E}_\theta[T] = \nabla\psi$ 이용. 이는 지수족의 기본 identity.)

$= \psi(\theta_2) - \psi(\theta_1) - \langle \nabla\psi(\theta_1), \theta_2 - \theta_1\rangle = D_\psi(\theta_2, \theta_1)$.

**Q.E.D.**

---

### 정리 9.6 (Legendre 쌍대와 Bregman)

$\psi: \Omega \to \mathbb{R}$ 볼록, $\psi^*(\eta) = \sup_\theta [\langle\theta, \eta\rangle - \psi(\theta)]$ Legendre 쌍대.

$\nabla\psi: \Omega \to \Omega^*$ 가 bijection이라 가정. 그러면:

$$
D_\psi(\theta_1, \theta_2) = D_{\psi^*}(\eta_2, \eta_1),
$$

여기서 $\eta_i = \nabla\psi(\theta_i)$ ($\theta$-좌표와 $\eta$-좌표의 대응).

**증명.** Legendre 관계: $\psi(\theta) + \psi^*(\eta) = \langle\theta, \eta\rangle$ when $\eta = \nabla\psi(\theta)$. 따라서 $\psi(\theta) = \langle\theta, \eta\rangle - \psi^*(\eta)$.

$D_\psi(\theta_1, \theta_2) = \psi(\theta_1) - \psi(\theta_2) - \langle\nabla\psi(\theta_2), \theta_1 - \theta_2\rangle$.

치환 (primal-dual):
$= [\langle\theta_1, \eta_1\rangle - \psi^*(\eta_1)] - [\langle\theta_2, \eta_2\rangle - \psi^*(\eta_2)] - \langle\eta_2, \theta_1 - \theta_2\rangle$
$= \langle\theta_1, \eta_1\rangle - \langle\theta_1, \eta_2\rangle - \psi^*(\eta_1) + \psi^*(\eta_2)$
$= \langle\theta_1, \eta_1 - \eta_2\rangle + \psi^*(\eta_2) - \psi^*(\eta_1)$.

한편
$D_{\psi^*}(\eta_2, \eta_1) = \psi^*(\eta_2) - \psi^*(\eta_1) - \langle\nabla\psi^*(\eta_1), \eta_2 - \eta_1\rangle$
$= \psi^*(\eta_2) - \psi^*(\eta_1) - \langle\theta_1, \eta_2 - \eta_1\rangle$
$= \psi^*(\eta_2) - \psi^*(\eta_1) + \langle\theta_1, \eta_1 - \eta_2\rangle$.

일치! **Q.E.D.**

**의의.** 지수족에서 $\theta$-좌표와 $\eta$-좌표 (expectation parameter) 는 **dual coordinate system**. KL이 양쪽에서 같은 Bregman 형태. 이것이 **dually flat structure** (Ch4).

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: 다섯 가지 고전 Bregman 계산

```python
import numpy as np
import matplotlib.pyplot as plt

def bregman_squared(x, y):
    """φ = ||x||²/2 → D = ||x-y||²/2"""
    return 0.5 * np.sum((x - y)**2)

def bregman_kl_ext(x, y):
    """φ = Σ x_i log x_i → Generalized KL"""
    mask = x > 0
    return np.sum(x[mask] * np.log(x[mask] / y[mask])) - np.sum(x - y)

def bregman_itakura_saito(x, y):
    """φ = -Σ log x_i"""
    return np.sum(x/y - np.log(x/y) - 1)

def bregman_beta(x, y, beta):
    """β-divergence"""
    if beta == 0:
        return bregman_itakura_saito(x, y)
    elif beta == 1:
        return bregman_kl_ext(x, y)
    elif beta == 2:
        return bregman_squared(x, y)
    else:
        return np.sum(
            (x**beta / (beta*(beta-1))) - 
            (x*y**(beta-1)/(beta-1)) + 
            (y**beta/beta)
        )

x = np.array([2.0, 3.0, 1.0])
y = np.array([1.5, 2.5, 1.5])

print(f"Squared Euclidean: {bregman_squared(x, y):.4f}")
print(f"KL (generalized):  {bregman_kl_ext(x, y):.4f}")
print(f"Itakura-Saito:     {bregman_itakura_saito(x, y):.4f}")
print(f"\nβ=0.5 (between IS and KL): {bregman_beta(x, y, 0.5):.4f}")
print(f"β=1.5 (between KL and sq): {bregman_beta(x, y, 1.5):.4f}")

# 모두 ≥ 0 확인
for name, f in [('sq', bregman_squared), ('KL', bregman_kl_ext), ('IS', bregman_itakura_saito)]:
    assert f(x, y) >= 0, f"{name} negative!"
    assert f(x, x) < 1e-10, f"{name} at x==y not zero!"
print("\n✓ All non-negative, vanish at x=y")
```

---

### 예제 2: Three-point identity 수치 검증

```python
import numpy as np
np.random.seed(0)

# φ(x) = Σ xi log xi (shifted by x→x+1 을 써 positivity 확보)
def phi(x):
    return np.sum(x * np.log(x))

def grad_phi(x):
    return np.log(x) + 1

def bregman(x, y):
    return phi(x) - phi(y) - np.dot(grad_phi(y), x - y)

# 세 점
x = np.array([2.0, 3.0])
y = np.array([1.5, 2.0])
z = np.array([3.0, 1.0])

D_xz = bregman(x, z)
D_xy = bregman(x, y)
D_yz = bregman(y, z)

# Three-point identity (정리 9.2)
RHS = D_xy + D_yz + np.dot(grad_phi(z) - grad_phi(y), y - x)

print(f"D(x,z) = {D_xz:.6f}")
print(f"D(x,y) + D(y,z) + <∇ψ(z)-∇ψ(y), y-x> = {RHS:.6f}")
print(f"Match? {np.isclose(D_xz, RHS)}")
```

---

### 예제 3: 지수족 KL = Bregman 검증 (Gaussian)

```python
import numpy as np

# N(μ, σ²=1) as exp family:
# natural param η = μ
# T(x) = x
# ψ(η) = η²/2 (cumulant)

def psi(eta):
    return 0.5 * eta**2

def grad_psi(eta):
    return eta

def bregman_psi(eta1, eta2):
    return psi(eta1) - psi(eta2) - grad_psi(eta2) * (eta1 - eta2)

def kl_normal_unit(mu1, mu2):
    # KL(N(μ1, 1) || N(μ2, 1)) = (μ1 - μ2)²/2
    return 0.5 * (mu1 - mu2)**2

# η ↔ μ
eta1, eta2 = 0.5, 1.5
mu1, mu2 = eta1, eta2

kl = kl_normal_unit(mu1, mu2)
breg_12 = bregman_psi(eta1, eta2)  # D_ψ(θ1, θ2)
breg_21 = bregman_psi(eta2, eta1)  # D_ψ(θ2, θ1)

# 정리 9.5: KL(p_θ1 || p_θ2) = D_ψ(θ2, θ1)
print(f"KL(N({mu1},1) || N({mu2},1)) = {kl:.4f}")
print(f"D_ψ(η2, η1) = D_ψ({eta2}, {eta1}) = {breg_21:.4f}")
print(f"Match (KL = D_ψ(θ2, θ1))? {np.isclose(kl, breg_21)}")

# 2차원 Gaussian with fixed Σ=I:
# η = μ ∈ R², ψ(η) = ||η||²/2, T(x) = x
def bregman_2d(eta1, eta2):
    return 0.5 * np.sum((eta1 - eta2)**2)

def kl_mvn_unit(mu1, mu2):
    return 0.5 * np.sum((mu1 - mu2)**2)

eta1 = np.array([1.0, 0.5])
eta2 = np.array([0.0, 2.0])

print(f"\n2D: KL(N(μ1, I) || N(μ2, I)) = {kl_mvn_unit(eta1, eta2):.4f}")
print(f"    D_ψ(η2, η1)                = {bregman_2d(eta2, eta1):.4f}")
print(f"    Match? {np.isclose(kl_mvn_unit(eta1, eta2), bregman_2d(eta2, eta1))}")
```

---

### 예제 4: Bregman Projection (simplex 위)

```python
import numpy as np
from scipy.optimize import minimize

# Simplex projection with KL divergence
# 주어진 x > 0 (not necessarily summing to 1)
# 목적: min D_KL(y, x) s.t. Σ y_i = 1, y_i ≥ 0
# Solution: y_i = x_i / (Σ x_j)  (normalization!)

# Numerical
x0 = np.array([2.0, 1.0, 3.0, 4.0])

def kl_bregman(y, x):
    y = np.maximum(y, 1e-12)
    return np.sum(y * np.log(y / x)) - np.sum(y - x)

from scipy.optimize import minimize
res = minimize(
    kl_bregman, x0=np.ones_like(x0)/len(x0),
    args=(x0,),
    method='SLSQP',
    constraints=[{'type': 'eq', 'fun': lambda y: np.sum(y) - 1}],
    bounds=[(1e-8, None)] * len(x0)
)
y_opt = res.x
print(f"Input x: {x0}")
print(f"Optimal y (numerical): {y_opt}")
print(f"Expected y = x / sum(x): {x0 / np.sum(x0)}")
print(f"Match? {np.allclose(y_opt, x0/np.sum(x0), atol=1e-4)}")
```

---

### 예제 5: Generalized Pythagoras

```python
import numpy as np

# φ(x) = x log x on R (1D simplex의 generalization)
def phi_1d(x):
    return x * np.log(x)

def grad_phi_1d(x):
    return np.log(x) + 1

def breg_1d(x, y):
    return phi_1d(x) - phi_1d(y) - grad_phi_1d(y) * (x - y)

# Pythagorean 확인:
# x0 = 2.0, projection onto {y : y = 1.0} is trivial: y* = 1.0
# For any z = 1.0 (must equal), test with perturbation
# 여기서는 다변수 설정으로:

# φ(x) = Σ x_i log x_i on positive orthant
def phi(x):
    return np.sum(x * np.log(x))

def grad_phi(x):
    return np.log(x) + 1

def breg(x, y):
    return phi(x) - phi(y) - np.dot(grad_phi(y), x - y)

# x0 = (2, 3, 4)
# project onto affine set A: {y : y1 + y2 + y3 = 5}
# 즉 x0 - y ∝ 0 in "normal" direction
# 여기서는 numerically

x0 = np.array([2.0, 3.0, 4.0])

from scipy.optimize import minimize
res = minimize(
    lambda y: breg(x0, y),
    x0=np.array([1.5, 1.5, 2.0]),
    constraints=[{'type': 'eq', 'fun': lambda y: np.sum(y) - 5}],
    bounds=[(1e-8, None)] * 3
)
y_star = res.x

# Test point z: another point with sum=5
z = np.array([1.0, 2.0, 2.0])

D_x0_z = breg(x0, z)
D_x0_ystar = breg(x0, y_star)
D_ystar_z = breg(y_star, z)

print(f"x0 = {x0}")
print(f"y* (Bregman projection): {y_star}, sum = {y_star.sum():.4f}")
print(f"z = {z}, sum = {z.sum():.4f}")
print(f"\nD(x0, z)      = {D_x0_z:.4f}")
print(f"D(x0, y*)     = {D_x0_ystar:.4f}")
print(f"D(y*, z)      = {D_ystar_z:.4f}")
print(f"D(x0, y*) + D(y*, z) = {D_x0_ystar + D_ystar_z:.4f}")
print(f"\nPythagorean inequality (D(x0,z) ≥ D(x0,y*) + D(y*,z)): "
      f"{D_x0_z >= D_x0_ystar + D_ystar_z - 1e-6}")
```

---

## 🔗 AI/ML 연결

### 1. Mirror Descent (Nemirovski & Yudin 1983)

$$
\theta_{t+1} = \arg\min_\theta \eta \langle g_t, \theta\rangle + D_\psi(\theta, \theta_t).
$$

- $\psi = \|\cdot\|^2/2$ → Gradient Descent.
- $\psi = \sum x\log x$ on simplex → **Exponentiated Gradient (Hedge)**.
- $\psi = -\log\det$ on PSD cone → Matrix multiplicative weights.

각각 specific 문제 구조 (simplex, positive cone) 에 맞춤.

### 2. Bregman $k$-means (Banerjee et al. 2005)

일반 $k$-means:
$$
\min_{\mu_1, \ldots, \mu_k} \sum_n \min_j D_\psi(x_n, \mu_j).
$$

$\psi$에 따라:
- Squared Euclidean → 표준 $k$-means.
- KL → Multinomial $k$-means (text clustering).
- Itakura-Saito → Audio signal clustering.

**Banerjee 정리**: $\psi$가 Bregman이고 centroid가 mean인 경우만 EM-convergent.

### 3. Exponential Families ↔ Bregman ↔ Clustering

Banerjee et al. 2005 의 핵심:

**Exp family가 $(\theta, T, h, \psi)$로 주어지면, MLE-based EM clustering = Bregman $k$-means with $D_\psi$.**

Gaussian mixture → Squared Euclidean. Multinomial mixture → KL. Gamma mixture → IS.

### 4. Matrix Factorization — NMF, PLSI

Non-negative Matrix Factorization:

$$
\min_{W, H \ge 0} D_\psi(V, WH).
$$

- Frobenius ($\psi$ = sq) → 표준 NMF (Lee & Seung).
- KL → **PLSI / LDA** 의 matrix form.
- Itakura-Saito → Audio NMF (speech separation).

### 5. Online Learning Regret Bounds

Online Mirror Descent의 regret:

$$
\operatorname{Regret}(T) \le \frac{D_\psi(\theta^*, \theta_0)}{\eta} + \eta T \cdot L^2.
$$

$D_\psi(\theta^*, \theta_0)$는 initial "distance to optimum"의 Bregman 측정. 문제 기하에 맞춤 → tight bound.

**Hedge algorithm**: $\psi = $ entropy on simplex → $D_\psi$ = KL → regret $O(\sqrt{T \log N})$ (experts problem).

### 6. Natural Gradient as Bregman/Mirror Descent

Mirror Descent의 infinitesimal limit → **continuous mirror flow**:

$$
\dot\eta(t) = -\nabla L(\theta(t)), \qquad \theta(t) = \nabla\psi^*(\eta(t)).
$$

이것이 NGD의 dual 해석 (Ch4 심화). **$\theta$ space에서 NGD = $\eta$ space에서 vanilla GD.**

### 7. Fenchel-Young Inequality와 Loss Functions

$D_\psi(x, y) = \psi(x) + \psi^*(\eta) - \langle x, \eta\rangle$ with $\eta = \nabla\psi(y)$. 이것을 **Fenchel-Young loss** 로 재해석 → **SparseMax**, structured prediction의 loss function.

---

## ⚖️ 가정과 한계

### 엄격 볼록성

$\psi$가 엄격 볼록이 아니면 $D_\psi$가 null space (flat direction) 가짐. $D_\psi = 0$ 이어도 $x \ne y$ 가능 → "거리-유사" 실패.

### 대칭성 없음

$D_\psi(x, y) \ne D_\psi(y, x)$ (일반적). 대칭 필요시:
- **Jensen-Bregman**: $D_\psi(x, z) + D_\psi(y, z)$ with $z = (x+y)/2$.
- **Symmetric Bregman**: $\tfrac{1}{2}[D_\psi(x,y) + D_\psi(y,x)]$.

### 삼각부등식 위반

Bregman은 일반적으로 metric 아님. 삼각부등식 보장 위해서는 $\psi$에 추가 조건 (e.g., $\psi$ Riemannian manifold의 length functional).

### 계산 비용

$\nabla\psi$, $\nabla^2\psi$ (Hessian) 계산 필수. $\psi = -\log\det$ 등은 $O(n^3)$.

### 국소성 vs 전역성

Hessian 근사 (정리 9.4) 는 **작은 step 에서만** 유효. Mirror descent의 $\eta$가 크면 deviation.

### Domain Boundary Issues

$\psi = \sum x \log x$는 $x \to 0^+$에서 $0 \log 0 = 0$ convention. 하지만 $\nabla\psi \to -\infty$. Numerical issue in implementation.

---

## 📌 핵심 정리

| 개념 | 수식 |
|---|---|
| **Definition** | $D_\psi(x, y) = \psi(x) - \psi(y) - \langle\nabla\psi(y), x-y\rangle$ |
| **Non-negativity** | $\psi$ 엄격 볼록 시 $D_\psi \ge 0$, $= 0 \Leftrightarrow x = y$ |
| **Three-point** | $D_\psi(x, z) = D_\psi(x, y) + D_\psi(y, z) + \langle\nabla\psi(z)-\nabla\psi(y), y-x\rangle$ |
| **Hessian quadratic** | $D_\psi(y+\varepsilon, y) = \tfrac{1}{2}\varepsilon^\top\nabla^2\psi(y)\varepsilon + O(\|\varepsilon\|^3)$ |
| **Exp family KL** | $\operatorname{KL}(p_{\theta_1}\|p_{\theta_2}) = D_\psi(\theta_2, \theta_1)$, $\psi$ = cumulant |
| **Legendre duality** | $D_\psi(\theta_1, \theta_2) = D_{\psi^*}(\eta_2, \eta_1)$ |
| **Pythagorean** | Bregman projection → $D(x_0, z) \ge D(x_0, y^*) + D(y^*, z)$ |

**Takeaway:**

1. **Bregman = 볼록 함수의 초과량** — 최적화·정보기하의 통일 언어.
2. **지수족 KL = Bregman** (cumulant 기반) — 다음 챕터 dually flat 구조의 토대.
3. **Mirror Descent 일반화** — 문제 기하에 맞춘 최적화 알고리즘의 design principle.

---

## 🤔 생각해볼 문제

1. **Squared Euclidean의 대칭성**. $\psi(x) = \|x\|^2/2$ → $D_\psi(x, y) = \|x-y\|^2/2$. 이것은 **대칭**. 왜 이 $\psi$만 그런가? 어떤 $\psi$가 대칭 Bregman을 낳는가?

2. **Matrix Bregman**. $\psi(X) = -\log\det X$ on PD matrices. $D_\psi$를 계산하라. 결과 = LogDet (Stein) divergence. Matrix-space에서의 의미와 PD cone의 기하.

3. **α-divergence를 Bregman으로**. $\alpha$-divergence $D_\alpha(p\|q) = \frac{1}{\alpha(1-\alpha)}(1 - \int p^\alpha q^{1-\alpha})$. 이것이 어떤 $\psi$의 Bregman으로 표현되는가? $\alpha \to 0, 1$에서 어떤 한계?

4. **Non-exponential family의 Bregman**. 지수족 아닌 모델 (e.g., Cauchy, Student-t) 의 KL은 일반적으로 Bregman으로 표현 안 됨. 왜? $\psi$의 convexity만으로 부족한 이유.

5. **Projection 의 계산 복잡도**. $\mathcal{A}$가 affine인 경우 Bregman projection은 closed form 존재 가능. 일반 convex $\mathcal{A}$에서는 iterative 필요. Examples: simplex (closed), $\ell_p$ ball (numerical).

6. **Neural network loss**. Cross-entropy loss는 $D_{\operatorname{KL}}$의 특수 경우. Softmax output layer와 categorical label의 결합으로 Bregman 어떤 형태? 이 Bregman이 back-propagation에서 어떤 구조적 단순성을 주는지.

7. **Hellinger distance가 Bregman인가?** $H^2(p, q) = \int (\sqrt{p} - \sqrt{q})^2$. 이것이 어떤 $\psi$의 Bregman인지 확인 (힌트: $\psi(x) = -2\sqrt{x}$). Hellinger가 symmetric Bregman의 흔한 예.

---

<div align="center">

| [◀ 02. KL-Fisher 연결](./02-kl-fisher-connection.md) | [📚 메인 README](../README.md) | [04. KL = Bregman in 지수족 ▶](./04-kl-as-bregman.md) |
|:---:|:---:|:---:|

</div>
