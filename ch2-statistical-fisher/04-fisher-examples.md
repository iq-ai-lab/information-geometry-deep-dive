# 04. 주요 분포들의 Fisher 계산 — 정규·이항·다항·Dirichlet·지수족

> **"공식을 외우지 말고, 계산으로 몸에 익혀라."**

---

## 🎯 핵심 질문

**대표적인 확률분포 족에 대해 Fisher 정보행렬을 직접 계산할 수 있는가? 각 분포의 Fisher 구조에서 읽어낼 수 있는 기하학적·통계적 의미는?**

여섯 개 대표 분포:
1. **Bernoulli / 이항 $B(n, p)$**
2. **정규 $\mathcal{N}(\mu, \sigma^2)$** (1차원 및 다변량)
3. **Poisson $\operatorname{Poi}(\lambda)$**
4. **다항 $\operatorname{Mult}(n; p_1, \ldots, p_k)$**
5. **Dirichlet $\operatorname{Dir}(\alpha_1, \ldots, \alpha_k)$**
6. **일반 지수족** $p_\theta(x) = h(x)\exp(\theta^\top T(x) - \psi(\theta))$

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 분포 | AI에서의 등장 |
|---|---|
| **Bernoulli** | 이진 분류 (sigmoid), RBM, GAN discriminator |
| **정규** | VAE의 reparameterization, Gaussian policy (continuous RL), Diffusion model |
| **Poisson** | Count data, neural spike train, rare event modeling |
| **다항(Multinoulli)** | Softmax 분류, language model, categorical VAE |
| **Dirichlet** | Topic model (LDA), Bayesian softmax, Dirichlet process |
| **지수족** | **모든 위의 통합**. Natural gradient가 단순해지는 canonical 형태 (Ch4). |

각 분포의 Fisher를 직접 계산해 두면 구현 시 $F^{-1}$ 계산을 **closed form**으로 할 수 있다 — 거대 신경망에서는 어렵지만, policy network나 VAE에서는 실용적.

---

## 📐 수학적 선행 조건

- [02. Fisher 3가지 정의의 동치성](./02-fisher-3-equivalence.md) — 특히 정의 (B) $F = -\mathbb{E}[\operatorname{Hess} \log p]$
- [03. Fisher-Rao 계량](./03-fisher-rao-metric.md) — 변환법칙
- Gamma 함수의 digamma $\psi$, trigamma $\psi'$ 성질
- 다변수 Gaussian identities: $\operatorname{tr}(AB) = \sum_{ij} A_{ij}B_{ji}$, $\partial \log\det X = \operatorname{tr}(X^{-1} dX)$

---

## 📖 직관적 이해

### Fisher 계산의 레시피

1. 로그우도 $\ell_\theta(x) := \log p_\theta(x)$ 쓰기
2. 스코어 $s_i(x;\theta) = \partial_i \ell_\theta$ 계산
3. 공분산 $F_{ij} = \mathbb{E}_\theta[s_i s_j]$ 또는 음의 Hessian $F_{ij} = -\mathbb{E}_\theta[\partial_i \partial_j \ell]$
4. 기대값 취할 때 분포의 **모멘트 공식** 활용

**Tip.** 지수족의 경우 $F = \operatorname{Hess} \psi$로 $\psi$의 이차도함수만 구하면 됨 — 가장 쉬움.

### 분포별 Fisher 패턴

- **Location-scale family** ($\mu, \sigma$): $F_{\mu\sigma} = 0$ (블록 대각).
- **이산 simplex** ($p_1, \ldots, p_k$): $F = \operatorname{diag}(1/p_i)$ 형태.
- **지수족**: $F = \operatorname{Cov}[T(X)]$ (natural 좌표) — 충분통계량의 공분산.

---

## ✏️ 엄밀한 정의 — 분포별 Fisher

### 1. Bernoulli $B(1, p)$, $p \in (0, 1)$

로그우도: $\ell_p(x) = x \log p + (1-x) \log(1-p)$.

**Fisher (스칼라):**

$$
\boxed{F(p) = \frac{1}{p(1-p)}}
$$

유도: $s = \frac{x}{p} - \frac{1-x}{1-p}$, $\mathbb{E}[s^2] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p(1-p)}$.

**Fisher-Rao line element:** $ds^2 = \frac{dp^2}{p(1-p)}$. 변환 $p = \sin^2(\varphi/2)$ 하면 $ds^2 = d\varphi^2$ — simplex가 **원호** (길이 $\pi$).

---

### 2. 이항 $B(n, p)$

$n$번 독립 시행: $F_n(p) = n F(p) = \frac{n}{p(1-p)}$. (정리 3.6)

---

### 3. 정규 $\mathcal{N}(\mu, \sigma^2)$

**좌표 $(\mu, \sigma)$에서:**

$$
F = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^2 \end{pmatrix}
$$

**좌표 $(\mu, \sigma^2)$에서** (분산 매개변수):

$$
F = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}
$$

**자연 좌표 $(\eta_1, \eta_2) = (\mu/\sigma^2, -1/(2\sigma^2))$** 에서:

$$
F = \begin{pmatrix} \sigma^2 & 2\mu \sigma^2 \\ 2\mu \sigma^2 & 2\sigma^4 + 4\mu^2 \sigma^2 \end{pmatrix}
$$

(지수족 자연좌표에서는 $F = \operatorname{Cov}[T(X)]$, $T = (x, x^2)^\top$의 공분산).

---

### 4. 다변량 정규 $\mathcal{N}_d(\mu, \Sigma)$

$\theta = (\mu, \operatorname{vech}(\Sigma))$. **$\mu$ 블록**:

$$
F_{\mu_i \mu_j} = (\Sigma^{-1})_{ij}
$$

**$\mu$-$\Sigma$ 블록**: $F_{\mu \Sigma} = 0$ (직교!).

**$\Sigma$ 블록**:

$$
F_{\Sigma_{ab}, \Sigma_{cd}} = \tfrac{1}{2}(\Sigma^{-1})_{ac}(\Sigma^{-1})_{bd} + \tfrac{1}{2}(\Sigma^{-1})_{ad}(\Sigma^{-1})_{bc}
$$

(symmetrized form; $(\operatorname{vech}\Sigma)_{ab}$ 좌표에서).

**의의.** $\mu$와 $\Sigma$가 **independent** (Fisher 관점) — VAE posterior의 평균과 분산을 독립적으로 업데이트해도 NGD 관점 OK.

---

### 5. Poisson $\operatorname{Poi}(\lambda)$

$\ell_\lambda(x) = x \log \lambda - \lambda - \log(x!)$. $s = x/\lambda - 1$, $\mathbb{E}[s^2] = \operatorname{Var}(x)/\lambda^2 = 1/\lambda$.

$$
\boxed{F(\lambda) = 1/\lambda}
$$

**Fisher-Rao line element:** $ds^2 = d\lambda^2/\lambda$. 변환 $\mu = 2\sqrt{\lambda}$ 하면 $ds^2 = d\mu^2$ — **Euclidean**!

이것이 "variance-stabilizing transformation" ($\sqrt{\cdot}$) 의 information-geometric 의미. Anscombe (1948).

---

### 6. 다항 $\operatorname{Mult}(1; p_1, \ldots, p_k)$

구속 $\sum p_i = 1$. 독립 좌표 $p_1, \ldots, p_{k-1}$ (with $p_k = 1 - \sum_{i < k} p_i$).

$$
\boxed{F_{ij} = \frac{\delta_{ij}}{p_i} + \frac{1}{p_k}}, \qquad i,j \in \{1, \ldots, k-1\}.
$$

또는 chart 독립으로 simplex 위의 "상대 Fisher":

$$
F_{ij} = \frac{\delta_{ij}}{p_i}, \qquad i = 1, \ldots, k.
$$

(restricted to tangent of simplex: $\sum dp_i = 0$.)

---

### 7. Dirichlet $\operatorname{Dir}(\alpha_1, \ldots, \alpha_k)$

$p(x; \alpha) = \frac{1}{B(\alpha)} \prod_i x_i^{\alpha_i - 1}$, $\sum x_i = 1$.

로그우도: $\ell_\alpha(x) = \sum_i (\alpha_i - 1)\log x_i - \log B(\alpha)$, where $\log B(\alpha) = \sum_i \log \Gamma(\alpha_i) - \log \Gamma(\alpha_0)$, $\alpha_0 := \sum_i \alpha_i$.

$$
\boxed{F_{ij}(\alpha) = \delta_{ij}\, \psi'(\alpha_i) - \psi'(\alpha_0)}
$$

여기서 $\psi'$는 **trigamma 함수** $\psi'(z) = \frac{d^2}{dz^2} \log \Gamma(z)$.

**행렬 형태:** $F = D - \psi'(\alpha_0) \mathbf{1}\mathbf{1}^\top$, $D = \operatorname{diag}(\psi'(\alpha_i))$. **Rank-1 perturbation of diagonal** — Sherman-Morrison으로 $F^{-1}$ closed form.

---

### 8. 일반 지수족

$$
p_\theta(x) = h(x) \exp\bigl(\theta^\top T(x) - \psi(\theta)\bigr)
$$

**핵심 공식:**

$$
\boxed{F(\theta) = \operatorname{Hess}_\theta \psi(\theta) = \operatorname{Cov}_\theta[T(X)]}
$$

즉 **cumulant 함수 $\psi$의 Hessian** = **충분통계량 $T$의 공분산**.

(증명: 정규화 $\int h(x)\exp(\theta^\top T(x) - \psi(\theta))\, d\mu = 1$ 의 양변 로그 미분 → $\partial_i \psi = \mathbb{E}[T_i]$, 다시 미분 → $\partial_i \partial_j \psi = \operatorname{Cov}(T_i, T_j)$. 한편 $\partial_i \partial_j \log p = -\partial_i \partial_j \psi$이므로 정의 (B) 에 의해 $F = \operatorname{Hess}\psi$.)

> **중요.** 지수족에서 Fisher가 $\psi$의 Hessian이라는 사실이 **Legendre duality** (Ch4) 의 핵심 — expectation parameter 좌표와 natural parameter 좌표가 **Fisher-dual**.

---

## 🔬 정리와 증명

### 정리 5.1 (정규분포 Fisher)

$\mathcal{N}(\mu, \sigma^2)$에서 좌표 $(\mu, \sigma)$:

$$
F = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^2 \end{pmatrix}
$$

**증명.**

$$
\ell = -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}.
$$

1차 도함수:
$$
\partial_\mu \ell = \frac{x-\mu}{\sigma^2}, \qquad \partial_\sigma \ell = -\frac{1}{\sigma} + \frac{(x-\mu)^2}{\sigma^3}.
$$

2차 도함수:
$$
\partial_\mu^2 \ell = -\frac{1}{\sigma^2}, \quad \partial_\mu \partial_\sigma \ell = -\frac{2(x-\mu)}{\sigma^3}, \quad \partial_\sigma^2 \ell = \frac{1}{\sigma^2} - \frac{3(x-\mu)^2}{\sigma^4}.
$$

기대값 ($\mathbb{E}[(x-\mu)^2] = \sigma^2$):
$$
F_{\mu\mu} = -\mathbb{E}[\partial_\mu^2 \ell] = \frac{1}{\sigma^2}, \qquad F_{\mu\sigma} = -\mathbb{E}[\partial_\mu\partial_\sigma \ell] = 0,
$$
$$
F_{\sigma\sigma} = -\mathbb{E}[\partial_\sigma^2 \ell] = -\frac{1}{\sigma^2} + \frac{3\sigma^2}{\sigma^4} = \frac{2}{\sigma^2}.
$$

**Q.E.D.**

---

### 정리 5.2 (다변량 정규 Fisher의 블록 대각성)

$\mathcal{N}_d(\mu, \Sigma)$에서 $\mu$와 $\Sigma$ 블록 사이의 Fisher cross-term은 **0**.

**증명.**

$\ell_{\mu, \Sigma}(x) = -\tfrac{1}{2}\log\det(2\pi\Sigma) - \tfrac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)$.

$\partial_{\mu_i} \ell = (\Sigma^{-1})_{ij}(x_j - \mu_j)$.
$\partial_{\Sigma_{ab}} \ell$은 $x-\mu$의 **이차** 식. Score 공분산은

$$
F_{\mu_i, \Sigma_{ab}} = \mathbb{E}[\partial_{\mu_i} \ell \cdot \partial_{\Sigma_{ab}} \ell].
$$

$\partial_{\mu_i}\ell$은 $(x-\mu)$에 대해 **홀수 차수 (1차)** 식, $\partial_{\Sigma_{ab}}\ell$의 $x$-의존 부분은 $(x-\mu)$의 **짝수 차수 (2차)** 식. 곱은 3차식. 중심 가우시안의 모든 홀수 차수 모멘트는 0. **Q.E.D.**

**의의.** NGD에서 $\mu$ 업데이트와 $\Sigma$ 업데이트가 **분리**. VAE의 encoder가 $\mu$, $\log\sigma^2$를 별도 head로 출력하는 설계의 이론적 정당화.

---

### 정리 5.3 (지수족의 Fisher = cumulant Hessian)

$p_\theta(x) = h(x)\exp(\theta^\top T(x) - \psi(\theta))$에서

$$
F_{ij}(\theta) = \frac{\partial^2 \psi}{\partial \theta^i \partial \theta^j}.
$$

**증명.** $\log p_\theta = \log h(x) + \theta^\top T(x) - \psi(\theta)$.
$\partial_i \log p_\theta = T_i(x) - \partial_i \psi$, $\partial_i \partial_j \log p_\theta = -\partial_i \partial_j \psi$.

정의 (B): $F_{ij} = -\mathbb{E}_\theta[\partial_i \partial_j \log p_\theta] = \partial_i \partial_j \psi(\theta)$ ($\psi$는 $x$에 의존 안 하므로 기대값이 자기 자신). **Q.E.D.**

**따름 5.3.1.** $\psi$는 **convex** ($F \succeq 0$). 엄격히 convex ⟺ $T(X)$의 affine linearly independent ⟺ 모델이 정규칙 (identifiable).

---

### 정리 5.4 (Dirichlet Fisher의 rank-1 구조)

$\operatorname{Dir}(\alpha)$의 Fisher 행렬은

$$
F(\alpha) = D - \psi'(\alpha_0)\, \mathbf{1}\mathbf{1}^\top, \qquad D = \operatorname{diag}(\psi'(\alpha_i)).
$$

Sherman-Morrison으로:

$$
F^{-1}(\alpha) = D^{-1} + \frac{\psi'(\alpha_0)}{1 - \psi'(\alpha_0) \mathbf{1}^\top D^{-1} \mathbf{1}}\, D^{-1}\mathbf{1}\mathbf{1}^\top D^{-1}.
$$

**증명.** $\ell_\alpha(x) = \sum_i (\alpha_i - 1)\log x_i - \sum_i \log\Gamma(\alpha_i) + \log\Gamma(\alpha_0)$.

$\partial_i \partial_j \ell = -\delta_{ij} \psi'(\alpha_i) + \psi'(\alpha_0)$ (마지막 항은 $\partial_j \alpha_0 = 1$ 이용).

$-\mathbb{E}[\partial_i \partial_j \ell] = \delta_{ij}\psi'(\alpha_i) - \psi'(\alpha_0)$. 행렬 형태로 $F = D - \psi'(\alpha_0) \mathbf{1}\mathbf{1}^\top$. **Q.E.D.**

---

### 정리 5.5 (Poisson과 variance-stabilizing)

$\operatorname{Poi}(\lambda)$의 Fisher: $F(\lambda) = 1/\lambda$. $\mu = 2\sqrt{\lambda}$ 변환 후:

$$
\tilde F(\mu) = F(\lambda) \left(\frac{d\lambda}{d\mu}\right)^2 = \frac{1}{\lambda} \cdot \mu^2 = \frac{\mu^2}{\mu^2/4} = 4,
$$

wait, more carefully: $\lambda = \mu^2/4$, $d\lambda/d\mu = \mu/2$. $\tilde F(\mu) = (1/\lambda)(\mu/2)^2 = (4/\mu^2)(\mu^2/4) = 1$.

**결론:** $\tilde F(\mu) = 1$ (상수) — Euclidean 계량.

**의의.** $\sqrt{\cdot}$ 변환 후 Fisher 상수 ⟹ 데이터를 $\sqrt{X}$로 변환하면 분산이 $\lambda$에 덜 의존 (variance stabilization). 신경 스파이크 데이터 분석에서 선호.

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: 지수족 통합 계산기

```python
import sympy as sp
import numpy as np

def fisher_exponential_family(psi_expr, theta_syms):
    """
    지수족 p_θ(x) = h(x) exp(θ·T(x) - ψ(θ)).
    Fisher = Hess(ψ). psi_expr는 θ 심볼들의 SymPy 식.
    """
    k = len(theta_syms)
    F = sp.zeros(k, k)
    for i in range(k):
        for j in range(k):
            F[i, j] = sp.diff(psi_expr, theta_syms[i], theta_syms[j])
    return F

# 예: 정규분포 N(μ, 1), natural param η = μ
# ψ(η) = η²/2
eta = sp.Symbol('eta')
psi_normal = eta**2 / 2
F_N = fisher_exponential_family(psi_normal, [eta])
print(f"N(η, 1) Fisher: {F_N}")   # [[1]]

# 예: Bernoulli, natural param η = log(p/(1-p))
# ψ(η) = log(1 + e^η)
eta = sp.Symbol('eta')
psi_bern = sp.log(1 + sp.exp(eta))
F_B = fisher_exponential_family(psi_bern, [eta])
print(f"\nBern(η) Fisher: {sp.simplify(F_B)}")
# e^η / (1 + e^η)² = p(1-p)
# wait: F = p(1-p) in natural coord, but 1/(p(1-p)) in p-coord
# 이는 좌표변환 p = σ(η)에 따른 metric tensor 변환

# 예: Poisson, natural param η = log λ
eta = sp.Symbol('eta')
psi_pois = sp.exp(eta)
F_P = fisher_exponential_family(psi_pois, [eta])
print(f"\nPoi(η) Fisher: {F_P}")  # [[e^η]] = λ

# 예: 다변량 정규, fixed Σ=I, natural param η=μ
eta1, eta2 = sp.symbols('eta1 eta2')
psi_mvn = (eta1**2 + eta2**2) / 2
F_MVN = fisher_exponential_family(psi_mvn, [eta1, eta2])
print(f"\nN₂(η, I) Fisher:\n{F_MVN}")  # Identity
```

---

### 예제 2: Dirichlet Fisher의 수치 검증

```python
import numpy as np
from scipy.special import polygamma

def dirichlet_fisher(alpha):
    """Dir(α)의 Fisher 행렬 (정리 5.4)"""
    k = len(alpha)
    alpha_0 = np.sum(alpha)
    trigamma_i = polygamma(1, alpha)       # ψ'(α_i)
    trigamma_0 = polygamma(1, alpha_0)      # ψ'(α_0)
    F = np.diag(trigamma_i) - trigamma_0 * np.ones((k, k))
    return F

# Dirichlet(2, 3, 5)
alpha = np.array([2.0, 3.0, 5.0])
F = dirichlet_fisher(alpha)

print("Fisher matrix for Dir(2, 3, 5):")
print(F)
print(f"\neigenvalues: {np.linalg.eigvalsh(F)}")
print(f"det: {np.linalg.det(F):.6f}")
print(f"PSD? {np.all(np.linalg.eigvalsh(F) >= -1e-10)}")

# Sherman-Morrison 역행렬
F_inv_closed = (np.diag(1/polygamma(1, alpha)) 
                + polygamma(1, alpha.sum()) 
                / (1 - polygamma(1, alpha.sum()) * np.sum(1/polygamma(1, alpha)))
                * np.outer(1/polygamma(1, alpha), 1/polygamma(1, alpha)))

F_inv_direct = np.linalg.inv(F)

print(f"\n||F⁻¹_closed - F⁻¹_direct||_F = {np.linalg.norm(F_inv_closed - F_inv_direct):.2e}")
```

---

### 예제 3: 다변량 정규 Fisher — 블록 대각 확인

```python
import numpy as np

def mvn_fisher_mu_block(Sigma):
    """다변량 정규의 μ-블록: Σ⁻¹"""
    return np.linalg.inv(Sigma)

def mvn_fisher_sigma_block_vech(Sigma):
    """
    Σ 블록 (vech 좌표) — 정리 5.2
    F_{Σ_ab, Σ_cd} = (1/2)[Σ⁻¹_ac · Σ⁻¹_bd + Σ⁻¹_ad · Σ⁻¹_bc]
    """
    d = Sigma.shape[0]
    S_inv = np.linalg.inv(Sigma)
    
    # vech 인덱스 (a <= b)
    indices = [(a, b) for a in range(d) for b in range(a, d)]
    n = len(indices)
    F = np.zeros((n, n))
    
    for i, (a, b) in enumerate(indices):
        for j, (c, dd) in enumerate(indices):
            # full tensor
            val = 0.5 * (S_inv[a, c] * S_inv[b, dd] + S_inv[a, dd] * S_inv[b, c])
            # symmetry factor for off-diagonal vech entries
            if a != b: val *= 2
            if c != dd: val *= 2
            # 주의: 정확한 scaling은 vech vs vec 차이 — 여기서는 symbolic-level
            F[i, j] = val
    return F

Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
F_mu = mvn_fisher_mu_block(Sigma)
F_sig = mvn_fisher_sigma_block_vech(Sigma)

print(f"μ-block Fisher (Σ⁻¹):\n{F_mu}\n")
print(f"Σ-block Fisher (vech):\n{F_sig}\n")
print("Cross term (μ, Σ) = 0 (by Theorem 5.2)")
```

---

### 예제 4: Monte Carlo로 Fisher 검증 (Gaussian)

```python
import numpy as np
np.random.seed(42)

mu_true, sigma_true = 1.0, 2.0
N = 100000
X = mu_true + sigma_true * np.random.randn(N)

# 스코어 at (μ, σ) = (1, 2)
s_mu = (X - mu_true) / sigma_true**2
s_sigma = -1/sigma_true + (X - mu_true)**2 / sigma_true**3

# Empirical Fisher
s_stack = np.stack([s_mu, s_sigma], axis=1)
F_emp = (s_stack.T @ s_stack) / N

# Closed form
F_closed = np.array([[1/sigma_true**2, 0], [0, 2/sigma_true**2]])

print("Empirical Fisher (N=100000):")
print(F_emp)
print(f"\nClosed-form Fisher:")
print(F_closed)
print(f"\n‖diff‖_F = {np.linalg.norm(F_emp - F_closed):.4f}")
print("  (오차는 O(1/√N) 스케일)")
```

---

### 예제 5: Poisson Variance-Stabilizing 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

lambdas = np.linspace(0.5, 20, 100)

# Original: Fisher in λ
F_lambda = 1 / lambdas

# Transformed: μ = 2√λ, Fisher in μ = 1
F_mu = np.ones_like(lambdas)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(lambdas, F_lambda, 'b-', linewidth=2)
ax1.set_xlabel('λ')
ax1.set_ylabel('F(λ)')
ax1.set_title('Poisson Fisher in λ (non-constant)')
ax1.grid(alpha=0.3)

ax2.plot(2*np.sqrt(lambdas), F_mu, 'r-', linewidth=2)
ax2.set_xlabel('μ = 2√λ')
ax2.set_ylabel('F̃(μ)')
ax2.set_title('After √-transform: F̃ = 1 (Euclidean)')
ax2.set_ylim(0, 1.5)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/kind-dazzling-ritchie/poisson_variance_stabilize.png', dpi=100)
plt.close()
```

---

## 🔗 AI/ML 연결

### 1. Policy Gradient의 Gaussian Policy

Continuous control RL에서 policy $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$. Fisher block-diagonal 구조 덕에:

$$
F_{\theta\theta}^{-1} g = (F_\mu^{-1} g_\mu, F_\Sigma^{-1} g_\Sigma)
$$

— $\mu$, $\Sigma$ 네트워크 파라미터 업데이트가 분리. **TRPO / NPG 구현의 효율성 근거**.

### 2. Categorical Policy — Softmax Fisher

Softmax policy $\pi_\theta(a|s) = \operatorname{softmax}(\theta^\top \phi(s,a))$ 의 Fisher는 다항분포 Fisher의 확장:

$$
F(\theta) = \mathbb{E}_s\bigl[ \Phi_s^\top (\operatorname{diag}(\pi) - \pi \pi^\top) \Phi_s \bigr]
$$

여기서 $\Phi_s$는 feature 행렬. 이것이 **NGD for softmax classifiers**의 기본 형태.

### 3. VAE Encoder의 Fisher

VAE의 amortized posterior $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \operatorname{diag}(\sigma_\phi(x)^2))$. 정리 5.2 덕에 $\mu_\phi$, $\log\sigma_\phi$ 출력 분리 설계가 natural — Fisher 관점에서 이 parameterization이 **pre-conditioner 없이도 잘 학습**.

### 4. Dirichlet for Bayesian NN

Bayesian NN의 마지막 layer가 Dirichlet (e.g., Prior Networks): Fisher의 rank-1 구조 덕에 $F^{-1}$ closed form — 실시간 NGD 가능.

### 5. Exponential Family의 통합력

대부분의 표준 확률층 (Bernoulli output, Gaussian output, Categorical output, Poisson output) 은 지수족. Fisher = $\operatorname{Hess}\psi$ 공식으로 **일관된 자연경사 업데이트 라이브러리** 작성 가능 — TensorFlow Probability, PyTorch Distributions의 철학.

### 6. LDA와 Dirichlet Fisher

Latent Dirichlet Allocation 에서 topic prior Dir 및 문서-topic 분포 Dir. Variational EM의 M-step update rule이 Dirichlet Fisher의 trigamma 함수로 명시적 표현.

---

## ⚖️ 가정과 한계

### 좌표 선택의 실제성

- **Natural coord (θ)** 에서 Fisher = $\operatorname{Hess}\psi$, 계산 용이. 그러나 해석이 어려움 (natural param이 직관적이지 않음).
- **Expectation coord (η = E[T])** 에서 Fisher = $(\operatorname{Hess}\psi)^{-1}$ 같은 쌍대성. Ch4에서 정리.
- **Source coord** (예: $\sigma$ vs $\sigma^2$) — 연구자의 기호에 따라 다름. 변환법칙으로 항상 통합 가능.

### Trigamma 함수 수치 안정성

Dirichlet Fisher의 $\psi'(\alpha)$는 $\alpha \to 0$에서 **발산** ($\psi'(\alpha) \sim 1/\alpha^2 + \pi^2/6$). sparse prior ($\alpha \ll 1$) 에서 Fisher 조건수 폭발.

### Empirical Fisher vs 이론 Fisher

MLE 근처에서는 두 값이 일치하지만, **under-fit** 구간에서 empirical Fisher는 매우 편향. Kunstner et al. 2019. 실무에서는 **Monte-Carlo Fisher** (model samples로 평균) 사용 권장.

### 정칙성 조건 위반 사례

- **Beta(α, β) with α, β < 1**: 경계에서 분포가 $\infty$로 발산 → $\log p$의 2차 모멘트 미분 주의 필요.
- **Mixture 모델** $\sum_k \pi_k p_\theta^{(k)}$: Fisher가 **degenerate** (label switching) → 식별 가능성 문제 (identifiability).

### Overparameterized Models

딥러닝에서 $\dim(\theta) \gg \dim(\operatorname{data})$ → Fisher rank-deficient. Pseudo-inverse $F^+$ 또는 damping ($F + \lambda I$) 필요. K-FAC, Shampoo 등.

---

## 📌 핵심 정리

| 분포 | Fisher (주요 좌표) | 특징 |
|---|---|---|
| $B(1, p)$ | $1/(p(1-p))$ | simplex 경계서 발산 |
| $\mathcal{N}(\mu, \sigma^2)$ | $\operatorname{diag}(1/\sigma^2, 2/\sigma^2)$ | block-diagonal, hyperbolic 기하 |
| $\mathcal{N}_d(\mu, \Sigma)$ | $(\Sigma^{-1}, \cdot)$ block | μ-Σ 직교 |
| $\operatorname{Poi}(\lambda)$ | $1/\lambda$ | $\sqrt{}$ 변환시 Euclidean |
| $\operatorname{Mult}(p)$ | $\operatorname{diag}(1/p_i)$ (tangent) | spherical 기하 |
| $\operatorname{Dir}(\alpha)$ | $D - \psi'(\alpha_0)\mathbf{1}\mathbf{1}^\top$ | rank-1 perturbation |
| Exp Family | $\operatorname{Hess}\psi = \operatorname{Cov}[T]$ | **통합 공식** |

**핵심 Takeaway:**

1. **지수족 공식** $F = \operatorname{Hess}\psi$로 대부분의 Fisher 계산 통일.
2. **Block-diagonal 구조** (Gaussian 등) 활용하면 NGD 구현 간단.
3. **rank-1 perturbation** (Dirichlet) 등의 특수 구조는 Sherman-Morrison 활용.
4. 좌표 선택에 따라 Fisher 모양 다름 — 하지만 텐서성 덕에 항상 **변환 가능**.

---

## 🤔 생각해볼 문제

1. **좌표 변환 실습**. $\mathcal{N}(\mu, \sigma^2)$의 Fisher를 $(\mu, \tau)$, $\tau = \log \sigma$ 좌표로 변환하라. 이 새 좌표에서 Fisher가 어떻게 생겼는지 (상수 계수), 이것이 $\log$-variance 파라미터화 (VAE의 관행) 의 기하학적 의미.

2. **지수족의 cumulant 계산**. Bernoulli의 natural parameter $\eta = \log(p/(1-p))$, cumulant $\psi(\eta) = \log(1+e^\eta)$임을 보이고 $F(\eta) = \psi''(\eta) = \sigma(\eta)(1-\sigma(\eta)) = p(1-p)$임을 확인. 일반적으로 sigmoid의 도함수가 Fisher와 일치.

3. **지수족 Fisher의 기하학적 의미**. $F = \operatorname{Cov}[T(X)]$ 이므로 $F_{ij} > 0$인지 여부가 충분통계량 사이의 상관관계와 관련. 예: Gaussian의 $T = (x, x^2)$에서 $\operatorname{Cov}(x, x^2) = 2\mu \sigma^2$ 임을 확인하라.

4. **Cramér-Rao 활용**. $\mathcal{N}(\mu, 1)$에서 $F(\mu) = 1$, 따라서 unbiased estimator의 분산 하한 $= 1/N$. 표본평균 $\bar X$가 이 하한을 **달성**함을 보여라 (Cramér-Rao efficient estimator 개념).

5. **High-dim curse.** $d$-차원 Gaussian $\mathcal{N}_d(\mu, \Sigma)$의 Fisher 행렬 크기: $d + d(d+1)/2 = O(d^2)$. 이것을 실제로 저장·반전하는 것이 딥러닝 스케일에서 불가한 이유와 K-FAC/Shampoo의 Kronecker 근사가 어떤 구조를 가정하는지.

6. **Beta 분포 Fisher**. $\operatorname{Beta}(\alpha, \beta)$는 $\operatorname{Dir}(\alpha, \beta)$의 2차원 특수 경우. 그 Fisher를 명시적으로 써보고, $\alpha + \beta$가 커질 때 (sharp prior) Fisher가 어떻게 scale하는지 관찰.

7. **Score function 제로의 기하학**. $\mathbb{E}_\theta[s(X;\theta)] = 0$은 score가 접공간에서 "평균 0 벡터" 임을 의미. 이것과 Riemannian 접공간의 관계? 힌트: 다양체 접공간 $T_\theta\mathcal{S}$ 안에서 score가 basis 역할.

---

<div align="center">

| [◀ 03. Fisher-Rao 계량](./03-fisher-rao-metric.md) | [📚 메인 README](../README.md) | [05. Cramér-Rao와 기하학 ▶](./05-cramer-rao-geometry.md) |
|:---:|:---:|:---:|

</div>
