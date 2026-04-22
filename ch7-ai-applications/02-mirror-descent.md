# 02. Mirror Descent와 쌍대공간 최적화

> **"유클리드는 원을 그린다. Mirror descent는 여러 기하에 맞는 다양한 '공'을 고른다."**

---

## 1. 왜 이 주제인가?

Gradient descent는 유클리드 기하에 맞춘 제약 최적화:

$$
\theta_{k+1} = \arg\min_\theta \langle g_k, \theta \rangle + \frac{1}{\eta} \|\theta - \theta_k\|^2/2.
$$

이 식의 $\|\cdot\|^2/2$는 "Euclidean proximal term". 하지만 문제의 기하가 유클리드가 아니라면? Simplex, 양의 octant, 또는 KL 공간이라면?

**Mirror Descent** (Nemirovski & Yudin 1983)은 이 proximal term을 **Bregman divergence**로 바꾸어 일반 기하로 확장:

$$
\theta_{k+1} = \arg\min_\theta \langle g_k, \theta \rangle + \frac{1}{\eta} B_\phi(\theta, \theta_k).
$$

놀랍게도 **exp family에서 $\phi = \psi$ (cumulant)**로 선택하면 mirror descent가 **natural gradient descent와 동치**. 이는 Amari의 information geometry와 Nemirovski의 convex analysis가 같은 현상의 두 얼굴임을 밝히는 심오한 결과다.

---

## 2. 학습 목표

1. **Bregman divergence** $B_\phi(x, y) = \phi(x) - \phi(y) - \nabla\phi(y)^T(x-y)$의 정의와 성질.
2. **Mirror descent** 업데이트 유도.
3. **Dual space formulation**: $\theta^* = \nabla\phi(\theta)$, dual update then primal map-back.
4. **$\phi = \psi$에서 MD = NGD** 동치 증명.
5. **Exponentiated Gradient** (simplex), **EG on multi-armed bandit**, **Hedge algorithm** 예제.

---

## 3. 전제 지식

- **Ch4-03**: Legendre 변환, $\psi \leftrightarrow \psi^*$
- **Ch5-02**: Natural gradient
- **Convex analysis**: Convex conjugate, Fenchel-Young inequality.

---

## 4. 직관적 설명

### 4.1 "공"을 고르다

GD: $\|\theta - \theta_k\| \leq \varepsilon$ 유클리드 공. 공 중심에서 $g$ 방향으로 내려옴.

MD: $B_\phi(\theta, \theta_k) \leq \delta$ **Bregman 공**. $\phi$에 따라 공 모양이 다름:

- $\phi = \|x\|^2/2$: 유클리드 구.
- $\phi = x \log x$ (negative entropy): KL ball (simplex 기하).
- $\phi = \psi$ (cumulant of exp family): Fisher ball = NPG.

### 4.2 Primal-Dual View

Mirror descent는 **쌍대공간**에서 덧셈으로 진행, **원시공간**으로 돌아올 땐 비선형 변환.

1. $\theta_k^* := \nabla\phi(\theta_k)$ — 쌍대 좌표.
2. $\theta_{k+1}^* = \theta_k^* - \eta g_k$ — 쌍대에서 단순 gradient step.
3. $\theta_{k+1} = (\nabla\phi)^{-1}(\theta_{k+1}^*) = \nabla\phi^*(\theta_{k+1}^*)$ — 원시로 환원.

이 "**dual step → primal map-back**"이 mirror descent의 기하.

### 4.3 왜 Simplex에서 Exp?

Simplex $\Delta_{K-1} = \{p : p_k > 0, \sum p_k = 1\}$에서 GD는 제약을 깬다 (음수 or 정규화 위반). MD with $\phi = \sum p_k \log p_k$:

$$
p_{k+1}(i) \propto p_k(i) \exp(-\eta g_k(i)).
$$

**Exponentiated Gradient**. Simplex 자동 유지 + multiplicative 업데이트.

---

## 5. 엄밀한 정의와 정리

### 5.1 Bregman Divergence

**정의 5.1.** $\phi: \mathcal{X} \to \mathbb{R}$ strictly convex, differentiable. Bregman divergence:

$$
B_\phi(x, y) := \phi(x) - \phi(y) - \nabla\phi(y)^T(x - y).
$$

**성질**:
- $B_\phi(x, y) \geq 0$, $= 0 \iff x = y$ (Jensen).
- 비대칭 ($B_\phi(x, y) \neq B_\phi(y, x)$ 일반적).
- $\phi = \|x\|^2/2$면 $B = \|x-y\|^2/2$ (유클리드).
- $\phi = \sum x_i \log x_i - x_i$면 $B(x, y) = \sum x_i \log(x_i/y_i) - (x_i - y_i)$ (KL).

### 5.2 Mirror Descent

**정의 5.2 (Nemirovski & Yudin 1983).**

$$
\boxed{\theta_{k+1} = \arg\min_\theta \left[\langle g_k, \theta \rangle + \frac{1}{\eta} B_\phi(\theta, \theta_k)\right].}
$$

### 5.3 Dual 형태

**정리 5.3.** MD update의 해:

$$
\nabla\phi(\theta_{k+1}) = \nabla\phi(\theta_k) - \eta g_k.
$$

즉 **dual 좌표에서 Euclidean gradient step**, 그 후 primal로 map-back.

### 5.4 MD = NGD when $\phi = \psi$

**정리 5.4 (Amari-Cichocki 2010).** $\phi = \psi$ (exp family cumulant function)면 mirror descent (in $\eta$ coordinates) = natural gradient descent (in $\theta$ coordinates).

### 5.5 Convergence

**정리 5.5.** $f$ convex L-Lipschitz, $\phi$ $\sigma$-strongly convex. Mirror descent의 $T$-step average의 regret:

$$
\frac{1}{T}\sum_{k=1}^T f(\theta_k) - f(\theta^*) \leq O\left(\sqrt{\frac{L^2 \log|\Theta|}{\sigma T}}\right).
$$

Simplex $\Delta_{K-1}$, $\phi = $ negative entropy ($\sigma = 1$): $O(\sqrt{\log K / T})$. **Dimension-independent**이 GD의 $O(\sqrt{K/T})$보다 훨씬 우수 (simplex).

---

## 6. 증명

### 6.1 Dual form (정리 5.3)

MD update:

$$
\theta_{k+1} = \arg\min_\theta \langle g_k, \theta \rangle + \frac{1}{\eta}[\phi(\theta) - \phi(\theta_k) - \nabla\phi(\theta_k)^T(\theta - \theta_k)].
$$

$\theta_k$ 관련 상수 항 제거, $\theta$에 대해 편미분:

$$
g_k + \frac{1}{\eta}[\nabla\phi(\theta) - \nabla\phi(\theta_k)] = 0.
$$

$$
\nabla\phi(\theta_{k+1}) = \nabla\phi(\theta_k) - \eta g_k. \quad \square
$$

### 6.2 MD = NGD (정리 5.4)

**Exp family 설정**: 분포 $p_\theta \propto \exp(\theta^T T - \psi(\theta))$. 

- $\eta = \nabla\psi(\theta)$ (expectation parameter).
- $\phi = \psi$ in $\theta$-coordinate, 혹은 $\phi^* = \psi^*$ in $\eta$-coordinate.

**MD in $\theta$-coordinate with $\phi = \psi$**:

$$
\nabla\psi(\theta_{k+1}) = \nabla\psi(\theta_k) - \eta g_k,
$$

즉 $\eta_{k+1} = \eta_k - \eta g_k$ (쌍대 공간에서 Euclidean step).

Primal 환원: $\theta_{k+1} = \nabla\psi^*(\eta_{k+1})$.

**NGD의 연속 flow**:

$$
\dot\theta = -F(\theta)^{-1} g = -(\nabla^2\psi)^{-1} g.
$$

$\eta = \nabla\psi(\theta)$의 시간 미분: $\dot\eta = \nabla^2\psi(\theta) \dot\theta = -g$.

즉 **$\eta$ 공간에서 Euclidean flow** = NGD. MD와 정확히 같은 변화. $\square$

### 6.3 Exponentiated Gradient from MD

$\phi(p) = \sum p_i \log p_i$ (negative entropy on simplex).

$$
\nabla\phi(p) = (1 + \log p_1, \dots, 1 + \log p_K).
$$

MD update:

$$
1 + \log p_{k+1}(i) = 1 + \log p_k(i) - \eta g_k(i)
$$

$$
\Rightarrow p_{k+1}(i) = p_k(i) \exp(-\eta g_k(i)),
$$

정규화 후 simplex 제약 만족. **Exponentiated Gradient (EG)**.

### 6.4 Regret 분석 (정리 5.5)

Online convex optimization setup. Bregman potential의 smoothness + strong convexity로 three-point identity:

$$
B_\phi(u, \theta_{k+1}) + B_\phi(\theta_{k+1}, \theta_k) = B_\phi(u, \theta_k) + \nabla\phi(\theta_{k+1}) - \nabla\phi(\theta_k))^T(u - \theta_{k+1}).
$$

Iterative summing + Cauchy-Schwarz → regret bound. $\square$

---

## 7. 구체 예제

### 7.1 Simplex: EG on prediction with experts

$K$ experts, loss $\ell_i^t \in [0, 1]$. Hedge algorithm (Freund & Schapire 1997):

$$
p^{t+1}(i) \propto p^t(i) e^{-\eta \ell_i^t}.
$$

이는 EG = MD with negative entropy. Regret $O(\sqrt{T \log K})$.

### 7.2 Positive orthant: EG 변형

$\mathcal{X} = \mathbb{R}_+^n$. $\phi(x) = \sum x_i \log x_i$:

$$
x_i^{k+1} = x_i^k \exp(-\eta g_i^k).
$$

제약 없는 multiplicative weight.

### 7.3 L_∞ ball: $\phi(x) = \frac{1}{2}\|x\|_p^2$

$p > 2$ 등 다양한 norm. Online learning에서 최적 regret 분석.

### 7.4 KL ball (exp family = NGD)

$\phi(\theta) = \psi(\theta)$ cumulant. 

MD의 $\eta$-space Euclidean step = NGD 한 step. 특히:

- Gaussian family, $\psi(\theta) = \theta^2/2$ (fixed $\sigma$): $\phi = \theta^2/2$, MD = GD (trivial).
- Exp(\lambda)$: $\psi(\theta) = -\log(-\theta)$, MD가 특이 form.

### 7.5 RLHF에서 Soft Q-learning

Energy-based policy $\pi \propto \exp(Q/\tau)$. KL-regularized Bellman = Mirror descent in Q-space with $\phi$ = negative entropy scaled by $\tau$.

---

## 8. Python 코드 검증

### 8.1 EG on simplex (experts)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
K, T = 10, 500
eta = np.sqrt(np.log(K) / T)  # optimal rate

# Adversarial losses (random in [0,1])
losses = np.random.rand(T, K)

# EG
p = np.ones(K) / K
cum_loss_eg = 0
cum_losses_eg = []
for t in range(T):
    # Pick expert according to p (here we use expected loss)
    ell = np.sum(p * losses[t])
    cum_loss_eg += ell
    cum_losses_eg.append(cum_loss_eg)
    # Update
    p *= np.exp(-eta * losses[t])
    p /= p.sum()

# Best expert (hindsight)
best = np.argmin(losses.sum(0))
cum_best = np.cumsum(losses[:, best])

# Regret
regret = np.array(cum_losses_eg) - cum_best
plt.plot(regret, label='EG regret')
plt.plot(np.sqrt(np.arange(T) * np.log(K)), 'r--', label='$\\sqrt{T \\log K}$')
plt.legend(); plt.grid()
plt.xlabel('t'); plt.ylabel('regret')
plt.title('Exponentiated Gradient: sublinear regret')
```

**기대**: Regret ∝ $\sqrt{T \log K}$ (logarithmic in $K$).

### 8.2 EG vs projected GD on simplex

```python
import numpy as np

np.random.seed(1)
K = 20
# Objective: quadratic on simplex, minimum at some p*
Q = np.random.randn(K, K); Q = Q @ Q.T  # PSD
b = np.random.randn(K)
def loss(p):
    return 0.5 * p @ Q @ p + b @ p

def grad_loss(p):
    return Q @ p + b

# EG
p_eg = np.ones(K) / K
for t in range(200):
    g = grad_loss(p_eg)
    p_eg *= np.exp(-0.05 * g)
    p_eg /= p_eg.sum()

# Projected GD: step then project onto simplex
def project_simplex(v):
    # Sort, find threshold
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.where(u - cssv/(np.arange(len(u))+1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)

p_gd = np.ones(K) / K
for t in range(200):
    g = grad_loss(p_gd)
    p_gd = project_simplex(p_gd - 0.05 * g)

print(f"EG loss: {loss(p_eg):.4f}")
print(f"GD loss: {loss(p_gd):.4f}")
# 비슷한 수렴, 하지만 EG가 simplex boundary 안전하게
```

### 8.3 MD ≡ NGD for exponential family

```python
import numpy as np

# Gaussian with known variance, θ = μ
# ψ(θ) = θ²/2, η = ∇ψ(θ) = θ
# MD with φ = ψ: nablaψ(θ_{k+1}) = nablaψ(θ_k) - η g_k
#                  θ_{k+1} = θ_k - η g_k
# NGD: F = 1, F^{-1} = 1, so Δθ = -η g_k
# 동일!

# Bernoulli: θ = logit, ψ(θ) = log(1+e^θ), η = σ(θ)
# MD: η_{k+1} = η_k - lr · g_k
# Then θ_{k+1} = logit(η_{k+1})
# 
# NGD: F(θ) = σ(1-σ), so Δθ = -lr · g / (σ(1-σ))
# Let's verify they give same η updates

theta = 0.0  # initial logit
g = 0.5  # gradient
lr = 0.1
# MD:
eta = 1/(1 + np.exp(-theta))  # σ(θ)
eta_new = eta - lr * g
# But eta must be in (0, 1)
eta_new = np.clip(eta_new, 0.001, 0.999)
theta_md = np.log(eta_new / (1 - eta_new))

# NGD:
F = eta * (1 - eta)
theta_ngd = theta - lr * g / F

# For small lr, these should agree up to O(lr²)
print(f"MD: θ = {theta_md:.6f}")
print(f"NGD: θ = {theta_ngd:.6f}")
print(f"Diff: {abs(theta_md - theta_ngd):.2e}")
# Small difference due to finite lr (continuous limit they're equal)
```

---

## 9. AI/ML 연결

### 9.1 AdaGrad as MD

Duchi+ 2011: AdaGrad = MD with adaptive $\phi_t$ based on cumulative gradients. Per-feature learning rate.

### 9.2 Adam as Adaptive MD

Momentum + adaptive rate. Technically not pure MD, but geometric intuition similar.

### 9.3 RLHF / InstructGPT

PPO objective에 KL penalty = mirror descent regularization. KL ball trust region for policy update.

### 9.4 Online Learning / Bandit

Hedge, EXP3, Follow-the-Regularized-Leader (FTRL) = MD variants.

### 9.5 Implicit Bias of MD

MD with $\phi = \|x\|_p^2/2$ in linear classification converges to max-margin solution in $\ell_p$ norm. Implicit regularization of deep learning (Gunasekar+ 2018).

### 9.6 Variational Methods

Khan & Rue (2023) Bayesian Learning Rule: VI = MD on natural parameters with entropy potential. Unifies VI, SGD, Laplace approximation.

---

## 10. 흔한 오해와 함정

1. **"MD는 GD의 일반화"가 아니라 "다른 기하에서의 GD"**.
   - 유클리드 기하가 최적이 아닐 때 MD가 dimension-independent regret.

2. **$\phi$ 선택이 결정적**.
   - Simplex: negative entropy.
   - Positive orthant: $x\log x$ or $x^{1/2}$.
   - Stiefel manifold: retractive 구조.

3. **Bregman divergence는 metric 아님**.
   - 비대칭, 삼각 부등식 X. KL이 가장 유명한 예.

4. **Dual map $\nabla\phi$의 closed form**.
   - Simplex + entropy → softmax. Non-trivial for other potentials.

5. **MD in infinite-dim**.
   - Functional spaces (Hilbert, Banach) 확장 가능. 연산 복잡.

6. **Strong convexity 요구**.
   - $\phi$가 strictly convex만으로 충분 (정리 5.5는 strong convexity).

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **Bregman divergence 예제**: $\phi(x) = x \log x - x$의 Bregman을 계산하고 KL과 연결.

2. **MD = NGD 상세**: Gaussian ($\psi = \theta^2/(2\sigma^2)$)에서 MD와 NGD가 동일 업데이트를 주는 과정 상세 유도.

3. **Hedge regret 증명**: EG의 regret bound $O(\sqrt{T \log K})$를 정리 5.5 기반으로 증명.

4. **AdaGrad as MD**: AdaGrad 업데이트를 적절한 $\phi_t$로 MD로 표현.

5. **Continuous-time MD**: MD의 연속 flow $\dot\theta = -[\nabla^2\phi(\theta)]^{-1} \nabla f$가 NGD와 동일함을 증명.

6. **Non-convex MD**: Non-convex $f$에서 MD의 convergence 논의.

### 다음 단계

- **[03. VAE Geometry](./03-vae-geometry.md)**: Generative models.
- **[Ch7-04. RHMC](./04-riemannian-hmc.md)**: Fisher를 mass matrix로.

---

**참고문헌**

- Nemirovski, A., Yudin, D. (1983). *Problem Complexity and Method Efficiency in Optimization*.
- Beck, A., Teboulle, M. (2003). *Mirror Descent and Nonlinear Projected Subgradient Methods*.
- Amari, S. (1998). *Natural Gradient Works Efficiently*.
- Cichocki, A., Amari, S. (2010). *Families of Alpha-Beta-Gamma Divergences*.
- Bubeck, S. (2015). *Convex Optimization: Algorithms and Complexity*, Ch. 4.
- Duchi, J.+ (2011). *Adaptive Subgradient Methods* (AdaGrad).
- Khan, M., Rue, H. (2023). *The Bayesian Learning Rule*.
- Freund, Y., Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning* (Hedge).

---

[◀ 01. Natural Policy Gradient](./01-natural-policy-gradient.md) | [📚 README](../README.md) | [03. VAE Geometry ▶](./03-vae-geometry.md)
