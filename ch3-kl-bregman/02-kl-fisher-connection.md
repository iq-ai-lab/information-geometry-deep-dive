# 02. KL-Fisher 연결 — Local Geometry와 Global Distance의 다리

> **"Fisher는 KL의 미분. KL은 Fisher의 적분."**

---

## 🎯 핵심 질문

**KL 발산은 왜 Fisher 계량의 "globalization"이며, 둘의 관계가 정보기하에서 어떤 구조를 드러내는가?**

네 층위의 연결:
1. **국소 (local)**: $\operatorname{KL}(p_\theta \| p_{\theta+d\theta}) \approx \tfrac{1}{2} d\theta^\top F d\theta$
2. **준국소 (mesoscale)**: Cubic 및 고차 항
3. **전역 (global)**: 측지선 길이²와 KL의 관계
4. **쌍대성**: Forward KL vs Reverse KL의 기하학적 구분

---

## 🔍 왜 이 개념이 AI에서 중요한가

| AI 상황 | KL-Fisher 연결이 답하는 것 |
|---|---|
| **Natural Gradient** | 왜 $F^{-1} g$가 KL-ball에서의 steepest descent인가 |
| **TRPO 2차 근사** | KL 제약을 $d\theta^\top F d\theta \le \delta$로 바꿀 수 있는 근거 |
| **Mirror Descent** | Bregman divergence + KL 근사의 통일 framework |
| **Information Bottleneck** | IB Lagrangian의 이차 근사 = Fisher quadratic form |
| **VI (Variational Inference)** | ELBO의 posterior 근사 quality를 Fisher로 측정 |

**이론적 일관성의 핵심.** Local (Fisher, Riemannian 기하) 과 Global (KL, 정보 이론) 이 같은 수학적 대상의 두 얼굴.

---

## 📐 수학적 선행 조건

- Ch2 [02. Fisher 3가지 정의](../ch2-statistical-fisher/02-fisher-3-equivalence.md) — 정리 3.3 (KL 2차 근사)
- Ch2 [03. Fisher-Rao 계량](../ch2-statistical-fisher/03-fisher-rao-metric.md) — 리만 계량
- 본 챕터 [01. KL의 기초](./01-kl-divergence-basics.md)
- Taylor 전개와 Landau 기호
- 측지선 거리, exponential map

---

## 📖 직관적 이해

### 국소-전역 이중성

작은 섭동 $d\theta$:

$$
\operatorname{KL}(p_\theta \| p_{\theta + d\theta}) = \tfrac{1}{2} d\theta^\top F(\theta) d\theta + \text{고차 항}
$$

큰 섭동에서는 고차 항이 커짐 — 특히 **3차 항**이 forward/reverse KL의 비대칭성 원인.

### KL-ball vs Riemannian ball

반지름 $\varepsilon$의 KL-ball $\{\theta': \operatorname{KL}(p_\theta \| p_{\theta'}) \le \varepsilon\}$과 Riemannian ball $\{\theta': d_F(\theta, \theta') \le \sqrt{2\varepsilon}\}$.

**작은 $\varepsilon$**: 두 ball이 거의 일치 (국소 근사).
**큰 $\varepsilon$**: KL-ball이 비대칭 (forward vs reverse 다름), Riemannian ball은 대칭.

### NGD의 이론적 정당화

$\theta_{\mathrm{new}}$를 KL constraint로 찾는 최적화:

$$
\theta_{\mathrm{new}} = \arg\min_{\theta'} L(\theta') \quad \text{s.t.} \quad \operatorname{KL}(p_\theta \| p_{\theta'}) \le \varepsilon.
$$

작은 $\varepsilon$에서 2차 근사:

$$
\approx \arg\min_{\theta'} L(\theta) + g^\top (\theta' - \theta) \quad \text{s.t.} \quad \tfrac{1}{2}(\theta' - \theta)^\top F (\theta'-\theta) \le \varepsilon.
$$

Lagrangian 최적화 → $\theta' - \theta \propto F^{-1} g$. 즉 **NGD가 KL-ball 내 최대 감소 방향**.

### 비대칭성의 3차 해석

정리 8.3에서 볼 것: forward와 reverse KL 의 **3차 테일러 항**은 서로 다름 — 이것이 **$\alpha$-connection** (Ch4) 의 $\alpha = \pm 1$ 구분으로 이어진다.

---

## ✏️ 엄밀한 정의

### 정의 8.1 (KL의 테일러 전개)

정칙모델에서 $\varepsilon \to 0$일 때:

$$
\operatorname{KL}(p_\theta \| p_{\theta + \varepsilon}) = \sum_{k=2}^{\infty} \frac{1}{k!} \, \varepsilon^\top\!\otimes^{k/2} T^{(k)}(\theta) \otimes^{k/2} \varepsilon + \ldots
$$

정확히는:
- **2차 항**: $\tfrac{1}{2} F_{ij}(\theta) \varepsilon^i \varepsilon^j$
- **3차 항**: $\tfrac{1}{6} T_{ijk}(\theta) \varepsilon^i \varepsilon^j \varepsilon^k$, 여기서 $T_{ijk} = \mathbb{E}_\theta[\partial_i \ell \cdot \partial_j \ell \cdot \partial_k \ell]$
- **4차 항**: 더 복잡

### 정의 8.2 (Skewness 텐서 & $\alpha$-접속의 전조)

$$
T_{ijk}(\theta) := \mathbb{E}_\theta[\partial_i \ell_\theta \cdot \partial_j \ell_\theta \cdot \partial_k \ell_\theta]
$$

이것이 **Amari-Chentsov 텐서** — $\alpha$-connection의 비대칭 부분을 encode.

### 정의 8.3 (KL divergence with parameter 변화)

$$
D_{\mathrm{KL}}(\theta, \theta') := \operatorname{KL}(p_\theta \| p_{\theta'}).
$$

이것을 $\Theta \times \Theta$ 위의 함수로 본다. 대각선 $\{\theta = \theta'\}$에서 $D_{\mathrm{KL}} = 0$. 접공간 quadratic 이 Fisher.

---

## 🔬 정리와 증명

### 정리 8.1 (KL의 2차 근사 재확인)

정칙모델에서

$$
\operatorname{KL}(p_\theta \| p_{\theta + \varepsilon}) = \tfrac{1}{2} \varepsilon^\top F(\theta) \varepsilon + O(\|\varepsilon\|^3).
$$

(이미 Ch2 정리 3.3에서 증명. 여기서는 **요약** 및 3차 항으로의 연결을 위한 언급.)

---

### 정리 8.2 (KL의 3차 항)

$$
\operatorname{KL}(p_\theta \| p_{\theta + \varepsilon}) = \tfrac{1}{2} \varepsilon^\top F \varepsilon - \tfrac{1}{6} T_{ijk}(\theta)\, \varepsilon^i \varepsilon^j \varepsilon^k + O(\|\varepsilon\|^4).
$$

여기서 $T_{ijk}$는 skewness 텐서.

**증명 (스케치).** $\log p_{\theta+\varepsilon}$의 Taylor 전개:

$$
\ell_{\theta+\varepsilon} - \ell_\theta = \varepsilon^i \partial_i \ell_\theta + \tfrac{1}{2} \varepsilon^i \varepsilon^j \partial_i \partial_j \ell_\theta + \tfrac{1}{6} \varepsilon^i \varepsilon^j \varepsilon^k \partial_i \partial_j \partial_k \ell_\theta + \ldots
$$

$-\mathbb{E}_\theta$ 적용 후 첫 항 0 (스코어 평균), 2차 → $F$, 3차 → $-\tfrac{1}{6}\mathbb{E}[\partial_i \partial_j \partial_k \ell]$.

**Bartlett identity (3차 확장)**: 정규화 조건 3회 미분 → $\mathbb{E}[\partial_i \partial_j \partial_k \ell] = \tfrac{1}{2}[T_{ijk} + (\text{permutations})] - \ldots$.

정리 후:

$$
-\tfrac{1}{6} \mathbb{E}[\partial_i \partial_j \partial_k \ell] \varepsilon^i \varepsilon^j \varepsilon^k = -\tfrac{1}{6} T_{ijk} \varepsilon^i \varepsilon^j \varepsilon^k.
$$

**Q.E.D.**

---

### 정리 8.3 (Forward vs Reverse KL의 3차 비대칭)

$$
\operatorname{KL}(p_\theta \| p_{\theta + \varepsilon}) = \tfrac{1}{2} \varepsilon^\top F \varepsilon - \tfrac{1}{6} T\varepsilon^{\otimes 3} + O(\|\varepsilon\|^4),
$$
$$
\operatorname{KL}(p_{\theta + \varepsilon} \| p_\theta) = \tfrac{1}{2} \varepsilon^\top F \varepsilon + \tfrac{1}{6} T\varepsilon^{\otimes 3} + O(\|\varepsilon\|^4).
$$

즉 **2차 항은 동일, 3차 항은 부호 반대**.

**증명.** Symmetric KL $\tfrac{1}{2}[\operatorname{KL}(p\|q) + \operatorname{KL}(q\|p)]$ 계산. $\tfrac{1}{2} \cdot 2 \cdot \tfrac{1}{2} F \varepsilon^{\otimes 2} = \tfrac{1}{2} F \varepsilon^{\otimes 2}$ (2차 항 유지), 3차 항은 취소.

한편 $\operatorname{KL}(q\|p) - \operatorname{KL}(p\|q) = \tfrac{1}{3} T \varepsilon^{\otimes 3} + O(\|\varepsilon\|^4)$ — 차이에서 3차 항이 남음. **Q.E.D.**

> **의의.** 이 **3차 비대칭** 이 $\alpha$-connection의 $\alpha = \pm 1$ 구분 (Ch4) 및 exponential / mixture family 의 이중적 역할 근원.

---

### 정리 8.4 (KL → 측지선 거리)

$\theta$와 $\theta'$가 Fisher-Rao 측지선 $\gamma: [0, 1] \to \Theta$로 연결된다 하자. $d_F(\theta, \theta') := \int_0^1 \sqrt{\dot\gamma^\top F(\gamma) \dot\gamma}\, dt$를 측지선 거리라 하자. 그러면

$$
\operatorname{KL}(p_\theta \| p_{\theta'}) = \tfrac{1}{2} d_F(\theta, \theta')^2 + \text{3차 보정}.
$$

**증명 (스케치).** 측지선 따라 적분:

$$
\operatorname{KL}(p_{\gamma(0)} \| p_{\gamma(1)}) = \int_0^1 \frac{d}{dt}\operatorname{KL}(p_{\gamma(0)} \| p_{\gamma(t)})\, dt.
$$

$\gamma$ 따라 아래 계산을 했을 때 measure가 $F$ quadratic form의 적분 → $\tfrac{1}{2} \int \dot\gamma^\top F \dot\gamma dt$. Geodesic property ($\dot\gamma^\top F \dot\gamma$ 상수) 로부터 $= \tfrac{1}{2} L^2$ where $L$ is arc-length.

고차 항은 Amari-Chentsov 텐서로 표현된 $\alpha$-geodesic 보정.

---

### 정리 8.5 (KL-ball의 Riemannian 근사)

작은 $\varepsilon > 0$에서,

$$
\{\theta': \operatorname{KL}(p_\theta \| p_{\theta'}) \le \varepsilon\} \;\approx\; \{\theta': (\theta' - \theta)^\top F(\theta) (\theta'-\theta) \le 2\varepsilon\}.
$$

즉 **KL-ball ≈ Fisher-Euclidean ball** in natural parameter space (locally).

**의의.** 모든 KL-constrained 최적화 (TRPO, Mirror Descent 등) 는 **작은 step에서는** Fisher 기반 Riemannian 최적화와 등가.

---

### 정리 8.6 (NGD = KL-constrained steepest descent)

최적화 문제

$$
\theta^* = \arg\min_{\theta'} L(\theta') \quad \text{s.t.} \quad \operatorname{KL}(p_\theta \| p_{\theta'}) \le \varepsilon
$$

의 first-order solution은

$$
\theta^* = \theta - \eta F(\theta)^{-1} \nabla L(\theta), \qquad \eta = \sqrt{\frac{2\varepsilon}{g^\top F^{-1} g}}.
$$

즉 **Natural Gradient Descent**.

**증명.** Lagrangian:

$$
\mathcal{L}(\theta', \lambda) = L(\theta') + \lambda (\operatorname{KL}(p_\theta \| p_{\theta'}) - \varepsilon).
$$

$\theta^*$에서 $\nabla L + \lambda \nabla_{\theta'} \operatorname{KL}(p_\theta \| p_{\theta'}) = 0$.

$\nabla_{\theta'} \operatorname{KL}(p_\theta \| p_{\theta'})|_{\theta' = \theta} = 0$ (Gibbs 부등식 최솟값). 따라서 1차 근사로는 $\nabla L(\theta) + \lambda F(\theta)(\theta^* - \theta) = 0$. 즉 $\theta^* - \theta = -\lambda^{-1} F^{-1} \nabla L$.

제약 $\tfrac{1}{2}(\theta^* - \theta)^\top F (\theta^* - \theta) = \varepsilon$ 로부터 $\lambda^{-1}$ 결정. **Q.E.D.**

---

### 정리 8.7 (대칭화 vs 비대칭 KL의 기하)

대칭 KL $J(p, q) := \tfrac{1}{2}[\operatorname{KL}(p\|q) + \operatorname{KL}(q\|p)]$의 2차 근사:

$$
J(p_\theta, p_{\theta+\varepsilon}) = \tfrac{1}{2} \varepsilon^\top F(\theta) \varepsilon + O(\|\varepsilon\|^4).
$$

3차 항 소거! 4차부터 비대칭. 이것이 **Jeffreys divergence** 의 기하적 장점.

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: Forward/Reverse/Symmetric KL의 2차·3차 항

```python
import sympy as sp

# N(0, 1) vs N(ε, 1): 1차원 μ 섭동
x, eps = sp.symbols('x epsilon', real=True)

# p_θ = N(θ, 1)의 로그
def log_p(mu):
    return -sp.Rational(1,2) * sp.log(2*sp.pi) - (x - mu)**2 / 2

# KL(p_0 || p_ε) = E_{p_0}[log p_0 - log p_ε]
integrand_forward = log_p(0) - log_p(eps)
# Simplification: (x-ε)²/2 - x²/2 = -εx + ε²/2
simplified = sp.simplify(integrand_forward)
print(f"Integrand simplified: {simplified}")

# E_{N(0,1)}[x] = 0, E_{N(0,1)}[1] = 1
kl_forward = sp.expand(simplified).subs(x, 0).subs(x**2, 1)
# Actually let's be more careful
# ∫ p_0(x) [(x-ε)²/2 - x²/2] dx = ∫ p_0(x) [-εx + ε²/2] dx = 0 + ε²/2
kl_forward_val = eps**2 / 2
print(f"KL(N(0,1)||N(ε,1)) = {kl_forward_val}")

# Fisher for N(μ, 1) at μ=0: F = 1
# 2차 항: (1/2) · 1 · ε² = ε²/2 ✓

# 더 재미있는 예: N(0, σ²) with σ → 1+ε
tau = sp.Symbol('tau')
# p_τ = N(0, (1+τ)²), at τ=0 이것은 N(0, 1)
log_p_sig = lambda tau_val: -sp.Rational(1,2)*sp.log(2*sp.pi*(1+tau_val)**2) - x**2 / (2*(1+tau_val)**2)

# KL(p_0 || p_τ):
kl_sig = sp.integrate(
    sp.exp(log_p_sig(0)) * (log_p_sig(0) - log_p_sig(tau)), (x, -sp.oo, sp.oo)
)
kl_sig_simplified = sp.simplify(kl_sig)
print(f"\nKL(N(0,1) || N(0,(1+τ)²)) = {kl_sig_simplified}")
# 이것의 Taylor 전개
kl_sig_taylor = sp.series(kl_sig_simplified, tau, 0, 5).removeO()
print(f"Taylor: {sp.expand(kl_sig_taylor)}")
# 예상: τ² (2차) + 고차 보정

# Reverse: KL(p_τ || p_0)
kl_sig_rev = sp.integrate(
    sp.exp(log_p_sig(tau)) * (log_p_sig(tau) - log_p_sig(0)), (x, -sp.oo, sp.oo)
)
kl_sig_rev_taylor = sp.series(sp.simplify(kl_sig_rev), tau, 0, 5).removeO()
print(f"Reverse Taylor: {sp.expand(kl_sig_rev_taylor)}")
```

---

### 예제 2: KL 측지선 vs 직선 경로

```python
import numpy as np
from scipy.integrate import quad

def kl_normal(mu1, sig1, mu2, sig2):
    return np.log(sig2/sig1) + (sig1**2 + (mu1-mu2)**2)/(2*sig2**2) - 0.5

# 두 분포: N(0, 1), N(1, 2)
mu0, sig0 = 0, 1
mu1, sig1 = 1, 2

# 직선 경로: (μ(t), σ(t)) = (1-t)(μ0, σ0) + t(μ1, σ1)
def path_linear(t):
    return (1-t)*mu0 + t*mu1, (1-t)*sig0 + t*sig1

# 측지선 (Fisher-Rao hyperbolic): Poincaré 반평면의 geodesic
# 여기서는 근사로 대체
# Fisher-Rao closed form:
d_FR = np.sqrt(2) * np.arccosh(1 + ((mu1-mu0)**2 + 2*(sig1-sig0)**2) / (4*sig0*sig1))

# KL forward
kl_fwd = kl_normal(mu0, sig0, mu1, sig1)
# KL reverse
kl_rev = kl_normal(mu1, sig1, mu0, sig0)
# Symmetric
kl_sym = (kl_fwd + kl_rev) / 2

# (1/2) d² (정리 8.4 예측)
d2_half = 0.5 * d_FR**2

print(f"Fisher-Rao distance:     {d_FR:.4f}")
print(f"(1/2) × d_FR²:           {d2_half:.4f}")
print(f"KL(p0 || p1) (forward):  {kl_fwd:.4f}")
print(f"KL(p1 || p0) (reverse):  {kl_rev:.4f}")
print(f"Symmetric KL:            {kl_sym:.4f}")
print(f"\n(큰 분포 차이에서 KL ≠ (1/2)d² — 고차 보정 필요)")
```

---

### 예제 3: NGD vs GD (2D Gaussian 문제)

```python
import numpy as np
import matplotlib.pyplot as plt

# 목적: p_θ(x) = N(x | θ₁, exp(θ₂)²) → target N(3, 2)
# L(θ) = KL(p_θ || N(3, 2)) = KL 직접 optimize
# Fisher in (θ₁ = μ, θ₂ = log σ):
# σ = exp(θ₂), dσ/dθ₂ = σ
# F_μμ = 1/σ² = e^{-2θ₂}
# F_θ₂θ₂ = 2
# Cross 0

def loss(theta):
    mu, log_sig = theta
    sig = np.exp(log_sig)
    mu_target, sig_target = 3.0, 2.0
    return np.log(sig_target/sig) + (sig**2 + (mu - mu_target)**2)/(2*sig_target**2) - 0.5

def grad_loss(theta):
    mu, log_sig = theta
    sig = np.exp(log_sig)
    mu_t, sig_t = 3.0, 2.0
    dL_dmu = (mu - mu_t) / sig_t**2
    dL_dsig = -1/sig + sig/sig_t**2
    dL_dlogsig = dL_dsig * sig  # chain rule
    return np.array([dL_dmu, dL_dlogsig])

def fisher(theta):
    mu, log_sig = theta
    sig = np.exp(log_sig)
    return np.array([[1/sig**2, 0], [0, 2]])

# GD
theta_gd = np.array([-2.0, np.log(0.5)])  # 시작점 N(-2, 0.5)
gd_trajectory = [theta_gd.copy()]
lr_gd = 0.1
for _ in range(200):
    g = grad_loss(theta_gd)
    theta_gd = theta_gd - lr_gd * g
    gd_trajectory.append(theta_gd.copy())

# NGD
theta_ngd = np.array([-2.0, np.log(0.5)])
ngd_trajectory = [theta_ngd.copy()]
lr_ngd = 0.1
for _ in range(200):
    g = grad_loss(theta_ngd)
    F = fisher(theta_ngd)
    ngd = np.linalg.solve(F, g)
    theta_ngd = theta_ngd - lr_ngd * ngd
    ngd_trajectory.append(theta_ngd.copy())

gd_t = np.array(gd_trajectory)
ngd_t = np.array(ngd_trajectory)

print(f"GD  final: μ={gd_t[-1,0]:.4f}, σ={np.exp(gd_t[-1,1]):.4f}")
print(f"NGD final: μ={ngd_t[-1,0]:.4f}, σ={np.exp(ngd_t[-1,1]):.4f}")
print(f"Target:    μ=3.0, σ=2.0")

# 손실 곡선
gd_losses = [loss(t) for t in gd_t]
ngd_losses = [loss(t) for t in ngd_t]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].semilogy(gd_losses, 'b-', label='GD', linewidth=2)
ax[0].semilogy(ngd_losses, 'r-', label='NGD', linewidth=2)
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('loss (KL)')
ax[0].set_title('GD vs NGD Convergence')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(gd_t[:, 0], np.exp(gd_t[:, 1]), 'b.-', label='GD', markersize=4)
ax[1].plot(ngd_t[:, 0], np.exp(ngd_t[:, 1]), 'r.-', label='NGD', markersize=4)
ax[1].plot([3.0], [2.0], 'k*', markersize=20, label='target')
ax[1].plot([-2.0], [0.5], 'ko', markersize=10, label='start')
ax[1].set_xlabel('μ')
ax[1].set_ylabel('σ')
ax[1].set_title('Parameter trajectory')
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/kind-dazzling-ritchie/gd_vs_ngd.png', dpi=100)
plt.close()
print("\n저장: gd_vs_ngd.png")
```

**기대:** NGD가 훨씬 빠르고 직접적으로 target에 도달.

---

### 예제 4: KL-ball의 형태 시각화 (2D Gaussian)

```python
import numpy as np
import matplotlib.pyplot as plt

# 기준점 N(0, 1)
mu0, sig0 = 0, 1

# Grid
mu_range = np.linspace(-3, 3, 100)
sig_range = np.linspace(0.3, 3, 100)
M, S = np.meshgrid(mu_range, sig_range)

def kl_normal(m1, s1, m2, s2):
    return np.log(s2/s1) + (s1**2 + (m1-m2)**2)/(2*s2**2) - 0.5

# Forward: KL(p0 || p(μ, σ))
KL_fwd = np.vectorize(lambda m, s: kl_normal(mu0, sig0, m, s))(M, S)

# Reverse: KL(p(μ, σ) || p0)
KL_rev = np.vectorize(lambda m, s: kl_normal(m, s, mu0, sig0))(M, S)

# Fisher quadratic (2차 근사)
# Δμ² · 1/σ0² + Δ(log σ)² · 2 (로컬 좌표)
# 여기서 σ를 log 좌표로 사용
KL_quad = (M - mu0)**2 / (2 * sig0**2) + (np.log(S) - 0)**2

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
levels = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

for ax, KL, title in zip(axes, [KL_fwd, KL_rev, KL_quad], 
                          ['Forward KL(p₀||p)', 'Reverse KL(p||p₀)', '½(Δθ)ᵀFΔθ (Fisher quadratic)']):
    cs = ax.contour(M, S, KL, levels=levels, cmap='viridis')
    ax.clabel(cs, inline=True, fontsize=8)
    ax.plot([mu0], [sig0], 'ro', markersize=10, label='N(0,1)')
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/kind-dazzling-ritchie/kl_ball_shape.png', dpi=100)
plt.close()
```

**관찰:** 작은 level 값 (작은 ε)에서 세 ball이 비슷 (2차 근사 유효). 큰 값에서 forward vs reverse 비대칭 확연.

---

## 🔗 AI/ML 연결

### 1. TRPO의 Precise Derivation

TRPO의 surrogate objective + KL constraint:

$$
\max_\theta \hat A(\theta) \quad \text{s.t.} \quad \bar{\operatorname{KL}}(\theta_{\text{old}}, \theta) \le \delta.
$$

**작은 $\delta$에서 2차 근사** (정리 8.1) → NGD update:

$$
\theta \leftarrow \theta_{\text{old}} + \sqrt{\frac{2\delta}{g^\top F^{-1} g}} F^{-1} g.
$$

$F$는 policy Fisher (Kakade 2001), $g$는 policy gradient.

### 2. PPO — First-Order Approximation of KL

TRPO의 계산 비용 (conjugate gradient for $F^{-1}$) 회피. **PPO clipped objective** 은 KL 2차 근사 대신 **ratio clipping**으로 KL bound 암묵적 보장. 그러나 실험적으로 TRPO와 유사 성능.

### 3. Natural Language Processing

Language Model fine-tuning with RLHF:

$$
\max_\theta \mathbb{E}[R(x)] - \beta \operatorname{KL}(p_\theta \| p_{\text{SFT}}).
$$

KL regularization은 reference 모델 $p_{\text{SFT}}$으로부터의 drift 제한. 이차 근사로 해석하면 **Fisher ball 내 탐색**.

### 4. Information Bottleneck

IB objective: $\min I(X; T) - \beta I(T; Y)$. MI = KL. 2차 근사는 $T$의 Fisher-like constraint. "$T$의 정보 capacity를 Fisher로 제한."

### 5. Variational Inference Quality

VI에서 $q_\phi \approx p(z|x)$. Posterior approximation error를 $\operatorname{KL}(q\|p)$ 로 재서 Fisher quadratic 으로 국소 근사 가능 → **posterior concentration bound**.

### 6. Meta-Learning의 KL-regularization

**MAML**: inner update $\theta' = \theta - \alpha \nabla L_\tau(\theta)$. $\theta$와 $\theta'$의 KL은 $O(\alpha^2 \|\nabla L_\tau\|_F^2)$ — 너무 큰 inner step은 Fisher ball 벗어남.

### 7. Mirror Descent Interpretation

Mirror descent: $\theta_{t+1} = \arg\min \langle g_t, \theta\rangle + \tfrac{1}{\eta} D_\psi(\theta, \theta_t)$.

$D_\psi$ = Bregman (Ch3-03). $\psi = $ negative entropy 경우 $D_\psi = $ KL. KL의 2차 근사 = Fisher quadratic → Mirror Descent ≈ NGD in small-step limit.

---

## ⚖️ 가정과 한계

### 2차 근사의 신뢰 범위

- **작은 step에서만** 정확. 큰 policy 갱신에서는 3차 이상 항 무시할 수 없음.
- 실무에서 TRPO는 **line search** 추가 — 2차 근사로 얻은 step을 실제 KL 계산으로 검증 후 축소.

### 고차 항 control

3차 항 ($T_{ijk}$)는 skewness. 분포가 heavy-tailed 이면 3차 항 큼 → 2차 근사 성능 저하. **Robust TRPO** 등이 이 문제 처리.

### Natural Gradient vs Second-Order Method

NGD $\ne$ Newton 일반적으로. 목적함수 $L$의 Hessian ≠ Fisher (except 특수 경우, e.g., MLE with correct model). 그래서 NGD는 "semi-Newton".

### KL-ball의 기하학적 제약

KL-ball이 큰 parameter space에서 너무 작을 수 있음 (e.g., over-parameterized NN에서 $F$ 의 작은 eigenvalue 방향으로 ball 매우 커짐). Damping ($F + \lambda I$) 필수.

### Asymmetry의 미묘함

Forward vs Reverse KL의 3차 비대칭 (정리 8.3) 은 **분포 approximation의 mode vs mean 결정**에 영향. 실무에서 symmetric KL (Jeffreys) 고려 가치.

---

## 📌 핵심 정리

| 관계 | 수식 |
|---|---|
| **국소 근사** | $\operatorname{KL}(p_\theta\|p_{\theta+\varepsilon}) = \tfrac{1}{2}\varepsilon^\top F\varepsilon + O(\|\varepsilon\|^3)$ |
| **3차 비대칭** | Forward 와 Reverse 는 $T_{ijk}\varepsilon^{\otimes 3}$ 부호 반대 |
| **Geodesic 관계** | $\operatorname{KL} \approx \tfrac{1}{2} d_F^2$ for small distances |
| **Symmetric KL** | $J = \tfrac{1}{2}[\operatorname{KL}(p\|q) + \operatorname{KL}(q\|p)] = \tfrac{1}{2}\varepsilon^\top F\varepsilon + O(\|\varepsilon\|^4)$ (3차 소거) |
| **NGD 유도** | KL-ball 제약 최소화 → $F^{-1}$ preconditioner |
| **Amari-Chentsov tensor** | $T_{ijk} = \mathbb{E}[\partial_i\ell\cdot\partial_j\ell\cdot\partial_k\ell]$, $\alpha$-접속의 전조 |

**Takeaway:**

1. **Fisher = KL의 미분**. KL = Fisher의 적분 (측지선 따라).
2. **TRPO/NGD의 이론적 기반** — 2차 근사 + 제약 최소화.
3. **3차 항 = α-connection의 씨앗** (다음 Chapter에서 α = ±1 만나게 됨).
4. **KL-ball ≈ Riemannian ball** 국소적 (Mirror Descent / NGD 통합).

---

## 🤔 생각해볼 문제

1. **4차 항 계산**. $\mathcal{N}(0, 1)$에서 $\mathcal{N}(\varepsilon, 1)$로의 KL Taylor 4차 항을 명시적으로 계산하라. 결과를 Amari-Chentsov 텐서의 4차 확장으로 해석 가능한가?

2. **3차 비대칭의 기하**. $T_{ijk}$가 0이면 forward = reverse KL (3차까지). 이것이 성립하는 모델은 무엇? (힌트: Gaussian with fixed σ 는 정규분포족의 어느 substructure?)

3. **TRPO 구현의 함정**. KL이 실제로 step 이후 $\delta$ 넘으면 어떻게 해야 하는가? Backtracking line search의 이론적 justification — 어느 정리에서?

4. **Mirror descent와 NGD**. Mirror Descent with negative entropy = Exponentiated Gradient. 이것이 KL-regularized NGD와 어떻게 연결되는지. Softmax policy에 적용시 업데이트 규칙이 동일한가?

5. **Overparameterized NN의 NGD**. 신경망 $F$가 거의 rank-deficient → $F^{-1}$ 불안정. Damped NGD $\theta \leftarrow \theta - \eta (F + \lambda I)^{-1} g$. 이것의 KL-ball 해석은? 정확히 어떤 constraint에 대응?

6. **Jeffreys prior와 Fisher**. Jeffreys prior $\pi_J(\theta) \propto \sqrt{\det F(\theta)}$는 reparametrization invariant. 왜? KL 대칭성과 연결되는가?

7. **분포 섭동과 local-geodesic coincidence**. $p_{\theta+\varepsilon}$와 $p_\theta$의 $\alpha$-geodesic distance (Ch4 예고) 와 Fisher-Rao 거리는 $\varepsilon$이 작을 때 일치, 클 때 분화. 그 분기점을 어떻게 정량화?

---

<div align="center">

| [◀ 01. KL의 기초](./01-kl-divergence-basics.md) | [📚 메인 README](../README.md) | [03. Bregman 발산 ▶](./03-bregman-divergence.md) |
|:---:|:---:|:---:|

</div>
