# 05. α-divergence와 Rényi divergence — KL의 일반화 가족

> **"KL은 무한한 divergence 가족의 한 점일 뿐. α를 바꾸면 다른 기하가 열린다."**

---

## 🎯 핵심 질문

**KL은 α-divergence 가족의 특수 경우이며, α-divergence는 어떻게 정보기하의 α-connection 구조로 확장되는가?**

$$
\boxed{
\begin{aligned}
\text{α-divergence} \quad D_\alpha(p \| q) &= \frac{1}{\alpha(1-\alpha)} \left[1 - \int p^\alpha q^{1-\alpha}\, d\mu\right] \\[4pt]
\text{Rényi divergence} \quad R_\alpha(p \| q) &= \frac{1}{\alpha - 1} \log \int p^\alpha q^{1-\alpha}\, d\mu
\end{aligned}}
$$

특수 한계:
- $\alpha = 1$: $D_\alpha = R_\alpha = \operatorname{KL}(p \| q)$.
- $\alpha = 0$: $D_\alpha = \operatorname{KL}(q \| p)$ (reverse), $R_\alpha = -\log \int_{\{p > 0\}} q$ (mass).
- $\alpha = 1/2$: $D_{1/2}$ = Hellinger² (대칭), $R_{1/2} = -2\log(1 - \tfrac{1}{2}H^2)$.
- $\alpha = 2$: $D_2 = \tfrac{1}{2}\chi^2$ divergence.

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 문제 | α-divergence의 역할 |
|---|---|
| **VAE 일반화** | β-VAE, IWAE, Rényi-VAE (Li & Turner 2016) — α로 tightness-diversity tradeoff |
| **Robust ML** | 작은 α (mode-seeking) vs 큰 α (mass-covering). Outlier 영향 조절 |
| **GAN objectives** | f-GAN — 임의 f-divergence로 GAN 확장 (Nowozin 2016) |
| **Bayesian inference** | α-VI: α≠1에서 posterior approximation의 다른 trade-off |
| **Privacy (DP)** | Rényi Differential Privacy (Mironov 2017) — composition이 Rényi에서 additive |
| **Information theory** | Rényi 엔트로피, min-entropy, Hartley 엔트로피의 unified framework |

**KL의 한계 극복.** 모드 문제 (mode collapse vs blurring), outlier 민감성, tail behavior — α 선택으로 tune.

---

## 📐 수학적 선행 조건

- [01-04] 모두
- f-divergence 개념 (Csiszár)
- Hölder 부등식
- Convex analysis 기초
- Chernoff bound 기본 개념

---

## 📖 직관적 이해

### α의 "mixing ratio" 해석

$\int p^\alpha q^{1-\alpha}$ = **geometric interpolation** between $p$ and $q$. α는 "어느 쪽으로 기울어질 것인가":

- $\alpha \to 1$: $p$-dominated — "$p$에서 본 $q$의 차이" (forward KL).
- $\alpha \to 0$: $q$-dominated — "$q$에서 본 $p$의 차이" (reverse KL).
- $\alpha = 1/2$: symmetric, Bhattacharyya coefficient.
- $\alpha > 1$: $p$에서 **$q$가 매우 작은 영역**에 민감 (tail-sensitive).
- $\alpha < 0$: $q$-tail sensitive.

### Mode vs Mean Seeking Spectrum

VAE의 $\operatorname{KL}(q \| p)$ (reverse) → mode collapse 경향.
$\operatorname{KL}(p \| q)$ (forward) → mean/blurring.
**α ∈ (0, 1)**: 중간 — 적절히 조절.

Rényi-VAE (Li & Turner 2016): α > 1에서 tighter ELBO, α < 1에서 더 diverse posterior.

### Rényi vs α divergence 차이

- **α-divergence**: f-divergence 가족 — Pinsker-type, convexity, DPI 만족.
- **Rényi divergence**: log 형태 — composition과 moment generating function에 적합.

두 divergence는 **monotone 변환**으로 관련:

$$
R_\alpha(p \| q) = \frac{1}{\alpha - 1} \log(1 - \alpha(1-\alpha) D_\alpha(p \| q)).
$$

작은 divergence에서 $R_\alpha \approx D_\alpha \cdot \alpha$.

### α-family의 기하학

Amari: **α-connection** $\nabla^{(\alpha)}$ 은 $\alpha = 1$에서 **e-connection** (exponential geodesic), $\alpha = -1$에서 **m-connection** (mixture geodesic). 일반 $\alpha$는 둘의 "convex combination":

$$
\nabla^{(\alpha)} = \frac{1+\alpha}{2} \nabla^{(e)} + \frac{1-\alpha}{2} \nabla^{(m)}.
$$

**α-divergence**는 **α-connection의 유도된 canonical divergence**. 이 구조가 Ch4의 핵심.

---

## ✏️ 엄밀한 정의

### 정의 11.1 (α-divergence, Amari form)

$\alpha \in \mathbb{R}$, $p, q$ 밀도:

$$
D_\alpha(p \| q) := 
\begin{cases}
\frac{1}{\alpha(1-\alpha)}\left(1 - \int p^\alpha q^{1-\alpha}\, d\mu\right) & \alpha \ne 0, 1 \\[4pt]
\operatorname{KL}(p \| q) & \alpha = 1 \\[4pt]
\operatorname{KL}(q \| p) & \alpha = 0
\end{cases}
$$

$\alpha = 1$, $\alpha = 0$의 한계는 L'Hopital로 확인 (아래 정리).

### 정의 11.2 (Rényi divergence)

$\alpha \in (0, \infty) \setminus \{1\}$:

$$
R_\alpha(p \| q) := \frac{1}{\alpha - 1} \log \int p^\alpha q^{1-\alpha}\, d\mu.
$$

$\alpha = 1$: $R_1(p \| q) := \operatorname{KL}(p \| q)$.
$\alpha = \infty$: $R_\infty(p \| q) := \sup_x \log(p(x)/q(x))$ (log max-ratio).
$\alpha = 0$: $R_0(p \| q) := -\log \int_{\{p > 0\}} q\, d\mu$.

### 정의 11.3 (f-divergence 재정리)

Csiszár (1963): $f: [0, \infty) \to \mathbb{R}$ convex, $f(1) = 0$. f-divergence:

$$
D_f(p \| q) := \int q(x) f\!\left(\frac{p(x)}{q(x)}\right) d\mu(x).
$$

**α-divergence는 특수한 f-divergence**: 

$$
f_\alpha(r) = \frac{r^\alpha - \alpha r - (1-\alpha)}{\alpha(\alpha-1)}.
$$

### 정의 11.4 (α-family 모델)

$\alpha$-family of distributions: natural parameter에서 $\alpha$-linear. Exp family ($\alpha = 1$) 과 mixture family ($\alpha = -1$) 의 일반화.

---

## 🔬 정리와 증명

### 정리 11.1 (α → 1 한계 = KL)

$$
\lim_{\alpha \to 1} D_\alpha(p \| q) = \operatorname{KL}(p \| q).
$$

**증명.** $\int p^\alpha q^{1-\alpha} = \int p \cdot (q/p)^{1-\alpha}$. Taylor in $\alpha$ near 1:

$$
(q/p)^{1-\alpha} = \exp((1-\alpha)\log(q/p)) = 1 + (1-\alpha)\log(q/p) + \tfrac{1}{2}(1-\alpha)^2 \log^2(q/p) + \ldots
$$

Integrate:
$$
\int p^\alpha q^{1-\alpha} = 1 + (1-\alpha)\mathbb{E}_p[\log(q/p)] + O((1-\alpha)^2) = 1 - (1-\alpha)\operatorname{KL}(p\|q) + O((1-\alpha)^2).
$$

따라서
$$
D_\alpha = \frac{1 - [1 - (1-\alpha)\operatorname{KL} + O((1-\alpha)^2)]}{\alpha(1-\alpha)} = \frac{(1-\alpha)\operatorname{KL} + O((1-\alpha)^2)}{\alpha(1-\alpha)} \to \operatorname{KL}.
$$

**Q.E.D.**

---

### 정리 11.2 (Rényi → 1 한계)

$$
\lim_{\alpha \to 1} R_\alpha(p \| q) = \operatorname{KL}(p \| q).
$$

**증명.** $R_\alpha = \frac{1}{\alpha-1}\log Z(\alpha)$, $Z(\alpha) := \int p^\alpha q^{1-\alpha}$. $Z(1) = \int p = 1$. $\log Z \to 0$.

L'Hopital: $\lim_{\alpha\to 1} \frac{\log Z(\alpha)}{\alpha - 1} = \frac{Z'(1)}{Z(1)} = Z'(1)$.

$Z'(\alpha) = \int p^\alpha q^{1-\alpha} \log(p/q)\, d\mu$. $Z'(1) = \int p \log(p/q) = \operatorname{KL}(p\|q)$. **Q.E.D.**

---

### 정리 11.3 (α-divergence의 비음성)

$D_\alpha(p \| q) \ge 0$, $D_\alpha = 0 \Leftrightarrow p = q$ ($\mu$-a.e.) for any $\alpha$.

**증명.** f-divergence 비음성 from Jensen: $f_\alpha$ convex이고 $f_\alpha(1) = 0$.

$D_{f_\alpha}(p\|q) = \mathbb{E}_q[f_\alpha(p/q)] \ge f_\alpha(\mathbb{E}_q[p/q]) = f_\alpha(1) = 0$. **Q.E.D.**

---

### 정리 11.4 (α-divergence의 Duality)

$$
D_\alpha(p \| q) = D_{1-\alpha}(q \| p).
$$

즉 **α ↔ (1-α) 변환으로 forward/reverse 전환**.

**증명.** $D_\alpha(p\|q) = \frac{1}{\alpha(1-\alpha)}(1 - \int p^\alpha q^{1-\alpha})$. 

$D_{1-\alpha}(q\|p) = \frac{1}{(1-\alpha)\alpha}(1 - \int q^{1-\alpha} p^{\alpha}) = D_\alpha(p\|q)$. ✓ **Q.E.D.**

**따름.** $\alpha = 1/2$에서 self-dual → **symmetric** divergence.

---

### 정리 11.5 (Hellinger² as α=1/2)

$$
D_{1/2}(p \| q) = \frac{1}{(1/2)(1/2)}\left(1 - \int \sqrt{pq}\right) = 4 \left(1 - \int \sqrt{pq}\right) = 2 H^2(p, q),
$$

여기서 $H^2(p, q) = \int (\sqrt{p} - \sqrt{q})^2 = 2(1 - \int\sqrt{pq})$는 **squared Hellinger**.

**의의.** Hellinger는 metric (dependencies among distributions) — symmetric α-divergence.

---

### 정리 11.6 (α = 2: χ² divergence)

$$
D_2(p \| q) = \frac{1}{2 \cdot (-1)}\left(1 - \int p^2 q^{-1}\right) = \frac{1}{2}\left(\int \frac{p^2}{q} - 1\right) = \frac{1}{2} \chi^2(p \| q).
$$

**χ² divergence**: 가설검정의 고전적 통계량. 두 번째 모멘트 의존 → outlier sensitive.

---

### 정리 11.7 (α-divergence와 Fisher: 국소 근사)

모든 $\alpha \ne 0, 1$에 대해

$$
D_\alpha(p_\theta \| p_{\theta + \varepsilon}) = \tfrac{1}{2} \varepsilon^\top F(\theta) \varepsilon + O(\|\varepsilon\|^3).
$$

즉 **2차 항은 α에 무관** — 모든 α-divergence가 같은 Fisher quadratic을 국소적으로.

3차 항에서 α 의존성:
$$
D_\alpha(p_\theta \| p_{\theta + \varepsilon}) = \tfrac{1}{2} \varepsilon^\top F \varepsilon + \frac{1-2\alpha}{6} T_{ijk} \varepsilon^i \varepsilon^j \varepsilon^k + O(\|\varepsilon\|^4).
$$

**증명 (스케치).** Taylor of $\int p^\alpha q^{1-\alpha}$ in $\varepsilon$. 2차 항은 $\frac{\alpha(1-\alpha)}{2} F \varepsilon^{\otimes 2}$. Prefactor $\frac{1}{\alpha(1-\alpha)}$로 cancel. 결과: $\frac{1}{2} F$. 3차에서 α-dependent symmetry term. **Q.E.D.**

> **의의.** 3차 항의 $\frac{1-2\alpha}{6}$ 계수: $\alpha = 1/2$에서 0 (symmetric), $\alpha = 0, 1$에서 $\pm \frac{1}{6}$ (반대 부호 = forward/reverse KL).

---

### 정리 11.8 (Rényi divergence의 Monotonicity in α)

Fixed $(p, q)$, $R_\alpha(p\|q)$는 $\alpha$의 non-decreasing function.

**증명.** $R_\alpha(p\|q) = \frac{1}{\alpha-1} \log \mathbb{E}_q[(p/q)^\alpha]$. Hölder's inequality로 $\mathbb{E}_q[(p/q)^\alpha]$의 log-convexity → $R_\alpha$ monotone.

**따름.**
- $R_0 \le R_1 = \operatorname{KL} \le R_2 \le R_\infty$.
- 작은 α: weak divergence. 큰 α: strong (tail-sensitive) divergence.

---

### 정리 11.9 (Data Processing Inequality for α-divergence)

$K$ Markov kernel:
$$
D_\alpha(Kp \| Kq) \le D_\alpha(p \| q), \quad R_\alpha(Kp \| Kq) \le R_\alpha(p\|q).
$$

**증명.** f-divergence DPI (Ch3-01 정리 7.4 의 일반화) + monotonicity in α.

---

### 정리 11.10 (Rényi Composition for DP)

두 mechanism $M_1, M_2$: 각각 $(\alpha, \epsilon_1)$-RDP, $(\alpha, \epsilon_2)$-RDP. Sequential composition:

$$
M_1 \circ M_2 \text{ is } (\alpha, \epsilon_1 + \epsilon_2)\text{-RDP}.
$$

**Rényi Differential Privacy**에서 composition이 **additive** — 표준 DP 대비 훨씬 tight bound.

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: α-divergence 가족 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def alpha_divergence(p, q, alpha):
    """D_α(p||q)"""
    p = np.asarray(p); q = np.asarray(q)
    if np.isclose(alpha, 1):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    elif np.isclose(alpha, 0):
        mask = q > 0
        return np.sum(q[mask] * np.log(q[mask] / p[mask]))
    else:
        return (1 - np.sum(p**alpha * q**(1-alpha))) / (alpha * (1 - alpha))

def renyi_divergence(p, q, alpha):
    if np.isclose(alpha, 1):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    else:
        return (1 / (alpha - 1)) * np.log(np.sum(p**alpha * q**(1-alpha)))

# 두 다항분포 예
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.1, 0.2, 0.3, 0.4])

alpha_vals = np.linspace(-0.5, 2.0, 100)
alpha_vals = alpha_vals[(alpha_vals > 1e-3) & (np.abs(alpha_vals - 1) > 1e-3)]

D_list = [alpha_divergence(p, q, a) for a in alpha_vals]
R_list = [renyi_divergence(p, q, a) if a > 0 else np.nan for a in alpha_vals]

# Special values
kl_pq = alpha_divergence(p, q, 1)   # = KL(p||q)
kl_qp = alpha_divergence(p, q, 0+1e-10)  # ≈ KL(q||p)
hell2 = alpha_divergence(p, q, 0.5)  # = 2 Hellinger²
chi2 = alpha_divergence(p, q, 2) * 2  # = χ²

print(f"KL(p||q)    = {kl_pq:.4f}")
print(f"KL(q||p)    = {kl_qp:.4f} (≈ D_{{α=0}})")
print(f"2 · Hell²   = {hell2:.4f} (= D_{{α=1/2}})")
print(f"χ²          = {chi2:.4f} (= 2 D_{{α=2}})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(alpha_vals, D_list, 'b-', linewidth=2, label='α-divergence')
axes[0].axhline(y=kl_pq, color='red', linestyle='--', alpha=0.5, label='KL(p||q) at α=1')
axes[0].axhline(y=kl_qp, color='green', linestyle='--', alpha=0.5, label='KL(q||p) at α=0')
axes[0].set_xlabel('α')
axes[0].set_ylabel('D_α(p || q)')
axes[0].set_title('α-divergence as function of α')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot([a for a in alpha_vals if a > 0], 
             [renyi_divergence(p, q, a) for a in alpha_vals if a > 0],
             'purple', linewidth=2, label='R_α')
axes[1].axhline(y=kl_pq, color='red', linestyle='--', alpha=0.5, label='KL(p||q) at α=1')
axes[1].set_xlabel('α')
axes[1].set_ylabel('R_α(p || q)')
axes[1].set_title('Rényi divergence (monotone)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/kind-dazzling-ritchie/alpha_divergence.png', dpi=100)
plt.close()
print("\n✓ Saved alpha_divergence.png")
```

---

### 예제 2: α=1 한계 검증

```python
import numpy as np

p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.1, 0.2, 0.3, 0.4])

def alpha_div(p, q, alpha):
    if abs(alpha - 1) < 1e-12:
        return np.sum(p * np.log(p/q))
    elif abs(alpha) < 1e-12:
        return np.sum(q * np.log(q/p))
    return (1 - np.sum(p**alpha * q**(1-alpha))) / (alpha*(1-alpha))

# α → 1
kl_exact = np.sum(p * np.log(p/q))
alpha_values = [0.9, 0.99, 0.999, 0.9999, 0.99999]
for a in alpha_values:
    d = alpha_div(p, q, a)
    err = abs(d - kl_exact)
    print(f"α = {a:.5f}: D_α = {d:.6f}, KL = {kl_exact:.6f}, |diff| = {err:.2e}")

# α → 0 (reverse KL)
kl_rev_exact = np.sum(q * np.log(q/p))
for a in [0.1, 0.01, 0.001, 0.0001]:
    d = alpha_div(p, q, a)
    err = abs(d - kl_rev_exact)
    print(f"α = {a:.5f}: D_α = {d:.6f}, KL(q||p) = {kl_rev_exact:.6f}, |diff| = {err:.2e}")
```

---

### 예제 3: α-divergence duality 검증

```python
import numpy as np

p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.1, 0.2, 0.3, 0.4])

def alpha_div(p, q, a):
    if abs(a - 1) < 1e-10: return np.sum(p * np.log(p/q))
    if abs(a) < 1e-10:     return np.sum(q * np.log(q/p))
    return (1 - np.sum(p**a * q**(1-a))) / (a*(1-a))

alphas = [0.2, 0.5, 0.7, 1.3, 2.0]

print(f"{'α':>6} {'D_α(p||q)':>12} {'D_{1-α}(q||p)':>15} {'diff':>10}")
for a in alphas:
    D1 = alpha_div(p, q, a)
    D2 = alpha_div(q, p, 1-a)
    print(f"{a:>6.2f} {D1:>12.6f} {D2:>15.6f} {abs(D1-D2):>10.2e}")

# α=1/2에서 대칭성
print(f"\nAt α=1/2: D(p||q) = D(q||p)?")
print(f"  D_1/2(p||q) = {alpha_div(p, q, 0.5):.6f}")
print(f"  D_1/2(q||p) = {alpha_div(q, p, 0.5):.6f}")
```

---

### 예제 4: Rényi monotonicity

```python
import numpy as np

p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.1, 0.2, 0.3, 0.4])

def renyi(p, q, a):
    if abs(a - 1) < 1e-10: return np.sum(p * np.log(p/q))
    return np.log(np.sum(p**a * q**(1-a))) / (a - 1)

alpha_vals = np.linspace(0.1, 5.0, 50)
R_vals = [renyi(p, q, a) for a in alpha_vals]

# Monotonicity check
diffs = np.diff(R_vals)
monotone = np.all(diffs >= -1e-10)
print(f"Rényi monotone non-decreasing in α: {monotone}")

print(f"\nR_α at specific α:")
for a in [0.5, 1.0, 1.5, 2.0, 5.0]:
    print(f"  R_{a} = {renyi(p, q, a):.4f}")

# R_∞ = log max (p/q)
R_inf = np.log(np.max(p/q))
print(f"  R_∞ = log max(p/q) = {R_inf:.4f}")
```

---

### 예제 5: Fisher 2차 근사 — α invariance

```python
import numpy as np

# N(0, 1) vs N(ε, 1)
# D_α 의 2차 항은 α-independent, 3차 항은 (1-2α)/6 · T

# Gaussian: T_μμμ = E[(∂_μ log p)³] = E[x³] = 0 (symmetric)
# 따라서 3차 항도 0 → α-independence in 2nd and 3rd order

# 더 재미있는 예: Bernoulli
# Bern(1/2) → Bern(1/2 + ε)
# score s = (x - p)/(p(1-p))
# T = E[s³] at p=1/2: E[(x - 1/2)³/(1/4)³] = (1/2·(1/2)³ + 1/2·(-1/2)³)/(1/64) = 0/(1/64) = 0
# Bern(1/2)에서도 T = 0 (symmetric)

# 비대칭 p = 0.3에서
p_base = 0.3

# Fisher: 1/(p(1-p))
F = 1 / (p_base * (1 - p_base))

# Skewness: T = E[s³]
# s = (x - p)/(p(1-p))
# E[s³] = p·((1-p)/(p(1-p)))³ + (1-p)·(-p/(p(1-p)))³
# = [p(1-p)³ - (1-p)p³] / (p(1-p))³
# = [p(1-p)³ - p³(1-p)] / (p(1-p))³
# = p(1-p)[(1-p)² - p²] / (p(1-p))³
# = [(1-p)² - p²] / [p²(1-p)²]
# = (1 - 2p) / [p²(1-p)²]

T = (1 - 2*p_base) / (p_base**2 * (1 - p_base)**2)

print(f"p = {p_base}")
print(f"Fisher F = {F:.4f}")
print(f"Skewness T = {T:.4f}")

# ε = 0.01
eps = 0.01
def D_alpha_bern(p1, p2, alpha):
    # KL 형태들
    if abs(alpha - 1) < 1e-10:
        return p1*np.log(p1/p2) + (1-p1)*np.log((1-p1)/(1-p2))
    if abs(alpha) < 1e-10:
        return p2*np.log(p2/p1) + (1-p2)*np.log((1-p2)/(1-p1))
    return (1 - (p1**alpha*p2**(1-alpha) + (1-p1)**alpha*(1-p2)**(1-alpha))) / (alpha*(1-alpha))

print(f"\nε = {eps}, 2nd-order approx = {0.5*F*eps**2:.6e}")
print(f"{'α':>6} {'D_α exact':>14} {'½Fε²':>14} {'diff':>12}")
for a in [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0]:
    Dval = D_alpha_bern(p_base, p_base + eps, a)
    quad = 0.5 * F * eps**2
    print(f"{a:>6.2f} {Dval:>14.6e} {quad:>14.6e} {(Dval - quad):>12.2e}")

# 2차 항은 모든 α에서 같음, 3차에서 α-dependent
```

---

### 예제 6: β-VAE 스타일 ELBO α 수정

```python
import numpy as np

# β-ELBO = E_q[log p(x|z)] - β · KL(q(z|x) || p(z))
# α-ELBO (Rényi VAE): 
#   ELBO_α = 1/(1-α) log E_{q(z|x)} [exp((1-α)(log p(x,z) - log q(z|x)))]
# α → 1 recovers standard ELBO

# 데모: 1D Gaussian model
def log_joint(x, z):
    """log p(x, z) = log p(x|z) + log p(z)"""
    return -0.5*(x - z)**2 - 0.5*z**2 - np.log(2*np.pi)

def log_q(z, mu, sig):
    return -0.5*((z - mu)/sig)**2 - np.log(sig) - 0.5*np.log(2*np.pi)

# Monte Carlo α-ELBO
def alpha_elbo(x, mu, sig, alpha, K=10000):
    z = mu + sig * np.random.randn(K)
    log_p = log_joint(x, z)
    log_q_val = log_q(z, mu, sig)
    ratio = log_p - log_q_val  # log(p(x,z)/q(z|x))
    
    if abs(alpha - 1) < 1e-10:
        return np.mean(ratio)
    else:
        # log-mean-exp
        w = (1 - alpha) * ratio
        return np.log(np.mean(np.exp(w - w.max()))) / (1 - alpha) + w.max() / (1 - alpha) * (1-alpha)
        # 더 간단히:
        return np.log(np.mean(np.exp(w))) / (1 - alpha)

x = 1.0
# Optimal q: q(z|x) = N(x/2, 1/√2) (conjugate posterior)
mu_opt, sig_opt = x/2, 1/np.sqrt(2)

# 다양한 α에서 ELBO
alpha_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
np.random.seed(0)
for a in alpha_vals:
    val = alpha_elbo(x, mu_opt, sig_opt, a)
    print(f"α = {a:.2f}: ELBO_α ≈ {val:.4f}")

# log p(x) = -0.5 (log Gaussian of x with var 2)
log_px_true = -0.5 * x**2 / 2 - 0.5*np.log(4*np.pi)
print(f"\nTrue log p(x) = {log_px_true:.4f}")
print(f"α = 1 (standard ELBO) matches log p(x) at optimal q")
print(f"α > 1 → tighter bound (underestimates less), α < 1 → looser but mass-covering")
```

---

## 🔗 AI/ML 연결

### 1. Rényi VAE (Li & Turner 2016)

VR bound:
$$
\mathcal{L}_\alpha(\phi) = \frac{1}{1-\alpha} \log \mathbb{E}_{q_\phi(z|x)}\!\left[\left(\frac{p_\theta(x, z)}{q_\phi(z|x)}\right)^{1-\alpha}\right].
$$

- $\alpha \to 1$: standard ELBO.
- $\alpha = 0$: IWAE lower bound with K=1 sample.
- $\alpha < 1$ (즉 $1 - \alpha > 0$): **tighter** bound (closer to log p(x)).
- $\alpha > 1$: looser but **mass-covering** posterior (다양한 mode).

### 2. α-VI / Power EP

Minka (2004)의 **Power EP** (α 발산 기반 Expectation Propagation). α 조절로 mode-seeking vs mass-covering 제어.

- $\alpha = 1$: EP (mean-seeking).
- $\alpha = 0$: reverse KL (mode-seeking, 표준 VI).
- $\alpha = 0.5$: Hellinger — symmetric, 균형.

### 3. Rényi Differential Privacy

Mironov (2017). $(\alpha, \epsilon)$-RDP:

$$
R_\alpha(M(D) \| M(D')) \le \epsilon \quad \forall \text{ adjacent } D, D'.
$$

**Composition**: $M_1 \circ M_2$가 $(\alpha, \epsilon_1 + \epsilon_2)$-RDP. Subsampling, Gaussian mechanism analyze 용이.

GPT/LLM 훈련에서 DP-SGD를 적용할 때 RDP accounting이 표준 (tfprivacy, opacus).

### 4. f-GAN (Nowozin 2016)

임의 f-divergence로 GAN 확장:

$$
\mathcal{L}_f(G, D) = \mathbb{E}_{p}[D(x)] - \mathbb{E}_{p_g}[f^*(D(x))].
$$

- f = KL → original GAN equivalent.
- f = χ² → LSGAN.
- f = Pearson χ² → specific variants.

### 5. Tsallis Entropy & Boltzmann Machines

Tsallis divergence ≈ α-divergence의 alternative form. Non-extensive statistical mechanics에서 중요.

### 6. Contrastive Learning의 InfoNCE

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\!\left[\log \frac{\exp(f(x,y^+))}{\sum_j \exp(f(x, y_j))}\right].
$$

이것이 MI의 lower bound ($\alpha = 1$ case). $\alpha$-InfoNCE 확장 가능 (Poole 2019).

### 7. Robust ML — α 선택

Outlier가 있는 데이터:
- $\alpha$ 작음 → 극단값 무시 (robust).
- $\alpha$ 큼 → 극단값 강조 (tail-sensitive).

α = 0.3~0.7 이 RNN/LLM 훈련에서 robustness 개선 (실험적).

### 8. Chernoff Bound 일반화

Rényi divergence $R_\alpha$는 Chernoff bound에서 등장:

$$
P(\text{Type I error}) \le \exp(-n R_\alpha^*), \qquad R_\alpha^* = \min_\alpha R_\alpha(p \| q).
$$

Hypothesis testing의 error exponent 이론.

---

## ⚖️ 가정과 한계

### α 선택의 tradeoff

- 작은 α → mode collapse, outlier에 둔감.
- 큰 α → tail overfit, outlier에 과민.
- "최적 α" 는 데이터 의존 → hyperparameter search.

### 수치 안정성

$p^\alpha q^{1-\alpha}$ 계산 시 $p, q \to 0$에서 underflow. Log-space: $\log \int \exp(\alpha \log p + (1-\alpha) \log q)$ — **log-sum-exp trick** 필수.

### f-divergence의 제약

모든 α-divergence는 Csiszár f-divergence 이므로:
- **DPI**: Markov 처리로 감소.
- **Convexity**: $(p, q)$에 jointly convex.
- **Chain rule**: KL과 유사.

**Wasserstein**는 f-divergence가 아니다 — 완전히 다른 class.

### Infinite Rényi ($\alpha = \infty$)

$R_\infty = \log \sup p/q$. 단 하나의 outlier $x^*$로 발산 가능 → unstable, 실무에서 흔히 미사용. **Max divergence** 개념.

### α-connection의 제한

α가 $\pm 1$ 아닌 경우 **비대칭, non-metric** 접속. α-geodesic은 실제 확률분포 공간의 "straight line" 이지만, Riemannian 의미의 최단 경로는 아님.

### Mode Coverage vs Mode Seeking

VAE에서 "mode coverage" (α < 1) 와 "sharpness" (α > 1) 의 tradeoff. 두 극단 모두 fail:
- 너무 작은 α: uninformative uniform posterior.
- 너무 큰 α: collapsed single-mode.

---

## 📌 핵심 정리

| α 값 | Divergence | 용도 |
|---|---|---|
| $\alpha = 0$ | $\operatorname{KL}(q\|p)$ (reverse) | VAE posterior |
| $\alpha = 1/2$ | $2H^2$ (Hellinger² × 2) | Symmetric, metric |
| $\alpha = 1$ | $\operatorname{KL}(p\|q)$ (forward) | Standard ML |
| $\alpha = 2$ | $\tfrac{1}{2}\chi^2$ | Hypothesis testing |
| $\alpha = \infty$ | $\log \sup(p/q)$ | Max divergence, DP |

**핵심 관계:**
- $D_\alpha(p \| q) = D_{1-\alpha}(q \| p)$ (duality).
- $R_\alpha$ monotone non-decreasing in α.
- $D_\alpha$의 2차 Fisher 근사 = **α-independent**.
- α-connection의 canonical divergence = α-divergence.

**AI Takeaways:**

1. **KL은 KL family의 특수 경우** — α 선택이 tradeoff 제어.
2. **RDP** = 프라이버시의 tight accounting.
3. **Rényi VAE, Power EP** = VI의 mode/mean-seeking control.
4. **Fisher는 α에 불변** (2차) — 왜 모든 α-divergence가 같은 "infinitesimal geometry".

---

## 🤔 생각해볼 문제

1. **Hellinger가 metric인가?** $H(p, q) = \sqrt{H^2(p, q)} = \sqrt{2 D_{1/2}(p\|q)}$가 metric (triangle ineq.) 인지 확인. 만약 그렇다면 $\sqrt{D_\alpha}$가 metric인 α는 어떤 값들?

2. **α → ∞ 한계**. Max divergence $R_\infty = \log \sup(p/q)$. 이것이 **uniform integrability** 실패와 어떤 관계? DP에서의 $\infty$-Rényi는 $\epsilon$-DP (원본 DP) 와 동등한가?

3. **VAE 목적함수 비교**. Standard ELBO ($\alpha = 1$), IWAE ($\alpha = 0$, K-sample), Rényi ELBO (general α). 각각의 posterior approximation quality가 어떻게 다른가? (Log-likelihood gap 비교.)

4. **f-GAN stability**. f-GAN에서 f = JS (original GAN) vs f = Wasserstein (WGAN). Wasserstein은 f-divergence 아니므로 Nowozin framework 벗어남. 왜 WGAN이 더 안정적인지 — α-divergence의 한계와 관련.

5. **Moment Generating Function과 Rényi**. $R_2(p\|q) = \log \int p^2/q$. 이것이 어떤 random variable의 moment generating function과 관련되는가? Chernoff bound에서의 역할.

6. **β-Divergence vs α-Divergence**. $\beta$-divergence (Cichocki):
$$
D_\beta(p\|q) = \frac{1}{\beta(\beta-1)} \int [p^\beta + (\beta-1) q^\beta - \beta p q^{\beta-1}]\, d\mu.
$$
$\beta = 1$은 generalized KL. $\beta = 2$는 squared error. α vs β의 관계와 각각의 선호 영역.

7. **Tsallis Entropy**. Tsallis-α entropy: $H_\alpha^T(p) = \frac{1}{\alpha - 1}(1 - \sum p_i^\alpha)$. Rényi와의 관계 (비선형). α-divergence와는 어떤 결합?

---

<div align="center">

| [◀ 04. KL = Bregman in 지수족](./04-kl-as-bregman.md) | [📚 메인 README](../README.md) | [Ch4-01. 지수족의 기하학적 정의 ▶](../ch4-exponential-duality/01-exponential-family-geometry.md) |
|:---:|:---:|:---:|

</div>
