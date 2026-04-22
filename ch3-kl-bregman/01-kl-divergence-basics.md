# 01. KL 발산의 기초 — 정의, 성질, 불변성

> **"KL은 거리가 아니다. KL은 정보 손실이다."**

---

## 🎯 핵심 질문

**KL 발산 $\operatorname{KL}(p \| q) = \mathbb{E}_p[\log(p/q)]$는 왜 "확률분포 간의 차이"를 측정하는 가장 기본적인 척도인가?**

$$
\boxed{\;\operatorname{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)}\, d\mu(x)\;}
$$

네 가지 핵심 관점:
1. **정보이론**: 잘못된 모델로 인한 코드 길이 증가 (cross-entropy - entropy)
2. **통계 추론**: 확률비의 기대 로그 = 우도비 검정 통계
3. **정보 기하**: Fisher 계량의 "globalization" — 2차 근사 = Fisher
4. **변분 추론**: ELBO 유도의 핵심 building block

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 영역 | KL의 역할 |
|---|---|
| **분류** | Cross-entropy loss = $\operatorname{KL}(y_{\mathrm{true}} \| p_{\mathrm{model}})$ + const |
| **VAE** | ELBO = $\log p(x) - \operatorname{KL}(q(z\|x) \| p(z\|x))$ |
| **TRPO/PPO** | KL trust-region: $\operatorname{KL}(\pi_{\mathrm{old}} \| \pi_\theta) \le \delta$ |
| **GAN** | Original GAN = JS-divergence (KL 기반) |
| **Language models** | Perplexity = $\exp(\operatorname{CE}) = \exp(\operatorname{KL}(\text{data}\|\text{model}) + H(\text{data}))$ |
| **Diffusion** | Forward process KL bound on data distribution |
| **Reinforcement Learning** | Maximum entropy RL: reward + $\alpha H(\pi) = $ KL to uniform |

**모든 현대 AI 손실 함수의 뿌리.** KL을 이해하지 못하면 VAE/TRPO/Diffusion 어느 하나도 제대로 이해할 수 없다.

---

## 📐 수학적 선행 조건

- Ch2 [02. Fisher 3가지 정의](../ch2-statistical-fisher/02-fisher-3-equivalence.md) — 특히 $\operatorname{KL}$의 2차 근사
- Jensen 부등식
- Absolute continuity $p \ll q$, Radon-Nikodym derivative
- 기본 확률론: 결합/조건부/주변 분포
- Entropy $H(p) = -\int p \log p$, Cross-entropy $H(p, q) = -\int p \log q$

---

## 📖 직관적 이해

### 정보이론적 해석 — 잘못된 코드의 대가

문자 $x$의 진짜 분포가 $p$인데, 코드를 $q$에 맞춰 설계했다면:

- **최적 코드 길이** (진짜 분포 아래): $H(p) = -\mathbb{E}_p[\log p]$ bits
- **잘못된 코드 길이**: $H(p, q) = -\mathbb{E}_p[\log q]$ bits
- **초과 비트 수**: $H(p, q) - H(p) = \mathbb{E}_p[\log(p/q)] = \operatorname{KL}(p \| q)$.

**"KL은 잘못된 모델을 쓰면 잃는 비트 수."** 항상 ≥ 0 (Jensen). 0 ⟺ $p = q$ (거의 곳곳).

### 통계적 해석 — 우도비 검정

표본 $X \sim p$에서 가설 $H_0: q$ vs $H_1: p$ 검정. Neyman-Pearson 우도비 $\log \frac{p(X)}{q(X)}$의 기댓값:

$$
\mathbb{E}_p \left[\log \frac{p}{q}\right] = \operatorname{KL}(p \| q).
$$

KL은 **진실과 가설을 구별하기 쉬운 정도**를 잰다.

### 비대칭성의 의미

$\operatorname{KL}(p \| q) \ne \operatorname{KL}(q \| p)$ — 이것이 KL이 "거리"가 아닌 이유.

- **Forward KL** $\operatorname{KL}(p \| q)$: $p$가 지지하는 모든 $x$에서 $q > 0$ 요구. 아니면 $\infty$. "zero-avoiding" — $q$가 $p$의 모든 mode 포함해야 함. → **mean-seeking**.
- **Reverse KL** $\operatorname{KL}(q \| p)$: $q$가 지지하는 곳에서만 $p > 0$ 요구. $q$가 mode 하나만 잡아도 OK. → **mode-seeking**.

VAE에서 $\operatorname{KL}(q \| p)$ (reverse) 사용 → posterior가 한 mode에 collapsing 경향.
EP/BP에서 $\operatorname{KL}(p \| q)$ (forward) 사용 → posterior가 spread out.

### 기하학적 해석 — 국소적 Fisher

Ch2에서 본 대로:

$$
\operatorname{KL}(p_\theta \| p_{\theta+d\theta}) \approx \tfrac{1}{2} d\theta^\top F(\theta) d\theta.
$$

즉 **KL의 2차 테일러 근사** = Fisher 이차 형식. KL은 Fisher의 "global" 버전이고, Fisher는 KL의 "local linearization".

---

## ✏️ 엄밀한 정의

### 정의 7.1 (KL 발산)

두 확률측도 $P, Q$가 공통 측도 $\mu$에 대해 밀도 $p = dP/d\mu$, $q = dQ/d\mu$ 를 가진다고 하자. $P \ll Q$ (absolute continuity) 이라면:

$$
\operatorname{KL}(P \| Q) := \int p(x) \log \frac{p(x)}{q(x)}\, d\mu(x) = \mathbb{E}_{X \sim P}\!\left[\log \frac{p(X)}{q(X)}\right].
$$

$P \not\ll Q$ (즉 $P(A) > 0$인데 $Q(A) = 0$인 $A$가 존재) 이면 $\operatorname{KL}(P\|Q) := +\infty$.

**관례:** $0 \log 0 = 0$ (측도 0의 기여는 0), $p \log(p/0) = +\infty$ if $p > 0$.

### 정의 7.2 (조건부 KL)

결합 확률 $p(x, y), q(x, y)$에서 **conditional KL**:

$$
\operatorname{KL}(p(Y|X) \| q(Y|X)) := \int p(x) \left[\int p(y|x) \log \frac{p(y|x)}{q(y|x)}\, dy\right] dx = \mathbb{E}_{X \sim p(X)}[\operatorname{KL}(p(\cdot|X) \| q(\cdot|X))].
$$

### 정의 7.3 (KL의 체인룰)

$$
\operatorname{KL}(p(X, Y) \| q(X, Y)) = \operatorname{KL}(p(X) \| q(X)) + \operatorname{KL}(p(Y|X) \| q(Y|X)).
$$

### 정의 7.4 (상호정보량으로서의 KL)

**Mutual Information** $I(X; Y)$는

$$
I(X; Y) := \operatorname{KL}(p(X, Y) \| p(X) \otimes p(Y)) = \mathbb{E}\!\left[\log \frac{p(X,Y)}{p(X)p(Y)}\right].
$$

즉 MI는 **결합분포와 독립분포의 KL 거리**.

---

## 🔬 정리와 증명

### 정리 7.1 (Gibbs 부등식: 비음성)

$\operatorname{KL}(p \| q) \ge 0$, 등호 ⟺ $p = q$ ($\mu$-a.e.).

**증명.** $\log$ 는 concave → $-\log$ 는 convex. Jensen:

$$
-\operatorname{KL}(p \| q) = \mathbb{E}_p\!\left[\log \frac{q}{p}\right] \le \log \mathbb{E}_p\!\left[\frac{q}{p}\right] = \log \int q = \log 1 = 0.
$$

즉 $\operatorname{KL}(p\|q) \ge 0$. 등호 ⟺ $q/p = $ const (거의 곳곳) ⟺ $p = q$ (정규화로부터).

**Q.E.D.**

---

### 정리 7.2 (KL의 convexity)

$\operatorname{KL}(p \| q)$는 $(p, q)$ 쌍에 대해 **jointly convex**: $\lambda \in [0, 1]$에서

$$
\operatorname{KL}(\lambda p_1 + (1-\lambda) p_2 \| \lambda q_1 + (1-\lambda) q_2) \le \lambda \operatorname{KL}(p_1 \| q_1) + (1-\lambda) \operatorname{KL}(p_2 \| q_2).
$$

**증명.** Log-sum inequality: $(a_1 + a_2) \log\frac{a_1+a_2}{b_1+b_2} \le a_1 \log\frac{a_1}{b_1} + a_2 \log\frac{a_2}{b_2}$를 적분.

**따름.** $q$ 고정시 $p \mapsto \operatorname{KL}(p\|q)$ convex. $p$ 고정시 $q \mapsto \operatorname{KL}(p\|q)$ convex.

---

### 정리 7.3 (Pinsker 부등식)

**Total Variation** 거리 $\operatorname{TV}(p, q) := \tfrac{1}{2}\int |p - q|$ 과 KL의 관계:

$$
\operatorname{TV}(p, q) \le \sqrt{\tfrac{1}{2}\operatorname{KL}(p \| q)}.
$$

즉 KL → 0이면 TV → 0 — KL 수렴이 더 강함.

**증명 (스케치).** $f(t) := \operatorname{KL}(p_t \| q)$, $p_t = (1-t)q + tp$. $f(0) = 0$, $f''(t) \ge 4 \operatorname{TV}^2/((1-t)+t \cdot \max p/q)$ 형태 하한으로부터 적분.

**의의.** KL를 쓰는 이유 중 하나 — KL 제어는 많은 다른 거리 (TV, Wasserstein in Lipschitz case) 를 제어.

---

### 정리 7.4 (Data Processing Inequality)

Markov kernel $K: \mathcal{X} \to \mathcal{Y}$, $p' = K \cdot p$, $q' = K \cdot q$라 하자. 그러면

$$
\operatorname{KL}(p' \| q') \;\le\; \operatorname{KL}(p \| q).
$$

즉 **데이터 처리는 KL을 감소시킨다**.

**증명.** Log-sum inequality의 conditional 버전:

$$
\int \left[\int k(y|x) p(x) dx\right] \log \frac{\int k(y|x) p(x) dx}{\int k(y|x) q(x) dx}\, dy \le \int \int k(y|x) p(x) \log\frac{p(x)}{q(x)}\, dx\, dy = \operatorname{KL}(p\|q).
$$

**Q.E.D.**

**의의.** **Markov 처리 ⟹ 정보 감소.** 이것이 Chentsov 유일성 정리 (Ch2-03) 의 기반.

---

### 정리 7.5 (Fisher 관계: KL의 이차 전개)

$p_\theta, p_{\theta+\varepsilon}$ (정칙모델), $\varepsilon \to 0$:

$$
\operatorname{KL}(p_\theta \| p_{\theta+\varepsilon}) = \tfrac{1}{2} \varepsilon^\top F(\theta) \varepsilon + o(\|\varepsilon\|^2).
$$

(정리 3.3 in Ch2-02와 동일. 요약.)

---

### 정리 7.6 (Chain rule 및 non-negativity 활용)

$\operatorname{KL}(p(X,Y) \| p(X)\otimes p(Y)) = I(X;Y) \ge 0$ 은 **서로 독립인 분포가 결합분포의 "KL-projection"** 이라는 의미:

$$
I(X; Y) = \min_{q_X \otimes q_Y} \operatorname{KL}(p(X,Y) \| q_X \otimes q_Y)
$$

(최소가 $q_X = p(X), q_Y = p(Y)$에서 달성.)

이것이 mutual information의 variational 표현의 기반.

---

### 정리 7.7 (Invariance under sufficient statistic)

$T$가 $\{p_\theta, q\}$에 대한 충분통계량이면:

$$
\operatorname{KL}(p_\theta \| q) = \operatorname{KL}(p_\theta^T \| q^T),
$$

여기서 $p^T$는 $T(X)$의 분포. **KL은 충분통계량으로 information loss 없이 변환**.

증명: Data Processing (정리 7.4) 의 양방향 적용 (충분통계량은 역방향 Markov도 존재).

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: 정규분포 간 KL의 closed form

```python
import numpy as np

def kl_normal(mu1, sig1, mu2, sig2):
    """KL(N(μ1,σ1²) || N(μ2,σ2²))"""
    return np.log(sig2/sig1) + (sig1**2 + (mu1-mu2)**2)/(2*sig2**2) - 0.5

# 검증 1: 자기 자신 → 0
print(f"KL(N(0,1) || N(0,1)) = {kl_normal(0, 1, 0, 1):.6e}")  # ≈ 0

# 비대칭성 확인
print(f"KL(N(0,1) || N(2,1)) = {kl_normal(0, 1, 2, 1):.4f}")  # 2.0
print(f"KL(N(2,1) || N(0,1)) = {kl_normal(2, 1, 0, 1):.4f}")  # 2.0 (대칭 in μ 변화)

# σ 다를때 비대칭
print(f"\nKL(N(0,1) || N(0,2)) = {kl_normal(0, 1, 0, 2):.4f}")
print(f"KL(N(0,2) || N(0,1)) = {kl_normal(0, 2, 0, 1):.4f}")

# Fisher 2차 근사와 비교
eps_vec = np.array([0.01, 0.01])  # dμ, dσ
F_at_01 = np.diag([1.0, 2.0])  # Fisher of N(0,1)
kl_quad = 0.5 * eps_vec @ F_at_01 @ eps_vec
kl_exact = kl_normal(0, 1, 0 + eps_vec[0], 1 + eps_vec[1])
print(f"\nSmall ε: exact KL={kl_exact:.6e}, ½εᵀFε={kl_quad:.6e}")
```

---

### 예제 2: Pinsker 부등식 수치 검증

```python
import numpy as np

def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def kl_discrete(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# 무작위 분포 쌍
np.random.seed(0)
n_trials = 500
k = 5

results = []
for _ in range(n_trials):
    p = np.random.dirichlet(np.ones(k))
    q = np.random.dirichlet(np.ones(k))
    tv = tv_distance(p, q)
    kl = kl_discrete(p, q)
    results.append((tv, kl))

results = np.array(results)
print(f"{'TV':>10} {'KL':>10} {'sqrt(KL/2)':>12} {'TV ≤ sqrt(KL/2)?':>20}")
for tv, kl in results[:10]:
    bound = np.sqrt(kl / 2)
    print(f"{tv:>10.4f} {kl:>10.4f} {bound:>12.4f} {str(tv <= bound):>20}")

# 전체에 대해
violations = np.sum(results[:,0] > np.sqrt(results[:,1]/2))
print(f"\nPinsker 위반 횟수: {violations}/{n_trials} (예상: 0)")
```

---

### 예제 3: Forward vs Reverse KL — Mode vs Mean seeking

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Target: 2-mode mixture
def target_density(x):
    return 0.5 * norm.pdf(x, -2, 0.5) + 0.5 * norm.pdf(x, 2, 0.5)

# Approximation family: 단일 Gaussian
def approx_density(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

x = np.linspace(-5, 5, 500)
p = target_density(x)
p /= np.sum(p) * (x[1] - x[0])

# Forward KL: KL(p || q) → q가 p의 모든 mode 덮기 (mean-seeking)
from scipy.optimize import minimize

def kl_pq(params, p, x):
    mu, sigma = params
    q = approx_density(x, mu, abs(sigma) + 1e-8)
    q /= np.sum(q) * (x[1] - x[0])
    mask = p > 1e-10
    return np.sum(p[mask] * np.log(p[mask] / q[mask])) * (x[1] - x[0])

def kl_qp(params, p, x):
    mu, sigma = params
    q = approx_density(x, mu, abs(sigma) + 1e-8)
    q /= np.sum(q) * (x[1] - x[0])
    mask = q > 1e-10
    return np.sum(q[mask] * np.log(q[mask] / p[mask])) * (x[1] - x[0])

# Forward (mean-seeking)
res_fwd = minimize(kl_pq, x0=[0, 1], args=(p, x), method='Nelder-Mead')
mu_fwd, sigma_fwd = res_fwd.x[0], abs(res_fwd.x[1])
print(f"Forward KL optimal: μ={mu_fwd:.3f}, σ={sigma_fwd:.3f}")

# Reverse (mode-seeking) — 초기값 중요
res_rev1 = minimize(kl_qp, x0=[-2.0, 0.5], args=(p, x), method='Nelder-Mead')
res_rev2 = minimize(kl_qp, x0=[+2.0, 0.5], args=(p, x), method='Nelder-Mead')

mu_rev1, sigma_rev1 = res_rev1.x[0], abs(res_rev1.x[1])
mu_rev2, sigma_rev2 = res_rev2.x[0], abs(res_rev2.x[1])
print(f"Reverse KL (init -2): μ={mu_rev1:.3f}, σ={sigma_rev1:.3f}")
print(f"Reverse KL (init +2): μ={mu_rev2:.3f}, σ={sigma_rev2:.3f}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, p, 'k-', label='target p(x)', linewidth=2)
axes[0].plot(x, approx_density(x, mu_fwd, sigma_fwd), 'b--', label='Fwd KL approx', linewidth=2)
axes[0].set_title('Forward KL: KL(p || q)\n→ mean-seeking')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(x, p, 'k-', label='target p(x)', linewidth=2)
axes[1].plot(x, approx_density(x, mu_rev1, sigma_rev1), 'r--', label='Rev KL (mode 1)', linewidth=2)
axes[1].set_title('Reverse KL: KL(q || p)\n→ mode-seeking (left)')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(x, p, 'k-', label='target p(x)', linewidth=2)
axes[2].plot(x, approx_density(x, mu_rev2, sigma_rev2), 'g--', label='Rev KL (mode 2)', linewidth=2)
axes[2].set_title('Reverse KL: KL(q || p)\n→ mode-seeking (right)')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/kind-dazzling-ritchie/fwd_vs_rev_kl.png', dpi=100)
plt.close()
```

**기대 결과:** Forward KL은 두 mode 사이 평균적인 Gaussian, Reverse KL은 한 mode 집중.

---

### 예제 4: Data Processing Inequality 실험

```python
import numpy as np

np.random.seed(1)

# 원본 분포 on {1, 2, 3, 4}
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.1, 0.2, 0.3, 0.4])

# Markov kernel: 노이즈 채널 (각 symbol이 0.2 확률로 랜덤 flip)
K = np.array([
    [0.85, 0.05, 0.05, 0.05],
    [0.05, 0.85, 0.05, 0.05],
    [0.05, 0.05, 0.85, 0.05],
    [0.05, 0.05, 0.05, 0.85],
])

# 적용
p_after = K @ p
q_after = K @ q

def kl(p, q):
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

kl_before = kl(p, q)
kl_after = kl(p_after, q_after)

print(f"KL(p || q) before:  {kl_before:.4f}")
print(f"KL(p' || q') after: {kl_after:.4f}")
print(f"DPI holds? (after ≤ before): {kl_after <= kl_before}")

# Identity kernel → 같음
K_id = np.eye(4)
print(f"\nKL after identity: {kl(K_id @ p, K_id @ q):.4f} (should = before)")

# Constant kernel (모든 정보 손실)
K_const = np.full((4, 4), 0.25)
print(f"KL after constant channel: {kl(K_const @ p, K_const @ q):.4e} (should ≈ 0)")
```

---

### 예제 5: MI의 KL 형태 검증

```python
import numpy as np

np.random.seed(42)

# 2D joint: correlated Bernoullis
# X, Y ∈ {0, 1}, P(X=Y) = 0.8
p_joint = np.array([
    [0.4, 0.1],  # P(X=0, Y=0), P(X=0, Y=1)
    [0.1, 0.4],  # P(X=1, Y=0), P(X=1, Y=1)
])
p_X = p_joint.sum(axis=1)
p_Y = p_joint.sum(axis=0)
p_indep = np.outer(p_X, p_Y)

# MI = KL(p_joint || p_X ⊗ p_Y)
mi_kl = np.sum(p_joint * np.log(p_joint / p_indep))

# Direct MI = H(X) + H(Y) - H(X,Y)
H_X = -np.sum(p_X * np.log(p_X))
H_Y = -np.sum(p_Y * np.log(p_Y))
H_XY = -np.sum(p_joint * np.log(p_joint))
mi_entropy = H_X + H_Y - H_XY

print(f"MI via KL formulation:       {mi_kl:.4f}")
print(f"MI via H(X) + H(Y) - H(X,Y): {mi_entropy:.4f}")
print(f"Match? {np.isclose(mi_kl, mi_entropy)}")
```

---

## 🔗 AI/ML 연결

### 1. Cross-Entropy Loss = KL + H(데이터)

$$
-\log p_\theta(y) = \operatorname{CE}(y, p_\theta) = H(y) + \operatorname{KL}(y \| p_\theta).
$$

$y$ (one-hot label)의 entropy는 $\theta$와 무관 → **minimizing CE = minimizing KL**. 모든 분류 학습의 기반.

### 2. VAE ELBO

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{\operatorname{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}} + \operatorname{KL}(q_\phi(z|x) \| p_\theta(z|x)).
$$

ELBO = 첫 두 항. 마지막 KL은 **posterior gap** — 아래로 유계. $q = p$ (posterior) 에서 tight.

### 3. TRPO Trust Region

$$
\max \mathbb{E}\!\left[\frac{\pi_\theta}{\pi_{\mathrm{old}}} A\right] \quad \text{s.t.} \quad \mathbb{E}_s[\operatorname{KL}(\pi_{\mathrm{old}}(\cdot|s) \| \pi_\theta(\cdot|s))] \le \delta.
$$

KL 제약이 monotonic policy improvement (TRPO 이론) 보장.

### 4. GAN과 Divergence 일반화

**Original GAN**: generator가 $\operatorname{JS}(p_{\mathrm{data}} \| p_\theta) = \tfrac{1}{2}\operatorname{KL}(p \| m) + \tfrac{1}{2}\operatorname{KL}(q \| m)$ 최소화, $m = (p+q)/2$. KL 기반.

**f-GAN (Nowozin 2016)**: 임의 f-divergence ($\chi^2$, $\alpha$-divergence 등) GAN으로 일반화.

### 5. Information Bottleneck

$$
\min_{T} I(X; T) - \beta I(T; Y)
$$

MI = KL 형태. Tishby & Zaslavsky (2015) 이후 deep learning의 representation learning의 이론적 framework.

### 6. Reinforcement Learning as Entropy-Regularized Optimization

Maximum Entropy RL 목적:

$$
\max_\pi \mathbb{E}_\pi[R(\tau)] + \alpha H(\pi) = \max_\pi \mathbb{E}_\pi[R(\tau)] - \alpha \operatorname{KL}(\pi \| \text{Uniform}).
$$

**Soft Actor-Critic**, **Soft Q-learning**의 기초.

### 7. Diffusion Models의 ELBO

Denoising diffusion probabilistic model (DDPM):

$$
-\log p_\theta(x_0) \le \operatorname{KL}(q(x_T|x_0) \| p(x_T)) + \sum_{t > 1} \operatorname{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1).
$$

각 시간 단계의 KL 조건부 Gaussian으로 단순화 → score matching.

### 8. Mutual Information Estimation

MINE (Belghazi 2018), InfoNCE (Oord 2018) — MI의 variational lower bound로 representation learning. **Contrastive learning** (SimCLR, MoCo) 의 이론적 근거.

---

## ⚖️ 가정과 한계

### Absolute Continuity 요구

$P \not\ll Q$이면 KL = $\infty$. 실무에서 $q_\theta$가 $0$이 되는 영역을 $p$가 지지하면 gradient signal 소실. → **ε-smoothing** 또는 **Wasserstein 기반** 전환 (WGAN).

### 비대칭성의 실무적 영향

- Forward KL (mean-seeking): 과도한 spread, 모델이 모든 mode 커버 시도 → 블러리한 결과.
- Reverse KL (mode-seeking): 한 mode에 collapsing, mode collapse in GANs.

**혼합 전략**: $\operatorname{KL}(p\|q) + \lambda \operatorname{KL}(q\|p)$ = **symmetric KL** (Jensen-Shannon의 전신).

### Estimation Difficulty

- Continuous KL은 밀도 함수 적분 — 일반적으로 불가.
- MC estimate $\hat{\operatorname{KL}} = \tfrac{1}{N}\sum \log(p(X_n)/q(X_n))$ 는 high variance. $X_n \sim p$일 때 $q$가 작은 영역에서 큰 값.
- **Density ratio estimation** (LSIF, NCE) 가 KL의 MC 개선.

### High-dimensional KL

$d$-차원 Gaussian 간 KL은 $d$에 선형 ($\operatorname{KL}(\mathcal{N}_d(0, I) \| \mathcal{N}_d(0, 2I)) = O(d)$). 의미 있는 비교가 어려움 → ratio 기반 metric 또는 KL per dimension 사용.

### KL vs 다른 divergences

| Divergence | 특성 | 사용처 |
|---|---|---|
| KL | $\infty$ 경향, 비대칭 | Cross-entropy, VAE, TRPO |
| TV | 유계, 대칭 | 실무에서 자주 사용 안 함 (non-smooth) |
| $\chi^2$ | 대칭화·유계, $O((p-q)^2/q)$ | 가설검정, 평균 차이 |
| JS | 유계 $\log 2$, 대칭 | GAN (original) |
| Wasserstein | metric, geometric | WGAN, OT |
| $\alpha$-divergence | KL family parameterize | Renyi, Tsallis |

각 상황에 맞는 divergence 선택이 핵심.

### 정보적 수치 안정성

- $\operatorname{KL}(\mathcal{N}(\mu_1, \sigma_1) \| \mathcal{N}(\mu_2, \sigma_2))$에서 $\sigma_2 \to 0$이면 발산 → **log-variance reparameterization** 필수.
- Softmax의 KL: logit space 정규화 ($\log$-sum-exp trick) 필수.

---

## 📌 핵심 정리

| 개념 | 수식 / 성질 |
|---|---|
| **정의** | $\operatorname{KL}(p\|q) = \mathbb{E}_p[\log(p/q)]$ |
| **비음성 (Gibbs)** | $\operatorname{KL} \ge 0$, 등호 ⟺ $p = q$ |
| **비대칭** | $\operatorname{KL}(p\|q) \ne \operatorname{KL}(q\|p)$ |
| **Convexity** | $(p, q)$에 jointly convex |
| **Pinsker** | $\operatorname{TV} \le \sqrt{\operatorname{KL}/2}$ |
| **Data Processing** | Markov kernel 아래 감소 |
| **Chain rule** | $\operatorname{KL}(p(X,Y)\|q) = \operatorname{KL}(p(X)\|q(X)) + \operatorname{KL}(p(Y\|X)\|q(Y\|X))$ |
| **Mutual Info** | $I(X;Y) = \operatorname{KL}(p(X,Y) \| p(X)\otimes p(Y))$ |
| **Fisher 근사** | $\operatorname{KL}(p_\theta \| p_{\theta+d\theta}) \approx \tfrac{1}{2} d\theta^\top F d\theta$ |

**Takeaway:**

1. KL은 **정보 이론적 거리** — "잘못된 코드의 초과 비트".
2. **비대칭 + convex + DPI** — 이 세 성질이 AI 손실 함수 설계의 핵심.
3. **Fisher의 globalization** — 미세 섭동은 Fisher quadratic, 큰 차이는 KL.
4. **Mutual Info 및 모든 엔트로피 기반 regularization의 모태**.

---

## 🤔 생각해볼 문제

1. **KL은 왜 거리가 아닌가?** 삼각 부등식 위반 예를 구체적으로 찾아라. 세 Bernoulli $p, q, r$로 $\operatorname{KL}(p\|r) > \operatorname{KL}(p\|q) + \operatorname{KL}(q\|r)$ 구현.

2. **Jensen-Shannon divergence**. $\operatorname{JS}(p, q) := \tfrac{1}{2}\operatorname{KL}(p\|m) + \tfrac{1}{2}\operatorname{KL}(q\|m)$, $m = (p+q)/2$. 이것이 대칭이고 유계 ($\le \log 2$) 임을 보이고, $\sqrt{\operatorname{JS}}$가 metric임을 확인 (증명은 스케치).

3. **Mode collapse의 KL 해석**. GAN에서 reverse KL $\operatorname{KL}(p_g \| p_{\mathrm{data}})$ 최소화시 mode collapse. Forward KL 으로 바꾸면 해결될까? Forward KL 계산의 어려움은?

4. **KL의 Wasserstein-유사 해석**. $\operatorname{KL}(p\|q) = \int p \log(p/q)$는 두 분포 간 "엔트로피 격차". Wasserstein-2는 "수송 비용". 언제 두 거리의 선호 방향이 바뀌는가? GAN 학습 안정성에서.

5. **KL의 경계 행동**. $\operatorname{KL}(\mathcal{N}(\mu, \sigma^2) \| \delta_0)$ (점 분포) = $\infty$. 이것의 해석과, density smoothing을 쓸 때 KL이 어떻게 behave?

6. **Reverse-mode autodiff로 KL 계산**. VAE에서 $\operatorname{KL}(q_\phi(z|x) \| \mathcal{N}(0, I))$의 closed form은? Gaussian family일 때 explicit한 공식 유도.

7. **Information Bottleneck의 Lagrangian 해석**. $\min_T I(X;T) - \beta I(T;Y)$의 solution curve (Information Plane) 이 $\beta$에 따라 어떻게 변하는지. $\beta \to \infty$ 한계에서 $T = Y$ 회귀.

---

<div align="center">

| [◀ Ch2-05. Cramér-Rao](../ch2-statistical-fisher/05-cramer-rao-geometry.md) | [📚 메인 README](../README.md) | [02. KL의 Fisher 2차 근사 ▶](./02-kl-fisher-connection.md) |
|:---:|:---:|:---:|

</div>
