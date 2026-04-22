# 03. Fisher-Rao 계량과 Chentsov 유일성 정리

> **"확률분포 공간 위의 자연스러운 거리는 유일하다 — 그리고 그것이 Fisher이다."**

---

## 🎯 핵심 질문

**통계다양체 $\mathcal{S} = \{p_\theta\}$ 위에 부과할 수 있는 "자연스러운" 리만 계량은 무엇이며, 왜 그것이 유일하게 Fisher인가?**

$$
\boxed{\;g_{ij}^{\mathrm{FR}}(\theta) \;=\; F_{ij}(\theta) \;=\; \mathbb{E}_\theta[\partial_i \ell_\theta \cdot \partial_j \ell_\theta]\;}
$$

**Chentsov (1972) 정리**: 통계적 불변성을 만족하는 리만 계량은 **상수배를 제외하고 Fisher 계량뿐**이다.

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 상황 | Fisher-Rao가 답하는 것 |
|---|---|
| **Natural Gradient** | Euclidean 경사의 "자연스러움"을 왜 바꾸어야 하는가? 파라미터 **reparametrization 불변**한 유일한 경사 = Fisher 기반. |
| **분포 간 거리 비교** | $\operatorname{KL}$은 비대칭·비거리. Fisher-Rao **측지선 거리**는 진짜 거리 (대칭·삼각부등식). 확산모델의 지각적 거리 등. |
| **Hyperbolic Embedding** | $\mathcal{N}(\mu, \sigma^2)$의 Fisher-Rao 기하 = **쌍곡 기하**. Poincaré embeddings, hyperbolic neural networks의 수학적 토대. |
| **분포강건 최적화** | Wasserstein ball vs KL ball vs Fisher ball — 불확실성 집합의 선택. |
| **Chentsov 유일성** | 어떤 계량을 쓰든 정보 처리에 합리적이면 결국 Fisher로 귀결 → 이론적 **피할 수 없음**. |

---

## 📐 수학적 선행 조건

- Ch1-03 [리만 계량](../ch1-manifold-riemannian/03-riemannian-metric.md) — 리만 계량, 길이, 측지선 정의
- Ch1-04 [접속과 Christoffel](../ch1-manifold-riemannian/04-connection-christoffel.md) — Levi-Civita 접속
- 본 챕터의 [02. Fisher 3가지 정의](./02-fisher-3-equivalence.md) — Fisher의 정의와 변환법칙
- Markov kernel, Sufficient statistic (충분통계량) 개념
- f-divergence 개념 (부록 수준)

---

## 📖 직관적 이해

### 왜 Euclidean 계량으로는 부족한가

파라미터 공간 $\Theta \subset \mathbb{R}^k$ 위의 Euclidean 거리 $\|\theta_1 - \theta_2\|_2$는 **좌표에 의존**. 예를 들어 $\mathcal{N}(\mu, \sigma)$과 $\mathcal{N}(\mu, \sigma^2)$ (분산 vs 표준편차) 두 매개화에서 같은 두 분포 사이의 "거리"가 달라진다.

반면 **분포 자체**의 공간 $\{p_\theta\}$ 위에는 매개화 무관한 거리가 있어야 한다. Fisher 계량이 바로 이 역할을 한다:

$$
ds^2 \;=\; \sum_{i,j} F_{ij}(\theta)\, d\theta^i d\theta^j
$$

이 $ds$는 **좌표 변환에 불변** (정리 3.5 in 02 문서) — 즉 $\mu$ 좌표에서든 $\sigma^2$ 좌표에서든 같은 길이.

### 정보적 직관

$p_\theta$와 $p_{\theta+d\theta}$의 Fisher-Rao 거리²는 **KL의 두 배** (대칭화):

$$
ds^2 \;\approx\; 2 \operatorname{KL}(p_\theta \| p_{\theta+d\theta}) \;=\; d\theta^\top F d\theta.
$$

즉 "두 분포를 구분하기 얼마나 쉬운가"를 잰다. Fisher가 크면 조금만 떨어져도 구분 쉬움 (가파른 지형). 작으면 멀리 떨어져도 구분 어려움 (평평한 지형).

### Chentsov 유일성

만약 Fisher와 다른 리만 계량을 쓴다면, **Markov kernel**을 적용했을 때 거리가 늘어나는 수학적 비정상성이 생긴다. 정보 이론의 자연스러운 요청 — "정보 처리는 정보를 줄인다(또는 보존한다)" (data processing inequality) — 과 호환되는 **유일한** 계량이 Fisher.

---

## ✏️ 엄밀한 정의

### 정의 4.1 (Fisher-Rao 계량)

정칙 통계다양체 $\mathcal{S} = \{p_\theta : \theta \in \Theta\}$ 위의 **Fisher-Rao 계량**은 각 점 $\theta$에서의 $(0,2)$-텐서

$$
g^{\mathrm{FR}}_{ij}(\theta) \;:=\; F_{ij}(\theta) \;=\; \mathbb{E}_\theta[\partial_i \ell_\theta(X) \cdot \partial_j \ell_\theta(X)].
$$

$F(\theta)$가 양의 정치 ($F \succ 0$) 라면 $(\mathcal{S}, g^{\mathrm{FR}})$은 **리만 다양체**가 된다.

### 정의 4.2 (Fisher-Rao 측지선 거리)

$\theta_0, \theta_1 \in \Theta$에 대해,

$$
d_{\mathrm{FR}}(\theta_0, \theta_1) \;:=\; \inf_{\gamma} \int_0^1 \sqrt{\dot\gamma(t)^\top F(\gamma(t))\, \dot\gamma(t)}\, dt,
$$

여기서 infimum은 $\gamma(0)=\theta_0$, $\gamma(1)=\theta_1$인 smooth curve 모두에 대해. 이것이 **진짜 거리** (positivity, symmetry, triangle inequality).

### 정의 4.3 (Markov 사상의 통계적 불변성)

두 측정공간 $(\mathcal{X}, \mu)$, $(\mathcal{Y}, \nu)$ 사이의 **Markov kernel** $K(y|x)$가 주어질 때, 모델 $\{p_\theta\} \subset \mathscr{P}(\mathcal{X})$는 pushforward

$$
q_\theta(y) \;=\; \int K(y|x) p_\theta(x)\, d\mu(x)
$$

로 $\mathscr{P}(\mathcal{Y})$의 모델이 된다.

**통계적 불변성**: 계량 $g$가 "통계적으로 불변"이라 함은 — 모든 Markov kernel $K$에 대해

$$
g^{(\mathcal{Y})}(\theta) \;\preceq\; g^{(\mathcal{X})}(\theta).
$$

(Markov 처리 후 계량은 작아질 수만 있다 = 정보 감소.)

**충분통계량 case**에서는 등호: $K$가 **충분통계량**이면 $g^{(\mathcal{Y})} = g^{(\mathcal{X})}$.

---

## 🔬 정리와 증명

### 정리 4.1 (Fisher-Rao 계량은 통계적 불변)

$K$가 Markov kernel이고 $q_\theta = K \cdot p_\theta$이면

$$
F^{(q)}(\theta) \;\preceq\; F^{(p)}(\theta).
$$

$K$가 충분통계량 $T: \mathcal{X} \to \mathcal{Y}$에 대응하면 등호.

**증명 (스케치).** Jensen 부등식. 조건부 기댓값 $\mathbb{E}[s_p | T(X)]$를 정의하면

$$
F^{(q)}_{ij} = \mathbb{E}[\mathbb{E}[s^p_i|T] \cdot \mathbb{E}[s^p_j|T]]
$$

이고 $F^{(p)}_{ij} = \mathbb{E}[s^p_i s^p_j]$. 조건부 기댓값의 수축 (tower rule + Jensen) 에 의해

$$
F^{(q)}_{ij} v^i v^j \;=\; \mathbb{E}[\mathbb{E}[v \cdot s^p | T]^2] \;\le\; \mathbb{E}[(v \cdot s^p)^2] \;=\; F^{(p)}_{ij} v^i v^j.
$$

충분통계량이면 $\sigma(T) \supseteq \sigma(s^p)$ 이어서 수축이 equality. **Q.E.D.**

---

### 정리 4.2 (Chentsov 1972: 유일성 정리)

$\mathcal{S}_n$을 $n$-점 유한 표본공간 위의 분포 simplex라 하자. 리만 계량 패밀리 $\{g^{(n)}\}_{n \ge 2}$가 다음을 만족한다 가정:

1. 각 $g^{(n)}$은 $\mathcal{S}_n$ 위의 smooth 리만 계량.
2. **Markov morphism 불변**: $K_{n \to m}$이 Markov kernel이면 $K^*g^{(m)} = g^{(n)}$ (충분한 morphism에 대해).

그러면 어떤 상수 $c > 0$이 존재해

$$
g^{(n)} \;=\; c \cdot F^{(n)} \qquad (\text{모든 } n \text{에 대해}).
$$

**증명 (개요).**

**Step 1.** $n = 2$ (Bernoulli) 에서 일반 불변 계량의 형태를 구한다. $\mathcal{S}_2 = \{(p, 1-p) : p \in (0,1)\}$ 은 $1$-차원이므로 임의의 리만 계량은 $g(p)\, dp^2$. Markov 불변성 (예: copying, permutation) 이 $g(p) = c / (p(1-p))$ 를 강제 ⟸ Fisher.

**Step 2.** $n \ge 3$ 을 $2 \cdot \binom{n}{2}$ 개의 Bernoulli embedding으로 덮는다. 각 2차원 submanifold에서 $g^{(n)}$은 $c \cdot F^{(n)}$로 제약됨.

**Step 3.** Consistency. 다른 embedding에서 유도되는 $c$들이 모두 같음을 보인다. ← Markov morphism으로 이들을 서로 변환.

상세 증명은 Chentsov (1972) 또는 Campbell (1986) 참조.

> **정리 4.2의 의의:** 정보처리에 "얌전한" 리만 계량은 **본질적으로 유일**. Fisher를 쓰는 것이 선택이 아니라 **필연**.

---

### 정리 4.3 (정규분포의 Fisher-Rao = 쌍곡기하)

$\mathcal{S} = \{ \mathcal{N}(\mu, \sigma^2) : \mu \in \mathbb{R}, \sigma > 0 \}$, 좌표 $(\mu, \sigma)$. 그 Fisher-Rao 계량은

$$
g^{\mathrm{FR}} = \frac{1}{\sigma^2}\, d\mu^2 + \frac{2}{\sigma^2}\, d\sigma^2.
$$

이 계량을 가진 $(\mu, \sigma) \in \mathbb{R} \times \mathbb{R}_{>0}$은 **쌍곡반평면 $\mathbb{H}^2$ 의 (스케일 변환된) 표준 계량**과 국소적으로 isometric.

**증명.** 02 문서 예제에서 이미 계산: $F_{\mu\mu} = 1/\sigma^2$, $F_{\mu\sigma} = 0$, $F_{\sigma\sigma} = 2/\sigma^2$. 따라서

$$
ds^2 = \frac{d\mu^2 + 2\,d\sigma^2}{\sigma^2}.
$$

좌표 변환 $\tilde\sigma = \sqrt{2}\,\sigma$ 또는 $y = \sigma$로 재스케일하면 표준 Poincaré 반평면 $ds^2 = (d\mu^2 + dy^2)/y^2$와 동형 (스칼라 인자 제외).

**따름 4.3.1.** $\mathcal{N}(\mu_1, \sigma_1)$와 $\mathcal{N}(\mu_2, \sigma_2)$ 사이의 Fisher-Rao 측지선 거리:

$$
d_{\mathrm{FR}}(p_1, p_2) \;=\; \sqrt{2}\, \operatorname{arccosh}\!\left( 1 + \frac{(\mu_1 - \mu_2)^2 + 2(\sigma_1 - \sigma_2)^2}{4\sigma_1 \sigma_2}\right).
$$

이 공식은 Rao (1945) 와 Atkinson & Mitchell (1981) 에서 유도.

---

### 정리 4.4 (Multinomial의 Fisher-Rao = 구면 기하)

$\mathcal{S}_n = \{p = (p_1, \ldots, p_n) : p_i \ge 0,\, \sum p_i = 1\}$의 내부에서 $\sqrt{p_i}$ 좌표로 바꾸자: $\xi_i = 2\sqrt{p_i}$. 그러면 $\sum \xi_i^2 = 4\sum p_i = 4$.

$\mathcal{S}_n$은 **반지름 2 구면 $S^{n-1}$의 first octant** (모든 $\xi_i \ge 0$) 로 embedding. 이 embedding에서 Fisher-Rao 계량은 **표준 구면 계량**.

**증명.** 좌표 변환 아래 Fisher 계량:

$$
F_{ij}(p) = \frac{\delta_{ij}}{p_i}
$$

(단, 중심좌표화 후). $\xi_i = 2\sqrt{p_i}$이면 $d\xi_i = dp_i / \sqrt{p_i}$, 즉 $d\xi_i^2 = dp_i^2 / p_i$. 따라서

$$
ds^2 = \sum_i \frac{dp_i^2}{p_i} = \sum_i d\xi_i^2
$$

(표준 유클리드 — simplex에서). 구속조건 $\sum \xi_i^2 = 4$에서 induced 계량은 구면 계량.

**따름.** $n$-카테고리 분포들의 Fisher-Rao 거리:

$$
d_{\mathrm{FR}}(p, q) = 2 \arccos\!\left( \sum_i \sqrt{p_i q_i}\right).
$$

$\sum_i \sqrt{p_i q_i} = \cos(\theta/2)$ 인 각 $\theta$ — 이것이 **Hellinger affinity** (Bhattacharyya coefficient) 와 정확히 일치!

---

### 정리 4.5 (측지선 방정식)

Fisher-Rao 계량의 Levi-Civita 접속의 Christoffel 기호:

$$
\Gamma^k_{ij}(\theta) = \tfrac{1}{2}\sum_\ell g^{k\ell}(\partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij}),
$$

측지선 $\gamma(t)$는 다음 ODE를 만족:

$$
\ddot\gamma^k + \sum_{ij} \Gamma^k_{ij}(\gamma) \dot\gamma^i \dot\gamma^j = 0.
$$

**의의.** Fisher-Rao 측지선은 "정보 관점에서 가장 효율적인 분포 변형 경로". 나중에 **e-geodesic** (mixture path) 과 **m-geodesic** (e-connection) 과 구별될 것이다 (Ch4).

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: 정규분포 Fisher-Rao 거리 — 공식 vs 수치적분

```python
import numpy as np
from scipy.integrate import quad

def fr_distance_normal_closed(mu1, sigma1, mu2, sigma2):
    """Rao-Atkinson-Mitchell closed-form (factor √2)."""
    num = (mu1 - mu2)**2 + 2*(sigma1 - sigma2)**2
    den = 4*sigma1*sigma2
    return np.sqrt(2) * np.arccosh(1 + num/den)

def fr_distance_normal_numeric(mu1, sigma1, mu2, sigma2, n_steps=200):
    """Path integral: straight line in (μ, σ) space."""
    def arc(t):
        mu    = mu1 + t*(mu2 - mu1)
        sigma = sigma1 + t*(sigma2 - sigma1)
        dmu   = mu2 - mu1
        dsig  = sigma2 - sigma1
        # ds² = dμ²/σ² + 2 dσ²/σ²
        return np.sqrt((dmu**2 + 2*dsig**2) / sigma**2)
    # straight-line 적분 (이것은 측지선이 아니라 상한!)
    dist, _ = quad(arc, 0, 1, limit=200)
    return dist

# Test
pairs = [
    (0, 1, 0, 2),
    (0, 1, 1, 1),
    (0, 0.5, 2, 1.5),
]
for mu1, sig1, mu2, sig2 in pairs:
    d_closed = fr_distance_normal_closed(mu1, sig1, mu2, sig2)
    d_numeric = fr_distance_normal_numeric(mu1, sig1, mu2, sig2)
    print(f"N({mu1},{sig1}) ~ N({mu2},{sig2}):  closed={d_closed:.4f}, straight-line={d_numeric:.4f}")
    print(f"  (straight-line >= closed is expected)")
```

**기대:** `straight-line` 거리는 측지선 거리의 **상한**이므로 `d_closed <= d_numeric`.

---

### 예제 2: Multinomial Fisher-Rao = Spherical 확인

```python
import numpy as np

def fr_distance_multinomial(p, q):
    """Closed form: 2·arccos(Σ √(p_i q_i))"""
    inner = np.sum(np.sqrt(p * q))
    inner = np.clip(inner, -1, 1)  # 수치 안정
    return 2 * np.arccos(inner)

def fr_spherical_distance(xi_p, xi_q):
    """구면 √p 좌표에서의 대원 거리 (반지름 2이므로 2·각도)"""
    cos_angle = np.dot(xi_p, xi_q) / (np.linalg.norm(xi_p) * np.linalg.norm(xi_q))
    cos_angle = np.clip(cos_angle, -1, 1)
    return 2 * np.arccos(cos_angle)  # 반지름 2의 arc length = 2 * angle

# 3-카테고리 예
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.1, 0.6, 0.3])

# √p 좌표
xi_p = 2 * np.sqrt(p)   # ||xi_p|| = 2
xi_q = 2 * np.sqrt(q)

print(f"||xi_p||² = {np.sum(xi_p**2):.6f}  (expected 4)")
print(f"||xi_q||² = {np.sum(xi_q**2):.6f}  (expected 4)")

d_mult = fr_distance_multinomial(p, q)
d_sphe = fr_spherical_distance(xi_p, xi_q)

print(f"\nFR distance (direct): {d_mult:.6f}")
print(f"FR distance (spherical): {d_sphe:.6f}")
print(f"Match? {np.isclose(d_mult, d_sphe)}")

# Hellinger affinity
print(f"\nBhattacharyya coef (Σ√(pq)) = {np.sum(np.sqrt(p*q)):.4f}")
print(f"  cos(d_FR / 2) = {np.cos(d_mult/2):.4f}  (should match!)")
```

---

### 예제 3: Chentsov 불변성의 실험적 검증

```python
import numpy as np

def fisher_bernoulli(p):
    """Bernoulli Fisher: F(p) = 1/(p(1-p))"""
    return 1 / (p * (1 - p))

# 모델: p_θ = θ on {0, 1}, θ ∈ (0, 1)
# Markov kernel K: 충분통계량 (identity) -> 동치
# Markov kernel K: 데이터 손실 (항상 0 출력) -> Fisher 감소

theta_vals = np.linspace(0.1, 0.9, 9)

# 케이스 1: Identity kernel (충분)
F_after_identity = [fisher_bernoulli(t) for t in theta_vals]

# 케이스 2: Noisy channel
# K(0|0) = 0.9, K(1|0) = 0.1, K(0|1) = 0.1, K(1|1) = 0.9
# q_θ(1) = 0.1·(1-θ) + 0.9·θ = 0.1 + 0.8θ
# 같은 Bernoulli 형태이므로 F on q = 1/(q(1-q))
# 체인룰: F_θ(q) = (dq/dθ)² · F(q) = 0.8² / (q(1-q))
F_after_noise = [0.8**2 / (0.8*t + 0.1) / (1 - 0.8*t - 0.1) for t in theta_vals]

print(f"{'θ':>6} {'F (identity)':>15} {'F (noisy)':>15} {'ratio':>10}")
for t, f_id, f_ns in zip(theta_vals, F_after_identity, F_after_noise):
    print(f"{t:>6.2f} {f_id:>15.4f} {f_ns:>15.4f} {f_ns/f_id:>10.4f}")

print("\n노이즈 후 Fisher가 감소 -> Markov monotonicity 확인")
```

**기대:** 두 번째 열 (noisy) 이 항상 첫 번째 열 보다 작음 (또는 같음) — **Chentsov 불변성 / Fisher monotonicity**.

---

### 예제 4: 쌍곡평면 Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Poincaré 반평면: (μ, σ), σ > 0
# 등거리 원 (측지선) 그리기
# μ-축 수직선 = 측지선 (중심이 μ-축에 있는 반원)

mu_vals = np.linspace(-3, 3, 100)

# (μ, σ) = (0, 1) 중심, 반지름 r인 쌍곡원
center = np.array([0, 1])
for r in [0.3, 0.6, 1.0, 1.5]:
    # (μ, σ) 공간의 쌍곡원 = euclidean 원이지만 중심·반지름이 다름
    # 쌍곡기하에서 (0,1) 중심 반지름 r 원: euclidean 중심 (0, cosh(r)), 반지름 sinh(r)
    eu_center_y = np.cosh(r)
    eu_radius = np.sinh(r)
    theta_param = np.linspace(0, 2*np.pi, 100)
    x_circle = 0 + eu_radius * np.cos(theta_param)
    y_circle = eu_center_y + eu_radius * np.sin(theta_param)
    ax.plot(x_circle, y_circle, label=f'hyperbolic r={r}')

# 측지선 예시 (두 점 연결: 중심이 μ축에 있는 반원)
# (μ=-1, σ=0.5) 와 (μ=1, σ=0.5) 연결: 중심 (0, 0), 반지름 √(1 + 0.25)
geo_center_x = 0
geo_radius = np.sqrt(1 + 0.25)
theta_half = np.linspace(0, np.pi, 100)
ax.plot(geo_center_x + geo_radius*np.cos(theta_half),
        geo_radius*np.sin(theta_half),
        'k--', label='geodesic')

# 기준점
ax.plot([0], [1], 'ro', markersize=8, label='N(0,1)')

ax.set_xlim(-3, 3)
ax.set_ylim(0, 3.5)
ax.set_xlabel('μ')
ax.set_ylabel('σ')
ax.set_title('Fisher-Rao on N(μ,σ²) = Hyperbolic Half-Plane')
ax.axhline(y=0, color='red', linewidth=0.5, linestyle=':')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('/sessions/kind-dazzling-ritchie/fisher_rao_hyperbolic.png', dpi=100)
plt.close()

print("✓ Hyperbolic visualization saved")
```

---

## 🔗 AI/ML 연결

### 1. Natural Gradient as Fisher-Rao Gradient

정의: $\tilde\nabla L(\theta) := F(\theta)^{-1} \nabla L(\theta)$.

**Amari (1998)**: $\tilde\nabla L$은 **Fisher-Rao 메트릭 하**에서의 steepest descent. 유일성 (Chentsov 4.2) 덕에 NGD는 "reparametrization-invariant 경사의 유일한 자연선택".

### 2. Hyperbolic Neural Networks (Ganea et al. 2018)

계층 구조 (tree) 데이터는 Euclidean 공간에 embed하기 부족 (볼륨 폭발). **Poincaré disk** ($\mathbb{H}^2$) 사용 시 Exponential family $\mathcal{N}$의 Fisher-Rao 기하와 직접 연결.

### 3. Wasserstein vs Fisher 기하

**Wasserstein-2** 계량: 최적수송 기반. 지지집합 이동 비용 관점.
**Fisher-Rao**: 정보/확률밀도 변화 관점.

두 계량의 보간 (Wasserstein-Fisher-Rao geometry) 이 **Unbalanced Optimal Transport** 의 수학적 기초 — 최근 확산모델·생성모델 연구 활발.

### 4. Information Bottleneck의 기하학적 해석

IB objective: $\min_T I(X; T) - \beta I(T; Y)$. 이때 $(X, T)$, $(T, Y)$ 결합분포 공간이 Fisher-Rao 다양체이며 IB 최적화는 이 기하 위의 변분 문제.

### 5. Mirror Descent와 Bregman 공식

$\operatorname{MD}$: $\theta_{k+1} = \arg\min_\theta \langle g_k, \theta \rangle + \frac{1}{\eta} D_\psi(\theta, \theta_k)$. Bregman $D_\psi$의 2차 근사가 Fisher (지수족에서 $\psi$ = cumulant, Ch4).

### 6. 정보적 Reparameterization의 안전성

Fisher의 변환법칙 (텐서성) 덕에 **parametrization 바꿔도 NGD 동등**:

```python
# Fisher-based update equivalent in any smooth coordinate system
# Euclidean SGD는 좌표계에 따라 다른 경로를 따르지만,
# NGD는 diffeomorphism 아래 불변.
```

이것이 딥러닝의 rescaling/reparametrization tricks (weight normalization, BN) 의 NGD 기반 재해석.

---

## ⚖️ 가정과 한계

### Chentsov 정리의 범위

- 원본 Chentsov 1972는 **유한 카테고리** 모델에 대한 정리. Campbell 1986, Ay-Jost-Lê-Schwachhöfer 2017 이 연속·비모수 모델로 확장.
- "통계적 불변성" 공리를 완화하면 다른 계량 (Wasserstein, $\chi^2$ 등) 도 가능.

### Fisher-Rao 거리 계산의 어려움

- **Closed form** 이 알려진 경우: 정규분포, multinomial, 감마분포, 지수분포 등. 대부분의 임의 모델에서는 **수치적분** 필요.
- 측지선 ODE 풀이가 고차원에서 매우 비쌈 → Natural gradient는 "로컬" 적용만.

### 쌍곡기하학의 한계

- 정규분포의 쌍곡기하 해석은 **scale-location family** 의 특수성; 일반 지수족으로 확장 안 됨.
- Natural parameter 좌표 vs expectation parameter 좌표의 구분 (Ch4) — 쌍곡 해석은 특정 좌표에 의존.

### 경계 이슈

- $\mathcal{S}_n$ simplex 의 **경계** ($p_i = 0$) 에서 Fisher-Rao 거리 발산.
- Softmax 신경망에서 very confident prediction 주변에서 Fisher-Rao 기하가 blow-up → Empirical Fisher 수치 불안정성의 근원.

---

## 📌 핵심 정리

| 개념 | 핵심 |
|---|---|
| **Fisher-Rao 계량** | $g_{ij} = F_{ij}$. 통계다양체의 canonical 리만 계량. |
| **통계적 불변성** | Markov kernel 아래 계량이 감소 (등호 ⟺ 충분통계량). |
| **Chentsov 유일성** | 통계 불변 리만 계량은 Fisher의 상수배뿐. |
| **정규분포 기하** | $\mathcal{N}(\mu, \sigma^2)$의 FR = 쌍곡평면 $\mathbb{H}^2$ (up to scale). |
| **Multinomial 기하** | $\mathcal{S}_n$의 FR = 반지름 2 구면 $S^{n-1}$의 first octant. |
| **KL과의 관계** | $\operatorname{KL}(p_\theta \| p_{\theta+d\theta}) \approx \tfrac{1}{2} ds^2$ (Fisher quadratic). |
| **AI 의의** | Natural Gradient의 유일성 보장. Hyperbolic NN, Mirror Descent 기초. |

---

## 🤔 생각해볼 문제

1. **왜 $c > 0$은 임의인가?** Chentsov 정리가 Fisher의 상수배 까지만 유일성을 주장하는 이유는? 만약 $F$ 대신 $2F$ 를 쓰면 natural gradient 업데이트가 어떻게 달라지는가? (힌트: learning rate 흡수)

2. **정규분포 측지선 explicit.** $\mathcal{N}(\mu_0, \sigma_0)$ 에서 $\mathcal{N}(\mu_1, \sigma_1)$ 로 가는 Fisher-Rao 측지선을 Poincaré 반평면 좌표에서 유도하라 — 결과: $\mu$-축에 중심을 둔 반원의 일부. 이것을 어떤 **중간 분포 경로**로 해석할 수 있는가?

3. **Simplex 경계 정규화.** $p_i \to 0$ 에서 Fisher 발산 문제를 실무에서 어떻게 피하는가? (Dirichlet prior, label smoothing, Laplace smoothing 등) 각 기법이 Fisher 기하에 어떤 효과를 주는지.

4. **유일성 유지의 본질.** Chentsov 공리 "Markov 감소성"을 빼면 어떤 다른 계량이 가능할까? $\alpha$-계량 ($\alpha \neq 0$) 은 왜 statistical 계량이 아닌가? (힌트: α-connection 은 계량이 아닌 비-메트릭 접속)

5. **고차원 저주.** $k$-parameter 모델에서 Fisher-Rao 볼륨 $\text{vol}(\mathcal{S}) = \int \sqrt{\det F}\, d\theta$는 Jeffreys prior와 연결. $k$가 크면 이 적분이 발산/특이점 문제. Bayesian 관점에서 어떤 의미?

6. **쌍곡 embedding의 실용성.** Poincaré ball로 knowledge graph embedding 시, Fisher-Rao 기하와 일치하는지? 실험적으로 임의 분포 데이터가 쌍곡공간 embedding에서 더 compact한 이유는?

7. **Fisher vs Wasserstein의 선택.** 언제 KL / Fisher 기반 손실이 적절하고, 언제 Wasserstein이 적절한가? GAN에서 둘의 성능 차이가 보이는 이유 — 지지집합이 달라질 때 KL이 발산하는 현상을 Fisher-Rao 관점에서 설명하라.

---

<div align="center">

| [◀ 02. Fisher 3가지 정의](./02-fisher-3-equivalence.md) | [📚 메인 README](../README.md) | [04. Fisher 계산 예제들 ▶](./04-fisher-examples.md) |
|:---:|:---:|:---:|

</div>
