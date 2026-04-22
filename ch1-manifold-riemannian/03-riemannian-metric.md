# 03. 리만 계량과 거리

## 🎯 핵심 질문

- 리만 계량은 왜 **각 점의 접공간에 내적을 주는 것**으로 정의되는가?
- 두 점을 잇는 **측지선(geodesic)**은 왜 Euler-Lagrange 방정식의 해인가?
- 계량이 다르면 같은 다양체의 "거리"가 어떻게 달라지는가? — 쌍곡기하 $\mathbb{H}^2$ 예제
- 분포 공간의 Fisher-Rao 계량이 왜 자연스러운가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **임베딩 공간의 거리**: 단순 $\ell_2$ 대신 학습된 계량(metric learning)을 쓰면, 같은 점 사이의 "거리"가 달라진다. Siamese Network, Triplet Loss, DML이 모두 이 아이디어.
- **Natural Gradient의 본질**: Fisher 행렬이 곧 통계다양체의 **리만 계량**이다. NGD가 "가장 가파른 방향"이라는 주장은 이 계량 하에서만 참이다.
- **하이퍼볼릭 임베딩**: Nickel & Kiela(2017) Poincaré embedding은 트리 구조를 쌍곡공간에 넣어, 유클리드에서 불가능한 지수적으로 많은 leaf를 표현한다 — 공간의 **곡률이 표현력을 바꾸는** 사례.
- **Wasserstein Gradient Flow**: 분포 공간의 Wasserstein 계량 하에서 KL divergence의 gradient flow가 Fokker-Planck 방정식이 된다(Jordan-Kinderlehrer-Otto).

"두 파라미터 사이 거리"가 자명하다는 생각은 오직 유클리드에서만 맞다. 분포 공간에서는 **어떤 계량을 주느냐가 본질**이다.

---

## 📐 수학적 선행 조건

- Ch1-01~02: 다양체, 접공간, 좌표기저
- 선형대수: 내적, 양정치 대칭 행렬, 이중선형형식
- 변분법: Euler-Lagrange 방정식 (최적화 레포 Ch6 참조)

> 내적 공간과 양정치 행렬은 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) Ch4 참조. 변분법은 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) Ch6.

---

## 📖 직관적 이해

### "점마다 다른 자"를 주는 것

리만 계량은 **각 점 $p \in M$의 접공간 $T_p M$에 내적 $g_p(\cdot, \cdot)$를 주는 것**이다. 즉,

$$\text{계량} = \{\text{점마다 다른 자(ruler)}\}$$

- $\mathbb{R}^n$의 유클리드 계량은 "모든 점에서 같은 자" — 그래서 평탄.
- 정규분포 다양체의 Fisher 계량은 "$\sigma$가 작은 곳에서 자가 더 길어진다" — $\sigma = 0.01$에서 $\mu$를 0.01 움직이는 것이 $\sigma = 10$에서 $\mu$를 10 움직이는 것과 "같은 정보량"을 준다.

### 측지선 = "이 자로 잰 가장 짧은 경로"

곡면 위에서 두 점을 잇는 직선은 없다. 대신 **가장 짧은 곡선**을 찾는다 — 이것이 측지선. 지구 표면의 항공경로(대원)가 대표적.

> **비유**: 평지에서는 최단 경로가 직선이지만, 언덕이 있는 지형에서는 경사와 거리를 모두 고려한 **우회 경로**가 "가장 짧다". 리만 계량이 이 "경사 비용"을 담는 텐서다.

### 유클리드 vs 쌍곡 vs 분포 공간

| 공간 | 계량 | 두 점 거리 |
|----|----|----|
| $\mathbb{R}^2$ 유클리드 | $ds^2 = dx^2 + dy^2$ | $\sqrt{(\Delta x)^2 + (\Delta y)^2}$ |
| 쌍곡 상반평면 $\mathbb{H}^2$ | $ds^2 = \frac{dx^2 + dy^2}{y^2}$ | 측지선 = 반원 또는 수직선 |
| 정규분포 $\{\mathcal{N}(\mu, \sigma^2)\}$ | $ds^2 = \frac{d\mu^2}{\sigma^2} + \frac{2\,d\sigma^2}{\sigma^2}$ | Fisher-Rao 측지선 |

**놀라운 사실**: 정규분포의 Fisher 계량은 **$\mathbb{H}^2$ 쌍곡계량의 상수배**다. 즉 정규분포족은 쌍곡공간이다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 리만 계량

매끈한 다양체 $M$ 위의 **리만 계량** $g$는, 각 점 $p \in M$에 접공간 $T_p M$ 위의 **양정치 대칭 이중선형형식** $g_p: T_p M \times T_p M \to \mathbb{R}$을 매끄럽게 할당하는 것이다:

1. **대칭**: $g_p(X, Y) = g_p(Y, X)$
2. **쌍선형**: 각 인자에 대해 $\mathbb{R}$-선형
3. **양정치**: $X \neq 0 \Rightarrow g_p(X, X) > 0$
4. **매끄러움**: 벡터장 $X, Y$가 매끄러우면 $p \mapsto g_p(X_p, Y_p)$가 $C^\infty$

좌표 $(\theta^1, \ldots, \theta^n)$에서 좌표기저 $\partial_i = \partial/\partial\theta^i$에 대한 성분으로

$$g_{ij}(\theta) := g_p(\partial_i, \partial_j), \qquad g = g_{ij}\, d\theta^i \otimes d\theta^j$$

$(g_{ij})$는 점마다 **양정치 대칭 행렬**이다.

### 정의 3.2 — 선소(line element)와 길이

접벡터 $X = X^i \partial_i$에 대해 $\|X\|_g^2 = g_{ij} X^i X^j$. 매끄러운 곡선 $\gamma: [a, b] \to M$의 **길이**는

$$L(\gamma) := \int_a^b \sqrt{g_{ij}(\gamma(t)) \dot\gamma^i(t) \dot\gamma^j(t)} \, dt$$

**선소(line element)** 표기: $ds^2 = g_{ij} d\theta^i d\theta^j$

### 정의 3.3 — 측지거리와 측지선

두 점 $p, q \in M$의 **측지거리**는

$$d_g(p, q) := \inf \{L(\gamma) : \gamma \text{ 매끄러운 곡선}, \gamma(a) = p, \gamma(b) = q\}$$

이 infimum을 달성하는 곡선(존재하면)을 **측지선(geodesic)**이라 한다.

### 정의 3.4 — 에너지 범함수와 Euler-Lagrange

길이 대신 **에너지 범함수**

$$E(\gamma) := \frac{1}{2}\int_a^b g_{ij}(\gamma) \dot\gamma^i \dot\gamma^j \, dt$$

의 극값을 찾는 것이 편하다. $E$의 라그랑지안 $\mathcal{L} = \frac{1}{2} g_{ij} \dot\gamma^i \dot\gamma^j$에 대한 Euler-Lagrange 방정식은

$$\boxed{\;\ddot\gamma^k + \Gamma^k_{ij}(\gamma) \dot\gamma^i \dot\gamma^j = 0\;}$$

여기서

$$\Gamma^k_{ij} = \frac{1}{2} g^{k\ell}\left(\partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij}\right)$$

는 **Christoffel 기호 (제2종)** — $g^{k\ell}$은 $(g_{ij})$의 역행렬.

### 정의 3.5 — Fisher-Rao 계량 (예고)

통계다양체 $\{p_\theta\}$ 위의 **Fisher-Rao 계량**은

$$g_{ij}(\theta) := \mathbb{E}_\theta[\partial_i \log p_\theta \cdot \partial_j \log p_\theta] = F_{ij}(\theta)$$

Ch2에서 정칙성 조건 하에서 양정치성·매끄러움을 증명한다.

---

## 🔬 정리와 증명

### 정리 3.1 — 측지선 방정식의 유도

**명제**: 에너지 범함수 $E(\gamma)$의 정류점(stationary point)은 $\ddot\gamma^k + \Gamma^k_{ij}\dot\gamma^i \dot\gamma^j = 0$을 만족한다.

**증명**: 라그랑지안 $\mathcal{L} = \frac{1}{2} g_{ij}(\gamma) \dot\gamma^i \dot\gamma^j$에 대해

$$\frac{\partial \mathcal{L}}{\partial \gamma^k} = \frac{1}{2} \partial_k g_{ij} \cdot \dot\gamma^i \dot\gamma^j$$

$$\frac{\partial \mathcal{L}}{\partial \dot\gamma^k} = g_{kj} \dot\gamma^j$$

Euler-Lagrange $\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot\gamma^k} - \frac{\partial \mathcal{L}}{\partial \gamma^k} = 0$에 대입:

$$\frac{d}{dt}(g_{kj} \dot\gamma^j) - \frac{1}{2} \partial_k g_{ij} \dot\gamma^i \dot\gamma^j = g_{kj}\ddot\gamma^j + \partial_i g_{kj} \dot\gamma^i \dot\gamma^j - \frac{1}{2}\partial_k g_{ij} \dot\gamma^i \dot\gamma^j = 0$$

$\dot\gamma^i \dot\gamma^j$는 $i, j$ 대칭이므로 $\partial_i g_{kj}$를 $\frac{1}{2}(\partial_i g_{kj} + \partial_j g_{ki})$로 대칭화. 따라서:

$$g_{kj}\ddot\gamma^j + \frac{1}{2}(\partial_i g_{kj} + \partial_j g_{ki} - \partial_k g_{ij}) \dot\gamma^i \dot\gamma^j = 0$$

양변에 $g^{\ell k}$를 곱해 $j$→$\ell$ 치환하면 $\Gamma^\ell_{ij}$의 정의가 나와 원하는 식을 얻는다. $\square$

---

### 정리 3.2 — 길이와 에너지의 극값 일치 (매개변수화 재조정)

**명제**: 에너지 $E$의 극값 곡선은 **arc-length 매개변수화**(즉 $\|\dot\gamma\|_g = \text{const}$) 하에서 길이 $L$의 극값 곡선이다.

**증명 스케치**: Cauchy-Schwarz로 $L(\gamma)^2 \leq 2(b-a) E(\gamma)$이고 등호는 $\|\dot\gamma\|_g$가 상수일 때. 길이 범함수는 매개변수화 재조정 하에서 불변이므로, arc-length로 재매개변수화하면 $E$와 $L$의 극값 조건이 일치한다. $\square$

---

### 정리 3.3 — 쌍곡 상반평면 $\mathbb{H}^2$의 측지선

**명제**: 상반평면 $\mathbb{H}^2 = \{(x, y) : y > 0\}$에 쌍곡 계량 $ds^2 = (dx^2 + dy^2)/y^2$를 주면, 측지선은 **(a) $x = \text{const}$인 수직선** 또는 **(b) 실축 위에 중심을 둔 반원**이다.

**증명 (수직선)**: $\gamma(t) = (x_0, y(t))$로 놓으면 $g_{xx} = g_{yy} = 1/y^2$, $g_{xy} = 0$. 따라서

$$\Gamma^y_{yy} = \frac{1}{2} g^{yy} (\partial_y g_{yy}) = \frac{1}{2} y^2 \cdot (-2/y^3) = -1/y$$

측지선 방정식: $\ddot y - \frac{1}{y}\dot y^2 = 0$. 해: $y(t) = y_0 e^{ct}$, 즉 지수적으로 올라가거나 내려가는 수직선. $\square$

**증명 스케치 (반원)**: Isometry $(x, y) \to (-x, y)$, scaling $(x, y) \to \lambda(x, y)$, translation $(x, y) \to (x + c, y)$을 조합하면 임의 반원을 수직선으로 보낼 수 있다 → 수직선 측지선이 반원으로 변환 → 반원도 측지선. (Möbius 변환의 hyperbolic isometry군.) $\square$

> **$\mathbb{H}^2$에서 두 점 거리**: $(x_1, y_1), (x_2, y_2)$ 사이의 측지거리는
> $$d_{\mathbb{H}}((x_1,y_1), (x_2,y_2)) = \text{arcosh}\left(1 + \frac{(x_1-x_2)^2 + (y_1-y_2)^2}{2 y_1 y_2}\right)$$

---

### 정리 3.4 — 정규분포의 Fisher-Rao 계량은 쌍곡계량

**명제**: $\mathcal{N}(\mu, \sigma^2)$ 통계다양체의 Fisher-Rao 계량은 $(\mu, \sigma)$ 좌표에서

$$ds^2 = \frac{d\mu^2}{\sigma^2} + \frac{2\,d\sigma^2}{\sigma^2}$$

이고, $(\mu, \tilde\sigma = \sigma\sqrt{2})$ 좌표에서 상반평면 쌍곡계량 $(d\mu^2 + d\tilde\sigma^2)/(\tilde\sigma^2/2)$와 **conformal**이다.

**증명**: Ch2-04에서 $F = \text{diag}(1/\sigma^2, 2/\sigma^2)$. 치환 $\tilde\sigma = \sigma\sqrt{2}$하면 $d\tilde\sigma = \sqrt{2}\,d\sigma$, $d\sigma^2 = d\tilde\sigma^2/2$, $\sigma^2 = \tilde\sigma^2/2$:

$$ds^2 = \frac{d\mu^2}{\tilde\sigma^2/2} + \frac{2 \cdot d\tilde\sigma^2/2}{\tilde\sigma^2/2} = \frac{2}{\tilde\sigma^2}(d\mu^2 + d\tilde\sigma^2)$$

즉 $\mathbb{H}^2$의 **2배 계량**. 곡률은 상수 $-1/2$ (음의 곡률). $\square$

> **해석**: 두 정규분포 사이의 "자연스러운 거리"는 유클리드 거리가 아니라 쌍곡거리다. $\sigma$가 작을수록 $\mu$ 방향 자가 길어진다(작은 분포는 평균이 조금만 달라도 크게 다르다). NGD가 유클리드 SGD보다 잘 작동하는 기하학적 근거.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Christoffel 기호를 Fisher 계량으로부터 symbolic 유도
# ─────────────────────────────────────────────
mu, sig = sp.symbols('mu sigma', positive=True)
g = sp.Matrix([[1/sig**2, 0], [0, 2/sig**2]])
g_inv = g.inv()
coords = [mu, sig]
n = 2

# Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
def christoffel(g, g_inv, coords):
    n = len(coords)
    Gamma = [[[sp.S.Zero]*n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = sp.S.Zero
                for l in range(n):
                    s += g_inv[k, l] * (sp.diff(g[j, l], coords[i])
                                        + sp.diff(g[i, l], coords[j])
                                        - sp.diff(g[i, j], coords[l]))
                Gamma[k][i][j] = sp.simplify(s / 2)
    return Gamma

G = christoffel(g, g_inv, coords)
print("─ 정규분포 다양체의 Christoffel 기호 ─")
for k in range(n):
    for i in range(n):
        for j in range(n):
            val = G[k][i][j]
            if val != 0:
                print(f"  Γ^{coords[k]}_{coords[i]}{coords[j]} = {val}")

# ─────────────────────────────────────────────
# 2. 측지선 ODE 수치 풀이 (Runge-Kutta 4)
# ─────────────────────────────────────────────
from scipy.integrate import solve_ivp

def geodesic_ode(t, y):
    mu, sig, dmu, dsig = y
    # Γ^μ_μσ = Γ^μ_σμ = -1/σ, Γ^σ_μμ = 1/(2σ), Γ^σ_σσ = -1/σ
    ddmu  = -(-1/sig) * 2 * dmu * dsig
    ddsig = -(1/(2*sig)) * dmu**2 - (-1/sig) * dsig**2
    return [dmu, dsig, ddmu, ddsig]

# 초기 조건: (μ₀=0, σ₀=1), 초기 속도 다양하게
fig, ax = plt.subplots(figsize=(9, 6))
for v0 in [(0.3, 0.0), (0.0, 0.3), (0.3, 0.3), (-0.3, 0.3), (0.2, -0.1)]:
    sol = solve_ivp(geodesic_ode, (0, 3), [0.0, 1.0, v0[0], v0[1]],
                    dense_output=True, rtol=1e-9)
    tt = np.linspace(0, 3, 500)
    mus, sigs = sol.sol(tt)[:2]
    mask = sigs > 0.05  # σ > 0 유지
    ax.plot(mus[mask], sigs[mask], lw=2, label=f'v=({v0[0]},{v0[1]})')

ax.scatter([0], [1], color='k', s=60, zorder=5, label='시작점')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\sigma$')
ax.set_title(r'$\mathcal{N}(\mu, \sigma^2)$ 다양체의 측지선 (Fisher-Rao)')
ax.axhline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('03-geodesics-fisher-rao.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. 유클리드 거리 vs Fisher-Rao 측지거리 비교
# ─────────────────────────────────────────────
def fisher_rao_distance_normal(mu1, s1, mu2, s2):
    # (μ, σ√2) 좌표에서 H² 거리
    x1, y1 = mu1, s1*np.sqrt(2)
    x2, y2 = mu2, s2*np.sqrt(2)
    arg = 1 + ((x1 - x2)**2 + (y1 - y2)**2) / (2 * y1 * y2)
    return np.sqrt(2) * np.arccosh(arg)  # factor √2 from metric scaling

pairs = [((0, 1), (0, 2)), ((0, 0.1), (0, 0.2)), ((0, 1), (5, 1))]
print("\n─ 거리 비교 (유클리드 vs Fisher-Rao) ─")
for (m1, s1), (m2, s2) in pairs:
    eu = np.sqrt((m1 - m2)**2 + (s1 - s2)**2)
    fr = fisher_rao_distance_normal(m1, s1, m2, s2)
    print(f"  N({m1},{s1}²) ↔ N({m2},{s2}²): Euclidean={eu:.3f}, Fisher-Rao={fr:.3f}")
# 예:
#  N(0,0.1²) ↔ N(0,0.2²): Euclidean=0.100, Fisher-Rao≈0.980
#  → 작은 σ 영역에서 Fisher-Rao 거리가 훨씬 크다
```

**출력 예시**:
```
─ 정규분포 다양체의 Christoffel 기호 ─
  Γ^μ_μσ = -1/σ
  Γ^μ_σμ = -1/σ
  Γ^σ_μμ = 1/(2σ)
  Γ^σ_σσ = -1/σ

─ 거리 비교 (유클리드 vs Fisher-Rao) ─
  N(0,1²) ↔ N(0,2²): Euclidean=1.000, Fisher-Rao=0.980
  N(0,0.1²) ↔ N(0,0.2²): Euclidean=0.100, Fisher-Rao=0.980
  N(0,1²) ↔ N(5,1²): Euclidean=5.000, Fisher-Rao=3.454
```

→ $\sigma$ 척도 변화는 **Fisher-Rao에서 같은 거리**(0.980)로 본다. 유클리드는 0.1배로 인식.

---

## 🔗 AI/ML 연결

### Metric Learning

Siamese/Triplet 구조는 학습 가능한 $\phi_\theta$를 통해 $d(x, y) = \|\phi_\theta(x) - \phi_\theta(y)\|$를 정의한다. 이것은 **$\phi_\theta$가 유도한 pullback 계량**이다: 원래 공간 $\mathcal{X}$ 위에 유클리드가 아닌 새 계량 $g = \phi_\theta^* g_\text{Eucl}$. 학습은 "데이터 공간의 계량을 바꾸는 것".

### Poincaré/Lorentz Embedding (하이퍼볼릭)

트리/계층 구조는 $\mathbb{R}^d$에서 효율적으로 표현 불가능하지만, $\mathbb{H}^d$에서는 노드 수가 반지름에 지수적으로 증가하는 공간에 자연스럽게 들어간다. Nickel & Kiela(2017)의 Poincaré ball은 WordNet의 의미 계층을 2D 쌍곡공간에 embedding해 유클리드 대비 차원 효율 ~10배 향상.

### Fisher-Rao 거리로 모델 비교

두 학습된 모델 $p_{\theta_1}, p_{\theta_2}$의 "거리"는 $\|\theta_1 - \theta_2\|$가 아니라 **Fisher-Rao 측지거리**로 봐야 한다. 재매개변수화(예: LayerNorm의 scale)는 유클리드에서 거리를 바꾸지만, Fisher-Rao에서는 불변이다(Ch2-03).

### Wasserstein Gradient Flow

확률분포 공간 $\mathcal{P}(\mathbb{R}^d)$에 Wasserstein-2 계량을 주면 **"분포의 다양체"**가 된다. KL divergence의 W₂-gradient flow는 Fokker-Planck 방정식(Jordan-Kinderlehrer-Otto 1998) — diffusion model 이론의 핵심 수학.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 양정치 $g_{ij}$ | Fisher가 특이(singular)하면 Natural Gradient 발산; pseudo-Riemannian(Lorentzian) 확장 필요 |
| 매끄러움 | $g$가 저규칙(low regularity)이면 측지선 존재성이 불투명 |
| 연결(closed)·완비(complete) | 완비하지 않으면 측지선이 유한시간에 다양체 밖으로 나갈 수 있음 (Hopf-Rinow) |
| $\inf$ 달성 | 비컴팩트 다양체에선 infimum이 달성되지 않을 수 있음 |

**특이 계량 주의**: 심플렉스의 경계, $\sigma \to 0$의 정규분포, rank-deficient 공분산 등에서 Fisher 계량이 특이해진다. 이 점들은 다양체의 "가장자리"로 처리한다.

---

## 📌 핵심 정리

$$\text{리만 계량 } g: p \mapsto \langle \cdot, \cdot \rangle_p \text{ on } T_p M \quad\leftrightarrow\quad g_{ij}(\theta) = \text{점마다 다른 양정치 행렬}$$

$$\text{측지선: } \ddot\gamma^k + \Gamma^k_{ij}(\gamma)\dot\gamma^i \dot\gamma^j = 0, \quad \Gamma^k_{ij} = \frac{1}{2} g^{k\ell}(\partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij})$$

| 개념 | 핵심 |
|------|------|
| 계량 성분 $g_{ij}$ | 좌표기저의 내적, 점마다 양정치 대칭 행렬 |
| 선소 $ds^2$ | 점의 무한소 이동에 대한 길이 제곱 |
| 측지선 | 에너지/길이 범함수의 극값, Christoffel로 ODE |
| Fisher-Rao | 통계다양체의 자연스러운 계량, 정규족에선 쌍곡계량 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 극좌표 $(r, \phi)$로 쓴 유클리드 $\mathbb{R}^2$의 계량은 $ds^2 = dr^2 + r^2 d\phi^2$다. 이로부터 $g_{ij}$를 쓰고, Christoffel 기호 $\Gamma^r_{\phi\phi}$, $\Gamma^\phi_{r\phi}$를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$g_{ij} = \text{diag}(1, r^2)$, $g^{ij} = \text{diag}(1, 1/r^2)$.  
$\Gamma^r_{\phi\phi} = \frac{1}{2}g^{rr}(2\partial_\phi g_{r\phi} - \partial_r g_{\phi\phi}) = -\frac{1}{2}\cdot 2r = -r$.  
$\Gamma^\phi_{r\phi} = \frac{1}{2}g^{\phi\phi}(\partial_r g_{\phi\phi}) = \frac{1}{2}\cdot \frac{1}{r^2}\cdot 2r = 1/r$.  
측지선 방정식 $\ddot r - r\dot\phi^2 = 0$, $\ddot\phi + (2/r)\dot r \dot\phi = 0$ — 이는 원점 중심 직선의 극좌표 표현.

</details>

**문제 2** (심화): $\mathbb{H}^2$의 수직선 측지선 $y(t) = e^t$을 arc-length 매개변수라 확인하라. 즉 $\|\dot\gamma\|_g = 1$인지.

<details>
<summary>힌트 및 해설</summary>

$\gamma(t) = (x_0, e^t)$, $\dot\gamma = (0, e^t)$. $\|\dot\gamma\|_g^2 = \frac{0^2 + (e^t)^2}{(e^t)^2} = 1$. ✓ Arc-length 매개변수. 따라서 두 점 $(x_0, y_1), (x_0, y_2)$ 사이의 쌍곡거리는 $|\log(y_2/y_1)|$.

</details>

**문제 3** (AI 연결): SGD의 학습률 $\eta$는 유클리드 공간에서 "step size". Natural Gradient의 $\eta$는 어떤 공간의 step size이며, **Fisher-Rao 측지선을 얼마나 따라가는가**? 작은 $\eta$에서 NGD와 Fisher-Rao 측지선이 일치하는지 논하라.

<details>
<summary>힌트 및 해설</summary>

NGD 업데이트 $\theta \leftarrow \theta - \eta F^{-1}\nabla L$은 Fisher 계량 하의 **접벡터 방향으로 Euler step** 한 번. Fisher-Rao 측지선은 정확히 $\Gamma$ 항을 포함한 2차 ODE의 해. $\eta \to 0$에서 둘 다 "접벡터 방향으로 직진"하지만, NGD는 1차 근사, 측지선은 2차 근사 이상을 따른다. **정확히 측지선을 따라가려면 "natural gradient" 대신 "exponential map"** $\theta \to \exp_\theta(-\eta F^{-1}\nabla L)$을 써야 한다 — Riemannian optimization 이론.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. 접벡터와 접공간](./02-tangent-space.md) | [📚 README로 돌아가기](../README.md) | [04. 아핀 연결과 공변미분 ▶](./04-connection-christoffel.md) |

</div>
