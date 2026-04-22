# 01. 다양체(Manifold)의 기초

## 🎯 핵심 질문

- 왜 "국소적으로 $\mathbb{R}^n$과 같다"는 조건이 필요한가?
- 차트(chart)와 아틀라스(atlas)는 무엇이고, 왜 전이함수(transition map)의 매끈함이 본질적인가?
- 왜 확률분포족 $\{p_\theta\}_{\theta \in \Theta}$를 다양체로 볼 수 있는가? — "분포를 점으로 보는" 것이 엄밀한가?
- $\mathbb{R}^n$으로는 표현할 수 없지만 다양체로는 자연스러운 공간의 예는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

확률분포 공간은 일반적으로 **유클리드 공간이 아니다**.

- **Softmax 출력 $\{p \in \mathbb{R}^K : p_i \geq 0, \sum p_i = 1\}$** 은 $(K-1)$-심플렉스다. 이것은 $\mathbb{R}^K$의 부분집합이지만, **$\mathbb{R}^{K-1}$과 동형인 다양체**일 뿐 벡터공간이 아니다.
- **공분산 행렬 공간 $\{\Sigma \in \mathbb{R}^{d \times d} : \Sigma \succ 0\}$** 는 **SPD 다양체**다. 이 위의 두 점을 단순 평균하면 양정치성이 깨질 수 있어, 유클리드 평균이 자연스러운 연산이 아니다.
- **정규분포족 $\{\mathcal{N}(\mu, \sigma^2) : \mu \in \mathbb{R}, \sigma > 0\}$** 은 상반평면 $\mathbb{H}^2 = \{(\mu, \sigma) : \sigma > 0\}$ 위의 점들이다. Fisher 계량은 이 공간에 **쌍곡기하**를 주어 유클리드 거리와 전혀 다른 거리를 만든다.

"분포를 점으로 보는" 것이 엄밀하려면, 먼저 **점이 놓이는 공간이 어떤 구조를 가지는가**를 정의해야 한다. 그것이 다양체다.

---

## 📐 수학적 선행 조건

- 위상공간의 기초: 열린집합, 연속, 하우스도르프, 제2가산
- 실공간의 미분가능성: $C^k$ 함수, $C^\infty$(매끈) 함수, 역함수 정리
- 절댓값·노름: $\mathbb{R}^n$에서의 거리

> 선형대수의 기본(벡터공간·선형사상)은 Chapter 1-02(접공간)부터 필요합니다.  
> → [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)는 Ch2 이후 권장.

---

## 📖 직관적 이해

### "국소적으로 $\mathbb{R}^n$"이라는 조건

지구 표면은 전체를 평면에 한 장의 지도로 펼칠 수 없다(위상적으로 구와 평면이 다름). 하지만 **내가 서 있는 주변은 평면으로 근사할 수 있고**, 서로 다른 지도를 **겹치는 영역에서 매끈하게 변환**할 수 있다. 이것이 다양체의 핵심 직관이다.

- 전체를 한 번에 다루는 것은 포기한다
- 대신 **국소 좌표계(차트)**의 모음으로 덮는다
- 차트가 겹치는 곳에서 **좌표 변환(전이함수)이 매끈**하면, 미분이 좌표계 선택에 무관해진다

### 분포 공간에 대한 비유

| 유클리드 공간 $\mathbb{R}^n$ | 다양체 $M$ |
|----|----|
| 전역 좌표계 하나 | 여러 국소 차트의 모음 |
| 두 점 사이 직선이 "거리" | 측지선(geodesic)이 "거리" (곡면 위의 가장 짧은 경로) |
| 평균 $\frac{x+y}{2}$가 자연 | 측지선의 중점이 자연 (Fréchet mean) |
| 덧셈·스칼라배 가능 | 국소적으로만 선형 구조 |

> **비유**: 정규분포 $\mathcal{N}(\mu, \sigma^2)$들이 이루는 공간은 종이에 그린 "$\mu$-$\sigma$ 평면"처럼 보이지만, 사실 **쌍곡면처럼 휘어져 있다**. 두 분포 사이의 진짜 거리는 유클리드 거리 $\sqrt{(\mu_1-\mu_2)^2 + (\sigma_1-\sigma_2)^2}$가 아니라, Fisher-Rao 측지선이다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 위상다양체 (Topological Manifold)

위상공간 $M$이 **$n$차원 위상다양체**라는 것은 다음을 모두 만족하는 것이다:

1. **하우스도르프(Hausdorff)**: 서로 다른 두 점 $p, q \in M$에 대해 서로소인 열린 근방 $U \ni p$, $V \ni q$가 존재한다
2. **제2가산(second countable)**: $M$의 위상에 가산 기저가 존재한다
3. **국소 유클리드(locally Euclidean)**: 모든 $p \in M$에 대해 열린 근방 $U \ni p$와 동형사상(homeomorphism) $\varphi: U \to \varphi(U) \subseteq \mathbb{R}^n$이 존재한다

쌍 $(U, \varphi)$을 **차트(chart)**, $\varphi$를 **국소좌표계(local coordinate)**라 한다.

### 정의 1.2 — $C^\infty$-호환성과 아틀라스

두 차트 $(U_\alpha, \varphi_\alpha)$, $(U_\beta, \varphi_\beta)$가 **$C^\infty$-호환**이라는 것은, $U_\alpha \cap U_\beta \neq \emptyset$이면 **전이함수(transition map)**

$$\varphi_\beta \circ \varphi_\alpha^{-1}: \varphi_\alpha(U_\alpha \cap U_\beta) \to \varphi_\beta(U_\alpha \cap U_\beta)$$

가 $\mathbb{R}^n$의 열린집합 사이의 $C^\infty$ 사상(즉 매끈한 사상)임을 뜻한다.

**아틀라스(atlas)** $\mathcal{A} = \{(U_\alpha, \varphi_\alpha)\}_{\alpha \in A}$는 $M = \bigcup_\alpha U_\alpha$를 덮고 모든 차트가 서로 $C^\infty$-호환인 차트의 모임이다. **극대 아틀라스**(더 이상 확장 불가능한 아틀라스)를 **매끈한 구조(smooth structure)**라 한다.

### 정의 1.3 — 매끈한 다양체

위상다양체 $M$과 그 위의 매끈한 구조 $\mathcal{A}$의 쌍 $(M, \mathcal{A})$을 **$n$차원 매끈한 다양체(smooth manifold)**라 한다.

### 정의 1.4 — 매끈한 사상

$M, N$이 매끈한 다양체일 때, 사상 $f: M \to N$이 $p \in M$에서 **매끈**하다는 것은, $p$ 주변 차트 $(U, \varphi)$와 $f(p)$ 주변 차트 $(V, \psi)$에 대해

$$\psi \circ f \circ \varphi^{-1}: \varphi(U \cap f^{-1}(V)) \to \psi(V)$$

가 **$\mathbb{R}^n$의 사상으로서 매끈**함을 뜻한다. 전이함수의 $C^\infty$-호환성이 이 정의의 **차트-독립성**을 보장한다.

### 정의 1.5 — 통계다양체 (Statistical Manifold)

확률분포족 $\mathcal{P} = \{p_\theta : \theta \in \Theta\}$가 **통계다양체**라는 것은, 다음을 만족하는 것이다:

1. 매개변수 공간 $\Theta \subseteq \mathbb{R}^n$이 열린집합
2. $\theta \mapsto p_\theta$가 단사(injective) — 서로 다른 $\theta$는 서로 다른 분포를 준다
3. $p_\theta(x)$가 $\theta$에 대해 $C^\infty$이고, 미분과 적분의 교환 등 **정칙성 조건(regularity)**을 만족

이때 $\Theta$가 $\mathcal{P}$의 전역 차트가 되어 $\mathcal{P}$는 $n$차원 매끈한 다양체 구조를 가진다.

---

## 🔬 정리와 증명

### 정리 1.1 — $S^n$이 매끈한 $n$차원 다양체

**명제**: $S^n = \{x \in \mathbb{R}^{n+1} : \|x\| = 1\}$은 매끈한 $n$차원 다양체이다.

**증명 스케치**: 북극 $N = (0, \ldots, 0, 1)$과 남극 $S = (0, \ldots, 0, -1)$을 제외한 두 차트를 **입체투영(stereographic projection)**으로 정의한다:

$$\varphi_N: S^n \setminus \{N\} \to \mathbb{R}^n, \quad \varphi_N(x_1, \ldots, x_{n+1}) = \left(\frac{x_1}{1 - x_{n+1}}, \ldots, \frac{x_n}{1 - x_{n+1}}\right)$$

$\varphi_S$는 $N$을 $-N$으로 바꾼 식. 이 두 차트는 $S^n \setminus \{N, S\}$에서 겹치고, 전이함수는

$$\varphi_S \circ \varphi_N^{-1}(y) = \frac{y}{\|y\|^2}$$

로 $y \neq 0$에서 $C^\infty$이다. $\square$

> **딥러닝 응용**: 단위 구면 $S^{d-1}$은 정규화된 임베딩 공간(예: CLIP, SimCLR의 임베딩이 $\ell_2$-정규화된 후 사는 공간)이다. 이 공간 위의 거리는 유클리드 거리가 아니라 구면 측지선이다.

---

### 정리 1.2 — 전이함수의 매끈함은 차트 선택의 "무관함"을 준다

**명제**: $f: M \to \mathbb{R}$이 정의 1.4의 의미에서 매끈하다는 것은, **임의의 한 쌍의 호환 차트**에서 매끈함을 확인하면 충분하다.

**증명**: 두 차트 $(U, \varphi)$, $(U', \varphi')$이 있다 하자. $f \circ \varphi^{-1}$이 매끈하면,

$$f \circ \varphi'^{-1} = (f \circ \varphi^{-1}) \circ (\varphi \circ \varphi'^{-1})$$

이고, $\varphi \circ \varphi'^{-1}$은 전이함수로서 $C^\infty$이다. **매끈한 함수들의 합성은 매끈**하므로 $f \circ \varphi'^{-1}$도 매끈하다. $\square$

> 이 정리가 **다양체 위의 미분이 좌표계에 의존하지 않는** 근본적 이유다. 차트가 바뀌어도 전이함수의 매끈함이 미분 구조를 보존한다.

---

### 정리 1.3 — 통계다양체는 매끈한 다양체

**명제**: 정의 1.5의 조건 하에서 $\mathcal{P} = \{p_\theta\}$는 매끈한 $n$차원 다양체이다.

**증명 스케치**: $\Theta$ 자체가 $\mathbb{R}^n$의 열린집합이므로 **단일 차트 $(\mathcal{P}, \varphi^{-1})$**, 여기서 $\varphi: \Theta \to \mathcal{P}$, $\varphi(\theta) = p_\theta$로 전역 차트를 구성하면 된다. 단사성(조건 2)이 $\varphi$의 역이 존재함을 보장하고, $\Theta$의 열린성이 국소 유클리드 조건을 만족시킨다. 하우스도르프·제2가산은 $\Theta \subseteq \mathbb{R}^n$의 성질에서 유전된다. $\square$

> **중요**: 통계다양체가 단일 차트로 덮이는 것은 "운 좋은" 경우다. 일반적으로 분포족이 **특이점(예: $\sigma \to 0$에서 정규분포의 Fisher 정보가 발산)**을 가지면 차트를 쪼개야 한다. 이것은 Ch2의 Fisher 정보 정칙성 가정으로 이어진다.

---

### 예시 3개

**예시 1 — $\mathcal{N}(\mu, \sigma^2)$ 다양체**
- $\Theta = \{(\mu, \sigma) : \mu \in \mathbb{R}, \sigma > 0\} = \mathbb{R} \times \mathbb{R}_{>0}$ — 상반평면
- 전역 차트: $\varphi(p_{\mu, \sigma}) = (\mu, \sigma)$
- 2차원 매끈한 다양체

**예시 2 — 심플렉스 $\Delta^{K-1}$**
- $\Delta^{K-1} = \{p \in \mathbb{R}^K : p_i \geq 0, \sum p_i = 1\}$의 **내부**(모든 $p_i > 0$)
- 차트: $\varphi(p) = (p_1, \ldots, p_{K-1})$ (마지막 좌표 제거)
- $(K-1)$차원 매끈한 다양체
- **경계**(어떤 $p_i = 0$)는 제외해야 매끈한 다양체가 된다

**예시 3 — SPD 다양체 $\text{Sym}_+^d$**
- $d \times d$ 대칭 양정치 행렬의 공간
- 대칭 조건은 $d(d+1)/2$개의 독립 성분, 양정치는 열린 조건
- $d(d+1)/2$차원 매끈한 다양체
- 공분산 행렬, Gaussian의 정확도(precision) 행렬이 사는 공간

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

# ─────────────────────────────────────────────
# 1. 구면 S² 위의 두 차트와 전이함수 검증
# ─────────────────────────────────────────────

def stereo_N(x):
    """북극 입체투영: S² \ {N} → ℝ²"""
    x1, x2, x3 = x
    return np.array([x1 / (1 - x3), x2 / (1 - x3)])

def stereo_N_inv(u):
    """역변환: ℝ² → S² \ {N}"""
    u1, u2 = u
    r2 = u1**2 + u2**2
    return np.array([2*u1 / (1 + r2), 2*u2 / (1 + r2), (r2 - 1) / (r2 + 1)])

def stereo_S(x):
    """남극 입체투영: S² \ {S} → ℝ²"""
    x1, x2, x3 = x
    return np.array([x1 / (1 + x3), x2 / (1 + x3)])

# 전이함수 φ_S ∘ φ_N⁻¹: y ↦ y/||y||²
def transition(y):
    return y / np.dot(y, y)

# 검증: S² 위의 점을 N-차트로 보내고, S-차트로 보낸 뒤, 전이함수와 일치하는지
rng = np.random.default_rng(0)
theta, phi = rng.uniform(0.1, np.pi - 0.1), rng.uniform(0, 2*np.pi)
p = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
y_N = stereo_N(p)
y_S_direct = stereo_S(p)
y_S_via_transition = transition(y_N)
print("─ S² 차트 전이함수 검증 ─")
print(f"직접 계산 φ_S(p)     : {y_S_direct}")
print(f"전이함수 y_N → y_S   : {y_S_via_transition}")
print(f"오차                : {np.linalg.norm(y_S_direct - y_S_via_transition):.2e}")

# ─────────────────────────────────────────────
# 2. SymPy로 전이함수의 야코비안이 가역임을 확인
# ─────────────────────────────────────────────
y1, y2 = sp.symbols('y1 y2', real=True)
r2 = y1**2 + y2**2
T = sp.Matrix([y1/r2, y2/r2])      # transition y ↦ y/||y||²
J = T.jacobian([y1, y2])
print("\n─ 전이함수 야코비안 ─")
sp.pprint(sp.simplify(J))
print(f"det(J) = {sp.simplify(J.det())}")   # -1/(y1²+y2²)² ≠ 0 → 가역

# ─────────────────────────────────────────────
# 3. N(μ, σ²) 다양체의 전역 차트 시각화
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (좌) 매개변수 공간 Θ = ℝ × ℝ₊
mus = np.linspace(-3, 3, 9)
sigmas = np.linspace(0.3, 3, 9)
MU, SIG = np.meshgrid(mus, sigmas)
axes[0].scatter(MU, SIG, c='b', s=15)
axes[0].set_xlabel(r'$\mu$')
axes[0].set_ylabel(r'$\sigma$')
axes[0].set_title(r'통계다양체 $\{\mathcal{N}(\mu,\sigma^2)\}$의 전역 차트')
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

# (우) 각 (μ, σ) 쌍에 대응하는 분포 밀도
x_grid = np.linspace(-6, 6, 400)
for mu_, sig_ in [(0, 1), (1, 0.5), (-1, 2), (2, 1.5)]:
    pdf = np.exp(-0.5*((x_grid - mu_)/sig_)**2) / (sig_*np.sqrt(2*np.pi))
    axes[1].plot(x_grid, pdf, label=rf'$\mathcal{{N}}({mu_}, {sig_}^2)$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('p(x)')
axes[1].set_title('각 점 = 하나의 분포')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01-manifold-basics.png', dpi=150, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
─ S² 차트 전이함수 검증 ─
직접 계산 φ_S(p)     : [0.4273 0.8107]
전이함수 y_N → y_S   : [0.4273 0.8107]
오차                : 1.11e-16

─ 전이함수 야코비안 ─
⎡y2² - y1²      -2·y1·y2  ⎤
⎢──────────    ───────────⎥
⎣(y1²+y2²)²    (y1²+y2²)² ⎦
...
det(J) = -1/(y1² + y2²)²
```

---

## 🔗 AI/ML 연결

### 정규화된 임베딩과 구면 다양체

CLIP, SimCLR 등 contrastive learning에서 임베딩은 $\ell_2$-정규화되어 **단위 구면 $S^{d-1}$** 위에 놓인다. 두 임베딩의 유사도로 쓰는 코사인 유사도 $\cos\theta = \langle u, v\rangle$는 **구면 측지거리** $\arccos(\langle u, v\rangle)$와 직접 대응한다. 유클리드 거리로 근사하면 비등방성이 생긴다.

### Softmax 출력과 심플렉스

분류 모델의 softmax 출력 $p \in \Delta^{K-1}$은 심플렉스 위의 점이다. KL divergence는 이 다양체에서 **Fisher-Rao 계량에 따른 자연스러운 divergence**로, 유클리드 거리와 본질적으로 다르다. Cross-entropy gradient는 $(p - y)$라는 간단한 형태를 가지지만, 이는 심플렉스 다양체의 e-좌표에서 본 것이다(Ch4).

### 공분산 행렬과 SPD 다양체

가우시안 혼합, 변분 오토인코더, Kalman 필터 등에서 공분산 행렬 $\Sigma$의 공간은 SPD 다양체다. 두 공분산의 단순 평균 $(\Sigma_1 + \Sigma_2)/2$는 SPD 다양체의 자연스러운 중점이 아니며, **affine-invariant 계량** 하의 측지선 중점(기하평균 $\Sigma_1^{1/2}(\Sigma_1^{-1/2}\Sigma_2 \Sigma_1^{-1/2})^{1/2}\Sigma_1^{1/2}$)이 올바른 중점이다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\Theta$가 열린집합 | 경계(예: $p_i = 0$의 심플렉스 경계, $\sigma = 0$)는 다양체가 아니거나 특이점 |
| 매개변수화의 단사성 | 과매개변수화된 모델(신경망)은 $\theta \neq \theta'$이어도 $p_\theta = p_{\theta'}$ 가능 → **부분다양체(quotient manifold)**로 처리 필요 |
| 정칙성(미분-적분 교환) | 경계가 매개변수에 의존하는 분포(예: 균등분포 $U(0, \theta)$)는 정규 exp family 밖 |
| $C^\infty$ | 실제로는 $C^2$면 Fisher 계량 유도에 충분, 그러나 고차 기하는 $C^\infty$ 필요 |

**특이점 주의**: 정규분포에서 $\sigma \to 0$이면 Fisher 정보가 발산, 심플렉스 경계에서 Fisher가 특이해진다. 이런 점들은 차트에서 제외되어야 한다.

---

## 📌 핵심 정리

$$M: \text{매끈한 } n\text{-다양체} \iff \underbrace{\text{국소적으로 }\mathbb{R}^n}_{\text{차트 존재}} + \underbrace{\text{전이함수 }C^\infty}_{\text{호환 아틀라스}} + \text{하우스도르프·제2가산}$$

| 개념 | 핵심 |
|------|------|
| 차트 $(U, \varphi)$ | 다양체의 국소 좌표계, $\varphi: U \to \mathbb{R}^n$ 동형사상 |
| 전이함수 | 겹치는 차트 사이의 좌표 변환 $\varphi_\beta \circ \varphi_\alpha^{-1}$ |
| 매끈한 구조 | 모든 전이함수가 $C^\infty$인 극대 아틀라스 |
| 통계다양체 | 매개변수화된 분포족 $\{p_\theta\}$; $\Theta$가 전역 차트 |
| 매끈한 사상 | 차트를 통해 $\mathbb{R}^n$의 매끈한 사상으로 환원 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 원 $S^1 \subset \mathbb{R}^2$에 대해 두 차트 (예: 상반원 + 하반원)를 구성하고, 전이함수가 매끈함을 직접 보여라.

<details>
<summary>힌트 및 해설</summary>

상반원 $U_+ = \{(x,y) \in S^1 : y > 0\}$에서 $\varphi_+(x,y) = x$, 하반원 $U_- = \{y < 0\}$에서 $\varphi_-(x,y) = x$. 좌/우 반원도 유사하게 추가해야 $S^1$ 전체를 덮는다. 전이함수는 $\varphi_\beta \circ \varphi_\alpha^{-1}: x \mapsto \pm\sqrt{1-x^2}$이 겹치는 영역 $(-1, 1)$에서 $C^\infty$. 각 $\pm 1$ 근방은 $y$-좌표로 매개변수화하는 별도 차트가 필요 — 최소 4개 차트로 $S^1$의 아틀라스를 만들 수 있다.

</details>

**문제 2** (심화): 심플렉스 $\Delta^{K-1}$의 내부에서 $\varphi(p) = (p_1, \ldots, p_{K-1})$를 좌표로 잡고, **softmax 매개변수화** $p_i = e^{\theta_i}/\sum_j e^{\theta_j}$를 두 번째 차트로 볼 때 전이함수를 구하고 매끈함을 논하라. $\theta$의 전체 shift $\theta \to \theta + c\mathbf{1}$이 $p$를 바꾸지 않는 사실이 무엇을 의미하는가?

<details>
<summary>힌트 및 해설</summary>

Softmax는 $\mathbb{R}^K \to \Delta^{K-1}$의 매끈한 사상이지만 **단사가 아니다**($\theta + c\mathbf{1}$ 불변). 따라서 $\mathbb{R}^K$ 전체가 $\Delta^{K-1}$의 차트가 될 수 없다. 표준 해결책은 **제약** $\theta_K = 0$을 두어 $\mathbb{R}^{K-1}$로 축소하거나, **부분공간** $\{\theta : \sum \theta_i = 0\}$으로 제한하는 것. 이 경우 $p_i$와 $\theta$ 사이의 변환은 매끈하고 단사이며, softmax는 exp family의 canonical parameter $\theta$와 expectation parameter $p$의 Legendre 쌍대 관계(Ch4)를 이루는 두 차트가 된다.

</details>

**문제 3** (AI 연결): 베르누이 분포족 $\{B(p) : 0 < p < 1\}$을 1차원 다양체로 볼 때, 차트로 (a) $p$ 자체 (b) log-odds $\theta = \log(p/(1-p))$ 두 가지가 가능하다. 전이함수와 그 역의 야코비안을 구하고, deep learning에서 logit vs probability 표현이 기하학적으로 무엇을 의미하는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

$\theta = \log\frac{p}{1-p}$, 역 $p = \sigma(\theta) = 1/(1+e^{-\theta})$. 야코비안은 $dp/d\theta = p(1-p)$로 $p = 0, 1$에서 0이 되며, **전이함수의 매끈함이 경계에서 깨진다** — 차트가 내부에 한정되는 이유. Deep learning에서 logit $\theta$와 probability $p$는 같은 베르누이 다양체의 두 좌표이고, logit 공간에서의 선형 결합이 probability 공간의 비선형 연산에 대응한다. Exp family에서 $\theta$는 canonical, $p$는 expectation parameter로 Legendre 쌍대(Ch4)다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 접벡터와 접공간 ▶](./02-tangent-space.md) |

</div>
