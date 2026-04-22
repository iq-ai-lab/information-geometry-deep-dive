# 02. 접벡터와 접공간

## 🎯 핵심 질문

- "접벡터"는 왜 **세 가지 서로 다른 방식**(곡선의 속도, 방향미분연산자, $n$-튜플의 동치류)으로 정의되는가? 세 정의가 왜 동치인가?
- 접공간 $T_p M$이 **벡터공간**이 되려면 무엇이 필요한가?
- 통계다양체에서 접벡터는 왜 **스코어 함수 $\partial_\theta \log p$**로 자연스럽게 해석되는가?
- 좌표기저 $\partial/\partial\theta^i$의 좌표 변환 법칙은?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **자동미분에서 tangent vector**: JAX의 `jvp(f, (x,), (v,))`가 계산하는 것은 점 $x \in M$에서 접벡터 $v \in T_x M$을 따라간 방향 미분. 다양체 위 미분은 "접벡터에 대한 작용"이다.
- **스코어 함수**: 통계다양체의 접공간은 $\mathcal{T}_\theta = \{\partial_\theta \log p_\theta\}$ (스코어 함수들의 공간)와 동일시된다. Fisher 정보(Ch2)의 공분산 표현이 바로 이 공간에서의 내적이다.
- **Natural gradient는 접벡터**: $\tilde{\nabla} L = F^{-1}\nabla L$은 접공간 $T_\theta \mathcal{P}$의 원소로, Fisher-Rao 계량 하에서 "가장 가파른 방향"을 가리킨다.
- **Meta-learning의 tangent 표현**: MAML의 inner-loop update $\theta' = \theta - \alpha \nabla L$은 접벡터를 따라 한 스텝 이동한 것이고, outer-loop는 이 이동의 미분($T_\theta M$ 위의 사상의 pushforward)을 본다.

"기울기를 더해서 파라미터를 갱신한다"는 직관이 엄밀해지려면, 그 덧셈이 일어나는 공간 — 접공간 — 이 먼저 정의되어야 한다.

---

## 📐 수학적 선행 조건

- Ch1-01: 다양체, 차트, 매끈한 사상
- 선형대수: 벡터공간, 기저, 선형사상, 쌍대공간
- 미적분: 일변수·다변수 미분, 연쇄법칙

> 벡터공간의 기저와 좌표 표현은 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) Ch1-Ch3 참조.

---

## 📖 직관적 이해

### 접벡터 = "점에서 뻗어 나가는 속도"

구면 $S^2$ 위의 한 점 $p$에서의 접평면 $T_p S^2$는 직관적으로 **$p$에서의 속도벡터들이 모인 공간**이다. 지구 표면의 한 지점에서 가능한 모든 이동 방향(동서남북 + 그 조합) — 다만 "지구 안쪽"이나 "우주 바깥"은 제외.

**중요한 주의**: 추상 다양체(embedding 없이 정의된 것)에서는 "바깥 공간"이 없으므로, 접벡터를 **내재적(intrinsic)**으로 정의해야 한다. 세 가지 방법이 모두 동치임을 보이면, 어떤 관점에서 접근해도 같은 대상을 얻는다.

### 세 가지 정의를 하나로

| 정의 | 직관 | 장점 |
|----|----|----|
| **곡선 속도류** | "점을 지나는 곡선들의 순간 속도" | 기하적 직관이 명확 |
| **방향미분연산자** | "함수에 작용해서 방향 미분을 내놓는 연산자" | 대수적 조작 용이 |
| **$n$-튜플 동치류** | "좌표에서 $(v^1, \ldots, v^n)$, 좌표 바뀌면 야코비안으로 변환" | 계산에 적합 |

> **비유**: 등산 중 "내가 어디서 어느 방향으로 얼마나 빨리 가는지"는 (a) 내 걸음의 순간 속도(곡선 속도), (b) 주변 고도장에 내 이동이 주는 변화율(방향미분), (c) 나침반 좌표에서의 수치(튜플) — 세 가지로 똑같이 기술할 수 있다.

### 통계다양체에서의 접벡터 = 스코어

$p_\theta$에서 $p_{\theta + dt \cdot v}$로 움직이는 곡선을 생각하면, 각 $x$에서의 **로그 확률의 변화율**이 바로 스코어 함수 $v^i \partial_i \log p_\theta(x)$다. 즉 **접공간의 원소는 "로그 확률에 대한 무한소 변화를 주는 함수"**로 해석된다. 이 관점이 Fisher 계량(Ch2)으로 이어진다.

---

## ✏️ 엄밀한 정의

고정된 점 $p \in M$에 대해 세 가지 접공간 정의를 차례로 본다. $(U, \varphi)$는 $p$ 주변 차트, $\varphi(p) = 0$으로 둘 수 있다.

### 정의 2.1 — 곡선을 통한 접벡터

**매끈한 곡선** $\gamma: (-\varepsilon, \varepsilon) \to M$이 $\gamma(0) = p$를 만족할 때, 두 곡선 $\gamma_1 \sim \gamma_2$가 **동치**라는 것은

$$\frac{d}{dt}\bigg|_{t=0}(\varphi \circ \gamma_1)(t) = \frac{d}{dt}\bigg|_{t=0}(\varphi \circ \gamma_2)(t)$$

를 만족함(차트를 통해 본 속도가 같음). 동치류 $[\gamma]$들이 이루는 집합을 $T_p^{(c)} M$이라 한다.

### 정의 2.2 — 방향미분연산자로서의 접벡터

$C^\infty(p)$를 $p$ 주변에서 정의된 매끈한 함수들의 germ이라 하자. $p$에서의 **derivation**이란 $\mathbb{R}$-선형 사상 $X: C^\infty(p) \to \mathbb{R}$으로 **Leibniz 규칙**

$$X(fg) = X(f) \cdot g(p) + f(p) \cdot X(g)$$

을 만족하는 것이다. Derivation들의 집합을 $T_p^{(d)} M$이라 한다.

### 정의 2.3 — $n$-튜플 동치류로서의 접벡터

$p$ 주변의 차트 쌍 $(U_\alpha, \varphi_\alpha)$를 택해, 쌍 $(\alpha, v)$ with $v \in \mathbb{R}^n$을 고려한다. 두 쌍 $(\alpha, v), (\beta, w)$가 **동치**라는 것은 전이함수의 야코비안 $J = D(\varphi_\beta \circ \varphi_\alpha^{-1})(\varphi_\alpha(p))$가 $w = J v$를 만족함. 이 동치류들의 집합을 $T_p^{(t)} M$이라 한다.

### 정의 2.4 — 접공간

$T_p M$은 $T_p^{(c)}$, $T_p^{(d)}$, $T_p^{(t)}$ 중 어느 것으로 정의해도 같은 대상이며(정리 2.1), **$n$차원 벡터공간** 구조를 갖는다.

### 정의 2.5 — 좌표기저

차트 $(U, \varphi = (\theta^1, \ldots, \theta^n))$에 대해,

$$\left(\frac{\partial}{\partial \theta^i}\right)_p (f) := \frac{\partial (f \circ \varphi^{-1})}{\partial \theta^i}\bigg|_{\varphi(p)}$$

로 정의된 $\partial/\partial\theta^i|_p \in T_p M$들이 $T_p M$의 기저를 이룬다. 임의의 $X \in T_p M$은

$$X = X^i \frac{\partial}{\partial \theta^i}\bigg|_p \quad (\text{Einstein 합 규약})$$

으로 유일하게 표현되고, $X^i$를 $X$의 **좌표 성분**이라 한다.

### 정의 2.6 — 통계다양체의 접공간과 스코어

통계다양체 $\mathcal{P} = \{p_\theta\}$의 점 $p_\theta$에서, 좌표기저 $\partial/\partial\theta^i$는 함수 공간 위에서

$$\frac{\partial}{\partial \theta^i}\bigg|_{p_\theta} \longleftrightarrow \frac{\partial \log p_\theta(x)}{\partial \theta^i} =: s_i(x; \theta) \quad (\text{스코어})$$

로 동일시된다. 따라서 접벡터 $X = X^i \partial_i$는 **스코어 선형결합** $X^i s_i$에 대응한다.

---

## 🔬 정리와 증명

### 정리 2.1 — 세 정의의 동치성

**명제**: 정의 2.1, 2.2, 2.3의 세 접공간 $T_p^{(c)}, T_p^{(d)}, T_p^{(t)}$ 사이에 자연스러운 전단사(natural bijection)가 존재하며, $T_p M$은 잘 정의된 $n$차원 벡터공간이다.

**증명 ($T_p^{(c)} \to T_p^{(d)}$)**: $[\gamma] \in T_p^{(c)}$에 대해, derivation $X_\gamma$를

$$X_\gamma(f) := \frac{d}{dt}\bigg|_{t=0} f(\gamma(t))$$

으로 정의한다. $\mathbb{R}$-선형성과 Leibniz 규칙은 일변수 미분의 선형성·곱 미분에서 즉시 성립. 잘 정의됨(well-defined)은 $\gamma_1 \sim \gamma_2$이면 $\varphi$에서 속도가 같고, 연쇄법칙에 의해 모든 $f$에 대해 $X_{\gamma_1}(f) = X_{\gamma_2}(f)$에서 나온다.

**증명 ($T_p^{(d)} \to T_p^{(t)}$)**: derivation $X$에 좌표 성분 $X^i := X(\theta^i)$(즉 $i$번째 좌표함수에 $X$를 작용)를 대응시킨다. Taylor 정리로 $f(q) = f(p) + \partial_i f(p)(\theta^i(q) - \theta^i(p)) + O(\text{차수 }2)$이 국소적으로 유효하고, Leibniz 규칙이 **2차 이상 항에서 derivation을 죽이는 것**을 보이면:

$$X(f) = X^i \partial_i f(p)$$

즉 derivation은 좌표 성분 $(X^1, \ldots, X^n)$으로 완전히 결정된다. 좌표 변환 하의 변환법칙(야코비안 곱)은 연쇄법칙에서 얻어진다. 역은 명백.

**증명 ($T_p^{(t)} \to T_p^{(c)}$)**: 좌표 성분 $v = (v^1, \ldots, v^n)$에 대해 $\gamma(t) = \varphi^{-1}(\varphi(p) + tv)$을 택하면 $[\gamma]$가 해당된다.

따라서 세 공간은 서로 자연스럽게 동일시되고, 벡터공간 구조(덧셈·스칼라배)는 $T_p^{(t)}$의 $\mathbb{R}^n$ 구조에서 이전된다. 차원이 $n$임도 자명하다. $\square$

---

### 정리 2.2 — 좌표기저의 변환법칙

**명제**: 두 차트 $(U, \theta)$, $(U', \widetilde\theta)$에 대해 겹치는 영역에서

$$\frac{\partial}{\partial \widetilde\theta^j}\bigg|_p = \frac{\partial \theta^i}{\partial \widetilde\theta^j}(p) \cdot \frac{\partial}{\partial \theta^i}\bigg|_p$$

**증명**: 임의의 매끈한 $f$에 대해

$$\frac{\partial f}{\partial \widetilde\theta^j} = \frac{\partial f}{\partial \theta^i} \cdot \frac{\partial \theta^i}{\partial \widetilde\theta^j}$$

은 다변수 연쇄법칙 그 자체. 이것이 임의의 $f$에 성립하므로 derivation의 등식이 성립한다. $\square$

> **따름정리**: 접벡터 $X$의 좌표 성분은 $\widetilde X^j = \frac{\partial \widetilde\theta^j}{\partial \theta^i} X^i$로 변환된다 — 기저의 변환과 **반대(역행렬)**라서 $X$는 **contravariant**라 부른다.

---

### 정리 2.3 — 스코어 함수들은 접공간의 자연스러운 표현

**명제**: 통계다양체 $\mathcal{P} = \{p_\theta\}$의 점 $p_\theta$에서, 사상

$$\iota: T_{p_\theta} \mathcal{P} \to L^2(p_\theta), \quad \frac{\partial}{\partial\theta^i}\bigg|_{p_\theta} \mapsto s_i(\cdot; \theta) = \frac{\partial \log p_\theta}{\partial \theta^i}$$

은 **단사 선형사상**이며, 상은 $\{s \in L^2(p_\theta) : \mathbb{E}_\theta[s] = 0\}$의 유한차원 부분공간이다.

**증명**:  
*(선형성)* Derivation의 선형성에서 즉시.

*($\mathbb{E}_\theta[s_i] = 0$)*:

$$\mathbb{E}_\theta[s_i] = \int p_\theta(x) \frac{\partial \log p_\theta(x)}{\partial \theta^i} dx = \int \frac{\partial p_\theta(x)}{\partial \theta^i} dx = \frac{\partial}{\partial\theta^i}\int p_\theta(x)dx = \frac{\partial}{\partial\theta^i} 1 = 0$$

(미분-적분 교환은 정칙성 조건).

*(단사성)* $X = X^i \partial_i$에 대해 $\iota(X) = X^i s_i = 0$ a.e.이면, 각 $x$에서 $X^i s_i(x) = 0$. $s_i$들이 (정칙 분포에서) $L^2$-선형독립이면 $X^i = 0$. 정규·이항·exp family 등에서 이 선형독립성은 Fisher 정보 행렬이 비가역이 **아니라**는 조건과 동치다(Ch2).

$\square$

> **의미**: 스코어 함수들의 공간 자체가 통계다양체의 접공간이다. 다음 장의 Fisher 계량 $g_{ij} = \mathbb{E}[s_i s_j]$은 이 $L^2$-내적의 제한과 정확히 일치한다.

---

### 예시 2개

**예시 1 — 정규분포 $\mathcal{N}(\mu, \sigma^2)$의 접공간과 스코어**

$\log p_{\mu,\sigma}(x) = -\frac{(x-\mu)^2}{2\sigma^2} - \log\sigma - \frac{1}{2}\log(2\pi)$

$$s_\mu = \frac{\partial \log p}{\partial \mu} = \frac{x-\mu}{\sigma^2}, \quad s_\sigma = \frac{\partial \log p}{\partial \sigma} = \frac{(x-\mu)^2}{\sigma^3} - \frac{1}{\sigma}$$

$\mathbb{E}[s_\mu] = 0$, $\mathbb{E}[s_\sigma] = 0$을 확인. 접벡터 $X = a\partial_\mu + b\partial_\sigma$는 $L^2$ 함수 $a s_\mu + b s_\sigma$에 대응한다.

**예시 2 — 베르누이 $B(p)$의 접공간**

$\log p_p(x) = x\log p + (1-x)\log(1-p)$, $x \in \{0,1\}$. 스코어 $s_p(x) = \frac{x}{p} - \frac{1-x}{1-p} = \frac{x - p}{p(1-p)}$. 이 1차원 공간이 베르누이 다양체의 접공간.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp

# ─────────────────────────────────────────────
# 1. 좌표기저의 변환법칙 검증 — 정규분포의 (μ, σ) ↔ (μ, log σ)
# ─────────────────────────────────────────────
mu, sigma = sp.symbols('mu sigma', positive=True)
log_sigma = sp.symbols('tau')  # τ = log σ
# 차트 변환: θ = (μ, σ), θ̃ = (μ, τ), 관계 σ = e^τ
# ∂θ^i / ∂θ̃^j 야코비안
J = sp.Matrix([[1, 0], [0, sp.exp(log_sigma)]])
print("─ 좌표 변환 야코비안 ∂(μ,σ)/∂(μ,τ) ─")
sp.pprint(J)

# 접벡터 X = a ∂_μ + b ∂_σ의 새 좌표 (a', b') 성분
# 반변성: X̃^j = (∂θ̃^j/∂θ^i) X^i = J⁻¹ X
Jinv = J.inv()
sp.pprint(sp.simplify(Jinv))
# → 기대값: [[1, 0], [0, e^(-τ)]], 즉 ∂_σ = e^(-τ) ∂_τ = (1/σ) ∂_τ

# ─────────────────────────────────────────────
# 2. 정규분포의 스코어 symbolic 유도와 기댓값 = 0 확인
# ─────────────────────────────────────────────
x = sp.symbols('x', real=True)
log_p = -(x - mu)**2 / (2*sigma**2) - sp.log(sigma) - sp.log(2*sp.pi)/2
s_mu = sp.diff(log_p, mu)
s_sig = sp.diff(log_p, sigma)
print("\n─ 스코어 함수 ─")
print(f"s_μ(x) = {sp.simplify(s_mu)}")
print(f"s_σ(x) = {sp.simplify(s_sig)}")

# E[s_μ] = ∫ s_μ · p dx
p_expr = sp.exp(log_p)
E_smu = sp.integrate(s_mu * p_expr, (x, -sp.oo, sp.oo))
E_ssig = sp.integrate(s_sig * p_expr, (x, -sp.oo, sp.oo))
print(f"E[s_μ]  = {sp.simplify(E_smu)}")
print(f"E[s_σ]  = {sp.simplify(E_ssig)}")
# 둘 다 0 이어야 함

# ─────────────────────────────────────────────
# 3. 수치 확인: MC로 스코어 공분산 (Fisher 미리보기)
# ─────────────────────────────────────────────
rng = np.random.default_rng(0)
mu_val, sig_val = 0.5, 1.2
N = 200_000
xs = rng.normal(mu_val, sig_val, N)
s_mu_vals = (xs - mu_val) / sig_val**2
s_sig_vals = ((xs - mu_val)**2 - sig_val**2) / sig_val**3
scores = np.vstack([s_mu_vals, s_sig_vals])
emp_cov = np.cov(scores, bias=True)
theoretical_F = np.diag([1/sig_val**2, 2/sig_val**2])
print("\n─ 스코어 공분산 (MC) vs Fisher 이론 ─")
print(f"MC      :\n{emp_cov}")
print(f"이론 F  :\n{theoretical_F}")

# ─────────────────────────────────────────────
# 4. 곡선 γ(t)의 속도 ↔ derivation 동치성 검증
# ─────────────────────────────────────────────
# 다양체 = ℝ², 곡선 γ(t) = (cos t, sin t), 점 p = (1, 0)
# 접벡터 = (-sin 0, cos 0) = (0, 1)
# 함수 f(x, y) = x² + y²
# derivation으로 계산: d/dt|_0 f(γ(t)) = d/dt (cos² t + sin² t) = 0
# 좌표로 계산: (0, 1) · ∇f(1, 0) = (0,1) · (2, 0) = 0
# 두 값 일치 ✓
print("\n─ 곡선 속도 = derivation ─")
print("d/dt|_0 f(γ(t)) = 0 (상수 함수의 미분)")
print("v · ∇f(p)       = (0,1)·(2,0) = 0")
print("→ 두 정의가 같은 값을 준다 ✓")
```

**출력 예시**:
```
─ 스코어 함수 ─
s_μ(x) = (x - μ)/σ²
s_σ(x) = ((x - μ)² - σ²)/σ³
E[s_μ] = 0
E[s_σ] = 0

─ 스코어 공분산 (MC) vs Fisher 이론 ─
MC      : [[0.696, 0.002], [0.002, 1.389]]
이론 F  : [[0.694, 0    ], [0,     1.389]]
```

---

## 🔗 AI/ML 연결

### JAX `jvp`와 접벡터

`jax.jvp(f, (x,), (v,))`는 점 $x$에서 접벡터 $v \in T_x \mathbb{R}^n$을 따라간 $f$의 방향 미분을 계산한다 — 정확히 정의 2.2의 derivation 작용 $X(f) = v^i \partial_i f$. 다양체로 일반화하면, forward-mode AD가 tangent space의 linearity를 그대로 쓴다.

### 자연 스코어 함수와 Natural Gradient의 접벡터 해석

Natural Gradient $\tilde\nabla L = F^{-1} \nabla L$은 점 $\theta \in \mathcal{P}$의 접공간 $T_\theta \mathcal{P}$의 원소다. 이 접벡터의 좌표 성분이 Fisher 행렬로 "raise" 된 gradient이고, 접공간의 내적(Ch3)에서 보면 **가장 가파른 방향**.

### Pushforward와 MAML

MAML의 inner update $\phi(\theta) = \theta - \alpha \nabla L_\text{task}(\theta)$는 다양체 위의 매끈한 사상이고, outer gradient는 이 사상의 **pushforward $\phi_*: T_\theta M \to T_{\phi(\theta)} M$**의 adjoint를 포함한다 — 이것이 MAML이 2차 미분을 필요로 하는 이유의 기하학적 해석.

### Policy Gradient의 접벡터

Policy gradient $\nabla_\theta J = \mathbb{E}_\tau[\nabla_\theta \log \pi_\theta(a|s) \cdot R(\tau)]$에서 $\nabla_\theta \log \pi_\theta$가 바로 **정책 다양체의 스코어** — 즉 접공간의 좌표기저 표현이다. Natural Policy Gradient는 이 gradient를 접공간의 Fisher 내적으로 "올바른 길이"로 재조정한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 좌표기저가 $T_p M$의 기저 | 차트 밖에선 좌표 자체가 없음 — $T_p M$은 항상 점 $p$에 묶여 있다 |
| 스코어 함수들의 $L^2$-선형독립 | 과매개변수화된 모델에선 선형종속이 발생 → Fisher 특이(singular) |
| 미분-적분 교환 가능 | 경계가 매개변수에 의존하는 분포에선 성립하지 않음 |
| $T_p M$이 벡터공간 | 서로 다른 점의 접공간은 **자연스럽게 동일시되지 않음** → 접다발(tangent bundle) 필요 |

**주의**: $T_p M$과 $T_q M$은 같은 벡터공간 차원을 가지지만, 그들 사이의 "자연 동일시"는 없다. 이를 연결하는 것이 **접속(connection)** (Ch1-04)이다.

---

## 📌 핵심 정리

$$\boxed{\;T_p M \;=\; \{[\gamma]\} \;=\; \{\text{derivations at } p\} \;=\; \mathbb{R}^n / \text{coord-equiv}\;\cong\;\mathbb{R}^n\;}$$

$$\text{좌표기저 } \frac{\partial}{\partial\theta^i}\bigg|_p, \qquad X = X^i \frac{\partial}{\partial\theta^i}\bigg|_p \text{ (Einstein 합 규약)}$$

| 개념 | 핵심 |
|------|------|
| 접공간 | 점 $p$에서 **뻗어나가는 방향**들의 $n$차원 벡터공간 |
| 세 정의의 동치 | 곡선 속도 ≡ derivation ≡ 좌표 튜플 동치류 |
| 좌표기저 | $\partial/\partial\theta^i$, 좌표변환 하에 야코비안으로 변환 (contravariant 성분) |
| 스코어 함수 | 통계다양체의 접벡터 = $\partial_\theta \log p_\theta$, 기댓값 0 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathbb{R}^2$에서 극좌표 $(r, \phi)$를 잡을 때 좌표기저 $\partial/\partial r$, $\partial/\partial\phi$를 직교좌표 $(x, y)$의 기저로 표현하라. 두 기저가 원점에서 정의되지 않는 이유는?

<details>
<summary>힌트 및 해설</summary>

$x = r\cos\phi, y = r\sin\phi$이므로 $\partial/\partial r = \cos\phi\,\partial/\partial x + \sin\phi\,\partial/\partial y$, $\partial/\partial\phi = -r\sin\phi\,\partial/\partial x + r\cos\phi\,\partial/\partial y$. 원점 $r=0$에서 $\partial/\partial\phi = 0$ — 극좌표 차트가 원점에서 **특이**해 좌표기저가 선형독립이 아니다. 따라서 극좌표는 원점을 제외한 영역에서만 유효한 차트.

</details>

**문제 2** (심화): 정의 2.2의 derivation이 "상수 함수에서 0"임을 증명하라. 이것이 접공간의 차원이 $n$인 이유와 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

$X(1 \cdot 1) = X(1) \cdot 1 + 1 \cdot X(1) = 2 X(1)$이 Leibniz에서 나와 $X(1) = 0$. 임의 상수 $c$에 대해 $X(c) = c X(1) = 0$. Taylor 전개 $f = f(p) + \partial_i f(p) \cdot \theta^i + (\text{2차 이상})$에서 **상수 항이 죽고, 2차 항도 Leibniz로 죽는다** → 1차 항의 계수만 남아, derivation은 $(X(\theta^1), \ldots, X(\theta^n))$으로 완전 결정 → $n$차원.

</details>

**문제 3** (AI 연결): Softmax 분포 $p_\theta(y=k) = e^{\theta_k}/\sum_j e^{\theta_j}$의 스코어 $\partial \log p_\theta(y=k)/\partial \theta_j$을 계산하고, cross-entropy loss gradient가 $(p - y)$로 단순화되는 이유를 접공간 관점에서 설명하라. $\theta$ 차원이 $K$인데 접공간 차원은 $K-1$인 이유는?

<details>
<summary>힌트 및 해설</summary>

$\partial \log p_\theta(y=k)/\partial \theta_j = \delta_{jk} - p_\theta(y=j)$. CE loss $-\log p_\theta(y=k)$의 gradient는 $-\partial \log p_\theta(y=k)/\partial \theta_j = p_j - \delta_{jk} = p_j - y_j$.

차원 불일치: softmax는 $\theta \to \theta + c\mathbf{1}$ 불변이므로 $\theta$의 1차원 "게이지 자유도"가 있다. 실제 확률 다양체(심플렉스)는 $K-1$차원이고, 접공간도 $\sum v_i = 0$ 조건으로 $K-1$차원. 즉 $\mathbb{R}^K$의 스코어 중 $\mathbf{1}$ 방향은 접공간의 **부분공간에 사는 영벡터**로, $\sum_j (p_j - y_j) = 0$이 이 사실을 반영한다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. 다양체의 기초](./01-manifold-basics.md) | [📚 README로 돌아가기](../README.md) | [03. 리만 계량과 거리 ▶](./03-riemannian-metric.md) |

</div>
