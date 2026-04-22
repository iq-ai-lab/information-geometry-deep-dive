# 01. 통계다양체의 정의

## 🎯 핵심 질문

- 확률분포족 $\{p_\theta\}_{\theta \in \Theta}$를 **"다양체"** 라고 부르려면 정확히 어떤 조건이 필요한가?
- 왜 "각 점 = 하나의 분포" 라는 해석이 엄밀하게 성립하는가? ($\theta \neq \theta'$이지만 $p_\theta = p_{\theta'}$인 경우 어떻게 처리?)
- 정규·이항·다항·Dirichlet·다변수정규는 각각 몇 차원 다양체이며, 어떻게 시각화되는가?
- **정칙성 조건(regularity)**이 왜 본질적인가 — 미분-적분 교환, 지지집합 불변 등

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **학습의 기하학적 해석**: 신경망 학습은 매개변수 공간 $\Theta \subseteq \mathbb{R}^d$에서의 최적화이지만, **본질적으로는 $\theta$가 유도하는 분포 $p_\theta(y|x)$의 공간에서 움직인다**. 분포 공간 위의 기하(Fisher-Rao)가 더 자연스러운 거리/각도를 준다.
- **과매개변수화와 quotient manifold**: 현대 신경망은 $\theta \neq \theta'$이어도 $p_\theta = p_{\theta'}$인 경우가 많다(순열 대칭, scaling 대칭). 이 경우 "실제 분포 다양체"는 매개변수 공간의 **quotient** $\Theta/\sim$이고, 차원이 줄어든다.
- **정칙성 가정의 실전 의미**: ReLU·균등분포·경계 의존 분포는 정칙성이 깨진다. 이럴 때 Fisher 정보가 특이하거나 발산하고, NGD·TRPO 같은 알고리즘이 실패한다.
- **Softmax는 심플렉스 위 다양체**: 분류기의 출력 공간은 $(K-1)$-심플렉스. $K$차원 logit 공간은 과매개변수화(shift invariance)로 인한 1차원 여유가 있다.

"분포 공간 위의 최적화"라는 말을 수학적으로 정당화하려면, 그 공간이 **매끈한 다양체**임을 먼저 보여야 한다.

---

## 📐 수학적 선행 조건

- Ch1-01~04: 다양체, 차트, 접공간, 리만 계량
- **[Probability Theory Deep Dive]** 측도, 확률측도, 확률밀도, 지지집합 — **필수**
- **[Mathematical Statistics Deep Dive]** 매개변수 모형, 스코어, 정칙 조건 — **필수**
- 해석학: Lebesgue 측도, 미분-적분 교환 정리 (Leibniz integral rule)

---

## 📖 직관적 이해

### "각 점 = 하나의 분포"

유클리드 공간 $\mathbb{R}^n$의 점은 **좌표 튜플** $(x_1, \ldots, x_n)$. 통계다양체의 점은 **하나의 확률분포** $p_\theta$. 겉보기에는 같은 $\theta = (\mu, \sigma)$로 기술되지만, **그 의미가 다르다**:

- $\mathbb{R}^2$의 점 $(0, 1)$: "수평 0, 수직 1"
- $\{\mathcal{N}(\mu, \sigma^2)\}$의 점 $(0, 1)$: "표준정규분포 전체 함수" $p(x) = e^{-x^2/2}/\sqrt{2\pi}$

두 공간은 같은 매개변수화를 공유하지만, **"거리"가 완전히 다르다**. Fisher-Rao 거리는 이 사실을 수학적으로 포착한다.

### 차원이 줄어드는 "겉보기 매개변수"

정규분포족 $\{\mathcal{N}(\mu, \sigma^2)\}$는 2차원. 하지만 **$\mathcal{N}(\mu, \sigma^2, \mu')$**(더미 매개변수 $\mu'$ 추가)처럼 쓸데없는 매개변수를 넣으면 $\Theta$는 3차원이지만 **실제 분포 다양체는 여전히 2차원**이다. 이 경우 $\theta \mapsto p_\theta$가 단사가 아니고, 진짜 다양체는 매개변수 공간의 **상(image)**인 2차원 부분공간.

> **비유**: 지도에 한 도시가 "서울, 대한민국, 아시아"로 라벨링 되어 있어도 도시 자체는 한 점이다. 매개변수화가 풍부해도 분포 공간의 차원은 늘어나지 않는다.

### 정칙성 — "좋은 다양체"의 조건

정칙성은 "Fisher 계량이 잘 정의되기 위한 기술적 조건" 이다:

1. **지지집합(support)이 $\theta$에 무관** — $\{x : p_\theta(x) > 0\}$가 $\theta$와 무관
2. **미분-적분 교환 가능** — $\partial_\theta \int = \int \partial_\theta$
3. **Fisher 정보가 유한·양정치**

균등분포 $U(0, \theta)$는 **지지집합이 $\theta$에 의존** → Fisher 발산 → 정칙 모형 아님.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 통계 모형 (Statistical Model)

측도공간 $(\mathcal{X}, \mathcal{B}, \mu)$ 위에서 지표 집합 $\Theta$가 주어졌을 때,

$$\mathcal{P} = \{p_\theta(x) : \theta \in \Theta\}$$

가 $\mu$에 대한 확률밀도족이면 이를 **통계 모형**이라 한다. 즉 각 $\theta$에 대해

$$p_\theta \geq 0, \quad \int_\mathcal{X} p_\theta(x) \, d\mu(x) = 1$$

($\mu$는 흔히 Lebesgue 또는 counting 측도.)

### 정의 1.2 — 매개변수 다양체로서의 통계다양체

다음 조건을 만족하면 $\mathcal{P}$를 **$n$차원 통계다양체**라 한다:

1. $\Theta \subseteq \mathbb{R}^n$이 **열린집합(open)**
2. 사상 $\theta \mapsto p_\theta$가 **단사(injective)** (= 서로 다른 $\theta$가 서로 다른 분포)
3. 임의의 $x$에 대해 $\theta \mapsto p_\theta(x)$가 $C^\infty$, 또는 최소 $C^2$ 
4. (정칙성) 정의 1.3의 조건들 만족

이 경우 $\Theta$가 $\mathcal{P}$의 **전역 차트** 역할을 하고, 단일 아틀라스 $\{(\mathcal{P}, \varphi^{-1})\}$로 매끈한 구조를 이룬다 (Ch1-01 정리 1.3).

### 정의 1.3 — 정칙 조건 (Regularity Conditions)

통계다양체가 **정칙(regular)**이라는 것은:

**(R1) 지지집합의 $\theta$-독립성**: $\{x : p_\theta(x) > 0\}$이 모든 $\theta$에 대해 같음.

**(R2) 미분-적분 교환**: 임의의 $\theta$ 주변 근방 $V$에서

$$\frac{\partial}{\partial \theta^i} \int_\mathcal{X} f(x) p_\theta(x) \, d\mu(x) = \int_\mathcal{X} f(x) \frac{\partial p_\theta(x)}{\partial \theta^i} \, d\mu(x)$$

가 $f \in \{1, \partial_j \log p_\theta, \partial_i \partial_j \log p_\theta\}$에 대해 성립.

**(R3) 스코어 함수의 $L^2$-존재**: $s_i(x;\theta) = \partial_i \log p_\theta(x)$가 $L^2(p_\theta)$ 함수, 즉

$$\mathbb{E}_\theta[s_i^2] < \infty$$

**(R4) Fisher 정보의 양정치**: $F_{ij}(\theta) = \mathbb{E}_\theta[s_i s_j]$가 $\Theta$의 각 점에서 양정치.

### 정의 1.4 — 부분 통계다양체와 Quotient 다양체

- **부분다양체**: $\Theta_0 \subseteq \Theta$가 $k$-차원 매끈한 부분다양체이면 $\mathcal{P}_0 = \{p_\theta : \theta \in \Theta_0\}$도 $k$-차원 부분 통계다양체.
- **Quotient**: $\theta \sim \theta' \iff p_\theta = p_{\theta'}$ 동치관계 하의 몫공간 $\Theta/\sim$이 진짜 분포 다양체. 과매개변수화된 모델에서 "실제 표현력의 차원"을 준다.

### 정의 1.5 — 정규 exponential family 다양체

$p_\theta(x) = \exp(\theta^T T(x) - \psi(\theta)) h(x)$ 꼴로 sufficient statistic $T$가 **선형 독립**이고 $\Theta = \{\theta : \psi(\theta) < \infty\}$의 내부(자연 매개변수 공간)가 열린집합이면 자동으로 정규(regular) 통계다양체. 자세한 기하 구조는 Ch4.

---

## 🔬 정리와 증명

### 정리 2.1 — 단사성과 quotient manifold 차원

**명제**: $\theta \mapsto p_\theta$가 **단사가 아닌** 경우(즉 $\ker := \{(\theta, \theta') : p_\theta = p_{\theta'}\}$이 $\Theta \times \Theta$의 비자명 부분집합), $\mathcal{P}$의 매개변수화로서의 "실제 차원"은 $\dim \Theta - \dim(\text{ker에서 유도된 표면})$.

**증명 스케치**: 동치관계 $\sim$의 몫공간 $\Theta/\sim$이 매끈한 quotient manifold가 되는 충분조건(슬라이스 정리, Slice Theorem)은 대수적 대칭군이 자유·적절하게 작용하는 것. 예시로 softmax $p(y|x;\theta)$에서 $\theta \to \theta + c\mathbf{1}$ 대칭의 군 $\mathbb{R}$이 자유로운 작용을 해 $\Theta/\sim \cong \mathbb{R}^{K-1}$. $\square$

> **딥러닝 함의**: ReLU 네트워크의 순열 대칭(뉴런 재배열), 스케일 대칭 등을 고려하면 실질적 분포 다양체의 차원은 파라미터 수보다 훨씬 작다. 이것이 NTK·Mode Connectivity의 배경 중 하나.

---

### 정리 2.2 — 정칙성 하의 스코어 기댓값 = 0

**명제**: 정칙 통계다양체에서 임의의 $\theta$에 대해

$$\mathbb{E}_\theta[s_i(X; \theta)] = 0$$

**증명**:

$$\mathbb{E}_\theta[s_i] = \int \frac{\partial \log p_\theta(x)}{\partial \theta^i} p_\theta(x) \, d\mu(x) = \int \frac{\partial p_\theta(x)}{\partial \theta^i} \, d\mu(x) \overset{(R2)}{=} \frac{\partial}{\partial \theta^i} \int p_\theta \, d\mu = \frac{\partial}{\partial \theta^i} 1 = 0 \quad \square$$

> 이 등식은 Ch2-02의 Fisher 3정의 동치성, Ch3의 KL-Fisher 연결의 **가장 기본적인 도구**. (R1) 지지집합 $\theta$-독립이 깨지면 미분-적분 교환이 실패해 이 등식도 깨진다.

---

### 정리 2.3 — 균등분포 $U(0, \theta)$는 정칙 모형이 아님

**명제**: $p_\theta(x) = \frac{1}{\theta}\mathbf{1}_{(0, \theta)}(x)$ ($\theta > 0$)의 지지집합은 $(0, \theta)$로 $\theta$에 의존 → (R1) 위배 → 정칙 아님.

**검증**: 스코어를 형식적으로 계산하면 $s(x;\theta) = -1/\theta$이고 $\mathbb{E}_\theta[s] = -1/\theta \neq 0$. 즉 정리 2.2가 실패한다. Fisher 정보도 $F(\theta) = 1/\theta^2$로 **표준 Cramér-Rao 하한 $\text{Var}(\hat\theta) \geq \theta^2$에 도달하지 못한다** — 실제 MVUE는 $O(1/n^2)$로 더 빠르게 수렴. 이는 정칙성 가정이 깨진 결과.

> **교훈**: "왜 Fisher-Rao가 만능이 아닌가"의 대표 사례. 지지집합이 매개변수 의존이면 다른 이론(e.g. location-scale estimation)이 필요.

---

### 예시 5개 — 구체 통계다양체

**예시 1 — 정규분포** $\{\mathcal{N}(\mu, \sigma^2)\}$
- $\Theta = \mathbb{R} \times \mathbb{R}_{>0}$, 2차원, 전역 차트
- 정칙: 지지집합 $= \mathbb{R}$ 고정, $C^\infty$, Fisher 양정치
- Ch1-03에서 본 대로 Fisher-Rao는 쌍곡계량

**예시 2 — 이항분포** $\{B(n, p) : 0 < p < 1\}$
- $\Theta = (0, 1)$, 1차원
- 정칙 (지지집합 $\{0, 1, \ldots, n\}$ 고정)
- Fisher $F(p) = n/(p(1-p))$

**예시 3 — 다항분포** $\{\text{Multi}(n, p) : p \in \Delta^{K-1, \circ}\}$
- $\Delta^{K-1, \circ}$은 심플렉스 내부(경계 $p_i = 0$ 제외)
- $(K-1)$차원 통계다양체
- 부분 차트: $(p_1, \ldots, p_{K-1})$, 마지막 좌표는 $p_K = 1 - \sum_{i<K} p_i$

**예시 4 — 다변수정규** $\{\mathcal{N}(\mu, \Sigma) : \mu \in \mathbb{R}^d, \Sigma \in \text{Sym}^+_d\}$
- $d + d(d+1)/2$차원
- $\Sigma$의 Cholesky 분해 $\Sigma = L L^T$로 국소 차트 구성 가능
- Fisher는 블록 대각 (Ch2-04 참조)

**예시 5 — Dirichlet** $\{\text{Dir}(\alpha) : \alpha_i > 0\}$
- $K$-차원 (심플렉스 위의 분포족 → $K$차원 매개변수)
- 정칙, 지지집합은 $\Delta^{K-1}$의 내부 고정
- Fisher는 trigamma 함수로 표현됨

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 정규분포 다양체를 2D 곡면으로 시각화
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))

# (좌) 매개변수 공간 Θ = ℝ × ℝ₊
ax1 = fig.add_subplot(131)
mu_grid = np.linspace(-2, 2, 15)
sig_grid = np.linspace(0.2, 3, 15)
MU, SIG = np.meshgrid(mu_grid, sig_grid)
ax1.scatter(MU, SIG, c='b', s=15, alpha=0.6)
ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'$\sigma$')
ax1.set_title(r'정규다양체 $\{\mathcal{N}(\mu,\sigma^2)\}$: 전역 차트 (2D)')
ax1.grid(True, alpha=0.3)

# (중) 각 점이 하나의 분포
ax2 = fig.add_subplot(132)
x_grid = np.linspace(-7, 7, 400)
for mu_, sig_, color in [(0, 1, 'b'), (-1.5, 0.5, 'g'), (1.5, 2, 'r'), (0, 0.3, 'm')]:
    pdf = np.exp(-0.5*((x_grid - mu_)/sig_)**2) / (sig_*np.sqrt(2*np.pi))
    ax2.plot(x_grid, pdf, color=color, label=rf'$({mu_},{sig_})$')
ax2.set_xlabel('x')
ax2.set_ylabel('density')
ax2.set_title('통계다양체의 각 점 = 하나의 분포')
ax2.legend()
ax2.grid(True, alpha=0.3)

# (우) 심플렉스 Δ² 삼각형 = Multi(n, p=(p1,p2,p3))의 2차원 통계다양체
ax3 = fig.add_subplot(133)
# 심플렉스 꼭짓점
corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
tri = plt.Polygon(corners, fill=False, edgecolor='k')
ax3.add_patch(tri)
# 내부 점들
rng = np.random.default_rng(0)
pts = rng.dirichlet([1, 1, 1], 120)
xy = pts @ corners
ax3.scatter(xy[:, 0], xy[:, 1], c='b', s=10, alpha=0.6)
for c, lbl in zip(corners, ['(1,0,0)', '(0,1,0)', '(0,0,1)']):
    ax3.annotate(lbl, c, xytext=(c[0]+0.02, c[1]+0.02), fontsize=9)
ax3.set_xlim(-0.1, 1.1); ax3.set_ylim(-0.1, 1.0)
ax3.set_aspect('equal')
ax3.set_title(r'$\Delta^2$: Multi$(n, (p_1,p_2,p_3))$의 2차원 다양체')
ax3.axis('off')

plt.tight_layout()
plt.savefig('01-statistical-manifold.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. 정칙성 (R2) 검증: 미분-적분 교환
# ─────────────────────────────────────────────
print("─ 정칙성 검증: ∫ ∂p dx = ∂ ∫ p dx = ∂(1) = 0 ─")
mu, sigma, x = sp.symbols('mu sigma x', real=True, positive=True)
p = sp.exp(-(x - mu)**2/(2*sigma**2)) / (sigma*sp.sqrt(2*sp.pi))
dp_dmu = sp.diff(p, mu)
integral = sp.integrate(dp_dmu, (x, -sp.oo, sp.oo))
print(f"  ∫ ∂p/∂μ dx = {sp.simplify(integral)}  (기대: 0)")

dp_dsigma = sp.diff(p, sigma)
integral2 = sp.integrate(dp_dsigma, (x, -sp.oo, sp.oo))
print(f"  ∫ ∂p/∂σ dx = {sp.simplify(integral2)}  (기대: 0)")

# ─────────────────────────────────────────────
# 3. 균등분포 U(0,θ)의 비정칙성 확인
# ─────────────────────────────────────────────
print("\n─ 균등분포 U(0,θ): 정리 2.2 실패 ─")
theta, y = sp.symbols('theta y', positive=True)
# 스코어: d/dθ log(1/θ)  = -1/θ  (0 < y < θ)
score_uniform = sp.diff(sp.log(1/theta), theta)
print(f"  스코어 s(y;θ) = {score_uniform}  ← 0에 대해 상수")
# E[s] = ∫₀^θ (-1/θ) · (1/θ) dy = -1/θ
E_s = sp.integrate(score_uniform * (1/theta), (y, 0, theta))
print(f"  E_θ[s] = {sp.simplify(E_s)}  (기대는 0, 실제: -1/θ)")
print("  → 정리 2.2 실패: 미분-적분 교환 (R2)가 지지집합 의존으로 깨짐")

# ─────────────────────────────────────────────
# 4. 단사성 실패 예 — softmax 과매개변수화
# ─────────────────────────────────────────────
print("\n─ Softmax의 1차원 shift 대칭성 ─")
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

theta1 = np.array([1.0, 2.0, 3.0])
theta2 = theta1 + 5.0  # 모든 성분에 5 더하기
print(f"  softmax(θ)     = {softmax(theta1)}")
print(f"  softmax(θ+5·1) = {softmax(theta2)}")
print(f"  두 분포 동일 → θ→θ+c1은 동치, quotient 차원 = K-1 = 2")
```

**출력 예시**:
```
─ 정칙성 검증 ─
  ∫ ∂p/∂μ dx = 0
  ∫ ∂p/∂σ dx = 0

─ 균등분포 U(0,θ): 정리 2.2 실패 ─
  스코어 s(y;θ) = -1/θ
  E_θ[s] = -1/θ
  → 정리 2.2 실패: (R2)가 지지집합 의존으로 깨짐

─ Softmax의 shift 대칭성 ─
  softmax(θ)     = [0.0900 0.2447 0.6652]
  softmax(θ+5·1) = [0.0900 0.2447 0.6652]
  → quotient 차원 = K-1 = 2
```

---

## 🔗 AI/ML 연결

### 과매개변수화된 신경망의 "실질 차원"

$\theta \in \mathbb{R}^{10^9}$의 GPT급 모델에서 순열·스케일 대칭을 고려한 **실질 분포 다양체의 차원**은 훨씬 작다. 이것이 (i) pruning이 성능 유지하며 가능한 이유, (ii) Mode Connectivity(서로 다른 극소값을 잇는 저손실 경로 존재)의 기하학적 배경, (iii) NTK 분석에서 학습 궤적이 저차원 부분공간에 집중되는 이유.

### Quotient 구조와 배치 정규화

BatchNorm은 레이어마다 scale/shift 매개변수를 재정규화한다 — 이는 **분포 다양체 위의 동일한 점을 여러 $\theta$로 표현하는 과매개변수화의 명시적 관리**. Fisher-Rao 관점에서 BN은 "표면적으로 달라보이는 $\theta$를 같은 분포로 인식"하도록 최적화를 바꾼다.

### Softmax 출력 다양체와 cross-entropy

분류기 출력 $p \in \Delta^{K-1}$은 $(K-1)$차원 심플렉스 다양체. Cross-entropy $H(y, p) = -\sum y_i \log p_i$의 gradient가 $(p - y)$로 단순해지는 이유는 softmax가 exponential family의 canonical parameter → expectation parameter 매핑이기 때문 (Ch4).

### 비정칙 모형과 생성 모델

GAN 생성기의 지지집합은 latent space의 저차원 manifold로 압축되어 데이터 공간의 일부에만 존재 → **"지지집합이 $\theta$에 의존"** 하는 전형적 비정칙 상황. 이것이 KL 발산이 무한대가 되는 원인이고, Wasserstein 거리 (지지집합 무관)로 해결하는 WGAN의 배경.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\Theta$ 열린 | 심플렉스 경계, $\sigma = 0$ 등에서 Fisher 발산 → 차트 축소 필요 |
| 단사 $\theta \mapsto p_\theta$ | 신경망처럼 과매개변수화된 경우 quotient로 처리 |
| (R1) 지지집합 $\theta$-무관 | 균등분포, truncated 분포, GAN에서 실패 |
| (R2) 미분-적분 교환 | dominated convergence의 가정 확인 필요 |
| (R4) Fisher 양정치 | 과매개변수화에선 특이 (singular) |

**정규 exp family의 "편안함"**: 위 모든 조건이 자동 만족 → Ch4에서 exp family 중심으로 이론이 전개되는 이유.

---

## 📌 핵심 정리

$$\boxed{\;\mathcal{P} = \{p_\theta : \theta \in \Theta\} \text{ 통계다양체} \iff \Theta \subseteq \mathbb{R}^n \text{ 열린} + \theta \mapsto p_\theta \text{ 단사} + C^\infty + \text{정칙성 (R1)-(R4)}\;}$$

| 개념 | 핵심 |
|------|------|
| 통계다양체 | 매개변수화된 분포족이 이루는 매끈한 다양체 |
| 전역 차트 | $\Theta \leftrightarrow \mathcal{P}$의 단일 동일시 |
| 정칙 조건 | 지지집합 $\theta$-무관, 미분-적분 교환, $L^2$ 스코어, 양정치 Fisher |
| Quotient | 과매개변수화 시 $\Theta/\sim$이 진짜 분포 다양체 |
| 비정칙 예 | $U(0, \theta)$, GAN의 저차원 지지집합 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 지수분포 $\{\text{Exp}(\lambda) : \lambda > 0\}$는 1차원 통계다양체다. 정칙성 (R1)-(R3)을 직접 확인하고, 스코어 $s(x;\lambda)$와 Fisher 정보를 손으로 계산하라.

<details>
<summary>힌트 및 해설</summary>

$p_\lambda(x) = \lambda e^{-\lambda x}, x > 0$. 지지집합 $(0, \infty)$이 $\lambda$에 무관 (R1) ✓. $\log p = \log\lambda - \lambda x$, $s = 1/\lambda - x$. $\mathbb{E}[s] = 1/\lambda - 1/\lambda = 0$ (R2 ✓). $F = \mathbb{E}[s^2] = \mathbb{E}[(1/\lambda - X)^2] = \text{Var}(X) = 1/\lambda^2$. 1차원 정규 정칙 모형.

</details>

**문제 2** (심화): 혼합분포 $p_\theta(x) = \theta p_1(x) + (1-\theta) p_2(x)$ ($p_1, p_2$ 고정)는 $\theta \in (0,1)$에서 1차원 통계다양체다. 이 모형이 **exponential family가 아닌** 이유를 설명하고, Fisher 정보가 $\theta \to 0$ 또는 $\theta \to 1$에서 발산할 조건을 논하라.

<details>
<summary>힌트 및 해설</summary>

혼합의 로그 $\log(\theta p_1 + (1-\theta)p_2)$은 $\theta$에 대해 **선형이 아니라 로그-선형 결합**이라 canonical form $\theta T(x) - \psi(\theta)$으로 안 떨어짐. 스코어 $s = (p_1 - p_2)/(\theta p_1 + (1-\theta)p_2)$. $\theta \to 0$일 때 $x \sim p_2$에서 분모 $\to p_2(x)$, $s \to p_1(x)/p_2(x) - 1$ → **$p_1$과 $p_2$가 거의 어긋난 지지집합을 가지면 $L^2$ 발산** → Fisher 발산. 일반적으로 혼합 모형은 mixture 연결 $\nabla^{(m)}$ 하에서 평탄하지만 e-연결 하에선 곡률 존재.

</details>

**문제 3** (AI 연결): 2층 ReLU MLP $f_\theta(x) = W_2 \sigma(W_1 x + b_1) + b_2$에서 뉴런 순열·스케일 대칭이 유도하는 동치관계를 구체적으로 기술하고, quotient manifold의 차원이 원본보다 얼마나 작은지 추정하라. 이것이 "실제 표현력의 차원"과 관련된다는 의미는?

<details>
<summary>힌트 및 해설</summary>

$W_1 \to P W_1, W_2 \to W_2 P^{-1}$ ($P$는 순열 행렬) 대칭 → $k!$개 원소가 같은 동치류 (뉴런 수 $k$). ReLU의 양수 homogeneity $\sigma(\alpha z) = \alpha \sigma(z)$ ($\alpha > 0$)로 $(W_1, W_2) \to (\alpha W_1, W_2/\alpha)$ 대칭 → 연속 대칭군 $(\mathbb{R}_+)^k$. Quotient 차원은 원본보다 $\log(k!) + k$ 정도 작다. 모델이 "실제로 구분해내는 분포"의 자유도는 파라미터 수보다 훨씬 작고, 이것이 pruning·distillation·lottery ticket 현상의 배경. Fisher 행렬 특이화도 같은 원인.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch1-04. 아핀 연결](../ch1-manifold-riemannian/04-connection-christoffel.md) | [📚 README로 돌아가기](../README.md) | [02. Fisher 3가지 정의의 동치 ▶](./02-fisher-3-equivalence.md) |

</div>
