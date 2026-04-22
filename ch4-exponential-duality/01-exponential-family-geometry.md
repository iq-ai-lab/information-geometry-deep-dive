# 01. Exponential Family의 기하학적 정의 — canonical parameter와 sufficient statistic

<div align="center">

> *"정규·감마·베르누이·다항·디리클레·푸아송·베타…  
> 통계학에서 중요한 거의 모든 분포가 지수족(exponential family)에 속한다.  
> 그런데 이것은 우연인가?*
>
> *답: 아니다. 지수족은 '정해진 모멘트 제약 하에서 최대 엔트로피 분포'로 자연스럽게 튀어나온다.  
> 그리고 기하학적으로는, **지수족이 분포 공간에서 아핀(affine) 부분다양체**이다.  
> 이것이 Amari의 정보기하 전체를 지탱하는 구조적 사실이다."*

</div>

---

## 🎯 핵심 질문

1. 지수족 $p(x \mid \theta) = \exp(\theta^T T(x) - \psi(\theta)) h(x)$의 엄밀한 정의는 무엇이고, 왜 "$\theta^T T(x)$"라는 **선형 결합**이 본질적인가?
2. **canonical parameter** $\theta$와 **sufficient statistic** $T(x)$는 기하학적으로 무엇을 포착하는가?
3. 왜 **유한 차원** $d$개의 통계량 $T_1, \dots, T_d$가 무한차원 함수 공간 안에서 $d$차원 다양체를 잘라내는가?
4. 정규·감마·베르누이·다항·디리클레·푸아송·지수 분포를 **어떻게 지수족 형식으로 재작성**하는가? 각 분포의 $(T, \theta, \psi, h)$를 손으로 쓸 수 있는가?
5. **curved exponential family**(예: $\mathcal{N}(\theta, \theta^2)$)와 **full exponential family**의 차이는?

---

## 🔍 왜 이 기하학이 AI에서 중요한가

| AI/ML 기법 | 지수족 구조가 하는 일 |
|-----------|------------------|
| **MLE 수렴성** | 지수족에서 음-로그우도는 $\theta$에 대해 볼록 → MLE가 유일 존재, 전역 최적점 |
| **Variational Inference** | Mean-field $q(\theta) = \prod_i q_i$에서 각 $q_i$를 지수족으로 두면 업데이트가 닫힌 형식 |
| **Exponentiated Family Neural Nets** | 분류기의 softmax, 로지스틱 시그모이드는 베르누이·다항 지수족의 **canonical 파라미터화** |
| **Natural Gradient** | $F(\theta) = \nabla^2 \psi(\theta)$ (Ch4-02에서 증명) — Fisher가 cumulant function의 헤시안 |
| **Energy-Based Models** | $p(x) \propto \exp(-E_\theta(x))$는 energy function을 sufficient statistic으로 보는 지수족 |
| **Diffusion Model** | $p_t(x) \propto \exp(-\frac{1}{2\sigma_t^2}\|x - \mu_t\|^2)$의 매 $t$마다 가우스 지수족 |

**직관:** 지수족은 "제약된 최대 엔트로피 분포"(MaxEnt)이다 (Ch6-04). 즉 sufficient statistic의 기댓값 $\mathbb{E}[T(X)] = \mu$만 알 때, 그 외에 대해 "가장 덜 전제하는" 분포이다 — 이것이 지수족이 물리학·정보이론·ML에서 보편적으로 등장하는 이유이다.

---

## 📐 수학적 선행 조건

| 개념 | 참조 |
|------|------|
| 측도·확률밀도·가측 함수 | Probability Theory Deep Dive Ch2 |
| **Radon-Nikodym 도함수**로서의 밀도함수 | Probability Theory Deep Dive Ch5 |
| **충분통계량(sufficient statistic)**의 Fisher-Neyman 인수분해 정리 | Mathematical Statistics Deep Dive Ch3 |
| 볼록함수·엄격볼록성 | Calculus & Optimization Deep Dive Ch4 |
| Multinomial, Dirichlet, Normal의 밀도 공식 | Probability Theory Deep Dive Ch3 |
| 통계다양체 $\mathcal{M} = \{p_\theta\}$ | **Ch2-01** |

---

## 📖 직관적 이해

### 1. 지수족은 "로그 밀도가 파라미터의 선형 함수"인 분포

로그우도를 써보자.

$$
\log p(x \mid \theta) = \theta^T T(x) - \psi(\theta) + \log h(x)
$$

여기서 **$x$를 고정하고 $\theta$의 함수**로 보면 — 이 식은 $\theta$의 아핀 함수 $\theta^T T(x)$에서 볼록함수 $\psi(\theta)$를 뺀 형태이다. 즉 로그우도가 $\theta$에 대해 **오목(concave)**이다.

$$
\underbrace{\log p(x \mid \theta)}_{\theta\text{의 오목함수}} = \underbrace{\theta^T T(x)}_{\theta\text{의 아핀항}} - \underbrace{\psi(\theta)}_{\theta\text{의 볼록항}} + \underbrace{\log h(x)}_{\theta\text{와 무관}}
$$

**이것이 MLE의 유일성과 볼록최적화로의 환원을 낳는 핵심이다.**

### 2. "로그 확률 공간"에서 지수족은 아핀

이것은 정보기하에서 가장 중요한 직관이다.

$$
\log p(x \mid \theta) - \log h(x) = \theta^T T(x) - \psi(\theta)
$$

**로그밀도가 $x$에 대해 $T_1(x), \dots, T_d(x)$의 선형 결합**이다. 이것은 분포 공간 $\mathcal{M}$이 "로그좌표에서 평평한" 부분다양체임을 뜻한다.

$$
\mathcal{E} = \{p_\theta(x) : \log p_\theta = \theta^T T + \text{base}, \theta \in \Theta\} \subset \mathcal{P}(\mathcal{X})
$$

여기서 $\mathcal{P}(\mathcal{X})$는 무한차원 확률분포 공간. 지수족 $\mathcal{E}$는 그 안에서 **$d$차원 아핀 부분공간**과 유사한 역할을 한다. **e-flat**("exponentially flat")이라는 표현이 나오는 이유이다 (Ch4-05).

### 3. 베르누이로 맛보기

$$
p(x \mid p) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

이대로는 지수족 형식이 아니다. 로그를 취하면:

$$
\log p(x \mid p) = x \log p + (1-x) \log(1-p) = x \log \frac{p}{1-p} + \log(1-p)
$$

**logit**이라고 알려진 $\theta = \log\frac{p}{1-p}$를 **canonical parameter**로 잡으면:

$$
\log p(x \mid \theta) = \theta \cdot x - \log(1 + e^\theta)
$$

즉 $T(x) = x$, $\psi(\theta) = \log(1+e^\theta)$, $h(x) = 1$. logistic 회귀의 sigmoid $\sigma(\theta) = e^\theta/(1+e^\theta)$가 곧 기댓값 $\mathbb{E}[X] = p$이다. 이것은 "expectation parameter" $\eta = \nabla\psi(\theta)$의 예고이다 (Ch4-02, Ch4-03).

### 4. "왜 유한 차원?"에 대한 직관

모든 확률분포의 공간 $\mathcal{P}(\mathcal{X})$는 무한차원이다. 그런데 충분통계량 $T: \mathcal{X} \to \mathbb{R}^d$가 정해지면, $\mathbb{E}_\theta[T(X)] = \mu$라는 $d$개 제약을 만족하는 분포들 중에서 **최대 엔트로피**를 고르면 지수족이 튀어나온다 (Ch6-04).

즉 지수족 다양체 $\mathcal{E}_T$는 "관찰 가능한 $T$-통계량만으로 분포를 결정"하는 **$d$차원 단면(slice)**이다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 (자연 지수족, Natural Exponential Family)

표본공간 $(\mathcal{X}, \mathcal{B}(\mathcal{X}))$와 기준측도(reference measure) $\nu$ ($\nu = $ 르베그 or 셈 측도), 가측 함수 $T: \mathcal{X} \to \mathbb{R}^d$와 $h: \mathcal{X} \to [0, \infty)$가 주어졌을 때, **자연 지수족**은

$$
\boxed{\;p(x \mid \theta) = \exp(\theta^T T(x) - \psi(\theta)) h(x), \quad \theta \in \Theta \subseteq \mathbb{R}^d\;}
$$

로 정의되는 분포족 $\{P_\theta\}_{\theta \in \Theta}$이다. 단,

$$
\psi(\theta) := \log \int_\mathcal{X} \exp(\theta^T T(x)) h(x) \, d\nu(x) < \infty
$$

이며 **자연 파라미터 공간**(natural parameter space)은

$$
\Theta := \{\theta \in \mathbb{R}^d : \psi(\theta) < \infty\}
$$

이다.

- $T(x)$: **sufficient statistic**(충분통계량)
- $\theta$: **canonical / natural parameter**(정준 / 자연 파라미터)
- $\psi(\theta)$: **cumulant function / log-partition function**(누율함수)
- $h(x)$: **base measure / carrying density**(기저 측도)

### 정의 1.2 (최소 / minimal 지수족)

$T_1, \dots, T_d$가 **아핀 독립**(affinely independent)이면, 즉 어떤 $(a_0, a_1, \dots, a_d)$에 대해

$$
a_0 + a_1 T_1(x) + \dots + a_d T_d(x) = 0 \quad (\nu\text{-a.e.}) \implies a_i = 0 \; \forall i
$$

를 만족하면 지수족을 **minimal**이라 한다. minimal이지 않으면 차원을 줄일 수 있다.

### 정의 1.3 (regular 지수족)

$\Theta$가 **열려 있고**(open) minimal이면 **regular**(정규) 지수족이라 한다. 이 글은 특별한 언급이 없으면 regular를 전제한다.

### 정의 1.4 (curved exponential family)

full 지수족 $\mathcal{E}$ 안에서 $k < d$ 차원 매끈한 부분다양체 $\mathcal{N} \subset \mathcal{E}$는 **curved exponential family**이다. 예: $\mathcal{N}(\mu, \mu^2) = \{(\theta_1, \theta_2) : \theta_1 = \frac{-1}{2\mu}, \theta_2 = \frac{1}{2\mu^2}\}$는 full 2차원 정규족 안의 1차원 곡선.

---

## 🔬 정리와 증명

### 정리 1.5 (cumulant function의 잘 정의됨)

$\Theta = \{\theta : \psi(\theta) < \infty\}$는 **볼록 집합**(convex set)이고, $\psi: \Theta \to \mathbb{R}$는 **볼록 함수**(convex)이다.

**증명.** $\theta_1, \theta_2 \in \Theta$와 $\lambda \in [0, 1]$에 대해 Hölder 부등식을 $p = 1/\lambda$, $q = 1/(1-\lambda)$로 쓰면:

$$
\begin{aligned}
\int e^{(\lambda \theta_1 + (1-\lambda)\theta_2)^T T(x)} h(x) d\nu(x) &= \int \left(e^{\theta_1^T T(x)}\right)^\lambda \left(e^{\theta_2^T T(x)}\right)^{1-\lambda} h(x) d\nu(x) \\
&\le \left(\int e^{\theta_1^T T} h \, d\nu\right)^\lambda \left(\int e^{\theta_2^T T} h \, d\nu\right)^{1-\lambda} < \infty
\end{aligned}
$$

따라서 $\lambda \theta_1 + (1-\lambda)\theta_2 \in \Theta$ (볼록성) 그리고 로그를 취하면

$$
\psi(\lambda \theta_1 + (1-\lambda)\theta_2) \le \lambda \psi(\theta_1) + (1-\lambda)\psi(\theta_2)
$$

즉 $\psi$는 볼록. $\quad\blacksquare$

> **참고.** $\psi$가 **엄격** 볼록임은 $T$가 minimal일 때 성립한다 (Ch4-02에서 Hölder 등호 조건으로 증명).

### 정리 1.6 (Fisher-Neyman 인수분해 정리)

$T$가 $\{P_\theta\}$에 대해 **충분통계량(sufficient)**임과 밀도가 $p(x \mid \theta) = g(T(x), \theta) h(x)$ 꼴로 인수분해됨이 동치이다.

**따름 따라서** 자연 지수족 $p(x \mid \theta) = \exp(\theta^T T(x) - \psi(\theta)) h(x)$는 $g(T, \theta) = \exp(\theta^T T - \psi(\theta))$로 인수분해되므로 **$T$는 항상 충분통계량**이다.

(증명은 Math Stat Deep Dive Ch3 참조. 여기서는 사용.)

### 정리 1.7 (지수족은 통계다양체)

자연 지수족 $\mathcal{E} = \{p_\theta : \theta \in \Theta\}$는 regular(정규)일 때 $d$차원 매끈 통계다양체이다 (Ch2-01의 R1–R7 만족).

**증명 스케치.**
1. **R1 (모수화):** $\theta \mapsto p_\theta$는 단사 — minimal이므로 $\theta_1 \neq \theta_2 \Rightarrow$ 어떤 $x$에서 $\theta_1^T T(x) \neq \theta_2^T T(x)$ $\Rightarrow p_{\theta_1} \neq p_{\theta_2}$.
2. **R2 (매끈성):** $\psi$가 $\Theta$ 내부에서 **해석적(real-analytic)**임을 $\partial_\theta$를 $\int$ 밑으로 교환하여 보일 수 있다 (지배수렴정리).
3. **R3 (support 공통):** $h(x) d\nu(x)$-지지대(support)가 $\theta$와 무관. 모든 지수족에서 성립.
4. **R4 (적분/미분 교환):**
   $$
   \partial_\theta \int e^{\theta^T T - \psi(\theta)} h \, d\nu = \int \partial_\theta [\cdot] h \, d\nu
   $$
   regular 지수족에서 $\Theta$의 임의 compact 부분집합에서 지배함수 $C \cdot e^{C\|T\|}$가 적분 가능.
5. **R5–R7:** 마찬가지로 regular 가정에서 $\nabla \psi, \nabla^2\psi$가 정의되고 연속 (Ch4-02에서 엄밀히).

$\quad\blacksquare$

### 정리 1.8 (주요 분포의 지수족 표현)

아래 표의 각 분포는 자연 지수족이다.

| 분포 | 밀도 | $T(x)$ | $\theta$ (canonical) | $\psi(\theta)$ | $h(x)$ |
|------|------|--------|------|---------|-------|
| **Bernoulli**($p$) | $p^x(1-p)^{1-x}$ | $x$ | $\log\frac{p}{1-p}$ | $\log(1+e^\theta)$ | $1$ |
| **Binomial**($n, p$) | $\binom{n}{x}p^x(1-p)^{n-x}$ | $x$ | $\log\frac{p}{1-p}$ | $n\log(1+e^\theta)$ | $\binom{n}{x}$ |
| **Poisson**($\lambda$) | $\frac{\lambda^x e^{-\lambda}}{x!}$ | $x$ | $\log\lambda$ | $e^\theta$ | $\frac{1}{x!}$ |
| **Geometric**($p$) | $p(1-p)^x$ | $x$ | $\log(1-p)$ | $-\log(1-e^\theta)$ | $1$ |
| **Exponential**($\lambda$) | $\lambda e^{-\lambda x}$ | $x$ | $-\lambda$ | $-\log(-\theta)$ | $1$ |
| **Normal**($\mu, \sigma^2$) (known $\sigma$) | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$ | $x/\sigma^2$ | $\mu$ | $\mu^2/(2\sigma^2)$ | $\frac{e^{-x^2/(2\sigma^2)}}{\sqrt{2\pi}\sigma}$ |
| **Normal**($\mu, \sigma^2$) (both) | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$ | $(x, x^2)$ | $(\mu/\sigma^2, -1/(2\sigma^2))$ | $-\theta_1^2/(4\theta_2) - \frac{1}{2}\log(-2\theta_2)$ | $1/\sqrt{2\pi}$ |
| **Gamma**($\alpha, \beta$) | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $(\log x, x)$ | $(\alpha-1, -\beta)$ | $\log\Gamma(\theta_1+1) - (\theta_1+1)\log(-\theta_2)$ | $1$ |
| **Multinomial**$(n; p_1, \dots, p_K)$ | $\frac{n!}{\prod x_i!}\prod p_i^{x_i}$ | $(x_1, \dots, x_{K-1})$ | $(\log\frac{p_i}{p_K})$ | $n\log(1+\sum e^{\theta_i})$ | $\frac{n!}{\prod x_i!}$ |
| **Dirichlet**$(\alpha_1, \dots, \alpha_K)$ | $\frac{\Gamma(\sum\alpha_i)}{\prod\Gamma(\alpha_i)}\prod x_i^{\alpha_i-1}$ | $(\log x_1, \dots, \log x_K)$ | $(\alpha_i - 1)$ | $\sum\log\Gamma(\theta_i+1) - \log\Gamma(\sum(\theta_i+1))$ | $1/\prod x_i$ |

**증명 (Normal, both parameters).** 로그밀도 전개:

$$
\log p(x \mid \mu, \sigma^2) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2} = \frac{\mu}{\sigma^2} x - \frac{1}{2\sigma^2}x^2 - \frac{\mu^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)
$$

$\theta_1 = \mu/\sigma^2$, $\theta_2 = -1/(2\sigma^2)$로 잡으면 $\sigma^2 = -1/(2\theta_2)$, $\mu = -\theta_1/(2\theta_2)$이므로

$$
\psi(\theta) = \frac{\mu^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2) = -\frac{\theta_1^2}{4\theta_2} - \frac{1}{2}\log(-2\theta_2) + \frac{1}{2}\log(2\pi)
$$

마지막 상수 $\frac{1}{2}\log(2\pi)$는 $h(x) = 1/\sqrt{2\pi}$로 흡수. $\blacksquare$

### 정리 1.9 (i.i.d. 표본의 지수족 구조)

$X_1, \dots, X_n \overset{\text{iid}}{\sim} p(\cdot \mid \theta)$ (지수족). **결합 분포 역시 지수족**이고:

$$
p(x_{1:n} \mid \theta) = \exp\left(\theta^T \underbrace{\sum_{i=1}^n T(x_i)}_{=: T_n} - n\psi(\theta)\right) \prod_{i=1}^n h(x_i)
$$

즉 **표본평균 $\frac{1}{n}T_n$**이 여전히 sufficient statistic. $\psi_n(\theta) = n\psi(\theta)$, $T_n(x_{1:n}) = \sum T(x_i)$.

**따름** Fisher 정보는 $n$에 비례: $F_n(\theta) = n F_1(\theta)$. 이것이 Ch2-05 CR 하한 $\text{Var}(\hat\theta) \succeq F_n^{-1} = F_1^{-1}/n$에서 $n$이 분모에 오는 이유.

### 정리 1.10 (Mean Value Parametrization, expectation parameter)

$\eta(\theta) := \mathbb{E}_\theta[T(X)]$는 regular 지수족에서 **$\theta \leftrightarrow \eta$가 미분동형(diffeomorphism)**이다.

(증명은 Ch4-03에서 Legendre 변환을 통해 완전히 전개. 여기서는 사실만.)

이 $\eta$는 expectation parameter, moment parameter, mean-value parameter라 부르며 dual coordinate이다. 예:

- Bernoulli: $\theta = \log\frac{p}{1-p}$, $\eta = p$
- Normal ($\sigma$ known): $\theta = \mu$, $\eta = \mu$
- Poisson: $\theta = \log\lambda$, $\eta = \lambda$

### 정리 1.11 (minimal 지수족의 identifiability)

minimal 지수족에서 $p_\theta = p_{\theta'}$ $\iff$ $\theta = \theta'$. 즉 파라미터가 **식별 가능(identifiable)**.

**증명.** $p_\theta = p_{\theta'}$ $\iff$ 모든 $x$에 대해 $\theta^T T(x) - \psi(\theta) = (\theta')^T T(x) - \psi(\theta')$ $\iff$ $(\theta - \theta')^T T(x) = \psi(\theta) - \psi(\theta') \text{ a.e.}$ 우변은 $x$에 무관한 상수, 좌변은 $x$의 함수. minimal 가정으로 $a_0 + a^T T = 0$을 만족하는 비자명 $(a_0, a)$가 없으므로 $\theta = \theta'$, $\psi(\theta) = \psi(\theta')$. $\blacksquare$

---

## 💻 NumPy / SymPy 구현으로 검증

### 코드 1: Bernoulli와 logit 좌표의 동치성

```python
import numpy as np

# 베르누이 분포: 두 좌표계
p = 0.7

# 1. 원래 좌표: P(X=1) = p, P(X=0) = 1-p
# 2. Canonical 좌표: θ = log(p/(1-p)), P(X=x) = exp(θ x - ψ(θ))
theta = np.log(p / (1 - p))
psi = np.log(1 + np.exp(theta))

# 양쪽 표현이 동일한지 확인
for x in [0, 1]:
    p_original = p**x * (1-p)**(1-x)
    p_canonical = np.exp(theta * x - psi)
    print(f"x={x}: original={p_original:.6f}, canonical={p_canonical:.6f}, diff={abs(p_original-p_canonical):.2e}")
# 두 값이 머신 정밀도로 일치
```

출력:
```
x=0: original=0.300000, canonical=0.300000, diff=5.55e-17
x=1: original=0.700000, canonical=0.700000, diff=1.11e-16
```

### 코드 2: SymPy로 Normal (both parameters)의 $(\theta, \psi)$ 유도

```python
import sympy as sp

mu, sigma, x = sp.symbols('mu sigma x', real=True, positive=True)

# 원래 밀도의 로그
log_p = -sp.Rational(1,2)*sp.log(2*sp.pi*sigma**2) - (x - mu)**2 / (2*sigma**2)
expanded = sp.expand(log_p)
print("로그밀도 전개:", sp.simplify(expanded))
# (μ/σ²)x + (-1/(2σ²))x² - μ²/(2σ²) - (1/2)log(2πσ²)

# Canonical 변환: θ₁ = μ/σ², θ₂ = -1/(2σ²)
theta1, theta2 = sp.symbols('theta1 theta2', real=True)
# 역변환
sigma2_expr = -1 / (2 * theta2)
mu_expr = -theta1 / (2 * theta2)

# ψ(θ) = μ²/(2σ²) + (1/2)log(2πσ²) - log h(x)의 x-독립 부분
psi_expr = mu_expr**2 / (2 * sigma2_expr) + sp.Rational(1,2)*sp.log(2*sp.pi*sigma2_expr)
psi_simplified = sp.simplify(psi_expr)
print("ψ(θ) =", psi_simplified)
# -θ₁²/(4 θ₂) - (1/2) log(-2 θ₂) + (1/2)log(2π)
# h(x) = 1/sqrt(2π)에 상수 흡수 → ψ(θ) = -θ₁²/(4θ₂) - (1/2) log(-2θ₂)
```

### 코드 3: i.i.d. 표본에서 sufficient statistic만으로 likelihood 계산

```python
import numpy as np

# 베르누이에서 n개 표본의 결합 likelihood가
# L(θ) = exp(θ · Σx - n·ψ(θ)) 로 축약됨을 확인
np.random.seed(42)
p_true = 0.3
n = 10000
X = np.random.binomial(1, p_true, size=n)

def log_lik_direct(theta, X):
    # 직접 p(x)^x (1-p)^(1-x) 곱
    p = 1 / (1 + np.exp(-theta))
    return np.sum(X * np.log(p) + (1 - X) * np.log(1 - p))

def log_lik_sufficient(theta, sumX, n):
    # 충분통계량만 이용
    psi = np.log(1 + np.exp(theta))
    return theta * sumX - n * psi

theta_test = np.linspace(-3, 3, 7)
sumX = X.sum()
print(f"{'theta':>7} {'direct':>12} {'sufficient':>12} {'diff':>10}")
for th in theta_test:
    ld = log_lik_direct(th, X)
    ls = log_lik_sufficient(th, sumX, n)
    print(f"{th:7.2f} {ld:12.3f} {ls:12.3f} {abs(ld-ls):10.2e}")
# 모든 θ에서 정확히 일치 — 원본 n개 관측치가 충분통계량 sum(X) 하나로 축약됨
```

### 코드 4: Gamma의 canonical 파라미터화

```python
import numpy as np
from scipy import stats

# Gamma(α=3, β=2)
alpha, beta = 3.0, 2.0
# Canonical: θ₁ = α - 1 = 2,  θ₂ = -β = -2
# T(x) = (log x, x)
# ψ(θ) = log Γ(θ₁+1) - (θ₁+1) log(-θ₂) = log Γ(3) - 3 log 2 = log 2 - 3 log 2 = -2 log 2

theta1, theta2 = alpha - 1, -beta
T_sample = lambda x: np.array([np.log(x), x])
psi_val = np.log(np.math.gamma(theta1 + 1)) - (theta1 + 1) * np.log(-theta2)

x_test = 1.5
# Scipy 밀도
scipy_pdf = stats.gamma.pdf(x_test, a=alpha, scale=1/beta)
# 지수족 형식: h(x) = 1/x (Gamma는 support가 x>0)
h_x = 1.0  # 일반 형식: β^α x^(α-1) e^(-βx) / Γ(α)
# 사실 Gamma를 위 canonical로 쓸 때 h(x) = 1/x이므로
h_x = 1 / x_test
exp_fam_pdf = np.exp(theta1 * np.log(x_test) + theta2 * x_test - psi_val) * h_x

print(f"SciPy pdf  : {scipy_pdf:.6f}")
print(f"Exp family : {exp_fam_pdf:.6f}")
print(f"diff       : {abs(scipy_pdf - exp_fam_pdf):.2e}")
```

### 코드 5: full vs curved 지수족

```python
import numpy as np
import matplotlib.pyplot as plt

# full N(μ, σ²) 족: (θ₁, θ₂) ∈ ℝ × ℝ_<0
# curved N(θ, θ²) 곡선: (θ₁, θ₂) = (θ/θ², -1/(2θ²)) = (1/θ, -1/(2θ²))

thetas = np.linspace(0.5, 3, 50)
theta1_curve = 1 / thetas
theta2_curve = -1 / (2 * thetas**2)

plt.figure(figsize=(7, 5))
# full family의 허용 영역 (θ₂<0)
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.fill_between([-3, 3], -3, 0, alpha=0.1, color='blue', label='full exp family (θ₂<0)')
plt.plot(theta1_curve, theta2_curve, 'r-', lw=2, label=r'curved: $\mathcal{N}(\mu, \mu^2)$')
plt.xlabel(r'$\theta_1 = \mu/\sigma^2$')
plt.ylabel(r'$\theta_2 = -1/(2\sigma^2)$')
plt.title('Full vs Curved Exponential Family')
plt.legend()
plt.tight_layout()
plt.savefig('curved_exp.png', dpi=120)
# full은 2차원, curved는 1차원 부분다양체
```

---

## 🔗 AI/ML 연결

### 1. Logistic Regression

$$
p(y = 1 \mid x; w) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}
$$

이것은 **조건부 베르누이 지수족**: $\theta = w^T x$(입력에 조건부 canonical 파라미터), $T(y) = y$, $\psi(w^T x) = \log(1 + e^{w^T x})$. 즉 **logit이 canonical parameter**이고 sigmoid가 expectation parameter $\eta = \nabla\psi$이다.

### 2. Softmax = Multinomial 지수족

$$
p(y = k \mid x; W) = \frac{e^{w_k^T x}}{\sum_j e^{w_j^T x}}
$$

$K-1$차원 canonical parameter $\theta_k = w_k^T x - w_K^T x$, sufficient statistic $T(y) = (\mathbb{1}\{y=1\}, \dots, \mathbb{1}\{y=K-1\})$, $\psi = \log(1 + \sum_{k<K} e^{\theta_k})$. **Cross-entropy loss가 음-로그우도** = $-\theta^T T(y) + \psi(\theta)$이고 $\theta$에 대해 볼록.

### 3. Energy-Based Model (EBM)

$$
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}, \quad Z(\theta) = \int \exp(-E_\theta(x)) dx
$$

에너지가 $\theta$에 선형이면($E_\theta(x) = -\theta^T T(x)$) **지수족**. Boltzmann machine, Ising model, CRF 모두 이 구조. $\log Z(\theta) = \psi(\theta)$가 cumulant function. 학습 시 $\nabla_\theta \log p_\theta(x) = T(x) - \mathbb{E}_{p_\theta}[T(X)]$ — sufficient statistic의 expected vs observed 차이 (Ch4-02).

### 4. Variational Autoencoder의 prior / posterior

VAE 표준에서 $p(z) = \mathcal{N}(0, I)$ (가우스 지수족), posterior $q(z \mid x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$ (가우스 지수족의 조건부 parameterization). ELBO 업데이트에서 mean parameter $\eta$ 업데이트는 natural gradient와 **정확히 같은 방향** (Ch4-05, Ch5).

### 5. 확산 모델 (Diffusion Model)

Forward process $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t)I)$는 **매 $t$마다 가우스 지수족**. Score $\nabla_x \log q_t(x) = -x - \mu_t / \sigma_t^2$가 canonical 파라미터의 공간적 미분.

### 6. Exponential Family Bandit

Multi-armed bandit에서 각 arm의 보상이 지수족을 따를 때 **Thompson sampling**의 사후분포가 conjugate prior와 함께 closed-form 업데이트. MLE 수렴 속도가 $F^{-1}$에 비례 (Ch2-05 CR).

---

## ⚖️ 가정과 한계

### 가정

1. **regular**: $\Theta$ 열려 있고 minimal. Non-regular 경우 (예: $\partial\Theta$에서의 지수족)는 boundary 처리 필요.
2. **지배 측도 $\nu$ 존재**: 모든 $P_\theta$가 공통의 $\sigma$-유한 측도에 대해 밀도를 가짐.
3. **support 불변**: $\{x : h(x) > 0\}$이 $\theta$에 무관.

### 한계

1. **Cauchy 분포는 지수족이 아님.** $p(x; \mu) = 1/(\pi(1 + (x-\mu)^2))$의 로그를 $\mu$의 선형 결합으로 쓸 수 없다. 따라서 Ch4-04의 e-connection·쌍대평탄성 이론이 **직접 적용되지 않는다** (curved family로만 취급 가능).

2. **Mixture model은 지수족이 아님.** $p(x) = \pi_1 \mathcal{N}(\mu_1, \sigma^2) + \pi_2 \mathcal{N}(\mu_2, \sigma^2)$의 로그가 $(\mu_1, \mu_2, \pi_1)$의 선형 결합이 아니다. 이것이 GMM의 EM이 local optimum을 가지는 이유 (Ch6-05).

3. **Uniform $\mathcal{U}(0, \theta)$는 비 regular 지수족.** Support가 $\theta$에 의존하므로 위의 이론이 실패. Fisher 정보가 정의 자체에 문제 ($\partial_\theta \log p$가 measure-zero 집합을 제외하고 0).

4. **$d \gg 1$에서 $\psi(\theta) = \log Z(\theta)$의 계산이 어려움.** Ising model, EBM 학습에서 partition function 추정이 NP-hard. **Contrastive divergence, score matching**이 이 한계를 우회.

5. **지수족의 유일성이 아님.** 같은 분포를 여러 방식으로 지수족으로 쓸 수 있음. minimal이 표준.

---

## 📌 핵심 정리

| 대상 | 공식 / 사실 |
|------|---------|
| 지수족 정의 | $p(x \mid \theta) = \exp(\theta^T T(x) - \psi(\theta)) h(x)$ |
| canonical parameter | $\theta \in \Theta = \{\theta : \psi(\theta) < \infty\}$, $\Theta$는 볼록 |
| sufficient statistic | $T(x) \in \mathbb{R}^d$, Fisher-Neyman 정리로 관측의 모든 정보를 담음 |
| cumulant function | $\psi(\theta) = \log \int e^{\theta^T T} h \, d\nu$, $\theta$의 볼록함수 |
| expectation parameter | $\eta = \mathbb{E}_\theta[T(X)] = \nabla\psi(\theta)$ (Ch4-02에서 증명) |
| minimal / regular | $T$가 아핀 독립 / $\Theta$ 열림 |
| curved | full 안의 $k < d$차원 부분다양체 |
| i.i.d. 결합 | 여전히 지수족, $T_n = \sum T(X_i)$, $\psi_n = n\psi$ |
| 아핀 구조 (기하) | 로그밀도가 $\theta$의 아핀 함수 $\Rightarrow$ 분포공간에서 **e-flat** |

**한 줄 요약:** 지수족은 **로그밀도가 $T(x)$의 선형 결합**이고, 그 결합 계수 $\theta$가 canonical 좌표이며 **cumulant 함수 $\psi$의 헤시안이 Fisher**이다 — 이것이 분포 공간의 아핀 구조 "e-flatness"의 기하학적 출발점이다.

---

## 🤔 생각해볼 문제

1. **(Bernoulli logit의 기하)** $p \in (0, 1)$에 대해 $\theta = \log\frac{p}{1-p}$. $p \to 0, 1$일 때 $\theta$는? 이것이 "지수족의 경계(boundary)"에서 무엇이 일어나는지에 대해 시사하는 바는?

2. **($\psi$ 로부터 모멘트)** $\psi(\theta) = \log\int e^{\theta T}$이므로 $\partial_\theta \psi = \mathbb{E}[T]$. 비슷하게 $\partial_\theta^2 \psi$는? 이것이 왜 Fisher 정보인가? (Ch4-02 완전 증명)

3. **(Multinomial의 차원)** $K$개 범주의 multinomial에서 왜 canonical parameter는 $K-1$차원인가? 제약 $\sum p_i = 1$의 역할을 minimal 조건으로 설명하시오.

4. **(Normal의 canonical)** $\theta_1 = \mu/\sigma^2$, $\theta_2 = -1/(2\sigma^2)$. $\theta \to (\mu, \sigma^2)$으로의 역변환 yakobian을 계산하고 (Ch4-03의 Legendre 변환 예고), 두 좌표계 간 Fisher 행렬의 변환을 추측해보시오.

5. **(Cauchy는 왜 지수족이 아닌가)** 엄밀히 증명: $p(x; \mu) = 1/(\pi(1 + (x-\mu)^2))$가 $p(x; \mu) = \exp(\theta(\mu)^T T(x) - \psi(\mu)) h(x)$ 꼴로 쓸 수 있다고 가정하고, $\mu$에 대한 로그밀도의 미분을 관찰하여 모순을 도출.

6. **(curved의 예: $\mathcal{N}(\mu, \mu^2)$)** 이 분포족이 full 가우스 족의 곡선임을 확인. $\mu \to 0, \infty$에서 이 곡선이 어떻게 full 공간의 경계에 접근하는가?

7. **(지수족의 Conjugate Prior)** $p(x \mid \theta) = \exp(\theta^T T(x) - \psi(\theta)) h(x)$에 대해 prior $\pi(\theta) \propto \exp(\theta^T \lambda_1 - \lambda_0 \psi(\theta))$가 conjugate임을 posterior의 형식으로 증명. Beta-Bernoulli, Gamma-Poisson, NIG-Normal 등 표준 예제와 대응시켜 보시오.

8. **(지수족과 MaxEnt의 동치)** $H(p) = -\int p \log p$ 최대화 with $\mathbb{E}_p[T_i] = \mu_i$, $\int p = 1$의 라그랑주 해가 지수족임을 유도. 이것이 Ch6-04에서 다뤄지지만, 왜 "자연스러운 분포"로 지수족이 튀어나오는지 설명.

---

<div align="center">

| [◀ Ch3-05. α-Rényi divergence](../ch3-kl-bregman/05-alpha-renyi-divergence.md) | [📚 메인 README](../README.md) | [02. Cumulant의 볼록성 ▶](./02-cumulant-convexity.md) |
|:---:|:---:|:---:|

</div>
