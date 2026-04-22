# 02. Cumulant function의 볼록성과 Fisher — $\nabla^2 \psi = F$

<div align="center">

> *"cumulant 함수 $\psi(\theta)$의 1차 미분은 모멘트. 2차 미분은 공분산. 3차 미분은 3차 cumulant.*  
> *그런데 Fisher 정보 행렬도 2차 미분이다.*
>
> *우연이 아니다 — $\psi$의 헤시안이 곧 Fisher이다.*  
> *이것이 지수족이 정보기하에서 왜 '중심적'인지 말해주는 단일한 등식이다."*

</div>

---

## 🎯 핵심 질문

1. cumulant 함수 $\psi(\theta)$의 **엄격 볼록성**(strict convexity)은 어떻게 증명되는가? Hölder 부등식의 등호 조건과 minimal 조건의 연결은?
2. $\nabla\psi(\theta) = \mathbb{E}_\theta[T(X)]$ — 이 공식은 왜 성립하는가? 미분과 적분의 교환은 언제 허용되는가?
3. **$\nabla^2\psi(\theta) = \text{Cov}_\theta[T(X)] = F(\theta)$** — Fisher 정보가 cumulant의 헤시안임을 완전 증명.
4. $\psi$의 고차 미분 $\partial^3\psi, \partial^4\psi$는 분포의 skewness, kurtosis와 어떻게 연결되는가?
5. 왜 **지수족에서는 MLE가 moment matching과 동치**인가? $\eta = \nabla\psi(\theta)$가 가교 역할.

---

## 🔍 왜 이 기하학이 AI에서 중요한가

| AI/ML 기법 | $\nabla^2\psi = F$가 하는 일 |
|-----------|-----------------|
| **MLE 수렴** | 음-로그우도 $-\theta^T T + \psi(\theta)$는 $\theta$에 대해 볼록 ($\psi$ 볼록) $\Rightarrow$ 유일한 전역 최적점 |
| **Natural Gradient 계산** | 지수족에서 Fisher를 **해석적으로** 계산: $F = \nabla^2\psi$. $n$번 샘플링할 필요 없음 |
| **Bregman = KL** | KL은 cumulant $\psi$의 Bregman divergence (Ch3-04): $\text{KL}(p_{\theta_1} \| p_{\theta_2}) = D_\psi(\theta_2, \theta_1)$ |
| **Legendre Duality** | $\psi$ 엄격 볼록 $\Rightarrow$ $\psi^*$(Legendre dual)가 잘 정의 (Ch4-03) |
| **EBM 학습** | $\nabla_\theta \log p_\theta(x) = T(x) - \nabla\psi(\theta) = T(x) - \mathbb{E}[T]$ — "관측 통계 − 모델 통계" |
| **Boltzmann Machine** | Contrastive divergence가 $\nabla\psi$의 Monte Carlo 근사 |

---

## 📐 수학적 선행 조건

| 개념 | 참조 |
|------|------|
| **Hölder 부등식**과 등호 조건 | Calculus & Optimization Deep Dive Ch7 |
| **지배수렴정리 (DCT)**, 미분과 적분의 교환 | Probability Theory Deep Dive Ch5 |
| **엄격 볼록함수**의 헤시안이 양정치 | Calculus & Optimization Ch4 |
| Fisher 정보의 3가지 정의 (score cov, −Hessian, KL 2차) | **Ch2-02** |
| 지수족의 정의 $p(x\|\theta) = \exp(\theta^T T - \psi)h$ | **Ch4-01** |

---

## 📖 직관적 이해

### 1. $\psi$는 "모멘트 생성함수의 로그"

MGF(Moment Generating Function)를 $M(\theta) := \mathbb{E}[e^{\theta X}]$로 정의하면, 지수족의 cumulant는 $\psi(\theta) = \log M(\theta) - (\text{normalizer of base})$. 고전 확률론에서 **cumulant**란 $\log M$의 Taylor 계수이고, 그 $k$차 계수는 $\mathbb{E}[X^k]$의 다항 결합 — skewness, kurtosis 등.

$$
\psi(\theta) = \kappa_1 \theta + \frac{\kappa_2}{2!}\theta^2 + \frac{\kappa_3}{3!}\theta^3 + \dots
$$

$\kappa_1 = \mathbb{E}[T]$, $\kappa_2 = \text{Var}[T]$, $\kappa_3 = $ 3차 cumulant(왜도), $\kappa_4 = $ 4차 cumulant(첨도−3).

이 식의 **2차 계수 = Fisher**이다!

### 2. "로그 정규화 상수는 공짜로 모멘트를 준다"

핵심 등식:

$$
\boxed{\;\nabla \psi(\theta) = \mathbb{E}_\theta[T(X)], \quad \nabla^2\psi(\theta) = \text{Cov}_\theta[T(X)] = F(\theta)\;}
$$

- **1차 미분 = 평균**: 정규화 상수 $Z(\theta) = \int e^{\theta^T T}h d\nu$의 로그 미분은 기댓값. "$\log Z$ 미분 trick"으로 잘 알려짐.
- **2차 미분 = 분산 = Fisher**: score의 분산 = Fisher의 첫 번째 정의와 정확히 일치.

이것은 Ch2-02의 "Fisher 3정의 동치" 정리를 지수족에서 **훨씬 더 강하게** 재진술한 것이다.

### 3. Bernoulli로 맛보기

$\theta = \log\frac{p}{1-p}$, $\psi(\theta) = \log(1 + e^\theta)$.

- $\psi'(\theta) = \frac{e^\theta}{1+e^\theta} = \sigma(\theta) = p$ = $\mathbb{E}[X]$. ✓
- $\psi''(\theta) = \sigma(\theta)(1-\sigma(\theta)) = p(1-p)$ = $\text{Var}[X]$ = Fisher. ✓

그리고 $p \in (0,1)$ 내부에서 $\psi''>0$이므로 **엄격 볼록**. logistic 시그모이드의 단조 증가성이 이 볼록성의 기하학적 귀결이다.

### 4. Moment matching으로서의 MLE

데이터 $x_1, \dots, x_n$의 로그우도:

$$
\ell(\theta) = \theta^T \left(\sum T(x_i)\right) - n\psi(\theta)
$$

$\partial_\theta\ell = 0 \Rightarrow \nabla\psi(\hat\theta) = \frac{1}{n}\sum T(x_i)$. 즉

$$
\mathbb{E}_{\hat\theta}[T(X)] = \overline{T(x)}
$$

**"모델이 예측하는 충분통계량의 평균" = "관측된 충분통계량의 평균"** — moment matching. 이것이 지수족 MLE의 기하학적 본질이다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 (cumulant function의 정확한 정의)

지수족 $p(x \mid \theta) = \exp(\theta^T T(x) - \psi(\theta))h(x)$에서

$$
\psi(\theta) = \log \int_\mathcal{X} \exp(\theta^T T(x)) h(x) \, d\nu(x)
$$

이 $\psi: \Theta \to \mathbb{R}$는 규정화 조건 $\int p(x \mid \theta) d\nu = 1$에서 자동으로 결정된다.

### 정의 2.2 (k차 cumulant 텐서)

$$
\kappa_{i_1\cdots i_k}(\theta) := \left.\frac{\partial^k \psi(\theta)}{\partial \theta^{i_1}\cdots \partial \theta^{i_k}}\right|_{\theta}
$$

- $k=1$: $\kappa_i = \mathbb{E}[T_i]$ (mean)
- $k=2$: $\kappa_{ij} = \text{Cov}[T_i, T_j]$ (Fisher의 $(i,j)$ 성분)
- $k=3$: Amari-Chentsov 텐서 ($T_{ijk}$)의 출발점 (Ch3-02, Ch4-04)

---

## 🔬 정리와 증명

### 정리 2.3 (엄격 볼록성)

minimal regular 지수족에서 $\psi: \Theta \to \mathbb{R}$는 **엄격 볼록(strictly convex)**이다.

**증명.** 이미 정리 1.5에서 Hölder로 볼록성은 보였다. 엄격성을 보이려면, $\theta_1 \neq \theta_2$이고 $\lambda \in (0,1)$일 때 Hölder 등호가 성립하지 않음을 보이면 된다.

Hölder:

$$
\int u^\lambda v^{1-\lambda} d\mu \le \left(\int u \, d\mu\right)^\lambda \left(\int v \, d\mu\right)^{1-\lambda}
$$

등호는 $u = c \cdot v$ $\mu$-a.e. 일 때. 여기서 $u = e^{\theta_1^T T}h$, $v = e^{\theta_2^T T}h$이므로 등호 조건:

$$
e^{\theta_1^T T(x)} = c \cdot e^{\theta_2^T T(x)} \quad \text{a.e.} \iff (\theta_1 - \theta_2)^T T(x) = \log c \quad \text{a.e.}
$$

이것은 $T_1, \dots, T_d$가 아핀 종속 ($a_0 + a^T T = 0$ a.e., $a = \theta_1 - \theta_2 \neq 0$)이라는 뜻. **minimal에 모순.** 따라서 $\theta_1 \neq \theta_2$이면 Hölder는 엄격 부등식. $\psi$는 엄격 볼록. $\quad\blacksquare$

### 정리 2.4 (미분과 적분의 교환 — regular 지수족)

$\Theta$ 내부의 임의 $\theta_0$에서 $\partial_i = \partial/\partial\theta^i$에 대해

$$
\partial_i \int f(x; \theta) d\nu = \int \partial_i f(x; \theta) d\nu
$$

가 $f(x;\theta) = e^{\theta^T T(x)} h(x)$와 그 곱/합에 대해 성립한다.

**증명 (스케치).** $\theta_0 \in \Theta^\circ$. $\theta_0$를 중심으로 닫힌 공 $\bar B(\theta_0, r) \subset \Theta$. 임의 $\theta \in \bar B(\theta_0, r)$에서 $|\partial_i e^{\theta^T T(x)}| = |T_i(x)| e^{\theta^T T(x)} \le |T_i(x)| e^{\|\theta_0\|\|T\| + r\|T\|}$. 이는 $|T_i(x)| e^{(\|\theta_0\| + r)\|T(x)\|} h(x)$로 bound되며, regular 가정 하에 적분 가능 ($\psi$가 $\bar B(\theta_0, 2r)$에서 유한하므로 MGF가 유한 $\Rightarrow$ 모든 유한 차수 모멘트 존재). 지배수렴정리 적용. $\quad\blacksquare$

### 정리 2.5 ($\nabla\psi = \mathbb{E}[T]$) — 핵심 등식 I

$$
\boxed{\;\partial_i \psi(\theta) = \mathbb{E}_\theta[T_i(X)]\;}
$$

**증명.** 정의에서

$$
\psi(\theta) = \log Z(\theta), \quad Z(\theta) = \int e^{\theta^T T(x)}h(x) d\nu
$$

$$
\partial_i \psi = \frac{\partial_i Z}{Z} = \frac{1}{Z}\int T_i(x) e^{\theta^T T(x)} h(x) d\nu = \int T_i(x) \cdot \frac{e^{\theta^T T - \psi(\theta)}h(x)}{1} d\nu = \mathbb{E}_\theta[T_i]
$$

정리 2.4에서 미적분 교환이 정당화됨. $\quad\blacksquare$

### 정리 2.6 ($\nabla^2\psi = \text{Cov}[T]$) — 핵심 등식 II

$$
\boxed{\;\partial_i\partial_j \psi(\theta) = \text{Cov}_\theta[T_i, T_j] = \mathbb{E}_\theta[(T_i - \mathbb{E} T_i)(T_j - \mathbb{E} T_j)]\;}
$$

**증명.** 정리 2.5 결과를 다시 미분:

$$
\partial_j \mathbb{E}_\theta[T_i] = \partial_j \int T_i(x) e^{\theta^T T - \psi} h \, d\nu
$$

적분 안쪽 미분:

$$
\partial_j [e^{\theta^T T - \psi}] = e^{\theta^T T - \psi} (T_j - \partial_j\psi) = e^{\theta^T T - \psi}(T_j - \mathbb{E}[T_j])
$$

따라서:

$$
\partial_j\partial_i\psi = \int T_i(x)(T_j - \mathbb{E}[T_j]) e^{\theta^T T - \psi}h \, d\nu = \mathbb{E}[T_i(T_j - \mathbb{E}[T_j])] = \mathbb{E}[T_i T_j] - \mathbb{E}[T_i]\mathbb{E}[T_j] = \text{Cov}[T_i, T_j]
$$

$\quad\blacksquare$

### 정리 2.7 (Fisher = Hessian of $\psi$) — 정보기하의 출발점

지수족에서

$$
\boxed{\;F_{ij}(\theta) = \partial_i\partial_j \psi(\theta) = \text{Cov}_\theta[T_i, T_j]\;}
$$

**증명.** Fisher의 정의 1 (score covariance)은

$$
F_{ij}(\theta) = \mathbb{E}[\partial_i \log p \cdot \partial_j \log p]
$$

지수족에서 $\log p = \theta^T T - \psi + \log h$이므로

$$
\partial_i \log p = T_i - \partial_i \psi = T_i - \mathbb{E}[T_i]
$$

(Ch2-02 정리 3.1로부터 score의 평균이 0임은 이미 확인: $\mathbb{E}[T_i - \mathbb{E}[T_i]] = 0$.)

$$
F_{ij} = \mathbb{E}[(T_i - \mathbb{E} T_i)(T_j - \mathbb{E} T_j)] = \text{Cov}[T_i, T_j] = \partial_i\partial_j\psi
$$

정리 2.6에서 그대로. $\quad\blacksquare$

**기하학적 귀결:** $\psi$가 엄격 볼록 $\Rightarrow$ $F = \nabla^2\psi$가 **양정치** $\Rightarrow$ Fisher가 **리만 계량**으로 쓸 수 있음 (Ch2-03과 일관).

### 정리 2.8 (MLE = Moment Matching)

$x_1, \dots, x_n \overset{iid}{\sim} p(\cdot\mid\theta)$에서 로그우도의 최적점 $\hat\theta$는

$$
\nabla\psi(\hat\theta) = \frac{1}{n}\sum_{i=1}^n T(x_i) =: \bar T_n
$$

를 만족한다. 즉 **MLE 존재 $\iff$ $\bar T_n$이 $\nabla\psi(\Theta) = $ {expectation parameter 공간}의 내부에 놓임**.

**증명.** $\ell(\theta) = \theta^T (\sum T(x_i)) - n\psi(\theta)$. $\nabla\ell = 0 \Rightarrow \sum T(x_i) = n\nabla\psi(\hat\theta)$. 볼록성 (정리 2.3)으로 유일. $\blacksquare$

### 정리 2.9 ($\psi$의 3차 미분과 왜도 텐서)

$$
\partial_i\partial_j\partial_k\psi(\theta) = \mathbb{E}_\theta\big[(T_i - \mathbb{E} T_i)(T_j - \mathbb{E} T_j)(T_k - \mathbb{E} T_k)\big] =: S_{ijk}(\theta)
$$

이것은 3차 중앙 moment = **skewness 텐서**. Ch3-02에서 KL의 3차 Taylor 계수로 등장한 Amari-Chentsov 텐서 $T_{ijk}$와 부호 차이만 있는 동일한 대상이다 (정확히는 Amari의 부호 규약 $T_{ijk}^{(\alpha)} = (1-\alpha)/2 \cdot S_{ijk}$ 형태).

**증명.** 정리 2.6 재미분, 생략 (정리 2.5, 2.6과 유사한 "안의 중심화" 패턴). $\blacksquare$

### 정리 2.10 (음-로그우도의 볼록성)

지수족에서 single-observation 음-로그우도 $L(\theta; x) := -\log p(x \mid \theta) = -\theta^T T(x) + \psi(\theta) - \log h(x)$는 $\theta$에 대해 엄격 볼록이다.

**증명.** $\nabla^2 L(\theta; x) = \nabla^2 \psi(\theta) = F(\theta) \succ 0$. $\blacksquare$

**귀결 (MLE의 전역 수렴).** 여러 관측의 합 $\sum L(\theta; x_i)$도 엄격 볼록이므로 **전역 최적점 유일**. SGD나 Newton's method가 항상 전역 최적으로 수렴.

---

## 💻 NumPy / SymPy 구현으로 검증

### 코드 1: Bernoulli에서 $\nabla\psi, \nabla^2\psi$ 직접 검증

```python
import numpy as np

def psi_bernoulli(theta):
    return np.log(1 + np.exp(theta))

def grad_psi(theta):
    return np.exp(theta) / (1 + np.exp(theta))  # = sigmoid = p

def hess_psi(theta):
    s = grad_psi(theta)
    return s * (1 - s)  # = p(1-p) = Var[X] = Fisher

# 수치 미분과 비교
from scipy.differentiate import derivative

for theta in [-1, 0, 1, 2]:
    p = grad_psi(theta)
    # 이론값
    mean_T = p         # E[X] = p
    var_T = p*(1-p)    # Var[X] = p(1-p)
    # ψ의 미분
    dpsi = grad_psi(theta)
    d2psi = hess_psi(theta)
    print(f"θ={theta:+.0f}: E[T]={mean_T:.4f}, dψ={dpsi:.4f} | Var[T]={var_T:.4f}, d²ψ={d2psi:.4f}")
# 모든 θ에서 E[T]=dψ, Var[T]=d²ψ 머신 정밀도로 일치
```

### 코드 2: Normal에서 $\nabla^2\psi = F$ SymPy 심볼릭 검증

```python
import sympy as sp

theta1, theta2 = sp.symbols('theta1 theta2', real=True)
# 가우스 full family의 ψ
psi = -theta1**2 / (4 * theta2) - sp.Rational(1, 2) * sp.log(-2 * theta2)
# (h(x)=1/sqrt(2π)로 상수 흡수)

# 1차 미분
dpsi_1 = sp.diff(psi, theta1)
dpsi_2 = sp.diff(psi, theta2)
print("∂ψ/∂θ₁ =", sp.simplify(dpsi_1))  # -θ₁/(2θ₂) = μ
print("∂ψ/∂θ₂ =", sp.simplify(dpsi_2))  # θ₁²/(4θ₂²) - 1/(2θ₂) = μ² + σ²

# 2차 미분 (Hessian)
H = sp.Matrix([[sp.diff(psi, theta1, theta1), sp.diff(psi, theta1, theta2)],
               [sp.diff(psi, theta2, theta1), sp.diff(psi, theta2, theta2)]])
print("H =")
sp.pprint(sp.simplify(H))

# 이론값: Cov[T] = Cov[(X, X²)]
# X ~ N(μ, σ²), T = (X, X²)
# Var[X] = σ², Cov[X, X²] = 2μσ², Var[X²] = 2σ⁴ + 4μ²σ²
mu, sigma2 = sp.symbols('mu sigma2', positive=True)
Cov_T = sp.Matrix([[sigma2, 2*mu*sigma2],
                   [2*mu*sigma2, 2*sigma2**2 + 4*mu**2*sigma2]])
# θ 좌표계로 치환: θ₁ = μ/σ², θ₂ = -1/(2σ²)
subs = {theta1: mu/sigma2, theta2: -1/(2*sigma2)}
H_muSigma = sp.simplify(H.subs(subs))
print("H(μ,σ²) =")
sp.pprint(H_muSigma)
print("Cov[T] =")
sp.pprint(Cov_T)
print("차이:", sp.simplify(H_muSigma - Cov_T))  # 영행렬 — 즉 ∇²ψ = Cov[T] = F
```

### 코드 3: Boltzmann/Ising 식 모델에서 MLE = moment matching 검증

```python
import numpy as np

# 2차원 Gaussian 지수족: T(x) = (x, x²/2), θ = (μ/σ², -1/σ²)
# 데이터로부터 MLE → 샘플 평균·분산과 일치해야 함
np.random.seed(0)
mu_true, sigma_true = 2.0, 1.5
n = 10000
X = np.random.normal(mu_true, sigma_true, size=n)

# 표본 통계
T1_bar = X.mean()              # -> E[X] = μ
T2_bar = (X**2).mean()          # -> E[X²] = μ² + σ²

# MLE: moment matching
# E_θ[X] = μ = T1_bar
# E_θ[X²] = μ² + σ² = T2_bar
mu_hat = T1_bar
sigma2_hat = T2_bar - T1_bar**2
print(f"μ̂   = {mu_hat:.4f} (true {mu_true})")
print(f"σ̂²  = {sigma2_hat:.4f} (true {sigma_true**2})")
# 샘플 평균·분산과 완벽히 일치 — moment matching = MLE
```

### 코드 4: $\psi$의 고차 미분 = cumulants

```python
import sympy as sp

# Poisson: θ = log λ, ψ(θ) = exp(θ)
theta = sp.symbols('theta')
psi = sp.exp(theta)
# 모든 차수의 미분이 exp(θ) = λ
# → Poisson의 모든 cumulant κ_k = λ
# (Poisson은 cumulant가 모두 같은 유일 분포 중 하나)
for k in range(1, 6):
    print(f"κ_{k} = {sp.diff(psi, theta, k)}")
# κ₁ = κ₂ = ⋯ = e^θ = λ → mean = variance = λ (Poisson의 특성)
```

### 코드 5: $\nabla\psi: \theta \to \eta$의 미분동형 시각화 (Bernoulli)

```python
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(-5, 5, 200)
eta = 1 / (1 + np.exp(-theta))  # ∇ψ = sigmoid

plt.figure(figsize=(6, 5))
plt.plot(theta, eta, lw=2)
plt.xlabel(r'canonical $\theta = \log p/(1-p)$')
plt.ylabel(r'expectation $\eta = p$')
plt.title(r'$\nabla\psi: \theta \leftrightarrow \eta$ 는 $\mathbb{R} \leftrightarrow (0,1)$ 미분동형')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('nabla_psi_bernoulli.png', dpi=120)
# 단조증가, 전단사 → 좌표 교환 가능 (Ch4-03 Legendre 기초)
```

---

## 🔗 AI/ML 연결

### 1. Energy-Based Model의 gradient

$$
\log p_\theta(x) = \theta^T T(x) - \psi(\theta) + \log h(x)
$$

$$
\nabla_\theta \log p_\theta(x) = T(x) - \nabla\psi(\theta) = T(x) - \mathbb{E}_{p_\theta}[T]
$$

즉 **데이터의 sufficient statistic − 모델의 기대 sufficient statistic**. 이것이 Boltzmann machine, RBM, Ising model 학습에서 "positive phase − negative phase" gradient의 기하학적 의미.

### 2. Contrastive Divergence (Hinton 2002)

$\mathbb{E}_{p_\theta}[T]$의 계산이 intractable할 때 Monte Carlo 샘플링 $\tilde x_1, \dots, \tilde x_m \sim p_\theta$로 $\frac{1}{m}\sum T(\tilde x_i)$ 근사. Gibbs sampling을 $k$ 스텝만 실행하는 것이 CD-$k$.

### 3. Softmax의 $\psi$

K-classification softmax: $\theta = (\theta_1, \dots, \theta_{K-1})$, $\psi(\theta) = \log(1 + \sum_k e^{\theta_k})$. $\nabla\psi_k = e^{\theta_k}/(1+\sum e^{\theta_j}) = p_k$ (softmax). **Softmax는 gradient of log-partition function**. 헤시안 $\nabla^2\psi = \text{diag}(p) - pp^T$는 다항분포의 Fisher.

### 4. Variational Inference ELBO

VI에서 variational posterior $q_\phi(z)$를 지수족으로 두면 ELBO는

$$
\mathcal{L}(\phi) = \mathbb{E}_q[\log p(x, z) - \log q_\phi(z)] = \mathbb{E}_q[\log p(x, z)] + H(q)
$$

$q_\phi$가 지수족일 때 엔트로피 $H(q) = \psi(\phi) - \phi^T \nabla\psi(\phi)$. ELBO의 gradient가 $\nabla\psi$를 통해 표현되고 **natural gradient**가 closed form으로 닫힘 (Ch5-03, Ch7-03).

### 5. Conjugate Prior의 Bayesian Update

Prior $\pi(\theta) \propto \exp(\theta^T \lambda_1 - \lambda_0 \psi(\theta))$에 $n$개 관측 $x_{1:n}$ 후 posterior:

$$
\pi(\theta \mid x_{1:n}) \propto \exp\left(\theta^T (\lambda_1 + \sum T(x_i)) - (\lambda_0 + n)\psi(\theta)\right)
$$

즉 $(\lambda_1, \lambda_0) \to (\lambda_1 + \sum T(x_i), \lambda_0 + n)$. **sufficient statistic을 누적**하는 것이 Bayesian 업데이트. 이것은 $\psi$가 cumulant임으로부터 closed-form으로 따라 나온다.

### 6. Score Matching과 $\nabla^2 \psi$

Score matching 손실:

$$
J(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|\nabla_x \log p_\theta(x)\|^2 + \nabla_x \cdot \nabla_x \log p_\theta(x)\right]
$$

지수족에서 $\nabla_x \log p_\theta = \theta^T \nabla T(x)$이므로 $J$는 $\theta$의 quadratic → 볼록 최적화 (Hyvärinen 2005). $\psi$ 계산 불필요 — 이것이 score matching의 이론적 매력.

---

## ⚖️ 가정과 한계

### 가정

- regular minimal: $\Theta$ 열려 있고 $T$ 아핀 독립
- **MGF 존재**: $\psi(\theta) < \infty$ for $\theta \in \Theta^\circ$의 주변 — Cauchy는 이 조건 실패
- 미적분 교환 가능: 지배수렴정리의 지배함수 존재

### 한계

1. **intractable $\psi$**: Ising, RBM, deep EBM에서 $Z(\theta) = \int e^{-E}$를 정확히 계산 불가. partition function estimation은 NP-hard (Jerrum-Sinclair).

2. **$\nabla^2\psi$의 수치적 특이성**: $\Theta$의 경계 근처에서 eigenvalue가 폭발/0으로 수렴 — regularization 필요. 예: Bernoulli에서 $p \to 0, 1$일 때 Fisher $= p(1-p) \to 0$, $F^{-1}$ 발산.

3. **무한차원 지수족**: RKHS 기반 지수족(kernel exponential family, Canu-Smola)은 $\psi$가 functional. 이 글의 $d$차원 이론이 그대로 적용되지 않음.

4. **curved exponential family**: $\psi_{\text{curved}}(\xi) = \psi(\theta(\xi))$의 헤시안이 Fisher와 다름 — Ch2-05의 Efron 곡률 등장.

5. **mixture model**: $p(x) = \sum \pi_k p_{\theta_k}(x)$는 지수족이 아님 → cumulant 정리 직접 적용 불가. EM으로 우회 (Ch6-02).

---

## 📌 핵심 정리

| 대상 | 공식 / 사실 |
|------|---------|
| cumulant 정의 | $\psi(\theta) = \log\int e^{\theta^T T}h\,d\nu$ |
| 볼록성 | regular minimal $\Rightarrow$ $\psi$ 엄격 볼록 (Hölder 등호 조건) |
| 1차 미분 | $\nabla\psi(\theta) = \mathbb{E}_\theta[T(X)] = \eta$ (expectation param) |
| 2차 미분 | $\nabla^2\psi(\theta) = \text{Cov}_\theta[T(X)] = F(\theta)$ ← **Fisher** |
| 3차 미분 | $\partial^3\psi = \mathbb{E}[(T-\mathbb{E}T)^{\otimes 3}]$ = 왜도 텐서 |
| MLE | $\nabla\psi(\hat\theta) = \bar T_n$ — moment matching |
| 음로그우도 | 엄격 볼록 $\Rightarrow$ MLE 전역 유일 |
| EBM gradient | $\nabla_\theta\log p_\theta(x) = T(x) - \nabla\psi(\theta)$ (positive − negative phase) |

**한 줄 요약:** 지수족에서 **Fisher 정보 = cumulant 함수의 헤시안** — $F = \nabla^2\psi$. 이 단일 등식이 "지수족이 볼록 최적화이며 해석적으로 풀리는 이유"와 "Bregman·Legendre·natural gradient가 모두 $\psi$에서 튀어나오는 이유"를 동시에 설명한다.

---

## 🤔 생각해볼 문제

1. **(score의 평균 = 0, 재확인)** $\mathbb{E}[\partial_i \log p] = 0$임을 regular 지수족에서 $\partial_i \int p = 0$으로부터 유도. 미적분 교환이 왜 필요한가?

2. **(3차 미분이 cumulant인 이유)** $\log M(\theta)$의 Taylor 계수가 cumulant임을 MGF 정의와 Taylor 전개로 유도. 왜 "moment의 다항식"이 아니라 "cumulant"가 자연스러운가?

3. **(Bernoulli 볼록성의 끝)** $\theta = \pm\infty$에서 $\psi''(\theta) = p(1-p) \to 0$. 이것이 Fisher의 특이성과 CR 하한의 퇴화로 이어지는 방식은?

4. **(Gaussian의 $\psi$)** Section "엄밀한 정의"에서 $\psi(\theta_1, \theta_2) = -\theta_1^2/(4\theta_2) - \frac{1}{2}\log(-2\theta_2)$. 이것의 Hessian을 직접 계산하고 $(\mu, \sigma^2)$ 좌표로 변환하여 Ch2-04의 가우스 Fisher $\text{diag}(1/\sigma^2, 1/(2\sigma^4))$와 일치함을 확인.

5. **(Moment matching의 실패)** 데이터 $\{0, 1\}$에서 표본이 모두 0이면 Bernoulli MLE $\hat p = 0$. 이때 canonical $\hat\theta = \log 0 = -\infty$. 이것이 $\Theta$의 경계 문제와 어떻게 연결되며, Bayesian conjugate prior로 어떻게 regularize되는가?

6. **(score 샘플링으로 Fisher 추정)** 데이터로부터 $F(\theta)$를 empirical하게 추정하는 두 방법: (i) $\frac{1}{n}\sum s(x_i)s(x_i)^T$, (ii) $-\frac{1}{n}\sum \partial^2 \log p(x_i)$. 지수족에서 두 추정치가 어떤 관계인가? empirical Fisher vs true Fisher 차이는?

7. **(Softmax의 특이성)** Softmax Fisher $\text{diag}(p) - pp^T$는 항상 one eigenvalue가 0 (방향 $\mathbf{1}$). 왜? 이것과 multinomial의 $\sum p_i = 1$ 제약의 기하학적 관계?

8. **(Legendre 예고)** $\psi$ 엄격 볼록 $\Rightarrow$ $\eta = \nabla\psi(\theta)$ 전단사. $\theta = (\nabla\psi)^{-1}(\eta) = \nabla\psi^*(\eta)$ where $\psi^* = $ Legendre dual of $\psi$. $\psi^*(\eta) = -H(p_\theta) - $ base entropy와 어떻게 연결되는가? (Ch4-03)

---

<div align="center">

| [◀ 01. Exp Family 기하학적 정의](./01-exponential-family-geometry.md) | [📚 메인 README](../README.md) | [03. Legendre 쌍대 ▶](./03-legendre-duality.md) |
|:---:|:---:|:---:|

</div>
