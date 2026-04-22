# 02. Natural Gradient의 유도 — 제약 최적화와 KKT

> **"The natural gradient is the steepest descent direction in a Riemannian manifold equipped with the Fisher information metric."** — Shun-ichi Amari, *Natural Gradient Works Efficiently in Learning* (1998)

---

## 1. 왜 이 주제인가?

앞 문서에서 유클리드 gradient $\nabla L(\theta)$는 **좌표계(parameterization)에 의존**하여, 같은 문제라도 $\theta$와 $\phi = A\theta$에서 서로 다른 최적화 경로를 그린다는 문제를 보았다. 해결책의 방향은 명확하다: **파라미터 공간이 아닌 분포 공간에서 거리를 재자**. 분포 공간의 무한소 거리는 KL divergence의 2차 근사, 즉 **Fisher 계량** $F(\theta)$로 결정된다.

그러나 Fisher 계량을 "어떻게" gradient에 반영하는가? 단순히 $F^{-1}$을 곱하는 것이 아니라, **엄밀한 최적화 원리**에서 자연스럽게 도출되어야 한다. 이 문서는 Amari (1998)의 유도를 따라, **제약 최적화 문제**로부터 natural gradient를 KKT 조건을 통해 유도한다.

**핵심 아이디어**: 유클리드 steepest descent는 "단위 유클리드 구 안에서 $L$을 가장 많이 줄이는 방향"이다. 이를 **단위 Fisher 타원체(Fisher ellipsoid)** 안으로 제약 조건만 바꾸면, steepest descent 방향이 자동으로 $F^{-1} \nabla L$이 된다. 이것이 **natural gradient의 탄생 원리**다.

---

## 2. 학습 목표

이 문서를 끝내면 다음을 할 수 있다.

1. Steepest descent를 **제약 최적화 문제**로 정식화할 수 있다.
2. **Lagrangian**과 **KKT 조건**으로부터 natural gradient $\tilde{\nabla} L = F^{-1} \nabla L$을 유도할 수 있다.
3. Natural gradient가 **Fisher 타원체 상의 최대 감소 방향**임을 기하학적으로 설명할 수 있다.
4. Step size $\eta$와 trust region 반경 $\varepsilon$의 관계 $\eta = \varepsilon / \sqrt{(\tilde{\nabla} L)^T F (\tilde{\nabla} L)}$를 도출할 수 있다.
5. Natural gradient가 **Newton's method**와 어떤 관계인지 (exp family MLE에서 같음) 설명할 수 있다.

---

## 3. 전제 지식

- **Ch4-02**: Fisher 정보 $F = \mathbb{E}[\nabla \log p \cdot \nabla \log p^T] = -\mathbb{E}[\nabla^2 \log p]$
- **Ch5-01**: 유클리드 gradient의 parameterization 의존성 문제
- **라그랑주 승수법 & KKT 조건**: 부등식 제약 최적화의 1차 필요조건
- **양정치 행렬의 성질**: $F \succ 0$이므로 $F^{-1}$ 존재, $x^T F x \geq 0$이 norm을 정의

---

## 4. 직관적 설명

### 4.1 Steepest descent의 일반화

**유클리드 버전.** "단위 거리만 가서 $L$을 가장 많이 줄이고 싶다"는 요구는 다음과 같다:

$$
\min_{d\theta} \nabla L(\theta)^T d\theta \quad \text{s.t.} \quad \|d\theta\|_2 \leq \varepsilon.
$$

Lagrangian $\mathcal{L} = \nabla L^T d\theta - \lambda(\varepsilon^2 - d\theta^T d\theta)$의 KKT 조건은 $\nabla L = 2\lambda d\theta$, 즉 $d\theta = -\frac{\varepsilon}{\|\nabla L\|} \nabla L$. 이것이 바로 유클리드 gradient descent다.

**Riemann 버전.** 문제는 "단위 거리"를 **어떤 계량으로 재는가**이다. 분포 공간에서 자연스러운 거리는 **KL divergence**:

$$
\text{KL}(p_\theta \| p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F(\theta) d\theta.
$$

따라서 "KL 반경 $\varepsilon$짜리 공 안에서 $L$을 가장 많이 줄이는 방향"은:

$$
\boxed{\min_{d\theta} \nabla L(\theta)^T d\theta \quad \text{s.t.} \quad d\theta^T F(\theta) d\theta \leq \varepsilon^2.}
$$

이 풀이가 **natural gradient**다.

### 4.2 기하학적 그림

- 유클리드 steepest descent: **원형 trust region** (등거리 원) 경계에서 $L$이 가장 작아지는 점으로 간다.
- Natural gradient: **타원형 trust region** (Fisher 타원) 경계에서 $L$이 가장 작아지는 점으로 간다. 타원의 축이 긴 방향(=Fisher 고유값이 작은 = 분포가 둔감한 방향)으로는 멀리 갈 수 있고, 축이 짧은 방향(=분포가 민감한 방향)으로는 조금만 간다.

이로써 "분포가 민감한 방향으로는 신중하게, 둔감한 방향으로는 과감하게" 움직이는 자동 step size 조절이 일어난다.

---

## 5. 엄밀한 정의와 정리

### 5.1 제약 최적화 문제

**문제 5.1 (Fisher ball 제약 steepest descent).** 주어진 $\theta \in \Theta$, $\nabla L(\theta) \in \mathbb{R}^n$, $F(\theta) \succ 0$, $\varepsilon > 0$에 대해:

$$
d\theta^* = \arg\min_{d\theta \in \mathbb{R}^n} \nabla L(\theta)^T d\theta \quad \text{s.t.} \quad d\theta^T F(\theta) d\theta \leq \varepsilon^2.
$$

### 5.2 Natural gradient의 정의

**정의 5.2 (Natural gradient; Amari 1998).** 선형 범함수 $dL = \nabla L^T d\theta$를 Fisher 계량의 쌍대성에 의해 벡터로 옮긴 것:

$$
\boxed{\tilde{\nabla} L(\theta) := F(\theta)^{-1} \nabla L(\theta).}
$$

### 5.3 메인 정리

**정리 5.3 (Natural gradient = KL-ball steepest descent).** 문제 5.1의 유일한 해는:

$$
d\theta^* = -\frac{\varepsilon}{\sqrt{\nabla L^T F^{-1} \nabla L}} F^{-1} \nabla L = -\eta^* \tilde{\nabla} L,
$$

여기서 **최적 step size**는:

$$
\eta^* = \frac{\varepsilon}{\sqrt{\nabla L^T F^{-1} \nabla L}} = \frac{\varepsilon}{\|\tilde{\nabla} L\|_F}.
$$

**결론**: Fisher 타원체 내부에서 $L$을 가장 많이 줄이는 방향은 **항상 $-F^{-1} \nabla L$ 방향**이며, 타원체 반경 $\varepsilon$을 따라 적응적으로 step size가 정해진다.

---

## 6. 증명

### 6.1 Lagrangian과 KKT

문제 5.1의 Lagrangian:

$$
\mathcal{L}(d\theta, \lambda) = \nabla L^T d\theta + \lambda(d\theta^T F d\theta - \varepsilon^2), \quad \lambda \geq 0.
$$

KKT 조건:

1. **Stationarity**: $\partial_\theta \mathcal{L} = \nabla L + 2\lambda F d\theta = 0$. 따라서:
   
   $$d\theta = -\frac{1}{2\lambda} F^{-1} \nabla L. \tag{6.1}$$

2. **Primal feasibility**: $d\theta^T F d\theta \leq \varepsilon^2$.

3. **Dual feasibility**: $\lambda \geq 0$.

4. **Complementary slackness**: $\lambda (d\theta^T F d\theta - \varepsilon^2) = 0$.

### 6.2 활성 제약

$\nabla L \neq 0$이면 (비탈출 케이스), 해는 반드시 **경계**에 있다: $\lambda > 0$이고 $d\theta^T F d\theta = \varepsilon^2$. 이유는 내부점은 Objective를 더 감소시킬 방향이 항상 존재하기 때문이다.

식 (6.1)을 제약에 대입:

$$
\left(-\frac{1}{2\lambda} F^{-1} \nabla L\right)^T F \left(-\frac{1}{2\lambda} F^{-1} \nabla L\right) = \varepsilon^2,
$$

$$
\frac{1}{4\lambda^2} \nabla L^T F^{-1} F F^{-1} \nabla L = \varepsilon^2,
$$

$$
\frac{1}{4\lambda^2} \nabla L^T F^{-1} \nabla L = \varepsilon^2,
$$

$$
\lambda^* = \frac{1}{2\varepsilon} \sqrt{\nabla L^T F^{-1} \nabla L}.
$$

### 6.3 최종 해

$\lambda^*$를 (6.1)에 대입:

$$
d\theta^* = -\frac{1}{2\lambda^*} F^{-1} \nabla L = -\frac{\varepsilon}{\sqrt{\nabla L^T F^{-1} \nabla L}} F^{-1} \nabla L.
$$

**유일성**: Objective는 선형이고 제약은 strictly convex quadratic이므로 $F \succ 0$ 하에서 해는 유일. $\square$

### 6.4 대안 유도: Fisher inner product

**정리 6.1 (Riesz representation).** Fisher 계량 $g(X, Y) = X^T F Y$를 내적으로 주자. 1-form $dL = \nabla L^T d\theta$를 Fisher 내적으로 표현하면:

$$
dL(d\theta) = g(\tilde{\nabla} L, d\theta) = (\tilde{\nabla} L)^T F d\theta \overset{!}{=} \nabla L^T d\theta \quad \forall d\theta,
$$

$$
\Rightarrow F \tilde{\nabla} L = \nabla L \Rightarrow \tilde{\nabla} L = F^{-1} \nabla L.
$$

즉 natural gradient는 **1-form $dL$을 Fisher 계량으로 vector로 옮긴 것**. 이 유도는 제약 최적화 없이도 $\tilde{\nabla} L$의 정의가 자연스러움을 보여준다. $\square$

### 6.5 Descent 성질

**Claim.** $\nabla L \neq 0$이면 $d\theta^* = -\eta^* F^{-1} \nabla L$은 descent 방향:

$$
\nabla L^T d\theta^* = -\eta^* \nabla L^T F^{-1} \nabla L < 0
$$

( $F^{-1} \succ 0$이므로 $\nabla L^T F^{-1} \nabla L > 0$ ). $\square$

---

## 7. 구체 예제

### 7.1 2D Gaussian MLE

데이터 $\{x_1, \dots, x_N\} \sim \mathcal{N}(\mu, \sigma^2)$, $\theta = (\mu, \sigma)$, 손실 $L(\theta) = -\frac{1}{N}\sum \log p(x_i | \theta)$.

**Fisher 행렬**:

$$
F(\mu, \sigma) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^2 \end{pmatrix}.
$$

**유클리드 gradient**:

$$
\nabla L = \begin{pmatrix} -(\bar{x} - \mu)/\sigma^2 \\ \ldots \end{pmatrix}.
$$

**Natural gradient**:

$$
\tilde{\nabla} L = F^{-1} \nabla L = \begin{pmatrix} \sigma^2 & 0 \\ 0 & \sigma^2/2 \end{pmatrix} \nabla L.
$$

특히 $\mu$ 방향의 natural gradient는 $-(\bar{x} - \mu)$ (표본 평균 방향으로 직접), step size가 $\sigma^2$로 스케일링된 결과. **$\sigma$가 작으면(분포가 뾰족하면) $\mu$에 대해 작은 step, 큰 $\sigma$면 큰 step** — 분포 민감도에 따라 자동 조절.

### 7.2 Trust region과 step size

주어진 trust region 반경 $\varepsilon = 0.1$에서 $\nabla L = (1, 2)^T$, $F = \text{diag}(4, 1)$이면:

$$
\tilde{\nabla} L = F^{-1} \nabla L = (0.25, 2)^T, \quad \|\tilde{\nabla} L\|_F^2 = \tilde{\nabla} L^T F \tilde{\nabla} L = 4(0.25)^2 + 1(2)^2 = 0.25 + 4 = 4.25,
$$

$$
\eta^* = 0.1 / \sqrt{4.25} \approx 0.0485, \quad d\theta^* = -0.0485 \cdot (0.25, 2)^T = -(0.0121, 0.0970)^T.
$$

Fisher norm 검증: $d\theta^{*T} F d\theta^* = 4(0.0121)^2 + 1(0.0970)^2 = 0.000586 + 0.00941 \approx 0.01 = \varepsilon^2$. ✓

### 7.3 Newton's method와의 관계

**관찰.** Exponential family MLE에서 $L(\theta) = -\log p(\theta) = -\theta^T T + \psi(\theta) - \log h$이므로

$$
\nabla^2 L = \nabla^2 \psi = F(\theta).
$$

즉 **Hessian = Fisher**. 따라서 Newton's step $-H^{-1} \nabla L = -F^{-1} \nabla L = -\tilde{\nabla} L$, **natural gradient와 Newton's method는 정확히 일치**.

**비exp family**에서도 Natural gradient는 Fisher(=기댓값 하의 Hessian)를 사용하므로 "expected Newton" 또는 "Generalized Gauss-Newton"의 변형으로 해석된다.

---

## 8. Python 코드 검증

### 8.1 KKT 수치 검증

```python
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
n = 5

# Fisher 행렬 (양정치) 생성
A = np.random.randn(n, n)
F = A @ A.T + np.eye(n) * 0.5

# 임의 gradient
grad_L = np.random.randn(n)

eps = 0.1

# --- 방법 1: 해석적 공식 ---
F_inv = np.linalg.inv(F)
nat_grad = F_inv @ grad_L
norm_F = np.sqrt(grad_L @ nat_grad)  # sqrt(grad^T F^-1 grad)
eta_star = eps / norm_F
d_theta_analytic = -eta_star * nat_grad

# --- 방법 2: 수치 최적화 (SLSQP) ---
def objective(d):
    return grad_L @ d

def constraint(d):
    return eps**2 - d @ F @ d  # >= 0

sol = minimize(
    objective,
    x0=np.zeros(n),
    constraints={'type': 'ineq', 'fun': constraint},
    method='SLSQP'
)
d_theta_numeric = sol.x

# --- 비교 ---
print("Analytic :", d_theta_analytic)
print("Numeric  :", d_theta_numeric)
print("차이 norm:", np.linalg.norm(d_theta_analytic - d_theta_numeric))

# Fisher norm 제약 활성도
print(f"Fisher norm² = {d_theta_analytic @ F @ d_theta_analytic:.6f}, eps² = {eps**2}")

# Objective 값
print(f"Objective (analytic) = {objective(d_theta_analytic):.6f}")
print(f"Objective (numeric)  = {objective(d_theta_numeric):.6f}")
```

**기대 출력**:
```
차이 norm: ~1e-7
Fisher norm² = 0.010000, eps² = 0.01
Objective 일치
```

### 8.2 Natural gradient vs 유클리드 gradient 비교 (Gaussian MLE)

```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(0)
X = np.random.normal(loc=3.0, scale=2.0, size=200)

def neg_log_lik(mu, sigma):
    return 0.5 * np.log(2*np.pi) + np.log(sigma) + 0.5 * np.mean(((X - mu)/sigma)**2)

def grad(mu, sigma):
    dmu = -np.mean((X - mu) / sigma**2)
    dsigma = 1/sigma - np.mean((X - mu)**2) / sigma**3
    return np.array([dmu, dsigma])

def fisher(mu, sigma):
    return np.array([[1/sigma**2, 0], [0, 2/sigma**2]])

# 최적화
lr = 0.05
theta_euc = np.array([0.0, 1.0])
theta_nat = np.array([0.0, 1.0])
path_euc = [theta_euc.copy()]
path_nat = [theta_nat.copy()]

for _ in range(200):
    mu, sigma = theta_euc
    theta_euc = theta_euc - lr * grad(mu, sigma)
    theta_euc[1] = max(theta_euc[1], 0.01)  # sigma > 0
    path_euc.append(theta_euc.copy())
    
    mu, sigma = theta_nat
    g = grad(mu, sigma)
    F_inv = np.linalg.inv(fisher(mu, sigma))
    theta_nat = theta_nat - lr * (F_inv @ g)
    theta_nat[1] = max(theta_nat[1], 0.01)
    path_nat.append(theta_nat.copy())

path_euc = np.array(path_euc)
path_nat = np.array(path_nat)

# 플롯
mu_grid = np.linspace(-1, 4, 80)
sigma_grid = np.linspace(0.3, 3, 80)
MU, SIGMA = np.meshgrid(mu_grid, sigma_grid)
L_grid = np.vectorize(neg_log_lik)(MU, SIGMA)

plt.figure(figsize=(9, 6))
plt.contour(MU, SIGMA, L_grid, levels=25, cmap='viridis', alpha=0.5)
plt.plot(path_euc[:, 0], path_euc[:, 1], 'r.-', label='Euclidean GD', markersize=3)
plt.plot(path_nat[:, 0], path_nat[:, 1], 'b.-', label='Natural GD', markersize=3)
plt.plot(3.0, 2.0, 'k*', markersize=15, label='True (μ=3, σ=2)')
plt.xlabel('μ'); plt.ylabel('σ')
plt.title('Euclidean vs Natural Gradient on Gaussian MLE')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
```

**관찰**: Euclidean GD는 초기 $\sigma \approx 1$에서 $\mu$ 방향으로만 빠르게 움직이다가 $\sigma$를 나중에 조정(지그재그). Natural GD는 **곧장 MLE로 수렴** — Fisher가 $\sigma$ 방향 step을 자동으로 크게 만들기 때문.

### 8.3 Newton ≡ Natural gradient (exp family 검증)

```python
from scipy.stats import norm
import numpy as np

# 고정 sigma=1 가정 하의 mu MLE
X = np.random.normal(2, 1, size=500)

def neg_ll(mu):
    return -norm.logpdf(X, loc=mu, scale=1).mean()

def grad_mu(mu):
    return -np.mean(X - mu)

def hess_mu(mu):
    return 1.0  # Hessian of neg log likelihood = 1/sigma^2 = 1

def fisher_mu(mu):
    return 1.0  # Fisher = 1/sigma^2 = 1

mu_newton = 0.0; mu_nat = 0.0
for it in range(5):
    mu_newton -= grad_mu(mu_newton) / hess_mu(mu_newton)
    mu_nat    -= grad_mu(mu_nat) / fisher_mu(mu_nat)
    print(f"it {it}: newton={mu_newton:.6f}, nat={mu_nat:.6f}, true={X.mean():.6f}")
```

**출력**: 두 수치가 모든 iteration에서 **소수점 오차 없이 일치**, exp family에서 Newton = Natural을 확인.

---

## 9. AI/ML 연결

### 9.1 Trust Region Policy Optimization (TRPO)

Schulman+ (2015)의 TRPO는 **정확히 이 문서의 제약 최적화 식**을 쓴다:

$$
\max_\theta \mathbb{E}_{\pi_\theta}[A^{\pi_{\text{old}}}(s,a)] \quad \text{s.t.} \quad \mathbb{E}_s[\text{KL}(\pi_{\text{old}} \| \pi_\theta)] \leq \delta.
$$

KL 제약을 2차 근사하면 $\frac{1}{2} d\theta^T F d\theta \leq \delta$이 되고, natural gradient가 해가 된다. TRPO는 conjugate gradient로 $F^{-1} \nabla L$을 근사 계산.

### 9.2 Natural Policy Gradient (Kakade 2001)

Kakade가 RL에 Amari의 natural gradient를 도입:

$$
\theta_{t+1} = \theta_t + \eta F^{-1} \nabla J(\theta_t),
$$

$J$는 expected return. 후에 PPO, ACKTR (K-FAC+actor-critic) 등으로 확장.

### 9.3 Variational Inference

ELBO 최대화에서 variational parameter $\lambda$에 대한 natural gradient:

$$
\lambda_{t+1} = \lambda_t + \eta F_q(\lambda_t)^{-1} \nabla_\lambda \text{ELBO}(\lambda_t),
$$

지수족 variational family에서는 $F_q^{-1} \nabla$가 **닫힌 형태**로 계산 가능 (Hoffman+ 2013의 SVI, Khan & Lin 2017의 Bayesian Learning Rule).

### 9.4 Meta-learning (MAML의 2차 항)

MAML의 gradient:

$$
\nabla_\theta L_{\text{meta}} = \nabla_\theta L(\theta - \alpha \nabla L) = (I - \alpha \nabla^2 L) \nabla L(\theta - \alpha \nabla L).
$$

$\nabla^2 L \approx F$로 근사하면 meta-gradient가 natural gradient flow와 관계된다 (Implicit MAML의 이론적 배경).

### 9.5 Second-order generalization: Shampoo, K-FAC, Fisher-SAM

- **K-FAC (Martens & Grosse 2015)**: 층별 block-diagonal Kronecker 근사 $F_\ell \approx A_\ell \otimes G_\ell$.
- **Shampoo (Gupta+ 2018)**: 각 텐서 축마다 preconditioner, Kronecker product 형태.
- **Fisher-SAM**: Sharpness-aware minimization의 Fisher 계량 확장.

모두 **정리 5.3의 해를 근사 계산**하는 방법이다.

---

## 10. 흔한 오해와 함정

1. **"Natural gradient는 그냥 $F^{-1}$ 전처리"가 아니다.**
   - 그건 결과일 뿐. 본질은 **KL ball 위 steepest descent**라는 제약 최적화 원리.

2. **$F$는 Hessian과 일반적으로 다르다.**
   - Exp family MLE에서만 $\nabla^2 L = F$. 일반 손실(예: cross-entropy의 feature map)에서는 다름.
   - Newton's method는 $H^{-1} \nabla L$, natural gradient는 $F^{-1} \nabla L$.

3. **Step size $\eta$는 trust region 반경 $\varepsilon$과 결합되어야 한다.**
   - 정리 5.3에서 $\eta^* = \varepsilon / \|\tilde{\nabla}L\|_F$. 단순히 $\eta$를 고정하면 trust region을 넘어설 수 있음.

4. **$F$ 추정의 편향.**
   - Empirical Fisher $\hat{F} = \frac{1}{N}\sum \nabla \log p_i \nabla \log p_i^T$는 $F = \mathbb{E}_p[\nabla \log p \nabla \log p^T]$의 편향된 추정치 (Kunstner+ 2019: "Limitations of the Empirical Fisher"). True Fisher는 $y \sim p_\theta$에서 샘플링 필요.

5. **Parameterization 불변성 ≠ 좌표 무관.**
   - Natural gradient의 **경로(trajectory)**는 좌표에 불변하지만, 수치 구현은 여전히 좌표를 고름. 그러나 다른 좌표를 골라도 같은 분포 경로로 수렴.

6. **$F$가 singular하면?**
   - Overparameterized NN에서 흔함. Damping $F + \delta I$ 혹은 pseudoinverse 사용. Martens & Grosse (2015)의 "damping" heuristic.

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **2D 증명**: $n = 2$, $F = \text{diag}(1, 4)$, $\nabla L = (1, 1)^T$, $\varepsilon = 1$일 때 $d\theta^*$와 $\eta^*$을 손으로 계산하라.

2. **Lagrangian 이중성**: 문제 5.1의 dual 문제를 쓰고, strong duality가 성립함을 Slater 조건으로 확인하라.

3. **Non-active constraint**: $\nabla L = 0$이면 KKT는 어떤 해를 주는가? 이것이 의미하는 바?

4. **좌표 변환 증명**: $\phi = A\theta$, $A$ 가역. 새 좌표에서 natural gradient가 $A^{-T} \tilde{\nabla}_\theta L$임을 보이고, 이동 방향 $A^{-1} d\theta^*_\phi = d\theta^*$임을 증명.

5. **KL ≠ Fisher norm (고차)**: $\text{KL}(p_\theta \| p_{\theta+d\theta}) = \frac{1}{2} d\theta^T F d\theta + \frac{1}{6} \sum T_{ijk} d\theta_i d\theta_j d\theta_k + O(d\theta^4)$의 3차 항을 Amari-Chentsov 텐서로 표현하라.

### 다음 단계

- **[03. Natural Gradient = KL Steepest Descent](./03-kl-steepest-descent.md)**: $\text{KL} \approx \frac{1}{2} d\theta^T F d\theta$의 엄밀한 2차 근사와 등가성 정리.
- **[04. Parameterization 불변성](./04-parameterization-invariance.md)**: 좌표 변환 하 natural gradient 경로의 완전 불변성 증명.
- **[05. K-FAC, Shampoo, 실전 구현](./05-kfac-shampoo.md)**: $F^{-1}$ 근사 기법들.

---

**참고문헌**

- Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*. Neural Computation 10(2).
- Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*. JMLR.
- Schulman, J., Levine, S., Moritz, P., Jordan, M., Abbeel, P. (2015). *Trust Region Policy Optimization*.
- Kunstner, F., Balles, L., Hennig, P. (2019). *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent*.
- Boyd, S., Vandenberghe, L. *Convex Optimization*, Ch. 5 (KKT).

---

[◀ 01. Euclidean Gradient의 문제](./01-euclidean-gradient-problem.md) | [📚 README](../README.md) | [03. KL Steepest Descent ▶](./03-kl-steepest-descent.md)
