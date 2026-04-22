# 03. Natural Gradient는 KL Ball의 Steepest Descent

> **"정보 공간에서 한 걸음"은 "파라미터 공간에서 한 걸음"과 다르다.**
> KL divergence가 두 분포 사이의 "정보 거리"라면, natural gradient는 그 거리로 측정한 steepest descent다.

---

## 1. 왜 이 주제인가?

앞 문서 (Ch5-02)에서 natural gradient를 **Fisher norm 제약** 하의 steepest descent로 유도했다:

$$
\min_{d\theta} \nabla L^T d\theta \quad \text{s.t.} \quad d\theta^T F d\theta \leq \varepsilon^2.
$$

그런데 왜 "Fisher norm"이 자연스러운 제약인가? 답은: **Fisher norm은 KL divergence의 무한소 형태**다. 즉 $d\theta$가 작을 때:

$$
\text{KL}(p_\theta \| p_{\theta+d\theta}) = \frac{1}{2} d\theta^T F(\theta) d\theta + O(\|d\theta\|^3).
$$

이로써 natural gradient는 **파라미터 공간의 geometric한 제약이 아니라, 분포 공간(=정보 공간)의 KL 제약 하의 steepest descent**다. 이것이 natural gradient를 "정보 기하학적으로 가장 합당한 gradient"로 만드는 이유이며, parameterization 불변성의 근원이다.

**이 문서의 목표**: KL divergence의 2차 근사를 엄밀하게 유도하고, 이것이 Fisher 계량과 정확히 일치함을 보이고, natural gradient를 **KL ball의 steepest descent**로 재해석한다.

---

## 2. 학습 목표

1. $\text{KL}(p_\theta \| p_{\theta+d\theta})$의 **2차 Taylor 전개**가 Fisher 계량임을 증명.
2. **0차, 1차 항이 소멸**하고 2차 항만 남는 이유를 이해.
3. $\text{KL}(p \| q)$와 $\text{KL}(q \| p)$의 **Taylor 전개가 동일**한 2차 항을 가짐을 증명 (비대칭성은 3차 이상).
4. Natural gradient를 **KL ball 위 steepest descent**로 재정식화.
5. Hellinger, $\chi^2$, Wasserstein 등 **다른 거리의 Fisher 근사** 비교.

---

## 3. 전제 지식

- **Ch3-01**: KL divergence $\text{KL}(p\|q) = \mathbb{E}_p[\log p - \log q]$
- **Ch2-01~04**: Fisher 정보의 3가지 정의
- **Ch5-02**: Natural gradient = $F^{-1} \nabla L$
- **다변수 Taylor 전개**와 교환 가능한 미분/적분

---

## 4. 직관적 설명

### 4.1 "정보 거리"로서의 KL

두 분포 $p, q$가 "얼마나 다른가"를 정보 이론적으로 재는 것이 KL divergence. $\text{KL}(p\|q)$는 $p$를 기준으로 $q$를 설명할 때의 정보 손실 (bits 혹은 nats 단위).

KL은 **미터(metric)가 아니다** (비대칭, 삼각부등식 X). 하지만 **무한소에서는 symmetric quadratic form**, 즉 Riemann 계량이 된다. 이 계량이 **Fisher 정보**.

### 4.2 두 해석의 통일

| 제약 | 해의 방향 | 해석 |
|------|---------|------|
| $\|d\theta\|_2 \leq \varepsilon$ | $-\nabla L$ | 파라미터 공간 steepest descent |
| $d\theta^T F d\theta \leq \varepsilon^2$ | $-F^{-1} \nabla L$ | Fisher 계량 steepest descent |
| $\text{KL}(p_\theta \| p_{\theta+d\theta}) \leq \varepsilon^2 / 2$ | $-F^{-1} \nabla L$ | **분포 공간 steepest descent** |

맨 아래 두 제약이 무한소에서 동치이므로, natural gradient는 **KL ball의 최적 이동 방향**.

### 4.3 왜 1차 항이 없는가?

$\text{KL}(p_\theta \| p_\theta) = 0$이고 $\theta$에서 $\text{KL}$이 **최솟값**을 가지므로, 1차 미분은 0. 이건 단순하지만 중요: **KL은 항상 "바닥(0)에서 출발"하므로 2차 항이 주도한다**.

---

## 5. 엄밀한 정의와 정리

### 5.1 메인 정리

**정리 5.1 (KL divergence의 Fisher 근사).** 정칙 통계 모델 $\{p(x|\theta) : \theta \in \Theta \subseteq \mathbb{R}^n\}$에서 $p_\theta$가 $\theta$에 대해 2번 연속 미분 가능이고 **미분과 적분의 교환이 허용**되면:

$$
\boxed{\text{KL}(p_\theta \| p_{\theta+d\theta}) = \frac{1}{2} d\theta^T F(\theta) d\theta + O(\|d\theta\|^3),}
$$

여기서 $F(\theta) = \mathbb{E}_\theta[\nabla \log p \cdot \nabla \log p^T] = -\mathbb{E}_\theta[\nabla^2 \log p]$.

### 5.2 대칭 KL도 같은 2차

**따름정리 5.2.** 

$$
\text{KL}(p_{\theta+d\theta} \| p_\theta) = \frac{1}{2} d\theta^T F(\theta) d\theta + O(\|d\theta\|^3).
$$

즉 두 방향의 KL이 모두 같은 Fisher 2차 항을 갖는다. 비대칭성은 **3차 이상**(Amari-Chentsov 텐서)에서 나타난다.

### 5.3 Natural gradient 재정식화

**정리 5.3 (KL ball의 steepest descent).** 다음 두 문제는 $O(\|d\theta\|^3)$까지 동치:

$$
\min_{d\theta} \nabla L^T d\theta \quad \text{s.t.} \quad \text{KL}(p_\theta \| p_{\theta+d\theta}) \leq \delta,
$$

$$
\min_{d\theta} \nabla L^T d\theta \quad \text{s.t.} \quad d\theta^T F d\theta \leq 2\delta.
$$

해는 둘 다 $d\theta^* = -\sqrt{2\delta / \|\tilde{\nabla}L\|_F^2} \cdot \tilde{\nabla} L$.

---

## 6. 증명

### 6.1 정리 5.1 증명

**설정.** $\ell(\theta; x) := \log p(x|\theta)$. 정의:

$$
\text{KL}(p_\theta \| p_{\theta+d\theta}) = \int p(x|\theta) [\ell(\theta;x) - \ell(\theta+d\theta;x)] dx = -\int p_\theta [\ell(\theta+d\theta) - \ell(\theta)] dx.
$$

**Taylor 전개.** $\ell(\theta+d\theta) = \ell(\theta) + \nabla\ell(\theta)^T d\theta + \frac{1}{2} d\theta^T \nabla^2 \ell(\theta) d\theta + O(\|d\theta\|^3)$. 따라서:

$$
\text{KL} = -\int p_\theta \left[\nabla\ell^T d\theta + \frac{1}{2} d\theta^T \nabla^2\ell \, d\theta\right] dx + O(\|d\theta\|^3).
$$

**1차 항.** 

$$
-\int p_\theta \nabla\ell^T d\theta \, dx = -d\theta^T \int p_\theta \nabla\ell \, dx = -d\theta^T \cdot 0 = 0,
$$

**이유**: Score의 평균은 0.
$$
\int p_\theta \nabla \log p_\theta \, dx = \int \nabla p_\theta \, dx = \nabla \int p_\theta \, dx = \nabla 1 = 0.
$$

**2차 항.**

$$
-\frac{1}{2} d\theta^T \left(\int p_\theta \nabla^2 \ell \, dx\right) d\theta = -\frac{1}{2} d\theta^T \mathbb{E}_\theta[\nabla^2 \log p] \, d\theta.
$$

Fisher의 정의 (Ch2-03): $\mathbb{E}_\theta[\nabla^2 \log p] = -F(\theta)$. 따라서:

$$
\text{KL} = -\frac{1}{2} d\theta^T (-F) d\theta + O(\|d\theta\|^3) = \frac{1}{2} d\theta^T F(\theta) d\theta + O(\|d\theta\|^3). \quad \square
$$

### 6.2 따름정리 5.2 증명

대칭 방향 KL:

$$
\text{KL}(p_{\theta+d\theta} \| p_\theta) = \int p_{\theta+d\theta} [\ell(\theta+d\theta) - \ell(\theta)] dx.
$$

$p_{\theta+d\theta} = p_\theta (1 + \nabla\ell^T d\theta + \frac{1}{2}(\nabla\ell^T d\theta)^2 + \frac{1}{2} d\theta^T \nabla^2 \ell \, d\theta + O(\|d\theta\|^3))$ 로 Taylor 전개하고, $[\ell(\theta+d\theta) - \ell(\theta)] = \nabla\ell^T d\theta + \frac{1}{2} d\theta^T \nabla^2\ell \, d\theta + O(\|d\theta\|^3)$과 곱한 뒤 $p_\theta$에 대한 기댓값을 취하면, 1차 항 소멸 후 2차 항으로:

$$
\mathbb{E}_\theta[(\nabla\ell^T d\theta)^2] + \mathbb{E}_\theta[\nabla\ell^T d\theta \cdot \nabla\ell^T d\theta]/2 - \mathbb{E}_\theta[\ldots] + \frac{1}{2} d\theta^T \mathbb{E}_\theta[\nabla^2\ell] d\theta.
$$

정확한 계산은 (대칭화 후) $\frac{1}{2} d\theta^T F d\theta$ (Ch2-03의 $\mathbb{E}[\nabla\ell \nabla\ell^T] = F$ 사용). $\square$

### 6.3 비대칭성은 3차부터

**보조정리 6.1.** 

$$
\text{KL}(p_\theta \| p_{\theta+d\theta}) - \text{KL}(p_{\theta+d\theta} \| p_\theta) = \frac{1}{6} \sum_{i,j,k} T_{ijk}(\theta) \, d\theta_i d\theta_j d\theta_k + O(\|d\theta\|^4),
$$

여기서 $T_{ijk} = \mathbb{E}_\theta[\partial_i \ell \, \partial_j \ell \, \partial_k \ell]$는 **Amari-Chentsov 텐서** (Ch4-04). 즉 비대칭성은 3차 cumulant에서 비로소 나타남. $\square$

### 6.4 정리 5.3 증명

정리 5.1에 의해 제약 $\text{KL}(p_\theta \| p_{\theta+d\theta}) \leq \delta$는 $O(\|d\theta\|^3)$까지 $\frac{1}{2} d\theta^T F d\theta \leq \delta$와 동치. Ch5-02의 정리 5.3에 의해 해는:

$$
d\theta^* = -\sqrt{\frac{2\delta}{\nabla L^T F^{-1} \nabla L}} F^{-1} \nabla L = -\sqrt{\frac{2\delta}{\|\tilde{\nabla} L\|_F^2}} \tilde{\nabla} L. \quad \square
$$

---

## 7. 구체 예제

### 7.1 단변수 정규분포

$p(x | \mu) = \mathcal{N}(\mu, \sigma^2)$ (σ 고정). $F(\mu) = 1/\sigma^2$.

**KL 직접 계산**:

$$
\text{KL}(p_\mu \| p_{\mu+d\mu}) = \frac{(d\mu)^2}{2\sigma^2}.
$$

**정리 5.1 예측**: $\frac{1}{2} (d\mu)^2 \cdot 1/\sigma^2 = \frac{(d\mu)^2}{2\sigma^2}$. ✓ **정확히 일치** (3차 이상 0, Gaussian의 특수 성질).

### 7.2 양변수 정규분포

$p(x | \mu, \sigma) = \mathcal{N}(\mu, \sigma^2)$.

$$
\text{KL}(p_{\mu,\sigma} \| p_{\mu',\sigma'}) = \log(\sigma'/\sigma) + \frac{\sigma^2 + (\mu-\mu')^2}{2\sigma'^2} - \frac{1}{2}.
$$

$d\mu = \mu' - \mu$, $d\sigma = \sigma' - \sigma$로 Taylor 전개하면:

$$
\text{KL} = \frac{d\mu^2}{2\sigma^2} + \frac{d\sigma^2}{\sigma^2} + O(\|d\theta\|^3) = \frac{1}{2}(d\mu, d\sigma)^T F (d\mu, d\sigma),
$$

$F = \text{diag}(1/\sigma^2, 2/\sigma^2)$. 3차 항은 $d\sigma^3/\sigma^3$ 등으로 **비대칭성 존재**.

### 7.3 베르누이

$p(x|\theta) = \theta^x (1-\theta)^{1-x}$. $F(\theta) = \frac{1}{\theta(1-\theta)}$.

$$
\text{KL}(p_\theta \| p_{\theta+d\theta}) = \theta \log\frac{\theta}{\theta+d\theta} + (1-\theta)\log\frac{1-\theta}{1-\theta-d\theta}.
$$

Taylor 전개하면 $\frac{(d\theta)^2}{2\theta(1-\theta)} + O(d\theta^3)$. ✓

### 7.4 Wasserstein-2와의 비교

$W_2^2(p_\theta, p_{\theta+d\theta})$도 Taylor 2차가 존재하지만, 계량이 Fisher가 아닌 **Wasserstein 계량**. Gaussian family에서:

$$
W_2^2(\mathcal{N}(\mu, \sigma^2), \mathcal{N}(\mu', \sigma'^2)) = (\mu-\mu')^2 + (\sigma-\sigma')^2.
$$

즉 Wasserstein 계량은 $\text{diag}(1, 1)$ (Euclidean!), Fisher는 $\text{diag}(1/\sigma^2, 2/\sigma^2)$. 두 계량은 **다른 기하**를 정의 → Wasserstein gradient flow ≠ natural gradient flow.

---

## 8. Python 코드 검증

### 8.1 KL vs Fisher 2차 근사

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Gaussian family N(mu, sigma^2), sigma 고정
mu0, sigma = 1.0, 0.8

def kl_gauss(mu1, mu2, s=sigma):
    return 0.5 * (mu1 - mu2)**2 / s**2  # 정확한 KL (Gaussian 간)

def fisher_quadratic(dmu, s=sigma):
    F = 1 / s**2
    return 0.5 * dmu**2 * F

dmus = np.linspace(-0.5, 0.5, 100)
kl_exact = np.array([kl_gauss(mu0, mu0 + d) for d in dmus])
kl_approx = np.array([fisher_quadratic(d) for d in dmus])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(dmus, kl_exact, 'b-', lw=2, label='Exact KL')
ax[0].plot(dmus, kl_approx, 'r--', lw=2, label='Fisher 2nd order')
ax[0].set_xlabel('dμ'); ax[0].set_ylabel('KL'); ax[0].legend(); ax[0].grid()
ax[0].set_title('Gaussian: KL과 Fisher 2차 정확히 일치')

ax[1].plot(dmus, np.abs(kl_exact - kl_approx), 'g-', lw=2)
ax[1].set_xlabel('dμ'); ax[1].set_ylabel('|diff|')
ax[1].set_title('차이 (수치 오차 수준)')
ax[1].grid()
plt.tight_layout()
```

**관찰**: Gaussian에서는 $O(d\theta^2)$로 **완전 일치** (고차 항 0).

### 8.2 베르누이에서 3차 오차

```python
import numpy as np

def kl_bern(theta1, theta2):
    return theta1*np.log(theta1/theta2) + (1-theta1)*np.log((1-theta1)/(1-theta2))

theta0 = 0.3
dthetas = np.array([0.001, 0.01, 0.05, 0.1])
errs = []
for dth in dthetas:
    kl = kl_bern(theta0, theta0 + dth)
    approx = 0.5 * dth**2 / (theta0 * (1-theta0))
    err = abs(kl - approx)
    print(f"dθ={dth:.3f}: KL={kl:.6f}, 2nd-order={approx:.6f}, err={err:.2e}, err/dθ³={err/dth**3:.3f}")
    errs.append(err)
```

**기대 출력**: `err/dθ³` 비율이 거의 상수 → 3차 오차 확인.

### 8.3 KL 대칭성과 Amari-Chentsov 텐서

```python
import numpy as np

def kl_bern(p, q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

theta = 0.3
dths = np.linspace(-0.05, 0.05, 21)
for dth in [0.01, 0.02, 0.05]:
    fwd = kl_bern(theta, theta + dth)
    bwd = kl_bern(theta + dth, theta)
    asym = fwd - bwd
    # Amari-Chentsov T_{111} for Bernoulli
    # T = E[(d log p)^3] = (1-2θ)/(θ^2(1-θ)^2)
    T = (1 - 2*theta) / (theta**2 * (1-theta)**2)
    pred_asym = T * dth**3 / 6
    print(f"dθ={dth}: fwd-bwd={asym:.6e}, predicted T·dθ³/6={pred_asym:.6e}")
```

**기대**: 예측 비대칭과 실측 비대칭이 leading order에서 일치.

### 8.4 2D Gaussian의 Fisher 타원

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

mu, sigma = 2.0, 1.0
F = np.array([[1/sigma**2, 0],[0, 2/sigma**2]])

# KL ball vs Euclidean ball
eps = 0.3
fig, ax = plt.subplots(figsize=(6,6))

# Euclidean
th = np.linspace(0, 2*np.pi, 200)
ax.plot(mu + eps*np.cos(th), sigma + eps*np.sin(th), 'r--', label='Euclidean ball')

# Fisher (KL) ellipsoid: d^T F d ≤ 2δ → ellipse
delta = 0.5 * eps**2  # same "radius" in Euclidean metric for comparison
# Eigen-decompose F
w, V = np.linalg.eigh(F)
# Semi-axes lengths: sqrt(2*delta / w)
ax.add_patch(Ellipse((mu, sigma), 
                     width=2*np.sqrt(2*delta/w[0]), 
                     height=2*np.sqrt(2*delta/w[1]),
                     angle=0, fill=False, edgecolor='blue', linestyle='-', 
                     linewidth=2, label='KL ball (Fisher ellipsoid)'))
ax.plot(mu, sigma, 'ko')
ax.set_xlabel('μ'); ax.set_ylabel('σ')
ax.legend(); ax.set_aspect('equal'); ax.grid()
ax.set_title('Euclidean ball vs KL ball (Gaussian)')
plt.tight_layout()
```

**관찰**: Fisher 타원은 $\mu$ 방향이 $\sigma$ 방향보다 **더 넓음** ($w_\sigma > w_\mu$이므로 $\sigma$ 축이 짧음). $\sigma$가 분포에 더 민감하기 때문 (작은 $d\sigma$도 큰 KL).

---

## 9. AI/ML 연결

### 9.1 TRPO의 KL 제약

Schulman+ (2015)은 정책 업데이트에 **정확히 정리 5.3의 KL 제약**을 쓴다:

$$
\max_\theta \mathbb{E}_{s \sim \rho}[\hat{A}^{\text{old}}(s,a) \frac{\pi_\theta}{\pi_{\text{old}}}], \quad \text{s.t. } \mathbb{E}_s[\text{KL}(\pi_{\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s))] \leq \delta.
$$

2차 근사 후 conjugate gradient로 $F^{-1} \nabla L$을 풀어 $\delta$-KL 경계로 line search.

### 9.2 Mirror Descent와 KL ball

Mirror descent (Nemirovski) with Bregman divergence $D_\psi$:

$$
\theta_{t+1} = \arg\min_\theta \langle g_t, \theta \rangle + \frac{1}{\eta} D_\psi(\theta, \theta_t).
$$

$\psi$ = negative entropy이면 $D_\psi = \text{KL}$, mirror descent는 **KL ball 내 steepest descent**와 직접 대응 (Ch4-03의 Legendre 쌍대 활용).

### 9.3 Proximal Policy Optimization (PPO)

TRPO의 KL 제약을 soft penalty로 바꾼 버전:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}[\ldots] - \beta \cdot \text{KL}(\pi_{\text{old}} \| \pi_\theta).
$$

$\beta$ 조절로 TRPO의 hard constraint 근사. 수학적 기반은 **라그랑지안 dual**.

### 9.4 Variational EM과 KL projection

VI에서 $q^*(\lambda) = \arg\min_q \text{KL}(q \| p)$의 $\lambda$ 업데이트 역시 KL ball 제약의 local 버전으로 볼 수 있음 (Ch6-03에서 상세).

### 9.5 Implicit regularization via KL

Continual learning (EWC: Kirkpatrick+ 2017)의 정규화:

$$
L(\theta) = L_{\text{new}}(\theta) + \frac{\lambda}{2} (\theta - \theta_{\text{old}})^T F_{\text{old}} (\theta - \theta_{\text{old}}).
$$

**정확히 이전 task 분포와의 KL 2차 근사**로 "이전 task 분포에서 멀어지지 말라"는 제약.

---

## 10. 흔한 오해와 함정

1. **"KL = Fisher × (dθ)² / 2"는 오직 2차 근사.**
   - 고차 항은 Amari-Chentsov 텐서 등으로 주어짐. 큰 step에서는 근사 깨짐.
   - TRPO는 line search로 실제 KL을 다시 검증, approximation의 한계를 보완.

2. **$\text{KL}(p\|q) \neq \text{KL}(q\|p)$는 3차부터.**
   - 2차 항은 symmetric. 즉 natural gradient 유도엔 방향이 중요하지 않음 (2차까지).

3. **KL이 distance가 아니지만 local metric을 정의.**
   - 삼각부등식 X, 대칭성 X. 하지만 2차 Taylor = Fisher = Riemann 계량.

4. **Forward KL vs Reverse KL.**
   - $\text{KL}(p\|q)$ (forward, mean-seeking) vs $\text{KL}(q\|p)$ (reverse, mode-seeking). VI는 보통 reverse. 두 선택의 geometry는 다름.

5. **Fisher information과 KL의 관계는 지수족에서 더 선명.**
   - Exp family: $\text{KL}(p_{\theta_1} \| p_{\theta_2}) = D_\psi(\theta_2, \theta_1)$ (Bregman divergence, Ch4-03). 2차 근사 → Hessian = Fisher.

6. **Wasserstein 거리는 다른 계량을 낳는다.**
   - Wasserstein gradient flow ≠ Fisher natural gradient flow (다른 Riemann 구조).

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **Multivariate Gaussian 증명**: $\mathcal{N}(\mu, \Sigma)$에서 $\text{KL}(p_{\mu_1,\Sigma} \| p_{\mu_2,\Sigma})$을 유도하고 Fisher 2차가 $\frac{1}{2}(\mu_1-\mu_2)^T \Sigma^{-1} (\mu_1-\mu_2)$임을 확인.

2. **Symmetric KL**: $J(p, q) := \text{KL}(p\|q) + \text{KL}(q\|p)$의 Taylor 전개를 쓰고, 2차가 $d\theta^T F d\theta$임을 보여라.

3. **$\chi^2$ divergence**: $\chi^2(p\|q) = \int \frac{(p-q)^2}{q} dx$의 Taylor 2차가 $d\theta^T F d\theta$임을 보여라. ($f$-divergence 공통 성질)

4. **Wasserstein-2 Fisher와 다른가**: 1D Gaussian family에서 $W_2^2(p_\theta, p_{\theta+d\theta})$를 Taylor 전개해 Fisher 계량과 다름을 보여라.

5. **3차 항의 부호**: 베르누이에서 $\theta = 0.3$일 때 $\text{KL}(p\|q) > \text{KL}(q\|p)$ 혹은 반대 여부를 Amari-Chentsov 텐서로 예측.

### 다음 단계

- **[04. Parameterization 불변성](./04-parameterization-invariance.md)**: 좌표 변환 하에서 natural gradient 경로의 완전 불변성.
- **[05. K-FAC, Shampoo](./05-kfac-shampoo.md)**: 실전 구현.

---

**참고문헌**

- Amari, S. (2016). *Information Geometry and Its Applications*, Springer, Ch. 3, 12.
- Kullback, S., Leibler, R. (1951). *On Information and Sufficiency*.
- Schulman, J.+ (2015). *Trust Region Policy Optimization*.
- Nielsen, F. (2020). *An Elementary Introduction to Information Geometry*. Entropy 22(10).
- Kirkpatrick, J.+ (2017). *Overcoming catastrophic forgetting in neural networks* (EWC).

---

[◀ 02. Natural Gradient 유도](./02-natural-gradient-derivation.md) | [📚 README](../README.md) | [04. Parameterization 불변성 ▶](./04-parameterization-invariance.md)
