# 04. Natural Gradient의 Parameterization 불변성

> **"자연 경사는 좌표를 묻지 않는다. 그것이 흐르는 곳은 분포 공간이지 파라미터 공간이 아니다."**

---

## 1. 왜 이 주제인가?

Ch5-01에서 유클리드 gradient descent는 좌표계에 따라 **서로 다른 경로**를 그린다는 문제를 제기했다. $\mathcal{N}(\mu, \sigma^2)$에서 $\sigma$ vs $\log \sigma$ 좌표를 택하면 같은 초기 분포에서 시작해도 다른 분포로 수렴한다. 이는 "gradient descent의 경로가 임의의 파라미터 선택에 의존"하는 철학적/실용적 문제다.

**Natural gradient의 핵심 주장**: 분포 공간(정보 공간)에서 본 경로는 **좌표 선택에 무관**하다. 즉 $\theta$ 좌표에서 NGD를 돌리나 $\phi = \phi(\theta)$ 좌표에서 NGD를 돌리나 **같은 분포 궤적**을 그린다.

이 문서에서는 그 증명을 엄밀하게 제시한다. Fisher 계량의 **텐서 변환 법칙**, natural gradient의 **공변적(covariant) 성질**, 그리고 연속 시간 flow의 불변성을 차례로 증명한다.

---

## 2. 학습 목표

1. Fisher 계량의 **공변 텐서 변환 법칙**을 유도.
2. Gradient는 **공변 벡터(1-form)**, natural gradient는 **반변 벡터(contravariant)**임을 구별.
3. $\theta \to \phi$ 변환 하에서 $\tilde{\nabla} L$이 **반변 텐서 법칙**을 따름을 증명.
4. 연속 시간 natural gradient flow $\dot\theta = -F^{-1}\nabla L$이 좌표 독립적인 분포 경로를 그림을 증명.
5. 이산 NGD 알고리즘도 1차 오차까지 불변임을 보임.

---

## 3. 전제 지식

- **Ch1-01, 02**: 다양체, 접공간, 좌표 변환
- **Ch1-04**: Riemann 계량의 텐서 변환
- **Ch2-05**: Fisher 계량의 공변성
- **Ch5-02**: Natural gradient 정의

---

## 4. 직관적 설명

### 4.1 좌표 변환의 기하학

$\theta$ 좌표와 $\phi$ 좌표, $\phi = \phi(\theta)$, Jacobian $J = \partial\phi/\partial\theta$.

- **함수 $L$**: 스칼라. $L(\theta) = L(\phi(\theta))$ (같은 분포에 같은 손실).
- **Gradient $\nabla L$**: 1-form (공변). $\nabla_\theta L = J^T \nabla_\phi L$.
- **접벡터 $d\theta$**: 반변 벡터. $d\phi = J \, d\theta$.
- **Fisher 계량 $F$**: (0,2) 텐서. $F_\theta = J^T F_\phi J$, 즉 $F_\phi = J^{-T} F_\theta J^{-1}$.

### 4.2 왜 유클리드는 깨지나

**Claim.** $\nabla L$ 자체는 반변 벡터가 아니다.

유클리드 gradient descent $d\theta = -\eta \nabla_\theta L$은 **공변 벡터를 반변 벡터처럼** 취급한다 (방향으로 씀). 이게 파라미터 의존성의 수학적 원인. 

$\theta$ 좌표에서 $d\theta = -\eta \nabla_\theta L$로 움직이면, $\phi$ 좌표에서는 $d\phi = J \, d\theta = -\eta J \nabla_\theta L = -\eta J J^T \nabla_\phi L \neq -\eta \nabla_\phi L$ (일반적으로 $J J^T \neq I$).

### 4.3 왜 natural gradient는 살아남나

Natural gradient $\tilde{\nabla} L = F^{-1} \nabla L$: $F^{-1}$이 공변을 반변으로 **옮긴다**(metric raising). 즉:

- $\nabla L$: 공변 → Jacobian 역변환 필요.
- $F^{-1}$: (2,0) 텐서 → 공변을 반변으로 변환.
- $F^{-1} \nabla L$: 순수 반변 벡터 → **$d\theta$와 같은 변환 법칙**.

결과: $\tilde{\nabla}_\phi L = J \tilde{\nabla}_\theta L$ — 올바른 반변 변환. 그러므로 방향으로 쓸 때 좌표 독립적.

---

## 5. 엄밀한 정의와 정리

### 5.1 텐서 변환 법칙

**법칙 5.1.** 좌표 변환 $\phi = \phi(\theta)$, $J = \partial\phi/\partial\theta$ (전역 가역, 매끄러움).

- **스칼라 $f$**: $f_\phi(\phi) = f_\theta(\theta)$.
- **1-form (공변 벡터)** $\omega_i$: $\omega_\phi^i = (J^{-T})^i_j \omega_\theta^j$.
- **벡터 (반변)** $X^i$: $X_\phi^i = J^i_j X_\theta^j$.
- **(0,2) 텐서** $g_{ij}$: $g_\phi = J^{-T} g_\theta J^{-1}$.
- **(2,0) 텐서** $g^{ij}$: $g_\phi^{-1} = J g_\theta^{-1} J^T$.

### 5.2 Fisher의 변환

**정리 5.2 (Fisher 공변성).** 

$$
F_\phi(\phi) = J^{-T} F_\theta(\theta) J^{-1}, \quad F_\phi^{-1} = J F_\theta^{-1} J^T.
$$

### 5.3 Gradient의 변환

**정리 5.3 (Gradient는 1-form).**

$$
\nabla_\phi L(\phi) = J^{-T} \nabla_\theta L(\theta).
$$

### 5.4 메인 정리: Natural Gradient의 반변성

**정리 5.4 (Natural Gradient 공변성; Amari 1998).**

$$
\boxed{\tilde{\nabla}_\phi L(\phi) = J \cdot \tilde{\nabla}_\theta L(\theta).}
$$

즉 natural gradient는 **순수 반변 벡터**처럼 변환된다.

### 5.5 Flow의 불변성

**정리 5.5 (Flow 불변성).** 연속 시간 natural gradient flow 

$$
\dot\theta = -F_\theta^{-1} \nabla_\theta L
$$

를 $\theta$ 좌표에서 풀면, $\phi$ 좌표로 변환한 궤적 $\phi(t) = \phi(\theta(t))$는 **$\phi$ 좌표의 natural gradient flow**:

$$
\dot\phi = -F_\phi^{-1} \nabla_\phi L.
$$

따라서 두 좌표계에서 시작점이 같은 분포면, **모든 시간 $t$에 대해 같은 분포**에 머문다.

---

## 6. 증명

### 6.1 정리 5.2 (Fisher 공변) 증명

Score 변환: $s_\phi(\phi) = \partial_\phi \log p = \partial_\phi \theta \cdot \partial_\theta \log p = J^{-1} s_\theta$ (since $\theta = \theta(\phi)$, $\partial\theta/\partial\phi = J^{-1}$).

**올바른 방향**: $\phi = \phi(\theta)$이면 $\theta = \theta(\phi)$, $\partial\theta/\partial\phi = (\partial\phi/\partial\theta)^{-1} = J^{-1}$. 따라서:

$$
s_\phi = \nabla_\phi \log p = (\partial\theta/\partial\phi)^T \nabla_\theta \log p = (J^{-1})^T s_\theta = J^{-T} s_\theta.
$$

Fisher:

$$
F_\phi = \mathbb{E}[s_\phi s_\phi^T] = \mathbb{E}[J^{-T} s_\theta s_\theta^T J^{-1}] = J^{-T} \mathbb{E}[s_\theta s_\theta^T] J^{-1} = J^{-T} F_\theta J^{-1}. \quad \square
$$

역행렬:

$$
F_\phi^{-1} = (J^{-T} F_\theta J^{-1})^{-1} = J F_\theta^{-1} J^T. \quad \square
$$

### 6.2 정리 5.3 (Gradient 1-form) 증명

$L(\phi) = L(\theta(\phi))$ (같은 분포에 같은 손실). Chain rule:

$$
\nabla_\phi L = (\partial\theta/\partial\phi)^T \nabla_\theta L = J^{-T} \nabla_\theta L. \quad \square
$$

### 6.3 정리 5.4 (Natural Gradient 반변) 증명

정리 5.2와 5.3 결합:

$$
\tilde{\nabla}_\phi L = F_\phi^{-1} \nabla_\phi L = (J F_\theta^{-1} J^T)(J^{-T} \nabla_\theta L) = J F_\theta^{-1} \nabla_\theta L = J \tilde{\nabla}_\theta L. \quad \square
$$

이것은 반변 벡터 변환 법칙 $X_\phi = J X_\theta$와 정확히 일치.

### 6.4 정리 5.5 (Flow 불변) 증명

$\phi(t) := \phi(\theta(t))$. 시간 미분:

$$
\dot\phi = J \dot\theta = J (-F_\theta^{-1} \nabla_\theta L) = -(J F_\theta^{-1}) \nabla_\theta L.
$$

정리 5.4에 의해 $J F_\theta^{-1} \nabla_\theta L = F_\phi^{-1} \nabla_\phi L \cdot J^T J^{-T} \cdot \ldots$ 더 단순하게:

$$
\dot\phi = J \tilde{\nabla}_\theta L \cdot (-1) = -\tilde{\nabla}_\phi L = -F_\phi^{-1} \nabla_\phi L. \quad \square
$$

즉 flow 방정식이 **좌표 동변적 형태**를 유지. 미분방정식의 유일성(Picard-Lindelöf)에 의해 두 좌표에서 푼 경로가 동일한 분포 경로를 산출.

### 6.5 이산 NGD의 1차 불변성

이산 업데이트 $\theta_{t+1} = \theta_t - \eta \tilde{\nabla}_\theta L$과 $\phi_{t+1} = \phi_t - \eta \tilde{\nabla}_\phi L$.

$\phi_t = \phi(\theta_t)$에서 출발, 한 step 후:

$$
\phi_{t+1} = \phi_t - \eta \tilde{\nabla}_\phi L = \phi(\theta_t) - \eta J \tilde{\nabla}_\theta L.
$$

$\phi(\theta_{t+1}) = \phi(\theta_t - \eta \tilde{\nabla}_\theta L) = \phi(\theta_t) - \eta J \tilde{\nabla}_\theta L + O(\eta^2)$.

따라서:

$$
\phi_{t+1} = \phi(\theta_{t+1}) + O(\eta^2).
$$

**결론**: 이산 NGD는 **$O(\eta)$ 정확도로 불변**, 연속 flow로 갈수록 정확해짐. $\square$

### 6.6 유클리드 gradient가 불변이려면

$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$. $\phi$ 좌표에서 이 경로가 $\phi_{t+1} = \phi_t - \eta \nabla_\phi L$이려면:

$$
J \nabla_\theta L = \nabla_\phi L = J^{-T} \nabla_\theta L \Rightarrow J = J^{-T} \Rightarrow J^T J = I,
$$

즉 Jacobian이 **직교(orthogonal)**여야만. 일반적 좌표 변환에서는 성립 X. 이것이 유클리드 GD의 parameterization 의존성의 기하학적 증명.

---

## 7. 구체 예제

### 7.1 정규분포: σ vs log σ

$\theta_1 = (\mu, \sigma)$, $\theta_2 = (\mu, \ell)$, $\ell = \log \sigma$, $\sigma = e^\ell$. Jacobian:

$$
J = \partial\theta_2/\partial\theta_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1/\sigma \end{pmatrix}, \quad J^{-1} = \begin{pmatrix} 1 & 0 \\ 0 & \sigma \end{pmatrix}.
$$

**Fisher ($\sigma$ 좌표)**: $F_1 = \text{diag}(1/\sigma^2, 2/\sigma^2)$.

**Fisher ($\ell$ 좌표)**: $F_2 = J^{-T} F_1 J^{-1} = \text{diag}(1/\sigma^2, \sigma \cdot (2/\sigma^2) \cdot \sigma) = \text{diag}(1/\sigma^2, 2)$.

**Gradient 변환**: $\nabla_{(\mu,\ell)} L = J^{-T} \nabla_{(\mu,\sigma)} L = \text{diag}(1, \sigma) \nabla_{(\mu,\sigma)} L$.

**Natural gradient ($\sigma$)**: $\tilde{\nabla}_1 = F_1^{-1} \nabla_1 L = \text{diag}(\sigma^2, \sigma^2/2) \nabla_1 L$.

**Natural gradient ($\ell$)**: $\tilde{\nabla}_2 = F_2^{-1} \nabla_2 L = \text{diag}(\sigma^2, 1/2) \cdot \text{diag}(1, \sigma) \nabla_1 L = \text{diag}(\sigma^2, \sigma/2) \nabla_1 L$.

**정리 5.4 검증**: $J \tilde{\nabla}_1 = \text{diag}(1, 1/\sigma) \text{diag}(\sigma^2, \sigma^2/2) \nabla_1 L = \text{diag}(\sigma^2, \sigma/2) \nabla_1 L = \tilde{\nabla}_2$. ✓

### 7.2 베르누이: θ vs logit

$\theta$ 좌표 vs $\eta = \log(\theta/(1-\theta))$ (logit). $\theta = \sigma(\eta) = 1/(1+e^{-\eta})$, $d\theta/d\eta = \theta(1-\theta)$, 따라서 Jacobian (1차원):

$$
J = d\eta/d\theta = 1/(\theta(1-\theta)), \quad J^{-1} = \theta(1-\theta).
$$

**Fisher ($\theta$)**: $F_\theta = 1/(\theta(1-\theta))$.

**Fisher ($\eta$)**: $F_\eta = J^{-2} F_\theta = \theta^2(1-\theta)^2 \cdot 1/(\theta(1-\theta)) = \theta(1-\theta)$.

**Natural gradient ($\theta$)**: $\tilde{\nabla}_\theta = \theta(1-\theta) \cdot \nabla_\theta L$.

**Natural gradient ($\eta$)**: $\tilde{\nabla}_\eta = 1/(\theta(1-\theta)) \cdot \nabla_\eta L = 1/(\theta(1-\theta)) \cdot \theta(1-\theta) \nabla_\theta L = \nabla_\theta L$.

**정리 5.4 검증**: $J \tilde{\nabla}_\theta = 1/(\theta(1-\theta)) \cdot \theta(1-\theta) \nabla_\theta L = \nabla_\theta L = \tilde{\nabla}_\eta$. ✓

### 7.3 기하학적 해석: Exponential family canonical

Exp family에서 $\theta$(canonical)와 $\eta = \nabla\psi$(expectation)는 Legendre 쌍대 (Ch4-03). Jacobian $J = \partial\eta/\partial\theta = \nabla^2 \psi = F(\theta)$.

**Natural gradient ($\theta$)**: $\tilde{\nabla}_\theta = F^{-1} \nabla_\theta L$.

**Natural gradient ($\eta$)**: $\tilde{\nabla}_\eta = J \tilde{\nabla}_\theta = F \cdot F^{-1} \nabla_\theta L = \nabla_\theta L$. 흥미롭게도 $\eta$ 좌표에서 natural gradient는 $\theta$ 좌표의 **유클리드 gradient**. 이것이 e-평탄 × m-평탄 쌍대의 결과.

---

## 8. Python 코드 검증

### 8.1 Gaussian에서 σ vs log σ 궤적 비교

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.normal(loc=3.0, scale=2.0, size=500)

def neg_log_lik(mu, sigma):
    return 0.5*np.log(2*np.pi) + np.log(sigma) + 0.5*np.mean(((X - mu)/sigma)**2)

def grad_musigma(mu, sigma):
    return np.array([-np.mean(X-mu)/sigma**2, 1/sigma - np.mean((X-mu)**2)/sigma**3])

def grad_mul(mu, l):  # ℓ = log σ
    sigma = np.exp(l)
    g = grad_musigma(mu, sigma)
    # Chain rule: ∂L/∂ℓ = ∂L/∂σ * dσ/dℓ = σ * ∂L/∂σ
    return np.array([g[0], sigma * g[1]])

def fisher_musigma(mu, sigma):
    return np.array([[1/sigma**2, 0],[0, 2/sigma**2]])

def fisher_mul(mu, l):
    sigma = np.exp(l)
    return np.array([[1/sigma**2, 0],[0, 2]])  # σ² · (2/σ²) = 2

# 시작: μ=0, σ=1 (ℓ=0)
theta_sigma = np.array([0.0, 1.0])
theta_logsigma = np.array([0.0, 0.0])  # ℓ=0 ↔ σ=1
lr = 0.2
path_s, path_l = [theta_sigma.copy()], [theta_logsigma.copy()]

for _ in range(150):
    # σ 좌표 NGD
    mu, sigma = theta_sigma
    g = grad_musigma(mu, sigma)
    Fi = np.linalg.inv(fisher_musigma(mu, sigma))
    theta_sigma = theta_sigma - lr * (Fi @ g)
    theta_sigma[1] = max(theta_sigma[1], 0.01)
    path_s.append(theta_sigma.copy())
    
    # ℓ 좌표 NGD  
    mu, l = theta_logsigma
    g = grad_mul(mu, l)
    Fi = np.linalg.inv(fisher_mul(mu, l))
    theta_logsigma = theta_logsigma - lr * (Fi @ g)
    path_l.append(theta_logsigma.copy())

path_s = np.array(path_s)
path_l = np.array(path_l)
path_l_as_sigma = np.stack([path_l[:,0], np.exp(path_l[:,1])], axis=1)  # ℓ→σ 변환

# Plot in (μ, σ) space
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(path_s[:,0], path_s[:,1], 'b.-', label='NGD in (μ, σ)', ms=3)
ax.plot(path_l_as_sigma[:,0], path_l_as_sigma[:,1], 'r.--', label='NGD in (μ, ℓ=log σ) → σ', ms=3)
ax.plot(3, 2, 'k*', ms=15, label='True')
ax.set_xlabel('μ'); ax.set_ylabel('σ'); ax.legend(); ax.grid()
ax.set_title('Natural Gradient: 좌표 선택에 무관한 궤적 (Ch5-01 유클리드와 대비)')
plt.tight_layout()

# 수치 차이
print(f"최종 (μ,σ) 좌표 NGD 결과:         ({path_s[-1,0]:.4f}, {path_s[-1,1]:.4f})")
print(f"최종 (μ,ℓ) 좌표 NGD 결과 (→σ):    ({path_l_as_sigma[-1,0]:.4f}, {path_l_as_sigma[-1,1]:.4f})")
print(f"MLE: μ={X.mean():.4f}, σ={X.std():.4f}")
```

**기대 결과**: 두 경로는 **분포 공간에서 거의 동일** (step size $\eta$의 고차 오차는 있음). Ch5-01에서 유클리드 GD는 서로 다른 경로를 그렸던 것과 대비.

### 8.2 베르누이: θ vs logit 검증

```python
import numpy as np

# 데이터
np.random.seed(42)
X = np.random.binomial(1, 0.7, size=300)
mean_x = X.mean()

def neg_ll_theta(theta):
    return -mean_x*np.log(theta) - (1-mean_x)*np.log(1-theta)

def grad_theta(theta):
    return -(mean_x/theta - (1-mean_x)/(1-theta))

def fisher_theta(theta):
    return 1/(theta*(1-theta))

# η = logit = log(θ/(1-θ))
def grad_eta(eta):
    theta = 1/(1+np.exp(-eta))
    # dL/dη = dL/dθ * dθ/dη = grad_theta * θ(1-θ)
    return grad_theta(theta) * theta*(1-theta)

def fisher_eta(eta):
    theta = 1/(1+np.exp(-eta))
    return theta*(1-theta)

# NGD in θ
theta, eta = 0.3, np.log(0.3/0.7)
lr = 0.1
for t in range(50):
    theta = theta - lr * grad_theta(theta)/fisher_theta(theta)
    theta = np.clip(theta, 0.01, 0.99)
    eta   = eta   - lr * grad_eta(eta)/fisher_eta(eta)

print(f"NGD in θ     → θ = {theta:.6f}")
print(f"NGD in η=logit → θ = {1/(1+np.exp(-eta)):.6f}")
print(f"True MLE: θ = {mean_x:.6f}")
```

**기대**: 두 결과가 소수점 4자리 이상 일치.

### 8.3 텐서 변환 법칙 수치 검증

```python
import numpy as np

np.random.seed(7)
n = 3

# 임의의 가역 Jacobian
J = np.eye(n) + 0.3*np.random.randn(n, n)

# 임의 Fisher (양정치)
A = np.random.randn(n, n)
F_theta = A@A.T + np.eye(n)

# 임의 gradient
g_theta = np.random.randn(n)

# 변환: θ → φ
F_phi = np.linalg.inv(J).T @ F_theta @ np.linalg.inv(J)  # J^-T F J^-1
g_phi = np.linalg.inv(J).T @ g_theta  # J^-T g

# Natural gradient
nat_theta = np.linalg.solve(F_theta, g_theta)
nat_phi   = np.linalg.solve(F_phi, g_phi)

# 이론 예측: nat_phi = J @ nat_theta
nat_phi_pred = J @ nat_theta

print("nat_phi (직접) :", nat_phi)
print("J @ nat_theta  :", nat_phi_pred)
print("차이 norm:", np.linalg.norm(nat_phi - nat_phi_pred))
```

**기대**: 차이가 `~1e-14` (기계 정밀도).

---

## 9. AI/ML 연결

### 9.1 BatchNorm과 parameterization

BatchNorm은 layer 출력을 정규화하여 **암묵적으로 parameterization을 바꾼다**. Kohler+ (2019)는 BN이 유사하게 Fisher-preconditioned gradient와 관계된 효과를 냄을 분석. NGD는 명시적으로 이 reparameterization 불변성을 갖는다.

### 9.2 LoRA / Low-rank adaptation

LoRA는 $W \leftarrow W + BA$, $B, A$ 저차원. Fisher가 $W$에 대한 것과 $(B, A)$에 대한 것이 다르므로, LoRA에서 NGD를 쓰려면 reparameterization 고려 필요 (Hu+ 2024).

### 9.3 VAE의 reparameterization trick

$z = \mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$. VAE 훈련시 $(\mu, \log\sigma)$ 좌표 사용 — 왜?

- $\sigma > 0$ 제약 자동 해결
- **Fisher가 log scale에서 더 등방(isotropic)** (7.1 예제: $F_\ell = \text{diag}(1/\sigma^2, 2)$, σ에 독립인 2)
- 즉 log scale ≈ "자연 좌표"의 근사, Adam 같은 유클리드 최적화가 더 잘 작동

**NGD 관점**: 어떤 좌표를 쓰든 **분포 경로는 같으므로**, 이는 단순히 수치 안정성 문제. NGD라면 좌표 선택이 중요하지 않다.

### 9.4 Invariant Policy Optimization

Kakade (2001)의 natural policy gradient의 핵심 장점은 **policy parameterization 불변성**. Softmax vs Gaussian policy 등 다른 표현을 써도 같은 정책 궤적. TRPO/PPO에서 이 속성이 실험 재현성을 높임.

### 9.5 Flat minima와 좌표 독립성

Dinh+ (2017, "Sharp minima generalize") — 좌표 변환으로 "평평/뾰족" 판단이 깨질 수 있음. Fisher 기반 sharpness (Fisher-SAM, Kim+ 2022)는 **parameterization 불변 sharpness measure**: $\text{tr}(F \Sigma)$ 등.

---

## 10. 흔한 오해와 함정

1. **"Natural gradient가 좌표에 무관"은 연속 시간 flow에서만 정확히 성립.**
   - 이산 NGD는 $O(\eta^2)$ 오차. 작은 $\eta$에선 거의 불변.

2. **불변성은 "계산 결과"가 아니라 "분포 경로".**
   - $\tilde{\nabla}_\theta$와 $\tilde{\nabla}_\phi$는 **수치적으로 다름** (다른 좌표니까). 하지만 지시하는 **분포는 같음**.

3. **모든 reparameterization이 유용하진 않다.**
   - 예: $\theta \to \theta^{1/3}$ 같은 비선형 변환은 Fisher를 singular하게 만들 수 있음. 실전에선 Jacobian이 well-conditioned인 좌표 선호.

4. **Parameterization 불변성 ≠ 모델 불변성.**
   - 모델 $p(x|\theta)$를 바꾸면 (예: Gaussian → mixture) 분포 궤적도 바뀜. 좌표 변환과 모델 변환은 다름.

5. **Riemann 계량을 바꾸면 flow도 바뀜.**
   - Fisher가 아닌 다른 계량 (예: Wasserstein)을 쓰면 flow가 다름. "parameterization 불변성"은 **주어진 계량(Fisher)에 대해** 말하는 것.

6. **수치 구현에서 $J$를 어떻게 쓰나.**
   - 대부분 자동미분(autograd)이 chain rule로 자동 처리. $F$와 $\nabla L$ 모두 새 좌표에서 계산하면 끝.

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **직교 변환**: $J^T J = I$ ($\phi = O\theta$, O 직교)이면 유클리드 gradient가 불변임을 보여라. 이것이 natural gradient의 특수 경우임을 확인.

2. **Softmax parameterization**: Categorical $p(k) = e^{\theta_k}/\sum_j e^{\theta_j}$에서 $\theta$ 좌표와 $\eta_k = p(k)$ 좌표 (simplex)의 Fisher 비교, natural gradient 불변 검증.

3. **EM/VI의 좌표**: VAE에서 $\theta_1 = (\mu, \sigma)$ vs $\theta_2 = (\mu, \log\sigma)$. ELBO에 대한 natural gradient가 두 좌표에서 같은 분포 궤적을 그리는지 수치로 검증.

4. **Non-affine 변환**: $\phi = \theta^3$ (n=1)에서 $\theta = 0$이면 $J = 0$, NGD가 정의 안 됨. 이 singularity를 어떻게 피할지 설명.

5. **Step size 스케일**: $\theta$ 좌표에서 $\eta = 0.01$이 $\phi = 100\theta$ 좌표에서 의미하는 바? trust region 해석에서 자명함을 설명.

### 다음 단계

- **[05. K-FAC, Shampoo, 실전 구현](./05-kfac-shampoo.md)**: Fisher 행렬의 대규모 근사.

---

**참고문헌**

- Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*.
- Amari, S. (2016). *Information Geometry and Its Applications*, Ch. 12.
- Dinh, L.+ (2017). *Sharp Minima Can Generalize For Deep Nets*.
- Kohler, J.+ (2019). *Exponential Convergence Rates for BN*.
- Kim, S.+ (2022). *Fisher SAM: Information Geometry and Sharpness Aware Minimisation*.
- Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*.

---

[◀ 03. KL Steepest Descent](./03-kl-steepest-descent.md) | [📚 README](../README.md) | [05. K-FAC, Shampoo ▶](./05-kfac-shampoo.md)
