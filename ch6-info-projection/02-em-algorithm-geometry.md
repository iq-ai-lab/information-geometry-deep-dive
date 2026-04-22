# 02. EM 알고리즘의 Information Geometry

> **"EM은 두 집합 사이의 탁구다. 한쪽은 관측 모델, 다른 한쪽은 사후 분포. 매 번 상대편으로 최단거리를 쏜다."**
> — Neal & Hinton, *A View of the EM Algorithm...* (1998)

---

## 1. 왜 이 주제인가?

EM (Expectation-Maximization) 알고리즘은 잠재변수 모델 $p(x, z | \theta)$의 최대우도 추정을 위한 고전적 기법이다. 표준 교과서에서는 **Jensen 부등식**으로 ELBO (Evidence Lower BOund)를 유도하고, 이를 교대로 최대화함으로써 증명한다.

Amari-Nagaoka (2000), Csiszár-Tusnády (1984)는 근본적으로 다른 관점을 제시했다: **EM은 두 정보 집합 사이의 m-projection과 e-projection의 교대**다. 이 기하학적 관점에서:

- E-step은 관측 데이터 위의 **m-projection** (조건부 분포로 잠재변수 보완).
- M-step은 모델 family 위의 **e-projection** (파라미터 갱신).
- Pythagoras 정리가 **단조 수렴성**을 자동으로 보장.

이 문서는 이 기하학적 해석을 엄밀히 증명하고, 고전 Jensen 기반 유도와 **완전 동치**임을 보인다. 이로써 VI, wake-sleep, Variational EM 등 확장들의 통일적 이해가 가능해진다.

---

## 2. 학습 목표

1. **ELBO와 KL**의 관계 $\log p(x|\theta) = \text{ELBO}(q, \theta) + \text{KL}(q\|p(z|x,\theta))$ 유도.
2. EM의 E-step을 **m-projection**으로 해석.
3. M-step을 **e-projection**으로 해석.
4. 두 projection의 교대가 **단조 감소하는 KL**를 보장함을 증명 (수렴성).
5. Neal-Hinton (1998)의 **partial / incremental EM**을 기하학적으로 설명.

---

## 3. 전제 지식

- **Ch6-01**: e-projection, m-projection, Pythagoras.
- **Jensen 부등식**과 기본 확률 이론.
- **잠재변수 모델** $p(x|θ) = \int p(x,z|θ)dz$.
- **기본 EM**: E-step, M-step 공식.

---

## 4. 직관적 설명

### 4.1 EM이 푸는 문제

잠재변수 $z$를 갖는 모델 $p(x, z|\theta)$에서 MLE:

$$
\hat{\theta} = \arg\max_\theta \log p(x|\theta) = \arg\max_\theta \log \int p(x, z|\theta) dz.
$$

$\int$ 때문에 직접 최대화 어려움. EM의 핵심 아이디어:

1. **보조분포** $q(z)$를 도입해 ELBO를 정의.
2. $q$를 움직여 ELBO $\uparrow$ (E-step).
3. $\theta$를 움직여 ELBO $\uparrow$ (M-step).

### 4.2 두 집합의 게임

두 집합을 정의:

- **Data manifold** $\mathcal{D}$: $q(z) p(x)$ 형태의 분포 (x가 관측된 상태, z는 임의의 $q$).  
- **Model manifold** $\mathcal{M}$: $p(x, z | \theta)$ 형태의 분포.

EM 매 iteration:

- **E-step**: $\mathcal{D}$ 위에서, $\mathcal{M}$ 현재 점에 **가장 가까운** $q$를 찾음 = **m-projection**.
- **M-step**: $\mathcal{M}$ 위에서, $\mathcal{D}$ 현재 점에 **가장 가까운** $\theta$를 찾음 = **e-projection**.

매 projection이 KL을 줄이므로 **단조 감소**. Pythagoras가 이 감소량을 **정확한 분해**로 설명.

### 4.3 왜 두 방향?

Forward KL: $\text{KL}(q\|p_\theta)$. 

- $q$ 고정, $\theta$ 움직임 → $\theta$에 대한 gradient가 $\log p$의 MLE gradient와 같음 (moment matching).
- $\theta$ 고정, $q$ 움직임 → $q^* = p_\theta(z|x)$ (사후분포).

이 둘이 각각 E-step과 M-step이다.

---

## 5. 엄밀한 정의와 정리

### 5.1 ELBO

$x$ 관측, $\theta$ 주어짐. 임의 $q(z)$에 대해:

$$
\log p(x|\theta) = \mathbb{E}_{q(z)}[\log p(x|\theta)] = \mathbb{E}_q\left[\log \frac{p(x,z|\theta)}{q(z)} \cdot \frac{q(z)}{p(z|x,\theta)}\right]
$$

$$
= \underbrace{\mathbb{E}_q[\log p(x,z|\theta) - \log q(z)]}_{=: \mathcal{L}(q, \theta) \text{ (ELBO)}} + \underbrace{\mathbb{E}_q[\log q(z) - \log p(z|x,\theta)]}_{= \text{KL}(q \| p(z|x,\theta))}.
$$

**정리 5.1 (ELBO 분해).** 

$$
\boxed{\log p(x|\theta) = \mathcal{L}(q, \theta) + \text{KL}(q \| p(z|x, \theta)).}
$$

KL ≥ 0이므로 $\mathcal{L}(q, \theta) \leq \log p(x|\theta)$, 등식은 $q = p(z|x,\theta)$일 때.

### 5.2 EM 반복 공식

**E-step**: $q^{(t+1)} := p(z | x, \theta^{(t)})$.

**M-step**: $\theta^{(t+1)} := \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta) = \arg\max_\theta \mathbb{E}_{q^{(t+1)}}[\log p(x, z | \theta)]$.

### 5.3 EM의 기하학적 정리 (Csiszár-Tusnády 1984; Amari 1995)

**정리 5.2.** 다음 집합들을 정의:

- $\mathcal{D} := \{q(z) \cdot \delta(x' - x) : q(z) \text{ is any density}\}$ — "empirical" family on $(x, z)$.
- $\mathcal{M} := \{p(x', z | \theta) : \theta \in \Theta\}$ — 모델 family.

($\delta(x'-x)$는 관측된 $x$를 강제하는 indicator.)

**E-step**: $\mathcal{D}$에서 $\mathcal{M}$의 현재 점 $p_{\theta^{(t)}}$로의 **m-projection**:

$$
q^{(t+1)} = \arg\min_{q \in \mathcal{D}} D(q \| p_{\theta^{(t)}}) = p(z | x, \theta^{(t)}).
$$

**M-step**: $\mathcal{M}$에서 $q^{(t+1)} \in \mathcal{D}$로의 **e-projection**:

$$
\theta^{(t+1)} = \arg\min_\theta D(q^{(t+1)} \| p_\theta) = \arg\max_\theta \mathbb{E}_{q^{(t+1)}}[\log p(x, z|\theta)].
$$

### 5.4 단조 수렴성

**정리 5.3 (EM Monotonicity; Dempster-Laird-Rubin 1977).** 각 iteration에서:

$$
\log p(x | \theta^{(t+1)}) \geq \log p(x | \theta^{(t)}).
$$

### 5.5 Pythagoras 기반 증명

**정리 5.4 (Amari 1995).** 매 iteration에서:

$$
D(q^{(t+1)} \| p_{\theta^{(t)}}) = D(q^{(t+1)} \| p_{\theta^{(t+1)}}) + D(p_{\theta^{(t+1)}} \| p_{\theta^{(t)}}),
$$

$\mathcal{M}$이 e-flat (exp family) 하에서. 따라서 두 번째 항 $\geq 0$이 **감소량**이다.

---

## 6. 증명

### 6.1 E-step = m-projection

**정리 5.2의 E-step 증명.** $\theta^{(t)}$ 고정, $q$ 움직임:

$$
D(q \| p_{\theta^{(t)}}) = \mathbb{E}_q[\log q(z) - \log p(x, z | \theta^{(t)})].
$$

$x$가 관측되어 상수이므로 위 KL은 $q$에 대해 다음과 등치 (상수 항 제외):

$$
\mathbb{E}_q[\log q(z) - \log p(z | x, \theta^{(t)})] + \text{const}(x, \theta^{(t)}) = \text{KL}(q \| p(z|x,\theta^{(t)})) + \text{const}.
$$

이를 $q$에 대해 최소화하면 $q^* = p(z | x, \theta^{(t)})$. $\square$

### 6.2 M-step = e-projection (exp family 경우)

$q^{(t+1)}$ 고정, $\theta$ 움직임:

$$
D(q^{(t+1)} \| p_\theta) = \mathbb{E}_{q^{(t+1)}}[\log q^{(t+1)}] - \mathbb{E}_{q^{(t+1)}}[\log p(x, z | \theta)].
$$

첫 항은 $\theta$ 무관 (상수). 따라서:

$$
\arg\min_\theta D(q^{(t+1)} \| p_\theta) = \arg\max_\theta \mathbb{E}_{q^{(t+1)}}[\log p(x, z | \theta)] = \theta^{(t+1)}.
$$

$\mathcal{M}$이 exp family (e-flat)에서 e-projection은 유일 (Ch6-01 정리 5.6). $\square$

### 6.3 Pythagoras 분해로 단조성

**정리 5.3 증명.** 정리 5.1에서:

$$
\log p(x | \theta^{(t+1)}) - \log p(x | \theta^{(t)}) = \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) + \text{KL}_{\text{new}} - \mathcal{L}(q^{(t+1)}, \theta^{(t)}) - \text{KL}_{\text{old}}.
$$

E-step: $\text{KL}_{\text{old}} = \text{KL}(q^{(t+1)} \| p(z|x, \theta^{(t)})) = 0$ (by E-step 정의, $q^{(t+1)} = p(z|x, \theta^{(t)})$).

M-step: $\mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)})$ (M-step은 $\mathcal{L}$을 $\theta$에 대해 최대화).

$\text{KL}_{\text{new}} \geq 0$ (KL non-negative).

따라서:

$$
\log p(x|\theta^{(t+1)}) - \log p(x|\theta^{(t)}) \geq 0. \quad \square
$$

### 6.4 EM의 고정점

고정점 $(\theta^*, q^*)$에서:

- E-step: $q^* = p(z|x, \theta^*)$.
- M-step: $\theta^* = \arg\max \mathbb{E}_{q^*}[\log p(x,z|\theta)]$.

이는 $\nabla_\theta \log p(x|\theta^*) = 0$과 동치 (score identity). 즉 EM 고정점 = MLE 정류점.

### 6.5 수렴 속도 (Pythagoras 확장)

**정리 6.1.** Exp family 가정 하에:

$$
D(q^{(t+1)} \| p_{\theta^{(t)}}) = D(q^{(t+1)} \| p_{\theta^{(t+1)}}) + D(p_{\theta^{(t+1)}} \| p_{\theta^{(t)}}).
$$

이는 정리 5.4. 증명: $q^{(t+1)}$는 $p_{\theta^{(t)}}$의 m-projection이지만, 여기선 $p_{\theta^{(t+1)}}$가 $q^{(t+1)}$의 e-projection on $\mathcal{M}$임을 쓴다. $\mathcal{M}$ e-flat이므로 Pythagoras (Ch6-01 정리 5.7) 성립:

$$
D(q^{(t+1)} \| p_{\theta^{(t)}}) = D(q^{(t+1)} \| p_{\theta^{(t+1)}}) + D(p_{\theta^{(t+1)}} \| p_{\theta^{(t)}}).
$$

두 번째 항이 **매 iteration KL 감소량**. 고정점 근처에서 $O(\|\Delta\theta\|^2)$ 2차 수렴.

---

## 7. 구체 예제

### 7.1 GMM (Gaussian Mixture Model)

$p(x|\theta) = \sum_k \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$, 잠재 $z \in \{1,\dots,K\}$ (assignment).

**E-step**: $q^{(t+1)}(z=k | x_i) = \gamma_{ik}^{(t+1)} = \frac{\pi_k^{(t)} \mathcal{N}(x_i | \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_j \pi_j^{(t)} \mathcal{N}(x_i | \mu_j^{(t)}, \Sigma_j^{(t)})}$.

**M-step**: 
- $\pi_k^{(t+1)} = \frac{1}{N}\sum_i \gamma_{ik}$
- $\mu_k^{(t+1)} = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$
- $\Sigma_k^{(t+1)} = \frac{\sum_i \gamma_{ik} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^T}{\sum_i \gamma_{ik}}$

**기하**: E-step은 각 $x_i$에 대한 **soft assignment** (m-projection of empirical to posterior). M-step은 **moment matching** (e-projection of expected statistics).

### 7.2 Mixture of Bernoullis

$p(x|\theta) = \sum_k \pi_k \prod_d \mu_{kd}^{x_d} (1-\mu_{kd})^{1-x_d}$.

E-step: 동일한 soft assignment.

M-step: $\mu_{kd}^{(t+1)} = \frac{\sum_i \gamma_{ik} x_{id}}{\sum_i \gamma_{ik}}$ (moment matching on each feature).

### 7.3 HMM의 Baum-Welch

Hidden Markov Model에서 EM은 Baum-Welch algorithm.

- **E-step**: Forward-Backward로 $q(z_t | x_{1:T}, \theta^{(t)})$ 계산.
- **M-step**: transition/emission 파라미터의 moment matching.

Pythagoras로 $\log p(x_{1:T})$의 단조 증가 증명.

### 7.4 PCA의 EM (Tipping & Bishop 1999)

Probabilistic PCA: $x = Wz + \epsilon$, $z \sim \mathcal{N}(0, I)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

EM으로 $W, \sigma^2$ MLE. E-step: 사후분포 $q(z|x)$ 계산 (Gaussian). M-step: $W^{(t+1)}$ 갱신.

해는 standard PCA로 수렴 (spectral decomposition과 동치, Roweis 1997).

---

## 8. Python 코드 검증

### 8.1 GMM with EM

```python
import numpy as np
from scipy.stats import multivariate_normal

# 합성 데이터: 2개의 Gaussian
np.random.seed(42)
N = 500
X = np.concatenate([
    np.random.multivariate_normal([-2, 0], [[1, 0.3],[0.3, 1]], 250),
    np.random.multivariate_normal([2, 1], [[1, -0.3],[-0.3, 1]], 250)
])

K, D = 2, 2
# 초기화
pi = np.ones(K) / K
mu = np.random.randn(K, D) * 2
Sigma = np.array([np.eye(D) for _ in range(K)])

def log_likelihood(X, pi, mu, Sigma):
    ll = 0
    for n in range(len(X)):
        p = sum(pi[k] * multivariate_normal.pdf(X[n], mu[k], Sigma[k]) for k in range(K))
        ll += np.log(p)
    return ll

log_liks = [log_likelihood(X, pi, mu, Sigma)]
for it in range(50):
    # E-step: γ_{ik}
    gamma = np.zeros((len(X), K))
    for k in range(K):
        gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mu[k], Sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)
    
    # M-step
    Nk = gamma.sum(axis=0)
    pi = Nk / len(X)
    mu = (gamma.T @ X) / Nk[:, None]
    for k in range(K):
        diff = X - mu[k]
        Sigma[k] = (gamma[:,k:k+1] * diff).T @ diff / Nk[k]
    
    log_liks.append(log_likelihood(X, pi, mu, Sigma))

import matplotlib.pyplot as plt
plt.plot(log_liks, 'b.-')
plt.xlabel('Iteration'); plt.ylabel('log p(X|θ)')
plt.title('EM: log-likelihood (monotone 단조 증가 검증)')
plt.grid()
print(f"Monotone? {all(log_liks[i+1] >= log_liks[i] - 1e-10 for i in range(len(log_liks)-1))}")
```

**기대 출력**: `Monotone? True`, 로그 우도가 매 iteration 단조 증가.

### 8.2 ELBO = log p(x) when q = posterior

```python
import numpy as np
from scipy.stats import norm

# Toy: p(x|z) = N(z, 1), p(z) = N(0, 1), observation x=2
x = 2.0

# True posterior: p(z|x) = N(x/2, 1/2)  (by conjugate Gaussian)
post_mu, post_sigma = 1.0, np.sqrt(0.5)

# log p(x) = -log(sqrt(2*pi*2)) - x^2/4
log_px = -0.5 * np.log(2*np.pi*2) - x**2/4
print(f"log p(x) = {log_px:.6f}")

# ELBO with q = posterior (exact)
def elbo(q_mu, q_sigma, x):
    # E_q[log p(x,z)] - E_q[log q(z)]
    # log p(x,z) = log p(z) + log p(x|z) = -0.5(z^2) - 0.5*log(2pi) - 0.5((x-z)^2) - 0.5*log(2pi)
    # E_q of that
    E_z2 = q_sigma**2 + q_mu**2
    E_xz2 = q_sigma**2 + (x - q_mu)**2
    E_log_p = -0.5*E_z2 - 0.5*np.log(2*np.pi) - 0.5*E_xz2 - 0.5*np.log(2*np.pi)
    E_log_q = -0.5 - 0.5*np.log(2*np.pi*q_sigma**2)  # entropy of Gaussian
    return E_log_p - E_log_q

elbo_at_post = elbo(post_mu, post_sigma, x)
print(f"ELBO at true posterior = {elbo_at_post:.6f}")
print(f"Diff = {abs(log_px - elbo_at_post):.2e}")
```

**기대**: 차이 `~1e-14` → E-step 후 ELBO = log p(x) 확인.

### 8.3 Pythagoras 분해 검증 (2-Gaussian 예제)

```python
import numpy as np

# Simple exp family: exponential distribution Exp(λ), one latent
# ... (여기선 교시적 목적으로 categorical 예)
# S = Categorical(4), M = product family

# 실제 복잡해서 개요만
# Jen Hui Ho's EM 검증: each iteration
# D(q^(t+1) || p_θ^(t)) = D(q^(t+1) || p_θ^(t+1)) + D(p_θ^(t+1) || p_θ^(t))
# 수치 검증은 GMM/HMM에서 가능
print("Pythagoras 검증은 GMM 혹은 exp family 모델에서 직접 구현 가능 (연습문제 3)")
```

---

## 9. AI/ML 연결

### 9.1 Variational EM

E-step에서 posterior이 closed-form이 아닐 때 (nonconjugate), variational approximation $q(z) \in \mathcal{Q}$ 사용. ELBO를 $q, \theta$ 모두에 대해 최대화. Ch6-03 VI로 연결.

### 9.2 Wake-Sleep (Hinton+ 1995)

- **Wake phase**: Data → latent → recognition model 업데이트 (m-projection 유사).
- **Sleep phase**: latent → data → generative model 업데이트 (e-projection 유사).

Helmholtz machine, VAE의 원조.

### 9.3 VAE (Kingma+ 2014)

VAE는 **amortized Variational EM**:

- E-step → encoder $q_\phi(z|x)$ (reparameterization trick으로 gradient).
- M-step → decoder $p_\theta(x|z)$.

동시 optimization, stochastic gradient. Ch7-03에서 상세.

### 9.4 Natural gradient EM

M-step을 Fisher natural gradient로 (Sato 2001, Honkela+ 2010). Exp family에서 closed form, non-exp에서 acceleration.

### 9.5 Online / Incremental EM (Neal-Hinton 1998)

Data를 mini-batch로 보고, 한 batch씩 E/M-step 교대. Stochastic EM, 대규모 데이터 처리.

---

## 10. 흔한 오해와 함정

1. **EM은 local optimum만 보장**.
   - 글로벌 최적은 아님. Initialization 중요 (k-means++ 등).

2. **Saddle point와 plateau**.
   - EM 고정점은 local max/min/saddle 모두 포함. Gradient 정보도 같이 봐야.

3. **ELBO 증가 ≠ posterior 근사 개선**.
   - E-step이 가정된 $\mathcal{Q}$에 제한되면 (VEM), KL이 완전히 0 안 됨. "Variational gap".

4. **M-step closed form 유무**.
   - Exp family 모델에선 closed form. 일반 모델은 다시 gradient-based.

5. **Complete-data likelihood vs observed-data**.
   - E-step은 $q(z)$로 $\log p(x, z|\theta)$의 기댓값. 관측 likelihood $\log p(x|\theta)$ 아님.

6. **"EM이 Bayesian"이 아니다**.
   - EM은 MLE (point estimate). Bayesian은 $p(\theta|x)$. VI가 posterior를 근사.

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **Jensen 유도와 projection 유도의 동치**: 교과서 Jensen 기반 유도와 정리 5.2의 projection 유도가 정확히 같은 업데이트를 주는지 보여라.

2. **GMM closed-form M-step**: GMM 경우 M-step이 exp family moment matching임을 유도.

3. **Pythagoras 분해 수치 검증**: Bernoulli mixture에서 매 EM iteration마다 $D(q\|p_{\text{old}}) = D(q\|p_{\text{new}}) + D(p_{\text{new}}\|p_{\text{old}})$을 코드로 검증.

4. **Saddle point 예제**: GMM에서 두 mode가 겹칠 때 EM이 saddle에 수렴하는 설정을 구성.

5. **Hard EM**: E-step에서 soft assignment를 argmax로 바꾸면 k-means. 이것이 m-projection의 degenerate case임을 설명.

### 다음 단계

- **[03. Variational Inference](./03-variational-inference.md)**: ELBO와 mean-field.
- **[05. Mixture Projection](./05-mixture-projection.md)**: GMM의 projection 해석 심화.

---

**참고문헌**

- Dempster, A., Laird, N., Rubin, D. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*.
- Csiszár, I., Tusnády, G. (1984). *Information Geometry and Alternating Minimization Procedures*.
- Amari, S. (1995). *Information Geometry of the EM and em Algorithms*.
- Neal, R., Hinton, G. (1998). *A View of the EM Algorithm that Justifies Incremental, Sparse, and other Variants*.
- Bishop, C. (2006). *Pattern Recognition and ML*, Ch. 9.
- Murphy, K. (2022). *Probabilistic ML: Advanced Topics*, Ch. 6.

---

[◀ 01. e-projection & m-projection](./01-e-m-projection.md) | [📚 README](../README.md) | [03. Variational Inference ▶](./03-variational-inference.md)
