# 05. Mixture Model과 Information Projection

> **"Mixture 모델은 본질적으로 '관측'과 '잠재'의 두 공간 사이를 왕복하는 생물이다. Information geometry는 이 왕복의 정체를 밝힌다."**

---

## 1. 왜 이 주제인가?

Ch6-02에서 EM 알고리즘이 **m-projection과 e-projection의 교대**임을 보였다. 이 통찰을 **Gaussian Mixture Model (GMM)** 같은 구체 모델에 적용하면, **수렴 속도, local minimum 구조, identifiability** 같은 문제를 information geometric으로 분석 가능하다.

또한 mixture는 **m-flat 다양체**의 자연스러운 예시 (Ch4-05에서 m-flat이란 $\eta = \mathbb{E}[T]$ 좌표에서 affine). Mixture 가중치 $\pi$는 정확히 m-좌표에 해당. 이 구조를 활용하면:

- GMM의 EM이 **이중 projection 궤적**.
- **Generalized Pythagoras**로 매 iteration의 수렴량 분해.
- GMM의 **non-identifiability**가 mixture manifold의 기하학적 degeneracy.

이 문서는 GMM, mixture of experts, mixture density network 등을 통해 **information projection의 실전 응용**을 정리한다.

---

## 2. 학습 목표

1. **Mixture manifold**가 m-flat임을 증명.
2. GMM EM의 E/M-step을 **이중 projection**으로 해석.
3. **Pythagoras로 수렴 속도 분석** 공식 유도.
4. **Label switching**과 identifiability의 기하학적 이해.
5. **Deep mixture models (MoE, MDN, DP-GMM)**과 NG의 연결.

---

## 3. 전제 지식

- **Ch4-05**: Dually flat, m-flat submanifold
- **Ch6-01**: e/m-projection, Pythagoras
- **Ch6-02**: EM = 이중 projection
- **GMM**: $p(x) = \sum_k \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$

---

## 4. 직관적 설명

### 4.1 Mixture는 "m-곡선" 위에 산다

$K$개 기저 분포 $q_1, \dots, q_K$ (고정)의 mixture:

$$
p_\pi(x) = \sum_{k=1}^K \pi_k q_k(x), \quad \pi \in \Delta_{K-1}.
$$

Mixture weight $\pi$는 **linear convex combination** → 분포 공간에서 **affine 부분집합** = **m-flat submanifold**.

실제로 $\eta$ (expectation) 좌표에서:

$$
\mathbb{E}_{p_\pi}[T] = \sum_k \pi_k \mathbb{E}_{q_k}[T] = \sum_k \pi_k \eta_k.
$$

$\eta$는 $\pi$의 affine 함수 → affine = m-flat.

### 4.2 GMM은 mixture + 파라미터

GMM: $p(x|\theta, \pi) = \sum_k \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$.

- 각 component $\mathcal{N}(\mu_k, \Sigma_k)$: **e-flat** (Gaussian exp family).
- Mixture weight: **m-flat** (simplex).

**결합**: 2가지 flatness의 **직교 곱** (product manifold).

### 4.3 EM의 기하 재조명

**E-step**: soft assignment $\gamma_{ik} = p(z=k|x_i, \theta^{(t)})$.
- 기하: **m-projection** of empirical $\hat{p}(x)$ onto product family (잠재 공간으로 "lifting").

**M-step**: $\pi_k^{(t+1)} = \bar\gamma_k$, $\mu_k^{(t+1)} = \sum \gamma x_i / \sum \gamma$, etc.
- 기하: **e-projection** in Gaussian manifold (moment matching) + m-projection on simplex (weight averaging).

---

## 5. 엄밀한 정의와 정리

### 5.1 Mixture manifold

**정의 5.1.** 기저 exp family $\{q_1, \dots, q_K\}$에 대해:

$$
\mathcal{M}_{\text{mix}} = \left\{p_\pi = \sum_k \pi_k q_k : \pi \in \Delta_{K-1}\right\}.
$$

**명제 5.2.** $\mathcal{M}_{\text{mix}}$는 **m-flat** 부분다양체.

### 5.2 GMM

**정의 5.3.** $K$-component GMM:

$$
p(x | \theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k),
$$

$\theta = (\pi, \{\mu_k\}, \{\Sigma_k\})$. $K$ 고정.

### 5.3 GMM EM의 이중 projection

**정리 5.4.** GMM EM 매 iteration:

- **E-step**: $q^{(t+1)}(z|x_i) = p(z|x_i, \theta^{(t)}) = \text{Cat}(\gamma_{i\cdot}^{(t+1)})$. 이는 joint manifold에서의 **m-projection**.

- **M-step**: 두 projection의 곱:
  - Mixture weight: $\pi^{(t+1)} = \bar\gamma$ (m-projection on simplex).
  - Component params: $(\mu_k^{(t+1)}, \Sigma_k^{(t+1)})$ = e-projection in Gaussian manifold (moment matching to $q^{(t+1)}$-weighted data).

### 5.4 Pythagoras-based 수렴량

**정리 5.5.** GMM EM 매 iteration에서 (Gaussian은 e-flat이므로 Ch6-01 정리 5.7):

$$
D(q^{(t+1)}(z, x) \| p^{(t)}(z,x)) = D(q^{(t+1)} \| p^{(t+1)}) + D(p^{(t+1)} \| p^{(t)}).
$$

두 번째 항 = **매 iteration KL 감소량** = 수렴 속도의 근본 measure.

### 5.5 Label Switching and Symmetry

**명제 5.6.** GMM은 **$K!$개 equivalent 해**를 가짐 (label permutation). Mixture manifold의 discrete symmetry는 non-identifiability의 원인.

이는 기하학적으로 $\mathcal{M}_{\text{mix}}/S_K$ (quotient manifold)에서 해결.

---

## 6. 증명

### 6.1 m-flatness of mixture

**명제 5.2 증명.** $p_\pi = \sum \pi_k q_k$, Sufficient statistic $T$에 대해:

$$
\mathbb{E}_{p_\pi}[T] = \sum_k \pi_k \mathbb{E}_{q_k}[T].
$$

$\eta$ 좌표 (expectation)에서 $\eta(\pi) = \sum_k \pi_k \eta_k$ — linear in $\pi$. 

$\pi$ 자체도 linear (simplex parameter), 따라서 $\eta(\pi)$ **affine 부분공간**. m-flat. $\square$

### 6.2 GMM은 m-flat이 아님

**주의**: Full GMM (파라미터 $\mu_k, \Sigma_k$ 포함)은 m-flat **아님**. 왜? Component 파라미터가 **$\mathcal{N}$ manifold의 점**이지 고정이 아니기 때문.

Mixture weight에 대해선 m-flat이지만, component 파라미터에 대해선 일반 manifold. 따라서 GMM의 EM의 **주된 m-projection 부분은 weight** 업데이트, **e-projection 부분은 component 업데이트**.

### 6.3 EM의 Pythagoras 분해

**정리 5.5 증명.** $q^{(t+1)}$는 현재 $\theta^{(t)}$ 하의 posterior이므로 $D(q \| p^{(t)})$ 최소화 = m-projection.

Next M-step: $\theta^{(t+1)}$는 $\arg\max \mathbb{E}_{q^{(t+1)}}[\log p(x,z|\theta)]$, 이는 e-projection $\Pi_\mathcal{M}^{(e)}(q^{(t+1)})$.

$\mathcal{M}$ (GMM manifold with fixed $K$)의 **local e-flatness** 가정 (Gaussian components) 하에 Pythagoras:

$$
D(q^{(t+1)} \| p^{(t)}) = D(q^{(t+1)} \| p^{(t+1)}) + D(p^{(t+1)} \| p^{(t)}).
$$

$\square$

### 6.4 수렴 속도

고정점 $\theta^*$ 근처에서 Taylor 전개:

$$
D(p^{(t+1)} \| p^{(t)}) \approx \frac{1}{2} (\theta^{(t+1)} - \theta^{(t)})^T F(\theta^*) (\theta^{(t+1)} - \theta^{(t)}).
$$

EM은 linear convergence rate $\rho$ ($0 < \rho < 1$) with $\rho = 1 - $ (fraction of missing information, Dempster+ 1977). Information geometric 해석: $\rho$는 **missing info / total info** 비율.

### 6.5 Label switching = quotient

Permutation $\sigma \in S_K$에 대해 $\pi_\sigma = (\pi_{\sigma(1)}, \dots)$, 같은 likelihood. Quotient manifold $\mathcal{M}/S_K$는 orbifold (modulo singular points 0-weight components).

---

## 7. 구체 예제

### 7.1 2-GMM 기하 시각화

$p(x) = \pi \mathcal{N}(-1, 1) + (1-\pi) \mathcal{N}(1, 1)$, $\pi \in [0,1]$.

**Mixture manifold**: $\pi$ 1D 곡선. Linear in density space.

**Pairs of Pythagoras**: EM iteration마다 weight update ($\pi$ 방향)과 component update (mean/variance) 직교.

### 7.2 GMM MLE는 non-convex

$K = 2$, 데이터 $\{-2, 2\}$. Likelihood landscape에 여러 local max:

- $(\mu_1, \mu_2) = (-2, 2)$ (정답, swap)
- $(\mu_1, \mu_2) = (2, -2)$ (label switch)
- $(\mu_1, \mu_2) = (0, 0)$ (degenerate)
- $\mu_1 \to $ data point, $\sigma_1 \to 0$ (degenerate, likelihood → $\infty$ without regularization)

기하: singular points on $\mathcal{M}_{\text{mix}}$가 EM 수렴을 방해.

### 7.3 Missing information fraction

Gaussian(known variance) 혼합에서 missing info fraction $\approx$ 두 component의 분리도 역비례. Clusters 잘 분리되면 fast convergence, 겹치면 slow.

### 7.4 Variational Bayes GMM

Dirichlet prior on $\pi$, NIW on $(\mu_k, \Sigma_k)$. VBEM은 mean-field + conjugate priors:

- Posterior on $\pi$: Dirichlet.
- Posterior on $\mu_k, \Sigma_k$: NIW.

ELBO가 단조 증가 + automatic model selection (small $\pi_k$ → component killed).

### 7.5 Infinite mixture (DP-GMM)

Dirichlet Process prior: $K \to \infty$. Information geometry of Bayesian nonparametric는 Hilbert-manifold framework (Amari 2014).

---

## 8. Python 코드 검증

### 8.1 GMM EM with KL decomposition tracking

```python
import numpy as np
from scipy.stats import multivariate_normal

np.random.seed(0)
# 2D 데이터 생성
N_per = 150
X1 = np.random.multivariate_normal([-2, 0], [[1, 0.3],[0.3, 1]], N_per)
X2 = np.random.multivariate_normal([2, 1], [[1, -0.3],[-0.3, 1]], N_per)
X = np.concatenate([X1, X2])
N = len(X)
K = 2
D = 2

# 초기화
pi = np.ones(K) / K
mu = X[np.random.choice(N, K, replace=False)]
Sigma = np.array([np.eye(D) for _ in range(K)])

def log_likelihood(X, pi, mu, Sigma):
    lik = np.zeros(len(X))
    for k in range(K):
        lik += pi[k] * multivariate_normal.pdf(X, mu[k], Sigma[k])
    return np.sum(np.log(lik))

def joint_log_prob(X, z, pi, mu, Sigma):
    # log p(x, z | θ)
    return np.log(pi[z]) + multivariate_normal.logpdf(X, mu[z], Sigma[z])

def kl_joint(gamma, theta_old, theta_new):
    # KL(q || p_new) - q는 γ, p는 joint
    pi_n, mu_n, Sigma_n = theta_new
    kl = 0
    for k in range(K):
        mask = gamma[:, k] > 1e-10
        if mask.sum() == 0:
            continue
        log_q = np.log(gamma[mask, k])
        log_p = np.log(pi_n[k]) + multivariate_normal.logpdf(X[mask], mu_n[k], Sigma_n[k])
        kl += np.sum(gamma[mask, k] * (log_q - log_p))
    return kl

log_liks, decompositions = [log_likelihood(X, pi, mu, Sigma)], []
for it in range(30):
    theta_old = (pi.copy(), mu.copy(), Sigma.copy())
    
    # E-step
    gamma = np.zeros((N, K))
    for k in range(K):
        gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mu[k], Sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)
    
    # Calculate D(q || p_old) before M-step
    D_before = kl_joint(gamma, None, theta_old)
    
    # M-step
    Nk = gamma.sum(axis=0)
    pi = Nk / N
    mu = (gamma.T @ X) / Nk[:, None]
    for k in range(K):
        diff = X - mu[k]
        Sigma[k] = (gamma[:,k:k+1] * diff).T @ diff / Nk[k]
    theta_new = (pi.copy(), mu.copy(), Sigma.copy())
    
    D_after = kl_joint(gamma, None, theta_new)
    D_theta_shift = D_before - D_after  # ≥ 0 (Pythagoras)
    
    decompositions.append((D_before, D_after, D_theta_shift))
    log_liks.append(log_likelihood(X, pi, mu, Sigma))

# 단조성 확인
print("Monotone log-lik?", all(log_liks[i+1] >= log_liks[i] - 1e-8 for i in range(len(log_liks)-1)))
print("\nIter | D(q||p_old) | D(q||p_new) | Shift (=D(p_new||p_old))")
for i, (b, a, s) in enumerate(decompositions[:10]):
    print(f"  {i:2d} | {b:10.4f} | {a:10.4f} | {s:10.4f}")
```

**기대**: Shift (= $D(p^{(t+1)}||p^{(t)})$) 값은 매 iteration에서 log-lik 증가량과 연결 (Pythagoras).

### 8.2 Label switching 시각화

```python
import numpy as np
# 같은 데이터에 다른 초기화로 EM 돌리면 다른 label 순서로 수렴
np.random.seed(1)
N = 200
X = np.concatenate([
    np.random.normal(-2, 0.5, N//2),
    np.random.normal(2, 0.5, N//2)
])

K = 2
results = []
for seed in range(5):
    np.random.seed(seed)
    pi = np.ones(K)/K
    mu = np.random.randn(K)*3
    var = np.ones(K)
    
    for it in range(30):
        # E-step
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = pi[k] * np.exp(-0.5*(X-mu[k])**2/var[k]) / np.sqrt(2*np.pi*var[k])
        gamma /= gamma.sum(axis=1, keepdims=True)
        # M-step
        Nk = gamma.sum(0); pi = Nk/N
        mu = (gamma.T @ X) / Nk
        for k in range(K):
            var[k] = np.sum(gamma[:,k] * (X-mu[k])**2) / Nk[k]
    results.append((sorted(mu), pi[np.argsort(mu)]))
    print(f"seed {seed}: μ sorted = {sorted(mu)}, π = {pi[np.argsort(mu)]}")
```

**관찰**: sorted 후 모든 초기값이 같은 MLE로 수렴 (label은 permute될 수 있음).

### 8.3 Singularity: variance collapse

```python
import numpy as np
# 초기화가 data point에 붙으면 σ → 0 → likelihood → ∞
X = np.array([1.0, 2.0, 3.0, 10.0])  # one outlier
# 10 근처에 component 하나, σ 작으면 likelihood 폭주
# 실전: σ에 lower bound or regularization needed
print("Singularity: GMM MLE는 regularization 없이는 unbounded.")
print("Solutions: (1) σ 하한, (2) prior (Bayesian GMM), (3) EM early stopping")
```

---

## 9. AI/ML 연결

### 9.1 Mixture Density Network (Bishop 1994)

Neural network로 mixture weight과 component params 출력:

$$
p(y|x) = \sum_k \pi_k(x) \mathcal{N}(y | \mu_k(x), \sigma_k^2(x)).
$$

Regression with multimodal output. EM 아닌 SGD로 학습.

### 9.2 Mixture of Experts (MoE)

Jacobs+ 1991, 최근 Switch Transformer (Fedus+ 2022). Gating network $g(x)$와 expert $f_k$:

$$
p(y|x) = \sum_k g_k(x) f_k(y | x).
$$

Sparse routing (top-$k$)으로 효율. GPT-4 MoE 추정. Information geometric 최적화: expert-specific Fisher + gate Fisher.

### 9.3 Gaussian Mixture Prior in VAE

VAE posterior을 GMM으로 하면 다모드 표현. Dilokthanakul+ 2016: GMVAE. ELBO에 mixture 처리 복잡.

### 9.4 Dirichlet Process GMM

$K$ 모르고 infinite. Blei & Jordan 2006, VB-DP-GMM. 자동 모델 선택.

### 9.5 Deep Clustering

Deep features → GMM EM. DEC (Xie+ 2016), VaDE (Jiang+ 2017).

---

## 10. 흔한 오해와 함정

1. **"GMM은 항상 unimodal posterior"는 거짓**.
   - Likelihood는 multimodal (label switching + local optima).

2. **K 선택 문제**.
   - BIC, AIC, cross-validation. DP-GMM은 자동이지만 prior sensitive.

3. **고차원에서 covariance matrix singular**.
   - Tied, diagonal, spherical covariance. 혹은 shrinkage.

4. **EM vs k-means**.
   - k-means = hard EM (Gaussian with $\sigma \to 0$, spherical). GMM은 soft + flexible covariance.

5. **Pythagoras는 e-flat component 가정**.
   - 일반 mixture (non-exp components)에선 완벽하게 성립하지 않음.

6. **Mixture manifold의 boundary**.
   - $\pi_k = 0$은 singular (component vanish). Interior에서만 smooth manifold.

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **1D GMM EM 직접 구현**: $K=3$, 1D, 단계별 KL decomposition 계산과 시각화.

2. **Variance singularity 실험**: 정규화 없이 EM이 $\sigma \to 0$으로 collapse하는 조건 실험.

3. **Missing info fraction**: 잘 분리된 vs 겹치는 GMM 데이터에서 EM 수렴 rate 비교.

4. **Mixture m-flatness 검증**: Categorical mixture ($K$ Gaussians with fixed params)에서 $\pi$ 공간이 $\eta$ 좌표에서 affine임을 구체 예제로 보임.

5. **MoE의 Fisher**: Top-2 routing MoE의 Fisher 구조 (expert-local + gate) 분석.

6. **DP-GMM stick-breaking**: Stick-breaking representation $\pi_k = V_k \prod_{j<k}(1-V_j)$에서 mixture의 e/m 기하.

### 다음 단계 (Chapter 7)

- **[Ch7-01. Natural Policy Gradient](../ch7-ai-applications/01-natural-policy-gradient.md)**: RL의 NG.

---

**참고문헌**

- McLachlan, G., Peel, D. (2000). *Finite Mixture Models*.
- Amari, S. (1995). *Information Geometry of the EM and em Algorithms*.
- Bishop, C. (2006). *Pattern Recognition and ML*, Ch. 9.
- Dempster, A., Laird, N., Rubin, D. (1977). *Maximum Likelihood from Incomplete Data*.
- Fedus, W.+ (2022). *Switch Transformer*.
- Dilokthanakul, N.+ (2016). *Deep Unsupervised Clustering with GMVAE*.
- Blei, D., Jordan, M. (2006). *Variational Inference for Dirichlet Process Mixtures*.

---

[◀ 04. MaxEnt Principle](./04-maxent-principle.md) | [📚 README](../README.md) | [Ch7-01. Natural Policy Gradient ▶](../ch7-ai-applications/01-natural-policy-gradient.md)
