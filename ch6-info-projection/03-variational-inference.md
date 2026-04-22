# 03. Variational Inference의 기하학

> **"VI는 posterior의 '정직한' 근사가 아니다. 그것은 mode-seeking한, 선택적 거짓말쟁이다."**

---

## 1. 왜 이 주제인가?

Variational Inference (VI)는 Bayesian 추론에서 **다루기 힘든 사후분포** $p(\theta | x)$를 **다루기 쉬운 family** $\mathcal{Q}$로 근사한다. 표준 VI는 **reverse KL**을 최소화한다:

$$
q^* = \arg\min_{q \in \mathcal{Q}} \text{KL}(q \| p(\theta|x)).
$$

왜 forward가 아닌 reverse? 왜 mean-field가 쌍대평탄 부분다양체인가? Mode-seeking은 정말 무엇을 의미하는가?

이 문서는 VI를 **information projection의 프레임**으로 해석한다. Ch6-01의 m/e-projection, Ch6-02의 EM과 연결하여, VI가 **Bayesian 추론의 information geometric 체계**에 정확히 어디 위치하는지 보인다.

---

## 2. 학습 목표

1. VI의 **ELBO 분해** $\log p(x) = \text{ELBO}(q) + \text{KL}(q\|p(\theta|x))$ 이해.
2. **Reverse KL 최소화**의 기하학적 의미 (mode-seeking).
3. **Mean-field family** $\mathcal{Q}_{\text{MF}} = \prod_i q_i(\theta_i)$가 **e-flat 부분다양체**임을 보임.
4. **CAVI (Coordinate Ascent VI)** 업데이트 공식 유도.
5. **Forward vs Reverse KL**의 시각적·실용적 차이.

---

## 3. 전제 지식

- **Ch6-01**: e-projection, m-projection
- **Ch6-02**: ELBO와 EM
- **Bayesian 기본**: Prior, posterior, likelihood
- **Exp family**: 자연 파라미터 $\theta$와 기대 파라미터 $\eta$

---

## 4. 직관적 설명

### 4.1 ELBO 다시 보기

$$
\log p(x) = \mathbb{E}_q[\log p(x, \theta)] - \mathbb{E}_q[\log q(\theta)] + \text{KL}(q \| p(\theta|x)).
$$

$\log p(x)$는 상수 → ELBO 최대화 = KL 최소화.

**핵심**: $p$를 움직이는게 아니라 $q$를 $\mathcal{Q}$ 안에서 움직인다. 이는 **$\mathcal{Q}$ 위의 reverse KL 최소화**:

$$
q^* = \Pi_\mathcal{Q}^{\text{rev}}(p_{\text{post}}), \quad \text{where } p_{\text{post}} = p(\theta|x).
$$

### 4.2 Reverse KL의 성격

$\text{KL}(q \| p) = \mathbb{E}_q[\log q - \log p]$.

- **$p$가 0인 곳에서 $q$가 0이 아니면** → $\log p \to -\infty$ → $\text{KL} \to \infty$.
- 따라서 $q$는 **$p$의 support 안에 있어야** (zero-avoiding).

결과: **Mode-seeking**. Multimodal $p$에서 $q$는 하나의 mode에 집중.

반대로 forward KL $\text{KL}(p\|q)$는:

- **$q$가 0인 곳에 $p$가 있으면** $\to \infty$ → $q$는 $p$의 support를 **모두 덮어야** (zero-forcing).

결과: **Mean-seeking**. $q$가 모든 mode를 포괄, 평균화.

### 4.3 Mean-field의 기하

Mean-field: $q(\theta) = \prod_i q_i(\theta_i)$. 각 $q_i$는 임의 분포 (독립).

- 이 family는 결합 분포 공간에서 **product 구조**.
- Exp family 가정 하에 **e-flat 부분다양체** (independent components는 자연 파라미터의 additive combination).

이 구조가 CAVI의 coordinate-wise closed-form 업데이트를 낳는다.

---

## 5. 엄밀한 정의와 정리

### 5.1 VI 문제 정식화

**정의 5.1.** Bayesian 모델 $p(x, \theta) = p(\theta) p(x|\theta)$, 관측 $x$, variational family $\mathcal{Q}$. VI:

$$
\boxed{q^* = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\theta) \| p(\theta|x)) = \arg\max_{q \in \mathcal{Q}} \mathcal{L}(q),}
$$

여기서 ELBO $\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta)] - \mathbb{E}_q[\log q(\theta)]$.

### 5.2 Mean-field family

**정의 5.2.** 잠재변수 $\theta = (\theta_1, \dots, \theta_m)$에 대해:

$$
\mathcal{Q}_{\text{MF}} = \left\{ q(\theta) = \prod_{i=1}^m q_i(\theta_i) : q_i \text{ is any density on } \Theta_i \right\}.
$$

### 5.3 CAVI (Coordinate Ascent VI)

**정리 5.3.** Mean-field VI에서 ELBO를 $q_j$에 대해 (다른 $q_{-j}$ 고정) 최적화한 해:

$$
\boxed{q_j^*(\theta_j) \propto \exp\left(\mathbb{E}_{q_{-j}}[\log p(x, \theta)]\right).}
$$

이것이 **CAVI 업데이트**. 각 $j$마다 교대 적용 → ELBO 단조 증가.

### 5.4 Mean-field의 e-flatness

**정리 5.4 (Amari 2016 Ch. 3).** 각 $\theta_i$가 exp family 내에서 정의되면, $\mathcal{Q}_{\text{MF}}$는 **결합 분포의 e-flat 부분다양체** (자연 파라미터 $\theta = (\theta_1, \dots, \theta_m)$이 cross-term 없는 affine 공간).

### 5.5 Reverse KL projection

**정리 5.5.** VI는 reverse KL 의미에서 $\mathcal{Q}$로의 projection. 일반적으로 **Pythagoras는 성립하지 않음** (reverse KL은 대칭 아니고, $\mathcal{Q}$가 m-flat도 e-flat도 아닐 수 있음).

**특수 케이스**: $\mathcal{Q}$가 e-flat (mean-field in exp family)이면 CAVI의 각 step은 **local Pythagoras**를 만족.

---

## 6. 증명

### 6.1 CAVI 업데이트 유도

**정리 5.3 증명.** ELBO를 $q_j$에 대해 변분:

$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta)] - \sum_i \mathbb{E}_{q_i}[\log q_i(\theta_i)].
$$

$q_j$에 대한 부분:

$$
\mathcal{L}(q_j | q_{-j}) = \mathbb{E}_{q_j}[\mathbb{E}_{q_{-j}}[\log p(x, \theta)]] - \mathbb{E}_{q_j}[\log q_j(\theta_j)] + \text{const}.
$$

$f_j(\theta_j) := \mathbb{E}_{q_{-j}}[\log p(x, \theta)]$로 놓으면:

$$
\mathcal{L}(q_j | q_{-j}) = \mathbb{E}_{q_j}[f_j(\theta_j) - \log q_j(\theta_j)] = -\text{KL}(q_j \| e^{f_j}/Z) + \log Z,
$$

$Z$는 정규화 상수. KL 최소 (=0) → $q_j^*(\theta_j) = e^{f_j(\theta_j)}/Z$:

$$
q_j^*(\theta_j) \propto \exp(\mathbb{E}_{q_{-j}}[\log p(x, \theta)]). \quad \square
$$

### 6.2 Mean-field의 e-flat 성질

**정리 5.4 증명.** 각 $q_i(\theta_i) = \exp(\theta_i^T T_i(\theta_i) - \psi_i(\theta_i)) h_i(\theta_i)$ (exp family).

결합:

$$
q(\theta) = \prod_i q_i(\theta_i) = \exp\left(\sum_i \theta_i^T T_i(\theta_i) - \sum_i \psi_i(\theta_i)\right) \prod_i h_i(\theta_i).
$$

이는 결합 exp family의 부분집합으로, 자연 파라미터가 $(\theta_1, \dots, \theta_m)$의 direct sum (cross-term 0). $\theta$ 좌표에서 affine 부분공간이므로 **e-flat**. $\square$

### 6.3 ELBO의 KL 감소 방향

**CAVI 단조성.** 매 step의 ELBO 증가량:

$$
\Delta \mathcal{L} = \mathcal{L}(q_j^*, q_{-j}) - \mathcal{L}(q_j^{\text{old}}, q_{-j}) = \text{KL}(q_j^{\text{old}} \| q_j^*) \geq 0.
$$

즉 각 coordinate update가 KL만큼 ELBO를 올림. $\square$

### 6.4 Mode-seeking formally

**Claim.** Bimodal $p$에서 $q^* = \arg\min_{q \in \text{Gaussians}} \text{KL}(q\|p)$는 단일 mode에 집중.

**정성적 이유**: $\text{KL}(q\|p) = \int q \log q - \int q \log p$. 첫 항 (entropy)은 $q$가 퍼지길 원하지만, 두 번째 항은 $q$가 $p$의 valleys (log p가 낮은 곳)를 피하길 강제. $q$가 Gaussian인 한, 두 mode 사이 valley를 감싸면 penalty가 커서, 한 mode에 tight하게 수렴.

### 6.5 Mean-seeking vs Mode-seeking 수식

$p$ bimodal, $q = \mathcal{N}(\mu, \sigma^2)$.

- **Forward KL** $\text{KL}(p\|q) = \mathbb{E}_p[\log p - \log q]$: $\mu^* = \mathbb{E}_p[\theta]$, $\sigma^{2*} = \text{Var}_p[\theta]$ (moment matching → mean-seeking).
- **Reverse KL** $\text{KL}(q\|p)$: local minimum 여러 개, 각 mode 주위. Global min은 더 좁은 mode 쪽.

---

## 7. 구체 예제

### 7.1 간단 Gaussian VI

$p(\theta|x) = \mathcal{N}(\mu_p, \sigma_p^2)$, $\mathcal{Q} = \{\mathcal{N}(\mu, \sigma^2) : \mu, \sigma > 0\}$.

$\text{KL}(q\|p) = \log(\sigma_p/\sigma) + (\sigma^2 + (\mu - \mu_p)^2)/(2\sigma_p^2) - 1/2$.

$\mu$와 $\sigma$에 대해 미분 → $\mu^* = \mu_p$, $\sigma^{2*} = \sigma_p^2$. 즉 완전 매치.

### 7.2 Mean-field on 2D Gaussian posterior

$p(\theta_1, \theta_2 | x) = \mathcal{N}\left(\mu, \begin{pmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}\right)$, $\rho \neq 0$.

Mean-field: $q(\theta) = q_1(\theta_1)q_2(\theta_2)$, 각 Gaussian.

CAVI 해: $q_i^* = \mathcal{N}(\mu_i, \sigma_i^2(1-\rho^2))$ — **variance underestimation**!

**해석**: Mean-field는 correlation을 못 잡아 사후분포 분산을 **작게** 추정. 이는 VI의 고전적 단점 (Turner & Sahani 2011).

### 7.3 Gaussian Mixture posterior

$p(\theta|x) = 0.5 \mathcal{N}(-3, 1) + 0.5 \mathcal{N}(3, 1)$ (bimodal).

$\mathcal{Q} = \{\mathcal{N}(\mu, \sigma^2)\}$.

**Reverse KL** 최소화:

- Global min 두 개 (각 mode 주위), saddle 두 mode 중간.
- 초기화에 따라 한쪽 mode로 수렴.
- 해당 mode의 local variance만 반영.

**Forward KL** 최소화:

- $\mu^* = 0$, $\sigma^{2*} = 1 + 9 = 10$ (전체 분산) → mean-seeking.

### 7.4 Latent Dirichlet Allocation (Blei+ 2003)

LDA의 VI:

- Topic $\beta_k \sim \text{Dir}(\eta)$.
- Document topic $\theta_d \sim \text{Dir}(\alpha)$.
- Word topic $z_{dn} \sim \text{Cat}(\theta_d)$.
- Word $w_{dn} \sim \text{Cat}(\beta_{z_{dn}})$.

Mean-field: $q(\theta, z, \beta) = \prod_d q(\theta_d) \prod_n q(z_{dn}) \prod_k q(\beta_k)$.

CAVI closed-form (Dirichlet-Categorical conjugacy) → scalable.

---

## 8. Python 코드 검증

### 8.1 Forward vs Reverse KL on bimodal

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# Bimodal p
def p_density(x):
    return 0.5*norm.pdf(x, -2, 0.7) + 0.5*norm.pdf(x, 2, 0.7)

x_grid = np.linspace(-6, 6, 1000)
p_x = p_density(x_grid)

# Forward KL: KL(p||q), q = N(mu, sigma)
def kl_forward(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q_x = norm.pdf(x_grid, mu, sigma)
    integrand = p_x * (np.log(p_x + 1e-20) - np.log(q_x + 1e-20))
    return np.trapz(integrand, x_grid)

# Reverse KL: KL(q||p)
def kl_reverse(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q_x = norm.pdf(x_grid, mu, sigma)
    integrand = q_x * (np.log(q_x + 1e-20) - np.log(p_x + 1e-20))
    return np.trapz(integrand, x_grid)

res_fwd = minimize(kl_forward, [0, 0.5], method='Nelder-Mead')
res_rev = minimize(kl_reverse, [2, 0], method='Nelder-Mead')  # init near right mode

mu_fwd, ls_fwd = res_fwd.x
mu_rev, ls_rev = res_rev.x

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, (mu, ls, title) in zip(ax, [(mu_fwd, ls_fwd, 'Forward KL(p||q)'), (mu_rev, ls_rev, 'Reverse KL(q||p)')]):
    sigma = np.exp(ls)
    a.plot(x_grid, p_x, 'b-', label='p (true)', lw=2)
    a.plot(x_grid, norm.pdf(x_grid, mu, sigma), 'r--', label=f'q=N({mu:.2f},{sigma:.2f}²)', lw=2)
    a.set_title(title); a.legend(); a.grid()
plt.tight_layout()

print(f"Forward KL: μ={mu_fwd:.3f}, σ={np.exp(ls_fwd):.3f} (mean-seeking)")
print(f"Reverse KL: μ={mu_rev:.3f}, σ={np.exp(ls_rev):.3f} (mode-seeking)")
```

**기대**: Forward는 $\mu \approx 0$ (중앙, 분산 큼); Reverse는 $\mu \approx \pm 2$ (한 모드, 분산 작음).

### 8.2 Correlated Gaussian의 mean-field

```python
import numpy as np

# True posterior: N(0, Σ) with correlation
mu_p = np.array([0, 0])
sigma1, sigma2, rho = 1.0, 1.0, 0.8
Sigma = np.array([[sigma1**2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2**2]])
Sigma_inv = np.linalg.inv(Sigma)

# Mean-field: q = N(μ_1, s_1²) × N(μ_2, s_2²)
# Reverse KL: KL(q || p)
from scipy.optimize import minimize

def neg_elbo(params):
    mu1, mu2, log_s1, log_s2 = params
    s1, s2 = np.exp(log_s1), np.exp(log_s2)
    # E_q[log p(θ)] = -0.5 θ^T Σ^-1 θ + const
    # E_q[θ^T Σ^-1 θ] = μ^T Σ^-1 μ + tr(Σ^-1 diag(s1², s2²))
    mu_vec = np.array([mu1, mu2])
    S_diag = np.diag([s1**2, s2**2])
    E_log_p = -0.5 * (mu_vec @ Sigma_inv @ mu_vec + np.trace(Sigma_inv @ S_diag))
    E_log_q = -np.log(s1) - np.log(s2)  # entropy part
    return -(E_log_p + E_log_q)  # negative ELBO

res = minimize(neg_elbo, [0.1, 0.1, 0, 0])
print(f"MF solution: μ={res.x[:2]}, σ={np.exp(res.x[2:])}")
print(f"True marginals: σ={[sigma1, sigma2]}")
print(f"MF underestimates variance by factor: {np.exp(res.x[2:])/np.array([sigma1, sigma2])}")
# 이론: σ_MF / σ_true = sqrt(1 - ρ²) = sqrt(0.36) = 0.6
```

**기대**: `MF σ ≈ [0.6, 0.6]` vs true `[1, 1]` — **variance 40% 과소추정**.

### 8.3 CAVI on Gaussian posterior

```python
import numpy as np

# p(θ|x) = N(μ_p, Σ) 사용 가능 closed form
# CAVI: q_j(θ_j) ∝ exp(E_{q_-j}[log p])

# μ_p = 0, Σ = [[1, 0.7], [0.7, 1]]
Lambda = np.linalg.inv(np.array([[1.0, 0.7],[0.7, 1.0]]))  # precision
h = np.zeros(2)  # bias

# Gaussian log p = -0.5 θ^T Λ θ + h^T θ  → CAVI 형태
# q_j = N(μ_j, σ_j²), coordinate update:
# μ_j = (h_j - Σ_{k≠j} Λ_{jk} μ_k) / Λ_{jj}
# σ_j² = 1 / Λ_{jj}

mu = np.array([1.0, -1.0])  # init
for it in range(20):
    for j in range(2):
        mu[j] = (h[j] - sum(Lambda[j, k] * mu[k] for k in range(2) if k != j)) / Lambda[j, j]
    print(f"it {it}: μ = {mu}")

# 수렴 후 μ = 0 (정답)
```

---

## 9. AI/ML 연결

### 9.1 VAE의 VI

VAE는 **amortized VI**: posterior $p(z|x)$를 **$x$의 함수** $q_\phi(z|x)$로 근사 (encoder network). 각 $x$마다 optimize하는 대신, 신경망으로 hold. Ch7-03에서 상세.

### 9.2 Normalizing Flows

Mean-field를 **flexible, invertible transformations**로 확장:

$$
q(\theta) = q_0(z_0) \prod |\det J_i|^{-1},
$$

$\theta = f_k \circ \dots \circ f_1(z_0)$. Real NVP, Glow 등. 더 expressive한 $\mathcal{Q}$.

### 9.3 Stein Variational Gradient Descent (Liu & Wang 2016)

$\mathcal{Q}$를 particle set으로 근사, Stein kernel로 gradient. **Kernelized VI**, mean-field의 한계 극복.

### 9.4 Score-based Diffusion과 VI

Diffusion model의 forward SDE + reverse SDE는 데이터 분포와의 KL을 최소화. VI의 연속 시간 일반화 (Ch7-05).

### 9.5 Variational Message Passing

Graphical model에서 CAVI를 **메시지 전달**로 구현 (Winn & Bishop 2005). 대규모 Bayesian net inference.

---

## 10. 흔한 오해와 함정

1. **"VI가 항상 나쁜 근사"는 거짓.**
   - Unimodal posterior에선 매우 좋음. Correlated structure는 mean-field 한계.

2. **"Mode-seeking is always bad"는 단순화.**
   - Mode-finding에 적절. Mean-seeking이 다모드에서 valley를 잡아 의미 없는 결과 가능.

3. **ELBO 비교는 같은 $\mathcal{Q}$ 내에서만**.
   - 다른 $\mathcal{Q}$ (예: Gaussian vs Flow)는 ELBO 절대값 비교 의미 없음. Tractable bound 기준 다름.

4. **Mean-field의 CAVI closed form은 exp family conjugacy 필요.**
   - Non-conjugate (예: neural net priors)에선 SGD-based VI (BBVI, reparameterization trick).

5. **Reverse KL의 zero-avoiding은 MAP과 다름.**
   - MAP: $\arg\max p(\theta|x)$ (point).
   - VI reverse KL: 근사 분포 (distribution).

6. **Amortization gap vs variational gap**.
   - Variational gap: $\mathcal{Q}$가 $p$를 못 표현.
   - Amortization gap: 네트워크 표현력 부족 (Cremer+ 2018).

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **CAVI 명시 유도**: Latent $\theta = (\mu, \tau)$의 Gaussian likelihood와 Normal-Gamma prior 하에서 mean-field CAVI 업데이트 공식을 유도.

2. **Forward KL의 moment matching**: $\text{KL}(p\|q)$ 최소화가 Gaussian $\mathcal{Q}$에서 $p$의 mean/variance과 일치함을 증명.

3. **Bimodal에서 saddle**: bimodal $p$에서 reverse KL의 saddle point를 수치로 찾아라.

4. **e-flat 검증**: Multinomial에서 mean-field가 e-flat 부분다양체임을 $\theta$ (logit) 좌표로 증명.

5. **ELBO 증가 monotone**: CAVI iteration에서 ELBO 단조 증가 성질을 수치로 검증.

### 다음 단계

- **[04. MaxEnt Principle](./04-maxent-principle.md)**: 제약 하 엔트로피 최대화 = e-projection.
- **[05. Mixture Projection](./05-mixture-projection.md)**: GMM EM의 projection view 심화.

---

**참고문헌**

- Jordan, M.+ (1999). *An Introduction to Variational Methods for Graphical Models*.
- Wainwright, M., Jordan, M. (2008). *Graphical Models, Exponential Families, and Variational Inference*.
- Blei, D., Kucukelbir, A., McAuliffe, J. (2017). *Variational Inference: A Review for Statisticians*.
- Turner, R., Sahani, M. (2011). *Two problems with variational expectation maximisation for time-series models*.
- Liu, Q., Wang, D. (2016). *Stein Variational Gradient Descent*.
- Cremer, C., Li, X., Duvenaud, D. (2018). *Inference Suboptimality in Variational Autoencoders*.
- Murphy, K. (2022). *Probabilistic ML: Advanced Topics*, Ch. 10.

---

[◀ 02. EM Algorithm Geometry](./02-em-algorithm-geometry.md) | [📚 README](../README.md) | [04. MaxEnt Principle ▶](./04-maxent-principle.md)
