# 01. e-projection과 m-projection

> **"KL divergence 최소화 문제는 어느 인자를 고정하느냐에 따라 두 개의 완전히 다른 기하학적 연산이 된다."**

---

## 1. 왜 이 주제인가?

Ch4에서 exponential family의 **이중 평탄성(dually flat)**과 **일반화 Pythagoras 정리**를 보았다. 이제 그 직접적 응용으로 **information projection**을 다룬다. KL divergence $D(p \| q) = \int p \log(p/q)$를 최소화할 때, 어느 인자를 움직이느냐에 따라 두 가지 projection이 정의된다:

- **m-projection**: $q$ 고정, $p$를 부분다양체 $M$ 위에서 움직여 $D(p \| q)$를 최소화.
- **e-projection**: $p$ 고정, $q$를 부분다양체 $M$ 위에서 움직여 $D(p \| q)$를 최소화.

이 비대칭성은 단순한 수학적 기교가 아니라, **EM 알고리즘, Variational Inference, 최대 엔트로피 원리, Diffusion Score Matching**의 수학적 기반이다. Amari는 이 두 projection이 서로 **Legendre 쌍대**임을 증명하여, 모든 통계 추론의 기하학적 통일 이론을 제시했다.

**이 문서의 목표**: e-projection과 m-projection의 엄밀한 정의, 유일성과 존재성 증명, 두 projection의 기하학적 비대칭성, 그리고 Pythagoras를 통한 **분해 정리**를 다룬다.

---

## 2. 학습 목표

1. **e-geodesic과 m-geodesic**을 정의하고 시각화할 수 있다.
2. **e-flat, m-flat 부분다양체**의 정의를 이해하고 예제를 구성할 수 있다.
3. **m-projection $\Pi_M^{(m)}(q)$와 e-projection $\Pi_M^{(e)}(p)$**의 정의를 내리고 존재성·유일성을 증명할 수 있다.
4. **일반화 Pythagoras 정리**를 이용해 projection의 기하학적 해석을 제시할 수 있다.
5. **두 projection의 비대칭성**이 AI 응용(VI mode-seeking, maxent, EM)에서 무엇을 의미하는지 설명할 수 있다.

---

## 3. 전제 지식

- **Ch4-03**: Legendre 쌍대와 $\theta \leftrightarrow \eta$
- **Ch4-04**: e-connection과 m-connection
- **Ch4-05**: Dually flat manifold, canonical divergence
- **Ch4-06**: 일반화 Pythagoras 정리 $D(P\|R) = D(P\|Q) + D(Q\|R)$
- **KL divergence의 성질**: 비음성, 비대칭성

---

## 4. 직관적 설명

### 4.1 KL의 비대칭을 기하로

$D(p \| q)$는 "$p$의 눈으로 본 $q$의 거리"다. $p$는 기대값을 주고, $q$는 비교 대상이다. 어느 쪽을 움직이느냐:

- **$p$를 움직이면**: 모든 가능한 $p \in M$ 중에서 $q$와 가장 "가까운" $p^*$를 찾음. 이때 $p^*$는 $q$의 **m-projection** on $M$.
- **$q$를 움직이면**: 모든 가능한 $q \in M$ 중에서 $p$와 "가장 가까운" $q^*$를 찾음. 이때 $q^*$는 $p$의 **e-projection** on $M$.

왜 "m"과 "e"인가? 답은 **어떤 geodesic에 수직으로 내려가는가**에 있다.

### 4.2 두 종류 geodesic의 충돌

- **e-geodesic** $\gamma_e$: $\theta$ 좌표 (canonical, $\nabla^{(e)}$-flat)에서 직선. Exp family에서 $p_t \propto p_0^{1-t} p_1^t$.
- **m-geodesic** $\gamma_m$: $\eta$ 좌표 (expectation, $\nabla^{(m)}$-flat)에서 직선. $p_t = (1-t)p_0 + t p_1$ (혼합).

두 geodesic은 **같은 두 분포를 연결하는 다른 경로**. Exp family에서 두 점 $p_0, p_1$을 잇는 e-geodesic과 m-geodesic은 다르다 (Gaussian에서 $\sigma$가 일정 ≠ 밀도의 convex 혼합).

### 4.3 Projection의 기하

- **m-projection** on $M$: $q$에서 $M$으로 **m-geodesic**을 따라 수직 내림. 즉 $M$ 위의 점 $\hat{p}$를 찾아 $q \to \hat{p}$를 m-geodesic으로 잇고, 이 geodesic이 $M$에 수직(=e-flat 부분공간에서 직교).
- **e-projection** on $M$: $p$에서 $M$으로 **e-geodesic**을 따라 수직 내림.

"**m-family에 투영할 땐 e-geodesic으로 내리고, e-family에 투영할 땐 m-geodesic으로 내린다**"는 쌍대성이 핵심.

---

## 5. 엄밀한 정의와 정리

### 5.1 Geodesic의 정의 (Ch4 복습)

**정의 5.1.** 두 분포 $p_0, p_1$을 잇는 경로:

- **e-geodesic**: $\theta$ 좌표에서 $\theta(t) = (1-t)\theta_0 + t\theta_1$. 분포 수준에서 $p_t \propto p_0^{1-t} p_1^t$ (정규화).
- **m-geodesic**: $\eta$ 좌표에서 $\eta(t) = (1-t)\eta_0 + t\eta_1$. 분포 수준에서 $p_t = (1-t)p_0 + t p_1$ (혼합, 단 이 혼합이 같은 exp family 안에 속해야 함).

### 5.2 Flat submanifold

**정의 5.2 (e-flat 부분다양체).** $M \subseteq \mathcal{S}$가 **e-flat**: $\theta$ 좌표에서 affine subspace. 즉 어떤 $\theta_0$와 $V \subseteq \mathbb{R}^n$에 대해 $M = \{\theta_0 + v : v \in V\}$.

**정의 5.3 (m-flat 부분다양체).** $M$이 **m-flat**: $\eta$ 좌표에서 affine subspace.

### 5.3 Projection 정의

**정의 5.4 (m-projection).** $q \in \mathcal{S}$, $M \subseteq \mathcal{S}$ m-flat이라 할 때, $q$의 $M$으로의 **m-projection**:

$$
\Pi_M^{(m)}(q) = \arg\min_{p \in M} D(p \| q).
$$

**정의 5.5 (e-projection).** $p \in \mathcal{S}$, $M \subseteq \mathcal{S}$ e-flat이라 할 때, $p$의 $M$으로의 **e-projection**:

$$
\Pi_M^{(e)}(p) = \arg\min_{q \in M} D(p \| q).
$$

### 5.4 존재성·유일성 정리

**정리 5.6 (Amari 2000, Theorem 3.8).** $\mathcal{S}$가 dually flat, $M$ e-flat이고 $p \in \mathcal{S}$일 때, e-projection $\Pi_M^{(e)}(p)$는 **존재하고 유일**. 또한 다음 **수직성 조건**으로 특징지어진다:

$$
\hat{q} = \Pi_M^{(e)}(p) \iff \hat{q} \in M \text{ and m-geodesic from } p \text{ to } \hat{q} \text{ is orthogonal to } M \text{ at } \hat{q}.
$$

대칭적으로 m-projection은 $M$이 m-flat일 때 유일.

### 5.5 Pythagoras와 projection

**정리 5.7.** $M$ e-flat, $\hat{q} = \Pi_M^{(e)}(p)$면 모든 $q \in M$에 대해:

$$
\boxed{D(p \| q) = D(p \| \hat{q}) + D(\hat{q} \| q).}
$$

이는 Ch4-06의 일반화 Pythagoras의 직접 응용이다.

---

## 6. 증명

### 6.1 정리 5.7 (Pythagoras) 증명

$\hat{q}$는 $p$의 e-projection이므로, $p$에서 $\hat{q}$로의 **m-geodesic**이 $M$에 수직.

$M$ e-flat이므로 $\hat{q}, q \in M$을 잇는 경로는 **e-geodesic**. 

두 geodesic의 수직성을 canonical divergence 공식으로 표현:

$$
D(p \| q) = \psi(\theta_q) + \psi^*(\eta_p) - \theta_q^T \eta_p.
$$

(Ch4-05의 canonical divergence)

이를 $D(p \| \hat{q}) + D(\hat{q} \| q)$로 분해:

$$
D(p \| \hat{q}) + D(\hat{q} \| q) = \left[\psi(\theta_{\hat{q}}) + \psi^*(\eta_p) - \theta_{\hat{q}}^T \eta_p\right] + \left[\psi(\theta_q) + \psi^*(\eta_{\hat{q}}) - \theta_q^T \eta_{\hat{q}}\right].
$$

$M$이 e-flat이므로 $\theta_q = \theta_{\hat{q}} + v$, $v \in V$ (부분공간). $\hat{q}$에서의 수직성은 $(\eta_p - \eta_{\hat{q}})^T v = 0 \quad \forall v \in V$:

$$
\theta_q^T \eta_{\hat{q}} + \theta_{\hat{q}}^T \eta_p = (\theta_{\hat{q}} + v)^T \eta_{\hat{q}} + \theta_{\hat{q}}^T \eta_p = \theta_{\hat{q}}^T \eta_{\hat{q}} + v^T \eta_{\hat{q}} + \theta_{\hat{q}}^T \eta_p.
$$

그리고 Legendre 등식 $\psi(\theta_{\hat{q}}) + \psi^*(\eta_{\hat{q}}) = \theta_{\hat{q}}^T \eta_{\hat{q}}$ (Ch4-03)을 대입. 모든 항 모아 정리하면:

$$
D(p \| \hat{q}) + D(\hat{q} \| q) = \psi(\theta_q) + \psi^*(\eta_p) - \theta_q^T \eta_p + v^T \eta_{\hat{q}} - v^T \eta_p = D(p\|q) + v^T(\eta_{\hat{q}} - \eta_p).
$$

수직성 $v^T(\eta_p - \eta_{\hat{q}}) = 0$에 의해 마지막 항 0. $\square$

### 6.2 유일성 증명

$\hat{q}_1, \hat{q}_2 \in M$ 둘 다 e-projection이라 하자. 정리 5.7을 $q = \hat{q}_2$에 적용:

$$
D(p \| \hat{q}_2) = D(p \| \hat{q}_1) + D(\hat{q}_1 \| \hat{q}_2).
$$

$\hat{q}_1, \hat{q}_2$ 모두 최솟값이므로 $D(p\|\hat{q}_2) = D(p\|\hat{q}_1)$, 따라서 $D(\hat{q}_1\|\hat{q}_2) = 0$, KL의 비음성+양성 명확성에 의해 $\hat{q}_1 = \hat{q}_2$. $\square$

### 6.3 존재성

$\mathcal{S}$ 열린 볼록, $M$ 닫힌 affine (e-flat), $D(p \| \cdot)$ $\theta$의 볼록 함수(Ch4-05). 강제성 (coercivity)은 Legendre 조건 하에 성립 (Rockafellar 1970 Ch.26). 따라서 $\arg\min$ 존재. $\square$

### 6.4 m-projection의 수직성

**정리 6.1.** $M$ m-flat, $q \in \mathcal{S}$, $\hat{p} = \Pi_M^{(m)}(q)$면 **모든 $p \in M$**에 대해 $p$에서 $\hat{p}$로의 **e-geodesic**이 $M$에 수직 @ $\hat{p}$:

$$
D(p \| q) = D(p \| \hat{p}) + D(\hat{p} \| q).
$$

증명은 정리 5.7과 대칭. $\square$

---

## 7. 구체 예제

### 7.1 정규분포: 평균 고정 e-family로 m-projection

**설정.** $\mathcal{S} = \{\mathcal{N}(\mu, 1) : \mu \in \mathbb{R}\}$ (분산 고정, $\mu$만 파라미터). 이 전체가 e-flat (canonical $\theta = \mu$, $\psi = \mu^2/2$).

부분다양체 $M = \{p : \mathbb{E}_p[X] = 0\} \cap \mathcal{S} = \{\mathcal{N}(0, 1)\}$ (단일 원소). 

임의 $q = \mathcal{N}(\mu_q, 1)$의 m-projection on $M$은 당연히 $\mathcal{N}(0,1)$, 수직성은 자명 (단일 원소).

**더 재밌는 예**: $\mathcal{S} = \{\mathcal{N}(\mu, \sigma^2)\}$ 전체, $M = \{\sigma = 1\}$ (e-flat). $q = \mathcal{N}(3, 4)$ (즉 $\sigma=2$)의 e-projection on $M$:

$$
\Pi_M^{(e)}(q) = \arg\min_{p \in M} D(q \| p) = \arg\min_{\mu} D(\mathcal{N}(3,4) \| \mathcal{N}(\mu, 1)).
$$

$D(\mathcal{N}(3,4)\|\mathcal{N}(\mu,1)) = \log(1/2) + (4 + (3-\mu)^2)/2 - 1/2$. $\mu$에 대한 미분 0 → $\mu = 3$. 즉 $\hat{q} = \mathcal{N}(3, 1)$.

### 7.2 m-projection: 데이터 평균 일치

**설정.** Empirical distribution $\hat{P} = \frac{1}{N}\sum \delta_{x_i}$, $M = $ Gaussian family. Gaussian MLE는 **$\hat{P}$의 m-projection on Gaussian family**:

$$
\hat{\theta} = \arg\min_{\theta} D(\hat{P} \| p_\theta) = \arg\max_\theta \mathbb{E}_{\hat{P}}[\log p_\theta] = \arg\max_\theta \frac{1}{N}\sum \log p_\theta(x_i).
$$

즉 **MLE = m-projection of empirical distribution onto model family**. Exp family에선 moment matching $\mathbb{E}_{p_\theta}[T] = \bar{T}$로 귀결 (Ch4-02).

### 7.3 e-projection: 정보 반영

**Maxent 예고 (Ch6-04):** 제약 $\mathbb{E}_q[T_i] = \mu_i$를 만족하는 $q$ 중에서 **uniform**(또는 prior $p_0$)와 가장 가까운 $q$는 $p_0$의 **e-projection**을:

$$
q^* = \arg\min_{q : \mathbb{E}_q[T_i] = \mu_i} D(q \| p_0).
$$

해는 $q^*(x) = p_0(x) \exp(\sum_i \lambda_i T_i(x) - \psi(\lambda))$ (exp family). 이것이 **maxent = e-projection** 해석.

### 7.4 VI는 reverse KL = e-projection?

VI: $q^* = \arg\min_q \text{KL}(q \| p)$. Forward KL $\text{KL}(p\|q)$가 아니라 **reverse**. 이는:

- $p$ (true posterior): 고정, $q$ (variational): 이동
- $q$를 variational family $\mathcal{Q}$ 안에서 이동 → $\mathcal{Q}$가 e-flat이면 $\Pi_\mathcal{Q}^{(m)}$에 해당?

실제로 $\min_q D(q\|p)$는 **m-projection**(q 움직임, 고정은 p)이 아니라 **reverse**이므로 다름. Amari 2016 Ch.3의 용어:

- $\min_{p \in M} D(p \| q) = \Pi_M^{(m)}(q)$ — m-projection
- $\min_{q \in M} D(p \| q) = \Pi_M^{(e)}(p)$ — e-projection
- $\min_{q \in M} D(q \| p) = $ **"reverse e-projection"** (일부 문헌에서 m-projection이라 혼용 주의)

이 문서는 Amari의 원 정의 (5.4, 5.5)를 따름. Ch6-03에서 VI의 분류 재논의.

---

## 8. Python 코드 검증

### 8.1 1D Gaussian e-projection

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# S = N(mu, sigma^2), M = {sigma = 1} (e-flat in natural coords)
q_mu, q_sigma = 3.0, 2.0  # 외부 점

# KL from q to p, with p = N(mu, 1)
def kl_qp(mu, q_mu=q_mu, q_sigma=q_sigma):
    return np.log(1/q_sigma) + (q_sigma**2 + (q_mu - mu)**2)/2 - 0.5

res = minimize_scalar(kl_qp)
print(f"e-projection μ* = {res.x:.6f} (이론: {q_mu})")
print(f"minimum KL = {res.fun:.6f}")
```

**기대 출력**: `μ* ≈ 3.000000`.

### 8.2 Categorical에서 m-projection

**설정.** $\mathcal{S} = \Delta_3$ (3-simplex), $M = \{p : p_1 = p_2\}$ (m-flat in $\eta=p$ coords).

```python
import numpy as np
from scipy.optimize import minimize

q = np.array([0.2, 0.5, 0.3])  # 외부 점

# M: p_1 = p_2, sum = 1 → 파라미터: p_1, p_3
def kl_pq(params, q=q):
    p1, p3 = params
    p2 = p1
    if p1 <= 0 or p3 <= 0 or (1 - 2*p1 - p3) < -1e-10:
        return 1e10
    p = np.array([p1, p2, p3])
    p = np.clip(p, 1e-10, 1)
    return np.sum(p * np.log(p/q))

# p_2 = p_1, p_3 = 1 - 2p_1 - ??? (실은 p_1+p_2+p_3=1 → p_3 = 1-2p_1)
# 1D: optimize over p_1
def kl_1d(p1, q=q):
    if p1 <= 0 or 1 - 2*p1 <= 0:
        return 1e10
    p = np.array([p1, p1, 1-2*p1])
    return np.sum(p * np.log(p/q))

from scipy.optimize import minimize_scalar
res = minimize_scalar(kl_1d, bounds=(0.001, 0.499), method='bounded')
p1_star = res.x
print(f"m-projection: p* = ({p1_star:.4f}, {p1_star:.4f}, {1-2*p1_star:.4f})")
print(f"min KL = {res.fun:.6f}")

# Pythagoras 검증: for any p in M, D(p||q) = D(p||p_star) + D(p_star||q)
p_star = np.array([p1_star, p1_star, 1-2*p1_star])
for p1_test in [0.1, 0.2, 0.3, 0.4]:
    p_test = np.array([p1_test, p1_test, 1-2*p1_test])
    lhs = np.sum(p_test * np.log(p_test/q))
    rhs = np.sum(p_test * np.log(p_test/p_star)) + np.sum(p_star * np.log(p_star/q))
    print(f"  p1={p1_test}: D(p||q)={lhs:.6f}, D(p||p*)+D(p*||q)={rhs:.6f}, 차이={abs(lhs-rhs):.2e}")
```

**기대**: Pythagoras 동등식이 **모든 $p \in M$**에 대해 기계 정밀도로 성립.

### 8.3 e-geodesic과 m-geodesic 비교

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 두 Gaussian
p0 = lambda x: norm.pdf(x, -2, 1)
p1 = lambda x: norm.pdf(x, 2, 1)

x = np.linspace(-6, 6, 400)

# m-geodesic: convex combination
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for t in np.linspace(0, 1, 5):
    pt_m = (1-t)*p0(x) + t*p1(x)
    ax[0].plot(x, pt_m, label=f't={t:.2f}', alpha=0.7)
ax[0].set_title('m-geodesic: $(1-t)p_0 + t p_1$')
ax[0].legend(); ax[0].grid()

# e-geodesic: p_t ∝ p_0^(1-t) p_1^t
for t in np.linspace(0, 1, 5):
    unnormalized = p0(x)**(1-t) * p1(x)**t
    pt_e = unnormalized / np.trapz(unnormalized, x)
    ax[1].plot(x, pt_e, label=f't={t:.2f}', alpha=0.7)
ax[1].set_title('e-geodesic: $p_0^{1-t} p_1^t$ (normalized)')
ax[1].legend(); ax[1].grid()
plt.tight_layout()
```

**관찰**: m-geodesic은 두 봉우리가 **동시에 존재**(혼합), e-geodesic은 **중간에 한 봉우리로 점점 이동** (곱셈적 평균).

---

## 9. AI/ML 연결

### 9.1 MLE as m-projection

Empirical distribution $\hat{P}$의 model $\{p_\theta\}$로의 m-projection = MLE. 이 해석이 **"데이터를 모델에 사영"**이라는 기하학적 직관을 제공.

### 9.2 VI와 projection

- **Forward KL ($\text{KL}(p\|q)$) 최소화** (이른바 M-projection onto q-family): mean-seeking (정답 모드 모두 커버).
- **Reverse KL ($\text{KL}(q\|p)$) 최소화**: mode-seeking (단일 모드에 집중). **VI가 이 방향을 주로 씀**.

이 차이는 Ch6-03에서 상세히.

### 9.3 EM의 두 projection

- E-step: $q(z|x) \leftarrow p(z|x, \theta^{\text{old}})$ — **m-projection**.
- M-step: $\theta \leftarrow \arg\max \mathbb{E}_q[\log p(x,z|\theta)]$ — **e-projection**.

두 projection의 교대로 ELBO 단조 증가 (Ch6-02).

### 9.4 MaxEnt = e-projection

제약 하의 엔트로피 최대화는 uniform prior로부터의 **e-projection**. Jaynes의 원리와 Amari의 기하가 만남 (Ch6-04).

### 9.5 Diffusion의 score matching

Fisher divergence $\mathbb{E}[\|\nabla \log p - \nabla \log q\|^2]$의 최소화는 local e-projection의 연속 시간 극한으로 해석 가능 (Ch7-05).

---

## 10. 흔한 오해와 함정

1. **Forward vs Reverse KL을 혼동**.
   - $\min_p D(p\|q)$ (p 움직임) = m-projection.
   - $\min_q D(p\|q)$ (q 움직임) = e-projection.
   - $\min_q D(q\|p)$ (reverse KL) ≠ 위 둘. VI의 경우.

2. **"m-projection"이 m-flat family에만 정의되는 건 아니다.**
   - Amari 정의 (5.4): $M$이 어떤 family든 수학적 정의 가능. 단 **Pythagoras 정리는 m-flat family에서만** 성립.

3. **수직성은 Fisher 내적으로 잰다.**
   - 유클리드 수직이 아님. 좌표 의존적 계산 주의.

4. **존재성과 고유성은 자동이 아니다.**
   - $M$이 닫혀있지 않으면 존재성 실패 가능 (boundary에서 infimum).
   - Dually flat 조건 중요.

5. **KL의 비음성과 projection의 최솟값이 0이 되는 것은 다르다.**
   - 최솟값 = 0은 $p \in M$일 때만. 일반적으로 $D(\hat{p}\|q) > 0$.

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **Categorical Pythagoras**: $\Delta_3$에서 $M = \{p_3 = 0\}$ (m-flat? e-flat?)에 대한 m-projection 계산, Pythagoras 확인.

2. **Gaussian에서의 이중 projection**: $\mathcal{S} = \{\mathcal{N}(\mu, \sigma^2)\}$, $M_1 = \{\mu = 0\}$ (e-flat or m-flat?), $M_2 = \{\mathbb{E}[X^2] = 1\}$ (e-flat or m-flat?). 두 경우 각각 projection 계산.

3. **MLE 재확인**: Bernoulli model에서 empirical $\hat{p}$의 Bernoulli family로의 m-projection이 MLE $\hat{\theta} = \bar{x}$임을 직접 증명.

4. **e-projection 수식**: Exp family $p_\theta$와 e-flat $M = \{\theta : \theta_1 = c\}$에 대해 외부 점 $p$의 e-projection 공식.

5. **Reverse KL의 projection 성질**: $\min_q D(q\|p)$의 조건부 mean-field 해 (Ch6-03 예고).

### 다음 단계

- **[02. EM 알고리즘의 Information Geometry](./02-em-algorithm-geometry.md)**: E-step = m-proj, M-step = e-proj.
- **[03. Variational Inference](./03-variational-inference.md)**: ELBO = reverse KL.
- **[04. MaxEnt](./04-maxent-principle.md)**: Jaynes ↔ Amari.

---

**참고문헌**

- Amari, S., Nagaoka, H. (2000). *Methods of Information Geometry*, AMS. Ch. 3.
- Amari, S. (2016). *Information Geometry and Its Applications*. Ch. 2, 3.
- Csiszár, I. (1975). *I-divergence geometry of probability distributions and minimization problems*.
- Murphy, K. (2022). *Probabilistic Machine Learning: Advanced Topics*. Ch. 10.

---

[◀ Ch5-05. K-FAC·Shampoo](../ch5-natural-gradient/05-kfac-shampoo.md) | [📚 README](../README.md) | [02. EM Algorithm Geometry ▶](./02-em-algorithm-geometry.md)
