# 7.4 리만 해밀토니안 몬테카를로 (Riemannian HMC)

[◀ 03. VAE Geometry](./03-vae-geometry.md) | [📚 README](../README.md) | [05. Diffusion Fisher ▶](./05-diffusion-fisher.md)

---

## 1. 왜 이것을 배우는가? (Motivation)

표준 해밀토니안 몬테카를로(HMC)는 **고정된 질량 행렬(mass matrix) $M = I$** 를 가정한다. 그러나 사후분포가 **비등방(anisotropic)** 이거나 **매우 다른 스케일의 파라미터**를 가질 때, 유클리드 HMC는 병약조건(ill-conditioning)으로 인해 **느린 혼합(slow mixing)** 을 보인다.

Girolami & Calderhead (2011)의 핵심 통찰은 이것이다:

> **사후분포 $p(\theta \mid x)$의 국소적 기하를 Fisher 정보 행렬 $G(\theta)$로 포착하고, 이를 해밀토니안의 질량 행렬로 사용한다.**

$$
H(\theta, p) = -\log p(\theta \mid x) + \frac{1}{2} \log |G(\theta)| + \frac{1}{2} p^\top G(\theta)^{-1} p
$$

이 **Riemannian Manifold HMC (RMHMC)** 는 정보기하가 베이지안 샘플링에 직접 응용된 가장 아름다운 예시 중 하나이며, Stan, PyMC3 등 현대 확률 프로그래밍의 NUTS 기본 옵션에 영향을 주었다.

---

## 2. 학습 목표 (Learning Objectives)

이 챕터를 마치면 다음을 할 수 있다:

1. 표준 HMC의 샘플링 효율과 혼합 시간을 해석할 수 있다.
2. Fisher 정보 $G(\theta)$를 질량 행렬로 채택하는 RMHMC의 **리만 해밀토니안**을 쓸 수 있다.
3. 계량 텐서가 **위치에 의존**할 때 발생하는 **implicit leapfrog** 적분기를 유도할 수 있다.
4. RMHMC의 혼합 시간이 비등방 사후분포에서 표준 HMC보다 개선됨을 정량적으로 설명할 수 있다.
5. SoftAbs 계량(Betancourt 2013), Lagrangian MC 등 현대적 변형을 이해한다.

---

## 3. 전제 지식 (Prerequisites)

- [5.1 Euclidean Gradient 문제](../ch5-natural-gradient/01-euclidean-gradient-problem.md)
- [5.4 파라미터화 불변성](../ch5-natural-gradient/04-parameterization-invariance.md)
- 해밀토니안 역학, 심플렉틱 적분, Störmer-Verlet/leapfrog
- MCMC 기본: 디테일드 밸런스, 에르고딕성
- [2.3 Fisher-Rao 계량](../ch2-statistical-fisher/03-fisher-rao-metric.md)

---

## 4. 직관적 이해 (Intuition)

### 4.1 표준 HMC 복습

HMC는 확률변수 $\theta$에 보조 운동량 $p \sim \mathcal{N}(0, M)$를 도입해 **증강된 결합분포**를 만든다:

$$
\pi(\theta, p) \propto \exp\!\left(-H(\theta, p)\right), \quad H = U(\theta) + \frac{1}{2} p^\top M^{-1} p
$$

여기서 $U(\theta) = -\log p(\theta \mid x)$는 **포텐셜 에너지**다. 심플렉틱 적분기(leapfrog)로 Hamilton 방정식을 $L$ 스텝 수치적으로 풀어 새 제안 $(\theta^*, p^*)$를 만든 후 Metropolis 교정을 수행한다.

**문제**: 만약 $U$의 헤시안(=사후분포의 공분산의 역)이 매우 **비등방적**이라면, $M = I$는 거의 평평한 방향과 매우 좁은 방향을 똑같이 탐색해 비효율적이다.

### 4.2 질량 = 기하

직관적으로, **질량이 큰 방향 = 해당 방향으로 움직이기 어려움 = 좁은 분포**. 즉 질량 행렬은 사후분포의 국소적 "폭"을 반영해야 한다:

$$
M(\theta) \approx \text{Cov}\!\left[\nabla U\right]^{-1} \text{ 또는 } \nabla^2 U
$$

Fisher 정보 $G(\theta) = \mathbb{E}[\nabla U \nabla U^\top]$ 는 정확히 이 정보를 제공한다(가우시안 근사 하에서).

### 4.3 기하학적 해석

**유클리드 HMC**: $\mathbb{R}^d$에서 움직이며, 거리는 $\sqrt{d\theta^\top d\theta}$.

**RMHMC**: 통계 다양체 $\mathcal{S}$ 위에서 움직이며, 거리는 $\sqrt{d\theta^\top G(\theta) d\theta}$. Fisher 계량을 따라 **측지선**을 근사하며 샘플링한다.

비유: 구 위에서 적도와 극점을 넘나들 때, 유클리드 좌표는 불균일하지만 리만 계량(위도/경도에 의존하는 $g$)을 쓰면 자연스럽다.

---

## 5. 엄밀한 전개 (Rigorous Development)

### 5.1 리만 해밀토니안

자연 다양체 위의 해밀토니안은 다음과 같이 정의된다(Girolami-Calderhead):

$$
\boxed{H(\theta, p) = U(\theta) + \frac{1}{2}\log |G(\theta)| + \frac{1}{2} p^\top G(\theta)^{-1} p}
$$

여기서:

- $U(\theta) = -\log p(x \mid \theta) - \log p(\theta)$ : 베이지안 포텐셜
- $\frac{1}{2}\log|G(\theta)|$ : 리만 부피 요소 (log determinant 보정)
- $p \sim \mathcal{N}(0, G(\theta))$ : **위치 의존 운동량**

**동치 분포 검증**: 운동량 $p$를 주변화하면

$$
\int e^{-H(\theta, p)} dp = e^{-U(\theta)} \cdot \frac{1}{|G(\theta)|^{1/2}} \cdot \underbrace{\int e^{-\tfrac{1}{2}p^\top G^{-1} p} dp}_{=(2\pi)^{d/2} |G|^{1/2}} \propto e^{-U(\theta)}
$$

즉 **$\theta$의 주변분포는 정확히 사후분포** $p(\theta \mid x)$다. log det 항이 정확히 Jacobian을 상쇄함에 주목.

### 5.2 Hamilton 방정식

$$
\frac{d\theta_i}{dt} = \frac{\partial H}{\partial p_i}, \qquad \frac{dp_i}{dt} = -\frac{\partial H}{\partial \theta_i}
$$

계산하면:

$$
\dot\theta = G(\theta)^{-1} p
$$

$$
\dot p_i = -\frac{\partial U}{\partial \theta_i} - \frac{1}{2}\mathrm{tr}\!\left(G^{-1} \frac{\partial G}{\partial \theta_i}\right) + \frac{1}{2} p^\top G^{-1} \frac{\partial G}{\partial \theta_i} G^{-1} p
$$

여기서 두 번째 항은 $\log|G|$의 미분에서, 세 번째 항은 운동 에너지의 $\theta$ 의존에서 나온다.

### 5.3 심플렉틱 보존

RMHMC의 시간 전개는 여전히 **심플렉틱 사상**이다(Hamilton 방정식의 성질). 따라서 이론적으로 연속적 흐름은 $\pi(\theta, p)$를 정확히 보존한다. 문제는 **수치 적분**이다.

### 5.4 Implicit Leapfrog

표준 leapfrog는 $M = I$에서 $\dot p = -\nabla U$가 $\theta$에 명시적으로 의존하지 않으므로 쉽다. RMHMC는 **운동 에너지조차 $\theta$에 의존**하므로 다음과 같은 **내재적(implicit) 적분기**를 필요로 한다:

$$
p_{n+1/2} = p_n - \frac{\epsilon}{2} \nabla_\theta H(\theta_n, p_{n+1/2})
$$

$$
\theta_{n+1} = \theta_n + \frac{\epsilon}{2}\!\left[\nabla_p H(\theta_n, p_{n+1/2}) + \nabla_p H(\theta_{n+1}, p_{n+1/2})\right]
$$

$$
p_{n+1} = p_{n+1/2} - \frac{\epsilon}{2}\nabla_\theta H(\theta_{n+1}, p_{n+1/2})
$$

첫 번째와 두 번째 스텝은 **고정점 반복**(fixed-point iteration)으로 풀어야 한다. 심플렉틱성과 시간 반전 대칭성을 유지하도록 설계된 **generalized leapfrog** (Leimkuhler-Reich 2004)다.

### 5.5 혼합 시간 이론

**가우시안 사후분포** $p(\theta \mid x) = \mathcal{N}(0, \Sigma)$ 에서 $\Sigma$의 조건수를 $\kappa = \lambda_{\max}/\lambda_{\min}$이라 하자.

**표준 HMC (M=I)** 의 혼합 시간: $T_{\text{mix}} \sim \kappa$ (비등방 조건수에 비례).

**RMHMC (M=G=Σ⁻¹)** 의 혼합 시간: $T_{\text{mix}} \sim 1$ (조건수 독립, 전이 후 완전 등방).

이는 NGD에서 Newton 방향이 조건수를 제거하는 것과 같은 원리다.

---

## 6. 증명 (Proofs)

### 6.1 정리: 주변화 후 목표분포 회복

**명제**: RMHMC의 평형분포에서 $\theta$의 주변분포는 $p(\theta \mid x) \propto e^{-U(\theta)}$이다.

**증명**: 결합 평형분포는

$$
\pi(\theta, p) \propto \exp(-H(\theta, p)) = e^{-U(\theta)} \cdot |G(\theta)|^{-1/2} \cdot e^{-p^\top G(\theta)^{-1} p / 2}
$$

$p$에 대해 주변화:

$$
\pi(\theta) = e^{-U(\theta)} |G|^{-1/2} \int e^{-\tfrac{1}{2}p^\top G^{-1} p} dp
$$

가우시안 적분은 $(2\pi)^{d/2} |G|^{1/2}$이므로:

$$
\pi(\theta) \propto e^{-U(\theta)} |G|^{-1/2} |G|^{1/2} = e^{-U(\theta)}
$$

즉 $\log|G|/2$ 보정항이 정확히 Jacobian을 제거한다. $\blacksquare$

### 6.2 정리: 심플렉틱성 하의 측정 보존

**명제**: Generalized leapfrog는 $\omega = dp \wedge d\theta$를 보존한다.

**증명 개요**: Generalized leapfrog는 세 개의 심플렉틱 사상의 합성:

1. $\Phi_1(\theta, p) = (\theta, p - \tfrac{\epsilon}{2}\nabla_\theta H(\theta, p'))$ (implicit)
2. $\Phi_2(\theta, p) = (\theta + \epsilon \bar v(\theta, p), p)$ where $\bar v$ symmetric
3. $\Phi_3 = \Phi_1$ with swapped roles

각각이 심플렉틱(Leimkuhler-Reich §VI)이므로 합성도 심플렉틱. 따라서 Metropolis 보정으로 정확한 샘플링이 가능하다. $\blacksquare$

### 6.3 정리: 조건수 제거 (Gaussian 한정)

**명제**: $p(\theta) = \mathcal{N}(0, \Sigma)$에 대해 $M = \Sigma^{-1}$을 택하면 해밀토니안 궤적은 원형 진동이 된다.

**증명**: 해밀토니안은

$$
H = \frac{1}{2}\theta^\top \Sigma^{-1} \theta + \frac{1}{2} p^\top \Sigma p
$$

Hamilton 방정식:

$$
\dot\theta = \Sigma p, \quad \dot p = -\Sigma^{-1}\theta
$$

$\ddot\theta = \Sigma \dot p = -\theta$이므로 모든 고유 방향이 같은 각진동수 $\omega = 1$로 진동한다. 즉 조건수와 무관하게 모든 방향이 동시에 탐색된다. $\blacksquare$

---

## 7. 예제 (Examples)

### 7.1 예제 1: 2D 비등방 가우시안

사후분포 $p(\theta) = \mathcal{N}(0, \Sigma)$, $\Sigma = \text{diag}(1, 100)$.

**유클리드 HMC** ($M=I$): 작은 방향(분산 1)에 맞춰 스텝 크기 조절 필요 → 큰 방향(분산 100)은 수백 스텝 걸쳐야 탐색됨.

**RMHMC** ($G = \Sigma^{-1} = \text{diag}(1, 0.01)$): 각 방향의 자연 시간 스케일 정규화 → 한 trajectory로 전체 탐색.

### 7.2 예제 2: 로지스틱 회귀

$\log p(y \mid X, \theta) = \sum_i [y_i \theta^\top x_i - \log(1 + e^{\theta^\top x_i})]$

피셔 정보:

$$
G(\theta) = X^\top \mathrm{diag}(\sigma(X\theta)(1-\sigma(X\theta))) X
$$

이는 **관측값 $y$에 의존하지 않음**(분포에 대한 기댓값)이므로 데이터 의존 MAP 주변에서도 잘 작동. Girolami-Calderhead 원 논문의 대표 벤치마크.

### 7.3 예제 3: 계층 모델의 퐁당(funnel) 분포

Neal's funnel:
$$
\log \sigma^2 \sim \mathcal{N}(0, 3^2), \quad x_i \mid \sigma^2 \sim \mathcal{N}(0, \sigma^2)
$$

$\log\sigma^2$의 값에 따라 $x_i$의 스케일이 극적으로 바뀜 → 유클리드 HMC는 "깔때기 목"을 탐색 못함. RMHMC는 국소 Fisher를 사용해 목을 통과할 수 있음(Betancourt 2013).

---

## 8. 코드 (Code)

```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def rmhmc_step(theta, U, grad_U, G, dG_dtheta, eps, L, n_fp=5):
    """
    Riemannian HMC one proposal step via generalized leapfrog.
    
    Args:
        theta: current position (d,)
        U(theta): potential energy (= -log posterior)
        grad_U(theta): gradient of U
        G(theta): Fisher information matrix (d, d)
        dG_dtheta(theta): list of d matrices, dG/dtheta_i
        eps: step size
        L: number of leapfrog steps
        n_fp: fixed-point iterations for implicit steps
    
    Returns:
        theta_new, p_new, H_new
    """
    d = len(theta)
    Gt = G(theta)
    # Sample momentum p ~ N(0, G(theta))
    cho, _ = cho_factor(Gt, lower=True)
    p = cho @ np.random.randn(d)
    
    H0 = U(theta) + 0.5*np.log(np.linalg.det(Gt)) + 0.5*p @ np.linalg.solve(Gt, p)
    
    for _ in range(L):
        # Step 1: p half-step (implicit in p)
        p_half = p.copy()
        for _ in range(n_fp):
            # dH/dtheta = grad_U + 0.5 tr(G^-1 dG) - 0.5 p^T G^-1 dG G^-1 p
            Gt = G(theta)
            Ginv_p = np.linalg.solve(Gt, p_half)
            dGs = dG_dtheta(theta)
            dH_dtheta = grad_U(theta).copy()
            for i, dGi in enumerate(dGs):
                dH_dtheta[i] += 0.5*np.trace(np.linalg.solve(Gt, dGi))
                dH_dtheta[i] -= 0.5*Ginv_p @ dGi @ Ginv_p
            p_half = p - 0.5*eps*dH_dtheta
        
        # Step 2: theta full-step (implicit in theta)
        theta_new = theta.copy()
        for _ in range(n_fp):
            Gt_new = G(theta_new)
            v = 0.5*(np.linalg.solve(G(theta), p_half) + np.linalg.solve(Gt_new, p_half))
            theta_new = theta + eps*v
        theta = theta_new
        
        # Step 3: p half-step (explicit with new theta)
        Gt = G(theta)
        Ginv_p = np.linalg.solve(Gt, p_half)
        dGs = dG_dtheta(theta)
        dH_dtheta = grad_U(theta).copy()
        for i, dGi in enumerate(dGs):
            dH_dtheta[i] += 0.5*np.trace(np.linalg.solve(Gt, dGi))
            dH_dtheta[i] -= 0.5*Ginv_p @ dGi @ Ginv_p
        p = p_half - 0.5*eps*dH_dtheta
    
    # Metropolis correction
    Gt = G(theta)
    H_new = U(theta) + 0.5*np.log(np.linalg.det(Gt)) + 0.5*p @ np.linalg.solve(Gt, p)
    
    return theta, p, H_new, H0


# Example: banana distribution
def banana_U(theta, a=1, b=1):
    x, y = theta
    return 0.5*((x/a)**2 + ((y + b*(x**2 - a**2))/1.0)**2)

def banana_grad(theta, a=1, b=1):
    x, y = theta
    gx = x/a**2 + 2*b*x*(y + b*(x**2 - a**2))
    gy = (y + b*(x**2 - a**2))
    return np.array([gx, gy])

def banana_G(theta, a=1, b=1):
    """Fisher approximation: outer product of score."""
    g = banana_grad(theta, a, b)
    return np.outer(g, g) + 1e-3*np.eye(2)  # regularized

# Run a chain... (full example available in repo)
```

### 8.2 SoftAbs 계량 (Betancourt 2013)

$G(\theta)$가 positive definite가 아닐 수 있는 경우(e.g. 관측 Fisher):

$$
G_{\text{SoftAbs}} = V \Lambda_{\text{softabs}} V^\top, \quad \Lambda_{\text{softabs}}(\lambda) = \lambda \coth(\alpha \lambda)
$$

음의 고유값을 부드럽게 절댓값으로 변환하여 항상 PD를 보장.

---

## 9. AI/ML 응용 (AI/ML Applications)

### 9.1 베이지안 딥러닝

- **Stochastic Gradient RMHMC (SGRHMC)** : 미니배치로 Fisher 근사.
- **Fisher as mass matrix** : K-FAC 방식으로 블록 대각화하여 대규모 네트워크에 적용 가능(Chen et al. 2014).
- **Flat minima 탐색**: Fisher의 고유값이 flatness를 반영; RMHMC는 자연스럽게 평평한 영역에서 오래 머뭄.

### 9.2 확률 프로그래밍 (Stan, PyMC)

Stan의 NUTS 알고리즘은 질량 행렬 적응(mass adaptation)으로 **warm-up** 기간 동안 추정된 사후 공분산을 사용. 이는 "부분적 RMHMC"로 볼 수 있음. 완전 RMHMC는 위치마다 행렬이 바뀌므로 비싸지만, 복잡한 계층 모델에서 필수.

### 9.3 불확실성 정량화

의료, 금융 등 고위험 영역에서는 단순 MAP이 아닌 **전체 사후분포**가 필요. RMHMC는 퐁당, 다봉 사후분포에서 안정적으로 샘플링하여 신뢰 구간과 예측 변동성을 정확히 제공.

### 9.4 변분 추론과의 비교

VI는 빠르지만 사후분포 근사에 편향 존재. RMHMC는 느리지만 점근적으로 무편향 샘플. **하이브리드**: VI로 $G(\theta)$를 근사한 후 RMHMC 초기화(Liu & Lee 2017).

### 9.5 조정된 랑주뱅 알고리즘 (MALA)

Girolami-Calderhead는 동일한 논문에서 **Manifold MALA (MMALA)** 도 제안:

$$
\theta' = \theta + \frac{\epsilon^2}{2} G^{-1}(\theta) \nabla \log p(\theta) + \epsilon G^{-1/2}(\theta) \xi
$$

이는 RMHMC의 1-step 한계. 딥러닝에서 **preconditioned SGLD**와 연결됨(Li et al. 2016).

### 9.6 확산 모델과의 교량

다음 문서(7.5)에서 다룰 score-based 확산 모델은 **랑주뱅 샘플링**에 기반하며, 위치 의존 **score network**가 사실상 위치 의존 preconditioner를 학습. RMHMC가 이론적 기반을 제공.

---

## 10. 흔한 오해 (Common Pitfalls)

### 10.1 "Fisher = Hessian이면 Newton과 동일"

다르다. Newton's method는 $\nabla^2 U$(관측 Hessian)을 쓰고, RMHMC는 expected Fisher $\mathbb{E}[\nabla U (\nabla U)^\top]$를 선호한다. 전자는 비가역(음의 고유값 가능), 후자는 PSD.

### 10.2 "Log det 항을 무시해도 돼"

안 된다. $|G(\theta)|^{1/2}$는 리만 부피 요소이며, 이를 무시하면 평형분포가 $p(\theta\mid x)$가 아닌 $p(\theta\mid x) \cdot |G|^{1/2}$가 된다.

### 10.3 "Implicit leapfrog의 고정점 반복 수렴 보장"

스텝 크기 $\epsilon$이 너무 크면 고정점 반복이 발산 → Metropolis 거부율 급증. 실전에서는 $\epsilon$을 적응적으로 조정(Betancourt 2013).

### 10.4 "RMHMC는 항상 HMC보다 빠르다"

RMHMC는 스텝 당 비용이 $O(d^3)$(행렬 역연산). 사후분포가 근사적으로 등방이면 유클리드 HMC가 실용적으로 빠르다. RMHMC의 이점은 **극단적 비등방, 곡률이 변화하는 사후분포**에서 나타남.

### 10.5 "Fisher가 특이하면 어떡하지?"

$G$가 low-rank이거나 특이 → SoftAbs, 정규화 $G + \lambda I$, 또는 경험적 Fisher 사용. 때로는 Jeffreys prior $p(\theta) \propto |G|^{1/2}$가 자연스럽게 이 문제를 해결.

---

## 11. 연습 문제 (Exercises)

**연습 11.1** (기본). 1D 가우시안 $p(\theta) = \mathcal{N}(0, \sigma^2)$에서 RMHMC의 해밀토니안을 쓰고, $G = 1/\sigma^2$임을 확인하라. 해석해로 Hamilton 방정식을 풀어라.

**연습 11.2** (증명). 리만 해밀토니안 $H$ 하에서 $\dot H = 0$임을 보여라 (에너지 보존). 힌트: 연쇄 법칙.

**연습 11.3** (계산). 로지스틱 회귀 $p(y=1\mid x, \theta) = \sigma(\theta^\top x)$의 Fisher 행렬을 유도하고, $\partial G/\partial \theta_i$를 구하라.

**연습 11.4** (코딩). Neal's funnel에서 유클리드 HMC와 RMHMC의 ESS(Effective Sample Size)를 비교하라. 체인 길이 10,000, burn-in 1,000.

**연습 11.5** (심화). SoftAbs 계량 $\Lambda = \lambda \coth(\alpha\lambda)$의 $\alpha \to 0$ 및 $\alpha \to \infty$ 극한을 분석하라.

**연습 11.6** (이론). Lagrangian Monte Carlo (Lan et al. 2015)가 RMHMC와 수학적으로 동치임을 보여라. 힌트: Legendre transform.

**연습 11.7** (응용). 베이지안 뉴럴네트워크에서 K-FAC 근사 Fisher를 RMHMC의 질량으로 사용하는 알고리즘을 설계하라. 계산 복잡도를 분석하라.

---

## 참고문헌

- Girolami, M., & Calderhead, B. (2011). "Riemann manifold Langevin and Hamiltonian Monte Carlo methods." *JRSS-B*, 73(2), 123-214.
- Betancourt, M. (2013). "A general metric for Riemannian manifold Hamiltonian Monte Carlo." *GSI 2013*.
- Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
- Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." *Handbook of MCMC*.
- Lan, S., et al. (2015). "Markov chain Monte Carlo from Lagrangian dynamics." *JCGS*, 24(2), 357-378.

---

[◀ 03. VAE Geometry](./03-vae-geometry.md) | [📚 README](../README.md) | [05. Diffusion Fisher ▶](./05-diffusion-fisher.md)
