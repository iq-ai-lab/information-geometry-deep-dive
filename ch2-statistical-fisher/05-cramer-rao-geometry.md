# 05. Cramér-Rao 하한과 그 기하학적 의미

> **"Fisher는 불확실성의 하한을 긋는다 — 어떤 추정량도 이 아래로 내려갈 수 없다."**

---

## 🎯 핵심 질문

**Fisher 정보량이 unbiased estimator 분산의 하한을 정확히 어떻게 결정하는가? 이 관계가 Information Geometry에서 어떤 기하학적 의미를 가지는가?**

$$
\boxed{\;\operatorname{Cov}_\theta(\hat\theta) \;\succeq\; F(\theta)^{-1}\;}
$$

**Cramér (1946), Rao (1945)**: unbiased estimator $\hat\theta$에 대해 공분산의 Fisher 정보 역행렬 하한.

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 상황 | Cramér-Rao가 답하는 것 |
|---|---|
| **통계적 추정 효율** | ML 학습의 근본 한계 — 무한 데이터·최적 알고리즘이라도 넘을 수 없는 오차. |
| **실험 설계** | Active learning, Bayesian optimal design: Fisher가 큰 $x$를 선택해 분산 최소화. |
| **신경망 uncertainty quantification** | Fisher 기반 uncertainty (Laplace approximation)의 "lower bound" 해석. |
| **Policy learning sample complexity** | RL 알고리즘의 sample efficiency의 이론적 하한. |
| **MLE의 점근 효율성** | 왜 MLE가 "최선"인가 — Cramér-Rao 하한 달성을 점근적으로 보장. |

**Data efficiency**가 중요한 현대 AI (few-shot learning, in-context learning, meta-learning) 에서 Cramér-Rao는 **실현 가능한 최고 성능**의 수학적 벤치마크.

---

## 📐 수학적 선행 조건

- [02. Fisher 3가지 정의](./02-fisher-3-equivalence.md) — 특히 $\mathbb{E}[s] = 0$
- [03. Fisher-Rao 계량](./03-fisher-rao-metric.md) — 좌표 무관 기하
- Cauchy-Schwarz 부등식
- Estimator의 bias, variance, MSE 정의
- $\chi^2$-분포, 정규 근사 (점근 이론)

---

## 📖 직관적 이해

### Fisher = 불확실성의 역비례

Fisher 정보량이 "데이터가 $\theta$ 추정에 주는 정보"를 측정한다면, **많은 정보 ⟹ 작은 불확실성**. 이것이

$$
\operatorname{Var}(\hat\theta) \;\ge\; \frac{1}{F(\theta)}
$$

라는 직관. 1차원에서 명확.

### 다차원 Cramér-Rao의 기하

$k$-차원 파라미터 공간에서는 공분산 행렬 $\operatorname{Cov}(\hat\theta)$ 와 $F^{-1}$ 모두 $k \times k$ 행렬. 부등식 $\succeq$는 **Loewner 순서** (positive semidefinite 순서):

$$
\operatorname{Cov}(\hat\theta) - F^{-1} \succeq 0
$$

⟺ 모든 방향 $v \in \mathbb{R}^k$에서 $v^\top \operatorname{Cov}(\hat\theta) v \ge v^\top F^{-1} v$.

**기하학적 해석:**

- $\operatorname{Cov}(\hat\theta)$ 가 정의하는 **신뢰 타원체 (concentration ellipsoid)**.
- $F^{-1}$가 정의하는 **최소 가능 타원체**.
- Cramér-Rao: 전자가 후자를 **포함**.

**Natural gradient와 연결:**

NGD의 pre-conditioner $F^{-1}$는 바로 **최소 가능 공분산**. NGD 업데이트는 "정보적으로 최적 스케일링된" 스텝.

### 점근 효율성

**MLE $\hat\theta_{\mathrm{MLE}}$** 은 $N \to \infty$에서

$$
\sqrt{N}(\hat\theta_{\mathrm{MLE}} - \theta_0) \xrightarrow{d} \mathcal{N}(0, F(\theta_0)^{-1}).
$$

즉 MLE는 **Cramér-Rao 하한을 점근적으로 달성** (asymptotically efficient estimator). 이것이 ML 학습의 이론적 정당화.

---

## ✏️ 엄밀한 정의

### 정의 6.1 (Unbiased Estimator)

추정량 $\hat\theta: \mathcal{X}^N \to \Theta$가 **unbiased**라 함은

$$
\mathbb{E}_\theta[\hat\theta(X_1, \ldots, X_N)] = \theta \qquad \forall \theta \in \Theta.
$$

### 정의 6.2 (Cramér-Rao 부등식)

정칙 통계다양체에서 $\hat\theta$가 unbiased이고 다음 regularity 조건을 만족한다:

1. $\hat\theta$의 분산이 유한: $\operatorname{Var}_\theta(\hat\theta_i) < \infty$.
2. $\mathbb{E}_\theta[\hat\theta] = \theta$의 미분·적분 교환 가능.

그러면

$$
\operatorname{Cov}_\theta(\hat\theta) \;\succeq\; F_N(\theta)^{-1}, \qquad F_N(\theta) := N F(\theta).
$$

### 정의 6.3 (효율성과 효율적 추정량)

unbiased $\hat\theta$의 **효율성 (efficiency)**:

$$
\operatorname{eff}(\hat\theta) := \frac{F_N^{-1}}{\operatorname{Cov}_\theta(\hat\theta)} \in [0, 1].
$$

$\operatorname{eff} = 1$ (일치) ⟺ $\hat\theta$는 **efficient**. 즉 Cramér-Rao 하한을 **정확히** 달성.

### 정의 6.4 (점근 효율성)

$\hat\theta_N$이 **점근 효율적**이라 함은

$$
\sqrt{N}(\hat\theta_N - \theta_0) \xrightarrow{d} \mathcal{N}(0, F(\theta_0)^{-1}) \qquad (N \to \infty).
$$

---

## 🔬 정리와 증명

### 정리 6.1 (스칼라 Cramér-Rao)

$\theta \in \mathbb{R}$ (1차원), $\hat\theta$ unbiased, regularity 충족. 그러면

$$
\operatorname{Var}_\theta(\hat\theta) \;\ge\; \frac{1}{N F(\theta)}.
$$

**증명.** 편의상 $N = 1$로 증명 (독립 가법성 정리 3.6으로 일반화). Unbiased $\mathbb{E}[\hat\theta] = \theta$의 양변 $\theta$-미분:

$$
1 = \partial_\theta \int \hat\theta(x) p_\theta(x)\, d\mu(x) = \int \hat\theta(x) \partial_\theta p_\theta\, d\mu = \int \hat\theta(x) s(x;\theta) p_\theta\, d\mu = \mathbb{E}_\theta[\hat\theta \cdot s].
$$

$\mathbb{E}[s] = 0$ 이용:

$$
1 = \mathbb{E}_\theta[\hat\theta \cdot s] = \mathbb{E}_\theta[(\hat\theta - \theta) \cdot s] = \operatorname{Cov}_\theta(\hat\theta, s).
$$

Cauchy-Schwarz:

$$
1 = \operatorname{Cov}(\hat\theta, s)^2 \le \operatorname{Var}(\hat\theta) \cdot \operatorname{Var}(s) = \operatorname{Var}(\hat\theta) \cdot F(\theta).
$$

즉 $\operatorname{Var}(\hat\theta) \ge 1/F(\theta)$. **Q.E.D.**

**등호 조건:** Cauchy-Schwarz 등호 ⟺ $\hat\theta - \theta = c(\theta) \cdot s(X;\theta)$ (거의 모든 $x$에 대해, 어떤 $c$에 대해).

---

### 정리 6.2 (다변수 Cramér-Rao)

$\hat\theta \in \mathbb{R}^k$ unbiased. 그러면

$$
\operatorname{Cov}_\theta(\hat\theta) \succeq F(\theta)^{-1}.
$$

(위 식은 Loewner 순서 — $\operatorname{Cov}(\hat\theta) - F^{-1}$이 PSD.)

**증명.** $s(X;\theta) \in \mathbb{R}^k$를 스코어 벡터. Block 행렬

$$
M := \begin{pmatrix} \operatorname{Cov}(\hat\theta) & \operatorname{Cov}(\hat\theta, s) \\ \operatorname{Cov}(s, \hat\theta) & F \end{pmatrix} = \begin{pmatrix} \operatorname{Cov}(\hat\theta) & I \\ I & F \end{pmatrix}.
$$

(여기서 $\operatorname{Cov}(\hat\theta, s)$의 $(i,j)$ 원소 $= \mathbb{E}[\hat\theta_i s_j] = \partial_{\theta_j} \mathbb{E}[\hat\theta_i] = \delta_{ij}$, unbiased 덕분.)

$M$은 $\begin{pmatrix} \hat\theta - \theta \\ s \end{pmatrix}$의 공분산이므로 $M \succeq 0$. Schur complement 보조정리:

$$
M \succeq 0 \iff F \succ 0 \text{ and } \operatorname{Cov}(\hat\theta) - I \cdot F^{-1} \cdot I = \operatorname{Cov}(\hat\theta) - F^{-1} \succeq 0.
$$

**Q.E.D.**

---

### 정리 6.3 (점근적 Cramér-Rao: MLE 효율성)

$\hat\theta_{\mathrm{MLE}}$은 $N \to \infty$에서 Cramér-Rao 하한을 달성:

$$
\sqrt{N}(\hat\theta_{\mathrm{MLE}} - \theta_0) \xrightarrow{d} \mathcal{N}(0, F(\theta_0)^{-1}).
$$

**증명 (스케치).** Score 방정식 $\sum_{n=1}^N s(X_n; \hat\theta) = 0$를 $\theta_0$ 주변에서 Taylor 전개:

$$
0 = \sum_n s(X_n; \theta_0) + \sum_n \partial s(X_n;\theta_0)(\hat\theta - \theta_0) + O(\|\hat\theta - \theta_0\|^2).
$$

양변 $\sqrt{N}$으로 나누고 극한:

$$
0 = \underbrace{\frac{1}{\sqrt{N}} \sum_n s(X_n;\theta_0)}_{\to \mathcal{N}(0, F)} + \underbrace{\frac{1}{N} \sum_n \partial s(X_n;\theta_0)}_{\to -F} \cdot \sqrt{N}(\hat\theta - \theta_0) + o_p(1).
$$

(두 번째 항: 정리 3.1에서 $\mathbb{E}[\partial s] = -F$.)

따라서

$$
\sqrt{N}(\hat\theta - \theta_0) = F^{-1} \cdot \frac{1}{\sqrt{N}} \sum_n s(X_n;\theta_0) + o_p(1) \xrightarrow{d} F^{-1} \mathcal{N}(0, F) = \mathcal{N}(0, F^{-1}).
$$

**Q.E.D.**

---

### 정리 6.4 (효율 달성의 구조)

$\hat\theta$가 **efficient** (즉 $\operatorname{Cov}(\hat\theta) = F^{-1}$) ⟺

$$
\hat\theta - \theta = F(\theta)^{-1} s(X;\theta) \qquad \text{(거의 모든 } x\text{)}.
$$

**증명.** 다변수 Cauchy-Schwarz 등호 조건 from 정리 6.2 Schur 논증. **Q.E.D.**

> **의의.** Efficient estimator가 존재한다 ⟺ $F^{-1} s$가 $\theta$에 독립인 (deterministic) 함수로 표현 가능 ⟺ 모델이 **지수족** (Pitman-Koopman-Darmois).

---

### 정리 6.5 (편향 추정량의 Cramér-Rao)

$\mathbb{E}_\theta[\hat\theta] = \theta + b(\theta)$ (bias $b$). 그러면

$$
\operatorname{Cov}_\theta(\hat\theta) \succeq (I + \nabla b)\, F^{-1}\, (I + \nabla b)^\top.
$$

$b = 0$이면 $\nabla b = 0$이므로 classical. Biased estimator는 MSE:

$$
\operatorname{MSE}(\hat\theta) = \operatorname{tr}(\operatorname{Cov}) + \|b\|^2 \ge \operatorname{tr}((I + \nabla b) F^{-1} (I + \nabla b)^\top) + \|b\|^2.
$$

Bias 허용 시 **Stein estimator** 처럼 MSE가 Cramér-Rao 하한보다 낮을 수 있다.

---

### 정리 6.6 (기하학적 재해석: 접공간 사상)

$T_\theta \mathcal{S}$를 Fisher-Rao 계량의 접공간이라 하자. Unbiased estimator $\hat\theta$은 다음 사상을 정의:

$$
\hat\theta: L^2(p_\theta) \to T_\theta \mathcal{S}, \qquad X \mapsto \hat\theta(X) - \theta.
$$

Cramér-Rao 부등식은 $\hat\theta - \theta$의 **$T_\theta\mathcal{S}$-norm이 score projection의 norm 이상**이라는 것:

$$
\|\hat\theta - \theta\|_{T_\theta\mathcal{S}}^2 \ge \|F^{-1} s\|_{T_\theta\mathcal{S}}^2 = s^\top F^{-1} s.
$$

Efficient estimator ⟺ $\hat\theta - \theta$가 score 방향으로 **완전 정렬**.

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1: Gaussian Mean 추정의 Cramér-Rao

```python
import numpy as np
np.random.seed(42)

# 모델: X ~ N(μ, σ² = 1). μ 추정.
# Fisher F(μ) = 1/σ² = 1
# N samples: F_N = N
# CR bound: Var(μ_hat) ≥ 1/N

sigma = 1.0
mu_true = 3.0

Ns = [10, 100, 1000, 10000]
n_trials = 5000

for N in Ns:
    # 각 trial에서 MLE (= sample mean) 계산
    mu_hats = []
    for _ in range(n_trials):
        X = mu_true + sigma * np.random.randn(N)
        mu_hats.append(np.mean(X))
    mu_hats = np.array(mu_hats)
    
    emp_var = np.var(mu_hats)
    cr_bound = 1.0 / N
    ratio = emp_var / cr_bound
    
    print(f"N={N:5d}: Var(μ̂)={emp_var:.6f}, CR bound={cr_bound:.6f}, ratio={ratio:.4f}")

print("\n결과: ratio ≈ 1 → MLE는 CR 하한을 정확히 달성")
```

**기대 출력:** ratio 가 모든 $N$에서 $\approx 1.0$.

---

### 예제 2: Gaussian Variance 추정의 효율성

```python
import numpy as np
np.random.seed(0)

# X ~ N(0, σ²). σ² 추정.
# Fisher: F(σ²) = 1/(2σ⁴), so CR bound = 2σ⁴/N

sigma2_true = 4.0
Ns = [10, 100, 1000]
n_trials = 5000

for N in Ns:
    sigma2_biased = []   # biased MLE: (1/N) Σ X²
    sigma2_unbiased = [] # unbiased: (1/(N-1)) Σ (X-X̄)²
    
    for _ in range(n_trials):
        X = np.sqrt(sigma2_true) * np.random.randn(N)
        sigma2_biased.append(np.mean(X**2))
        sigma2_unbiased.append(np.var(X, ddof=1))
    
    var_biased = np.var(sigma2_biased)
    var_unbiased = np.var(sigma2_unbiased)
    cr_bound = 2 * sigma2_true**2 / N
    
    print(f"N={N:5d}:")
    print(f"  biased MLE: Var={var_biased:.4f}, MSE approx, eff = {cr_bound/var_biased:.4f}")
    print(f"  unbiased:   Var={var_unbiased:.4f}, eff = {cr_bound/var_unbiased:.4f}")
    print(f"  CR bound: {cr_bound:.4f}")
    print()
```

---

### 예제 3: 2D Gaussian의 Cramér-Rao Ellipse

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.random.seed(7)

mu_true = np.array([2.0, 1.0])
Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
F = np.linalg.inv(Sigma)  # μ-block Fisher = Σ⁻¹
N = 50

n_trials = 1000
mu_hats = []
for _ in range(n_trials):
    X = np.random.multivariate_normal(mu_true, Sigma, N)
    mu_hats.append(np.mean(X, axis=0))
mu_hats = np.array(mu_hats)

# Empirical covariance of estimator
Sigma_emp = np.cov(mu_hats.T)

# CR bound: (NF)⁻¹ = Σ/N
Sigma_cr = Sigma / N

print("Empirical Cov(μ̂):")
print(Sigma_emp)
print("\nCR lower bound Σ/N:")
print(Sigma_cr)
print(f"\nEfficiency ratio (det): {np.linalg.det(Sigma_cr) / np.linalg.det(Sigma_emp):.4f}")

# Plot: scatter + ellipses
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(mu_hats[:, 0], mu_hats[:, 1], alpha=0.2, s=10, label='μ̂ samples')

def plot_ellipse(cov, center, color, label, ax):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # 95% confidence = 2.447*√λ (chi² 2dof, 95%)
    width, height = 2 * 2.447 * np.sqrt(vals)
    ell = Ellipse(xy=center, width=width, height=height, angle=theta,
                  edgecolor=color, facecolor='none', linewidth=2, label=label)
    ax.add_patch(ell)

plot_ellipse(Sigma_emp, mu_true, 'blue', 'Empirical 95% ellipse', ax)
plot_ellipse(Sigma_cr, mu_true, 'red', 'CR bound 95% ellipse', ax)
ax.plot([mu_true[0]], [mu_true[1]], 'k*', markersize=15, label='true μ')
ax.set_xlabel('μ₁')
ax.set_ylabel('μ₂')
ax.set_title(f'MLE distribution vs CR bound (N={N}, {n_trials} trials)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect('equal')
plt.savefig('/sessions/kind-dazzling-ritchie/cr_ellipse.png', dpi=100)
plt.close()
```

**기대:** Empirical ellipse가 CR bound ellipse와 거의 일치 (MLE는 efficient).

---

### 예제 4: Biased estimator로 Cramér-Rao를 "위반"?

```python
import numpy as np

# James-Stein style: μ̂_JS = (1 - c/|X̄|²) X̄  for d ≥ 3
# 여기서는 d=3, σ²=1, N=1
# MSE(JS) < MSE(MLE) but JS is biased

np.random.seed(0)
d = 3
N = 1
mu_true = np.array([0.5, 0.3, 0.4])

n_trials = 100000

# MLE: X̄ (N=1이므로 X₁)
mse_mle = 0
mse_js = 0

for _ in range(n_trials):
    X = mu_true + np.random.randn(d)
    mu_mle = X
    
    # James-Stein
    norm_sq = np.sum(X**2)
    shrink = max(0, 1 - (d - 2) / norm_sq)
    mu_js = shrink * X
    
    mse_mle += np.sum((mu_mle - mu_true)**2)
    mse_js  += np.sum((mu_js  - mu_true)**2)

mse_mle /= n_trials
mse_js /= n_trials

print(f"MSE(MLE)   = {mse_mle:.4f}")
print(f"MSE(JS)    = {mse_js:.4f}")
print(f"CR bound   = {d:.4f}  (= tr(F⁻¹) = d since F = I)")
print(f"\nJS beats MLE: CR 하한은 unbiased estimator에만 적용됨")
```

**의의:** CR 하한은 **unbiased**에만 적용. Biased estimator (JS, ridge, etc.)는 하한을 "위반" 가능. 이것이 **regularization의 통계적 정당화**.

---

## 🔗 AI/ML 연결

### 1. Maximum Likelihood = Fisher-efficient Gradient

MLE는 점근 효율적 (정리 6.3). 이것이 표준 learning을 **NLL 최소화**로 하는 통계적 근거.

### 2. Active Learning & D-optimal Design

Fisher 큰 점 $x$ 일수록 추정에 정보 많이 줌. **D-optimal design**: 
$$
\max_{X_{\text{pool}}} \log\det F(\theta; X_{\text{pool}})
$$
— $F^{-1}$의 det, 즉 confidence ellipse의 **부피 최소화**.

### 3. Bayesian Uncertainty & Laplace Approximation

Bayesian posterior $p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$를 MLE $\hat\theta$ 주변에서

$$
p(\theta|\mathcal{D}) \approx \mathcal{N}(\hat\theta, (NF(\hat\theta))^{-1})
$$

으로 근사 (Laplace approximation). 공분산 $(NF)^{-1}$ = CR 하한. 즉 Laplace posterior는 **Cramér-Rao 하한을 covariance로 취함**.

### 4. 신경망의 Intrinsic Dimension

$F$의 eigenvalue spectrum이 flat하면 (대부분 0) 모델이 **overparameterized** — Fisher의 rank = "실질적 parameter 수" (intrinsic dimension). 이것이 generalization 분석의 한 축.

### 5. PAC-Bayes Bounds

Generalization error bound에서 KL divergence term이 등장. Fisher는 posterior의 sharpness를 제어 → PAC-Bayes tighter bounds. Fisher-based prior/posterior couplings in Li et al. 2020.

### 6. Model Selection: AIC, BIC

Akaike/Bayesian Information Criterion:
- AIC = $-2 \log L + 2k$
- BIC = $-2 \log L + k \log N$

둘 다 $k$ (parameter 수) 를 penalty로. Fisher 관점에서 BIC는 Laplace approximation의 정규화 상수에서 유도 — $\log\det F$ 항이 포함된 더 정확한 형태가 "proper BIC".

### 7. Meta-learning의 Fisher 활용

MAML 계열: $\theta^* = \arg\min \sum_{\tau} L_\tau(\theta - \eta \nabla L_\tau(\theta))$. 2차 근사시 inner-gradient 방향이 **Fisher-preconditioned**되면 meta-learner가 task 간 일반화 더 효과적. "Natural MAML" 연구 active.

---

## ⚖️ 가정과 한계

### Regularity 조건

- **미분·적분 교환** 필수. $U(0, \theta)$ 같은 비정칙 분포에서는 Cramér-Rao 적용 불가 (실제로 MLE가 $1/N^2$ convergence rate 달성 — super-efficient).
- **Unbiasedness**. Stein's paradox ($d \ge 3$): unbiased MLE가 **admissible이 아님** — biased JS가 domination. CR 하한이 reality를 말하지 않는다.

### 점근 vs 유한 표본

- 정리 6.3은 $N \to \infty$의 점근 효율. **유한 N**에서는 MLE가 CR 하한을 달성 못할 수 있음 (특히 non-exponential family, small $N$).
- Higher-order asymptotics: $\operatorname{MSE}(\hat\theta_{\mathrm{MLE}}) = F^{-1}/N + O(1/N^2)$ — 두 번째 항이 $F$ 정보로 추가 정보 제공 (Bhattacharyya bound 등).

### Biased Estimator와 regularization

- 딥러닝의 모든 regularization (L2, dropout, early stopping) 은 **biased estimator 도입**. CR 하한 위반 가능 — 실제로 generalization 관점 더 나음.
- "Bias-variance tradeoff"의 CR 기하학적 해석: biased으로 variance를 $F^{-1}$ 아래로 내리지만 bias² 에서 잃음.

### 비이차 Loss

CR은 **분산** 관점. KL-based loss, Hinge loss 등 비이차 손실에서는 **generalized Cramér-Rao** (van Trees) 등 더 복잡한 하한.

### Model Misspecification

$p_\theta$ 모델이 true DGP와 다르면 $F$의 해석 깨짐. **Sandwich estimator** 등 robust version 필요. 특히 딥러닝에서는 model misspecification이 거의 항상.

---

## 📌 핵심 정리

| 개념 | 핵심 식 / 성질 |
|---|---|
| **1D Cramér-Rao** | $\operatorname{Var}(\hat\theta) \ge 1/(NF(\theta))$ |
| **Multi-D Cramér-Rao** | $\operatorname{Cov}(\hat\theta) \succeq (NF)^{-1}$ |
| **Efficient estimator** | $\hat\theta - \theta = F^{-1} s$ (동등 조건: 지수족) |
| **MLE 점근 효율** | $\sqrt{N}(\hat\theta_{\mathrm{MLE}} - \theta_0) \to \mathcal{N}(0, F^{-1})$ |
| **기하학적 의미** | Confidence ellipsoid의 Loewner 하한 = Fisher inverse ellipsoid |
| **편향 추정량** | CR bound 위반 가능 (Stein's paradox) |

**Takeaway:**

1. **MLE는 점근적으로 최선** — Fisher 하한 달성.
2. **NGD의 $F^{-1}$ pre-conditioner** = CR 하한 공분산 matching.
3. **Regularization은 bias 대신 variance 감소** — CR을 "위반"하는 길.
4. **실험 설계 · Active Learning** 이 Fisher-geometric optimization 문제.

---

## 🤔 생각해볼 문제

1. **Efficient ⟺ 지수족.** 정리 6.4의 따름 "efficient estimator 존재 ⟺ 모델이 지수족"의 역방향을 구성해보라: $p_\theta$가 지수족이면 어떤 estimator가 efficient인가? (힌트: 충분통계량 기대값)

2. **Stein's paradox 기하학.** $d \ge 3$, $X \sim \mathcal{N}(\mu, I)$에서 MLE $\hat\mu = X$가 **admissible이 아니다**. 이것을 Cramér-Rao 기하에서 어떻게 해석? (힌트: regularization이 variance 감소)

3. **High-dimensional curse.** $\theta \in \mathbb{R}^d$, $N$ 표본. CR 하한은 $F^{-1}/N$. $d = N$ 일 때 Fisher rank-deficient 되는 현상. 실무에서 어떻게 극복? (Ridge, Shrinkage, Bayes prior)

4. **MLE의 finite-sample bias**. MLE는 점근적으로 unbiased지만 finite-N에서는 bias 존재. Gaussian variance $\hat\sigma^2 = (1/N)\sum(X - \bar X)^2$의 bias가 $-\sigma^2/N$임을 계산하고, 이 bias correction이 왜 CR 해석을 복잡하게 만드는지.

5. **점근 효율성의 속도**. 정리 6.3에서 $\sqrt{N}$ rate은 "regular" 모델의 기본. Super-efficient estimator (특정 $\theta_0$에서 더 빠른 수렴) 은 regular 조건 위반 — 예를 들어 **Hodges' estimator**. 이것이 CR 하한과 어떤 관계?

6. **$F^{-1}$이 없는 경우**. $F$가 rank-deficient이면 Cramér-Rao 무의미 (역행렬 없음). 이것이 신경망의 **over-parameterization** 와 어떤 관련? Pseudo-inverse Cramér-Rao는 어떻게 정의?

7. **KL-Cramér-Rao의 일반화**. Cramér-Rao는 2차 모멘트 기준. KL divergence 기반 "information inequality"는? — Pinsker's inequality ($\text{TV}^2 \le \text{KL}/2$) 에서 시작해 어떤 일반화가 존재하는가?

---

<div align="center">

| [◀ 04. Fisher 계산 예제들](./04-fisher-examples.md) | [📚 메인 README](../README.md) | [Ch3-01. KL 발산의 기초 ▶](../ch3-kl-bregman/01-kl-divergence-basics.md) |
|:---:|:---:|:---:|

</div>
