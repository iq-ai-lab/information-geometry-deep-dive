# 02. Fisher 정보량의 3가지 정의와 동치성

> **"같은 양을 세 가지 다른 얼굴로 마주치면, 그것이 본질임을 의심하라."**

---

## 🎯 핵심 질문

**Fisher 정보량 $F_{ij}(\theta)$은 세 가지 전혀 달라 보이는 식으로 정의되는데, 왜 모두 같을까?**

$$
\boxed{
\begin{aligned}
\text{(A) 스코어 공분산:} \quad & F_{ij}(\theta) \;=\; \mathbb{E}_\theta\!\left[ \partial_i \ell_\theta(X) \, \partial_j \ell_\theta(X) \right] \\[2pt]
\text{(B) 음의 로그가능도 Hessian:} \quad & F_{ij}(\theta) \;=\; -\,\mathbb{E}_\theta\!\left[ \partial_i \partial_j \ell_\theta(X) \right] \\[2pt]
\text{(C) KL의 2차 근사:} \quad & \operatorname{KL}\!\bigl(p_\theta \,\|\, p_{\theta+d\theta}\bigr) \;=\; \tfrac{1}{2}\,d\theta^\top F(\theta)\, d\theta \;+\; o(\|d\theta\|^2)
\end{aligned}}
$$

여기서 $\ell_\theta(x) := \log p_\theta(x)$는 로그가능도.

세 정의의 동치성은 Information Geometry, 통계 추론, 최적화 이론의 **근간**이다.

---

## 🔍 왜 이 개념이 AI에서 중요한가

| 세 정의 | AI에서의 사용처 |
|---|---|
| (A) 스코어 공분산 | **Natural Gradient**: $\tilde{g} = F^{-1} g$에서 $F$를 $\hat{F} = \frac{1}{N}\sum_n \nabla \ell_n \nabla \ell_n^\top$ (Empirical Fisher) 으로 근사. K-FAC의 블록 구조가 여기서 유래. |
| (B) Hessian | **2차 최적화**: Newton 방법의 Hessian 대신 Fisher 사용 시 PSD 보장. 분류 문제의 softmax cross-entropy Hessian = Fisher. |
| (C) KL 2차 근사 | **TRPO / PPO**: KL 제약 $\operatorname{KL}(\pi_{\text{old}} \| \pi_\theta) \le \delta$를 이차 근사해 $d\theta^\top F d\theta \le 2\delta$로 변환. Mirror Descent의 프록시. |

세 정의가 **같다**는 사실이 "자연 경사 = KL 최급강하 = 정보 행렬" 이라는 세 해석을 하나로 묶는 접착제이다.

---

## 📐 수학적 선행 조건

- 이전 문서 [01. 통계다양체](./01-statistical-manifold.md)의 정칙성 조건 (R1)-(R4)
- Ch1의 [02. 접공간](../ch1-manifold-riemannian/02-tangent-space.md) — 스코어 $s_i = \partial_i \log p$가 접벡터와 대응한다는 관점
- Dominated Convergence Theorem (미분과 적분의 교환)
- Taylor 전개와 Landau 기호 $o(\cdot)$
- KL 발산의 정의 $\operatorname{KL}(p\|q) = \int p \log(p/q)\, d\mu$

---

## 📖 직관적 이해

### 세 얼굴의 같은 것

확률모델 $p_\theta$를 $\theta$에 대해 미분할 때 얻는 **로그우도 곡선의 곡률**을 세 가지 방식으로 잰다:

1. **스코어 요동 (A)** — $\partial_i \ell$이 기대값 $0$ 주변에서 얼마나 퍼져 있는가? 분산이 크면 데이터가 $\theta$를 "잘 구분"해줌.

2. **로그우도 꼭대기의 뾰족함 (B)** — MLE 주변에서 로그우도가 얼마나 가파르게 떨어지는가? Hessian이 음으로 크면(즉 $-\partial_i\partial_j\ell$이 양으로 크면) 좁은 봉우리.

3. **확률분포 공간에서의 거리 (C)** — $\theta$를 조금 흔들면 분포 $p_\theta$가 얼마나 멀어지는가? KL로 잰 거리²가 바로 Fisher quadratic form.

**(A)=(B)**: 스코어가 평균 $0$일 때, 분산과 "도함수 기대값"이 부호 반대로 같다. 이것은 $\int p_\theta\, d\mu = 1$이라는 **정규화 조건의 두 번 미분** 결과.

**(B)=(C)**: KL을 Taylor 전개하면 $0$차·$1$차 항이 사라지고 $2$차 항이 정확히 Hessian의 기대값이 된다. 이는 KL이 **분포끼리의 거리**가 아닌 **정보 손실**이라는 사실과 직결.

### 왜 세 정의가 하나로 수렴하는가

**핵심은 정규화 $\int p_\theta(x)\,d\mu(x) = 1$이다.** 이것을 두 번 미분하면:

$$
\partial_i \!\int p_\theta \, d\mu = 0 \implies \mathbb{E}_\theta[\partial_i \ell] = 0 \quad (\text{스코어 평균 0})
$$

$$
\partial_j \partial_i \!\int p_\theta \, d\mu = 0 \implies \mathbb{E}_\theta[\partial_j \partial_i \ell] + \mathbb{E}_\theta[\partial_i \ell \cdot \partial_j \ell] = 0
$$

**두 식을 빼면 (A) = (B).** 그리고 KL의 Taylor 전개에서 1차 항이 스코어 평균 $0$ 덕에 사라지고 2차 항에 (B)가 등장해 (C)까지 연결된다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 (Fisher 정보행렬의 3가지 정의)

정칙 통계다양체 $\mathcal{S} = \{p_\theta : \theta \in \Theta\}$에서 **Fisher 정보행렬** $F(\theta) = (F_{ij}(\theta))$은 다음 중 어느 하나로 정의된다. 정의 각각이 존재하려면 적절한 미분 가능성·적분 가능성 조건이 필요.

**(A) 스코어 공분산 정의**

$$
F_{ij}^{(\mathrm{A})}(\theta) \;=\; \mathbb{E}_\theta\!\bigl[\, s_i(X;\theta)\, s_j(X;\theta) \,\bigr] \;=\; \int \partial_i \log p_\theta(x) \cdot \partial_j \log p_\theta(x) \cdot p_\theta(x)\, d\mu(x).
$$

여기서 $s_i(x;\theta) := \partial_i \log p_\theta(x)$는 **스코어 함수(score function)**.

**(B) 음의 Hessian 정의**

$$
F_{ij}^{(\mathrm{B})}(\theta) \;=\; -\,\mathbb{E}_\theta\!\bigl[\, \partial_i \partial_j \log p_\theta(X) \,\bigr] \;=\; -\int \partial_i \partial_j \log p_\theta(x) \cdot p_\theta(x)\, d\mu(x).
$$

**(C) KL 2차 근사 정의**

$d\theta \to 0$일 때 $F^{(\mathrm{C})}(\theta)$를 유일하게 결정하는 2차 형식:

$$
\operatorname{KL}\bigl(p_\theta \,\|\, p_{\theta + d\theta}\bigr) \;=\; \tfrac{1}{2}\, (d\theta)^\top F^{(\mathrm{C})}(\theta)\, (d\theta) \;+\; o(\|d\theta\|^2).
$$

### 정의 3.2 (정칙성 조건, 강화판)

(A)(B)(C) 모두가 잘 정의되고 동치가 되려면 (R1)-(R4) 외에 다음이 필요:

- **(R5)** $p_\theta > 0$인 열린 영역 위에서 적분 · 미분 교환 가능 — Dominated Convergence의 가수 함수 존재.
- **(R6)** 스코어 $s_i$의 2차 모멘트가 유한: $\mathbb{E}_\theta[s_i^2] < \infty$.
- **(R7)** $\partial_i \partial_j \log p_\theta$가 $\theta$에 대해 연속이고 $\mathbb{E}_\theta[|\partial_i \partial_j \log p_\theta|] < \infty$.

이하에서 조건 (R1)-(R7)을 모두 **"정칙조건"**이라 부른다.

---

## 🔬 정리와 증명

### 정리 3.1 (정의 A와 B의 동치성)

정칙조건 하에서,

$$
F_{ij}^{(\mathrm{A})}(\theta) \;=\; F_{ij}^{(\mathrm{B})}(\theta).
$$

**증명.** 정규화 조건 $\int p_\theta(x)\, d\mu(x) = 1$의 양변을 $\theta_i$에 대해 미분:

$$
0 \;=\; \partial_i \int p_\theta\, d\mu \;\stackrel{(R5)}{=}\; \int \partial_i p_\theta\, d\mu \;=\; \int \frac{\partial_i p_\theta}{p_\theta} \cdot p_\theta\, d\mu \;=\; \mathbb{E}_\theta[\partial_i \log p_\theta]. \tag{1}
$$

즉, **스코어 평균은 0**: $\mathbb{E}_\theta[s_i] = 0$.

식 (1)을 다시 $\theta_j$에 대해 미분:

$$
0 \;=\; \partial_j \!\int \partial_i \log p_\theta \cdot p_\theta\, d\mu \;=\; \int \partial_j \partial_i \log p_\theta \cdot p_\theta\, d\mu \;+\; \int \partial_i \log p_\theta \cdot \partial_j p_\theta\, d\mu.
$$

두 번째 항에서 $\partial_j p_\theta = p_\theta \cdot \partial_j \log p_\theta$이므로:

$$
0 \;=\; \mathbb{E}_\theta[\partial_j \partial_i \log p_\theta] \;+\; \mathbb{E}_\theta[\partial_i \log p_\theta \cdot \partial_j \log p_\theta].
$$

따라서

$$
\mathbb{E}_\theta[\partial_i \log p_\theta \cdot \partial_j \log p_\theta] \;=\; -\,\mathbb{E}_\theta[\partial_j \partial_i \log p_\theta],
$$

즉 $F_{ij}^{(\mathrm{A})} = F_{ij}^{(\mathrm{B})}$. **Q.E.D.**

---

### 정리 3.2 (스코어 공분산 형태)

$\mathbb{E}_\theta[s_i] = 0$이므로 정의 (A)는 정확히 **공분산 행렬**이다:

$$
F_{ij}^{(\mathrm{A})}(\theta) \;=\; \operatorname{Cov}_\theta\bigl(s_i(X;\theta),\, s_j(X;\theta)\bigr).
$$

**따름.** $F$는 **대칭**이고 **양의 준정치(PSD)** 행렬이다. 즉 $v \in \mathbb{R}^k$에 대해

$$
v^\top F(\theta) v \;=\; \mathbb{E}_\theta\!\bigl[\, (v^\top s(X;\theta))^2 \,\bigr] \;\ge\; 0.
$$

등호 ⟺ $v^\top s(X;\theta) = 0$ (거의 모든 $x$). 이것이 정칙조건에서 **$F$가 양의 정치(PD)**가 되는 필요충분조건 — (R4) 스코어 선형독립성과 동치.

---

### 정리 3.3 (정의 B와 C의 동치성)

정칙조건 하에서,

$$
\operatorname{KL}\bigl(p_\theta \,\|\, p_{\theta + \varepsilon}\bigr) \;=\; \tfrac{1}{2}\, \varepsilon^\top F^{(\mathrm{B})}(\theta)\, \varepsilon \;+\; o(\|\varepsilon\|^2) \qquad (\varepsilon \to 0).
$$

**증명.** KL 정의를 분해:

$$
\operatorname{KL}(p_\theta \| p_{\theta+\varepsilon}) \;=\; \int p_\theta \log \frac{p_\theta}{p_{\theta+\varepsilon}}\, d\mu \;=\; \mathbb{E}_\theta[\ell_\theta - \ell_{\theta+\varepsilon}],
$$

단 $\ell_\theta(x) := \log p_\theta(x)$.

$\ell_{\theta+\varepsilon}(x)$를 $\theta$ 근방에서 Taylor 전개 (정리 (R7)이 보장):

$$
\ell_{\theta+\varepsilon}(x) \;=\; \ell_\theta(x) \;+\; \sum_i \varepsilon_i\, \partial_i \ell_\theta(x) \;+\; \tfrac{1}{2} \sum_{i,j} \varepsilon_i \varepsilon_j\, \partial_i \partial_j \ell_\theta(x) \;+\; R_3(x;\varepsilon),
$$

여기서 $R_3 = o(\|\varepsilon\|^2)$ (uniformly in $x$ 하에서 기대값 가능).

기대값:

$$
\begin{aligned}
\operatorname{KL}(p_\theta \| p_{\theta+\varepsilon}) &= -\,\mathbb{E}_\theta[\ell_{\theta+\varepsilon} - \ell_\theta] \\
&= -\sum_i \varepsilon_i\, \underbrace{\mathbb{E}_\theta[\partial_i \ell_\theta]}_{=\,0\ \text{(식 (1))}} \;-\; \tfrac{1}{2} \sum_{i,j} \varepsilon_i \varepsilon_j\, \mathbb{E}_\theta[\partial_i \partial_j \ell_\theta] \;-\; \mathbb{E}_\theta[R_3] \\
&= \tfrac{1}{2} \sum_{i,j} \varepsilon_i \varepsilon_j \cdot \bigl(-\mathbb{E}_\theta[\partial_i \partial_j \ell_\theta]\bigr) \;+\; o(\|\varepsilon\|^2) \\
&= \tfrac{1}{2}\, \varepsilon^\top F^{(\mathrm{B})}(\theta)\, \varepsilon \;+\; o(\|\varepsilon\|^2).
\end{aligned}
$$

**Q.E.D.**

---

### 정리 3.4 (대칭화된 KL — 자연 근사)

KL은 비대칭이지만, 두 방향의 KL을 평균하면 **대칭** Fisher quadratic가 나온다:

$$
\tfrac{1}{2}\bigl[\operatorname{KL}(p_\theta \| p_{\theta+\varepsilon}) \;+\; \operatorname{KL}(p_{\theta+\varepsilon} \| p_\theta)\bigr] \;=\; \tfrac{1}{2} \varepsilon^\top F(\theta) \varepsilon \;+\; o(\|\varepsilon\|^2).
$$

**증명.** 반대 방향 KL을 같은 방식으로 전개하면 $\varepsilon = -\varepsilon'$ 치환에 의해 1차 항이 $0$이고 2차 항이 $\tfrac{1}{2} F(\theta+\varepsilon) \approx \tfrac{1}{2} F(\theta)$로 일치. **Q.E.D.**

> **주의.** 단일 방향 KL도 이미 2차 항이 대칭 $F$이다 (정리 3.3). 비대칭성은 **3차 이상**에서 나타나며, 이것이 $\alpha$-접속의 $\alpha = \pm 1$ 구분으로 이어진다 (Ch4).

---

### 정리 3.5 (좌표 변환 법칙)

$\theta \mapsto \tilde\theta(\theta)$가 $C^1$-미분동형이면, 새 좌표에서 Fisher:

$$
\tilde F_{ab}(\tilde\theta) \;=\; \sum_{i,j} \frac{\partial \theta^i}{\partial \tilde\theta^a} \, \frac{\partial \theta^j}{\partial \tilde\theta^b}\, F_{ij}(\theta).
$$

즉 $F$는 $(0,2)$-텐서 (리만 계량과 같은 변환 법칙). **이것이 Fisher를 "계량"이라 부르는 이유** (다음 문서 03에서 Fisher-Rao 계량으로 정식화).

**증명.** 연쇄 법칙: $\partial_{\tilde a} \ell = \sum_i \frac{\partial \theta^i}{\partial \tilde\theta^a} \partial_i \ell$이므로 정의 (A)에 대입.

---

### 정리 3.6 (독립 표본의 가법성)

$X_1, \ldots, X_N \stackrel{iid}{\sim} p_\theta$의 결합분포 $p_\theta^{(N)}(x_1,\ldots,x_N) = \prod_n p_\theta(x_n)$에 대해,

$$
F^{(N)}(\theta) \;=\; N \cdot F(\theta).
$$

**증명.** $\log p_\theta^{(N)} = \sum_n \log p_\theta(x_n)$이므로 $\partial_i \log p_\theta^{(N)} = \sum_n s_i(x_n;\theta)$. 독립성에서 서로 다른 $n$의 스코어는 공분산 $0$이고, 각각의 공분산이 $F$. $N$개의 합이 $NF$.

> **의의.** 이것이 **Cramér-Rao 하한** $\operatorname{Var}(\hat\theta) \ge (NF)^{-1}$의 $1/N$ 스케일링의 근원.

---

## 💻 NumPy / SymPy 구현으로 검증

### 예제 1 (SymPy): 정규분포 $\mathcal{N}(\mu, \sigma^2)$에서 (A), (B) 직접 계산

```python
import sympy as sp

x, mu, sigma = sp.symbols('x mu sigma', real=True, positive=True)

# 로그가능도
log_p = -sp.Rational(1,2) * sp.log(2*sp.pi*sigma**2) - (x - mu)**2 / (2*sigma**2)

# 스코어
s_mu    = sp.diff(log_p, mu)      # (x - μ)/σ²
s_sigma = sp.diff(log_p, sigma)   # -1/σ + (x - μ)²/σ³

# Hessian 원소
H_mumu      = sp.diff(log_p, mu, 2)
H_musigma   = sp.diff(log_p, mu, sigma)
H_sigmasigma= sp.diff(log_p, sigma, 2)

# 기대값 계산 (정규분포 아래)
# 정규분포 모멘트: E[(x-μ)²] = σ², E[(x-μ)⁴] = 3σ⁴
def E_norm(expr):
    # (x - mu) 를 기준으로 기대값 계산
    poly = sp.expand(expr)
    # 치환: E[1] = 1, E[(x-mu)] = 0, E[(x-mu)²] = σ², E[(x-mu)⁴] = 3σ⁴
    u = sp.Symbol('u')
    repl = sp.expand(poly.subs(x - mu, u))
    result = 0
    for term in sp.Add.make_args(repl):
        coeff, powers = term.as_coeff_Mul()
        deg = sp.degree(term, u)
        if deg == 0:
            result += term
        elif deg == 2:
            result += term.subs(u**2, sigma**2)
        elif deg == 4:
            result += term.subs(u**4, 3*sigma**4)
        # 홀수 차수는 평균 0
    return sp.simplify(result)

# (A) 스코어 공분산
F_A_mu_mu       = sp.simplify(E_norm(s_mu * s_mu))
F_A_mu_sigma    = sp.simplify(E_norm(s_mu * s_sigma))
F_A_sigma_sigma = sp.simplify(E_norm(s_sigma * s_sigma))

print("=== (A) Score Covariance ===")
print(f"F_μμ     = {F_A_mu_mu}")       # 1/σ²
print(f"F_μσ     = {F_A_mu_sigma}")    # 0
print(f"F_σσ     = {F_A_sigma_sigma}") # 2/σ²

# (B) -Hessian
F_B_mu_mu       = sp.simplify(-E_norm(H_mumu))
F_B_mu_sigma    = sp.simplify(-E_norm(H_musigma))
F_B_sigma_sigma = sp.simplify(-E_norm(H_sigmasigma))

print("\n=== (B) -E[Hessian] ===")
print(f"F_μμ     = {F_B_mu_mu}")
print(f"F_μσ     = {F_B_mu_sigma}")
print(f"F_σσ     = {F_B_sigma_sigma}")

# 동치성 확인
print("\n=== (A) == (B) ? ===")
print(sp.simplify(F_A_mu_mu - F_B_mu_mu) == 0)             # True
print(sp.simplify(F_A_mu_sigma - F_B_mu_sigma) == 0)       # True
print(sp.simplify(F_A_sigma_sigma - F_B_sigma_sigma) == 0) # True
```

**기대 출력:**

```
=== (A) Score Covariance ===
F_μμ     = 1/σ²
F_μσ     = 0
F_σσ     = 2/σ²

=== (B) -E[Hessian] ===
F_μμ     = 1/σ²
F_μσ     = 0
F_σσ     = 2/σ²

=== (A) == (B) ? ===
True
True
True
```

---

### 예제 2 (NumPy): (C) KL → 2차 근사 검증

```python
import numpy as np
from scipy import stats

def kl_normal(mu1, sig1, mu2, sig2):
    """정규분포 간 정확한 KL (closed form)"""
    return np.log(sig2/sig1) + (sig1**2 + (mu1-mu2)**2)/(2*sig2**2) - 0.5

# 기준점
mu, sigma = 0.0, 1.0

# Fisher at (μ, σ) = (0, 1): diag(1, 2)
F = np.array([[1.0, 0.0], [0.0, 2.0]])

# 다양한 크기의 섭동
epsilons = np.logspace(-4, -1, 20)
ratio_list = []

for eps in epsilons:
    # 랜덤 방향
    rng = np.random.default_rng(0)
    dtheta = eps * rng.standard_normal(2)
    mu_new, sig_new = mu + dtheta[0], sigma + dtheta[1]
    if sig_new <= 0:
        continue
    
    kl_exact = kl_normal(mu, sigma, mu_new, sig_new)
    kl_quad  = 0.5 * dtheta @ F @ dtheta
    
    ratio_list.append((eps, kl_exact, kl_quad, kl_exact/kl_quad))

print(f"{'ε':>10} {'KL_exact':>15} {'½dθᵀFdθ':>15} {'ratio':>10}")
for row in ratio_list:
    print(f"{row[0]:>10.1e} {row[1]:>15.6e} {row[2]:>15.6e} {row[3]:>10.6f}")
```

**기대 동작:** $\varepsilon \to 0$일 때 마지막 `ratio` 열이 $1.0$에 수렴.

---

### 예제 3: Bernoulli 분포 $B(1, p)$에서 세 정의의 일치

```python
import sympy as sp

p = sp.Symbol('p', positive=True)
# X ∈ {0, 1}, P(X=1) = p
# log p_p(x) = x log p + (1-x) log(1-p)

# 스코어 s = ∂_p log p = x/p - (1-x)/(1-p)
# E[s²]:
# x=0 확률 1-p: s² = (1/(1-p))²
# x=1 확률 p:   s² = (1/p)²
# E[s²] = (1-p)·1/(1-p)² + p·1/p² = 1/(1-p) + 1/p = 1/(p(1-p))
F_A = 1/(1-p) + 1/p
F_A = sp.simplify(F_A)
print(f"(A) F_B = {F_A}")  # 1/(p*(1-p))

# -E[∂²_p log p]:
# ∂²_p log p = -x/p² - (1-x)/(1-p)²
# E[...] = -p/p² - (1-p)/(1-p)² = -1/p - 1/(1-p)
# -E[...] = 1/p + 1/(1-p) = 1/(p(1-p))
F_B = 1/p + 1/(1-p)
F_B = sp.simplify(F_B)
print(f"(B) F_B = {F_B}")

# (C) KL(p || p+ε) = p log(p/(p+ε)) + (1-p) log((1-p)/(1-p-ε))
eps = sp.Symbol('eps')
kl = p*sp.log(p/(p+eps)) + (1-p)*sp.log((1-p)/(1-p-eps))
kl_series = sp.series(kl, eps, 0, 3).removeO()
print(f"\n(C) KL 2차 근사: {sp.simplify(kl_series)}")
# 계수를 보면 ε²/2 항의 계수가 1/(p(1-p))

coeff_eps2 = kl_series.coeff(eps, 2)
print(f"    계수(ε²): {sp.simplify(coeff_eps2)}")
# coeff should be 1/(2*p*(1-p))
F_C = 2 * coeff_eps2  # because we have (1/2) F ε²
print(f"(C) F_B = {sp.simplify(F_C)}")

print(f"\n(A)=(B)=(C)? -> {sp.simplify(F_A - F_B) == 0 and sp.simplify(F_B - F_C) == 0}")
```

**기대 출력:**
```
(A) F_B = 1/(p*(1 - p))
(B) F_B = 1/(p*(1 - p))

(C) KL 2차 근사: eps²/(2*p*(1 - p))
    계수(ε²): 1/(2*p*(1 - p))
(C) F_B = 1/(p*(1 - p))

(A)=(B)=(C)? -> True
```

---

### 예제 4: Empirical Fisher vs True Fisher 비교 (AI 연결)

```python
import numpy as np
np.random.seed(42)

# True model: Normal(μ=0, σ=1)
N = 10000
X = np.random.randn(N)

# Parameters at evaluation
mu_hat, sigma_hat = 0.05, 1.02  # 추정치

# 스코어
s_mu    = (X - mu_hat) / sigma_hat**2
s_sigma = -1/sigma_hat + (X - mu_hat)**2 / sigma_hat**3

# Empirical Fisher = (1/N) Σ s_n s_n^T
s = np.stack([s_mu, s_sigma], axis=1)  # (N, 2)
F_emp = (s.T @ s) / N

# True Fisher: diag(1/σ², 2/σ²)
F_true = np.diag([1/sigma_hat**2, 2/sigma_hat**2])

print("Empirical Fisher:")
print(F_emp)
print("\nTrue Fisher:")
print(F_true)
print(f"\nFrobenius 오차: {np.linalg.norm(F_emp - F_true):.6f}")
# N=10000이면 오차 ~0.05 수준 (1/√N 스케일)
```

**의의.** 신경망 학습에서 **Empirical Fisher** (정의 A의 몬테카를로)는 **K-FAC**, **Shampoo**, **Natural Gradient Descent**의 실제 구현 근간.

---

## 🔗 AI/ML 연결

### 1. Natural Gradient — 정의 (A) 기반

$$
\tilde \nabla L(\theta) \;=\; F(\theta)^{-1} \nabla L(\theta), \qquad F(\theta) \approx \frac{1}{N}\sum_{n=1}^N s(x_n;\theta)\, s(x_n;\theta)^\top.
$$

Amari (1998) 증명: Natural gradient는 **KL 구면** $\operatorname{KL}(p_\theta \| p_{\theta+d\theta}) = \text{const}$ 위에서 $L$을 가장 빠르게 줄이는 방향. — **정의 (C)** 의 직접적 따름.

### 2. TRPO (Schulman et al. 2015) — 정의 (C) 기반

$$
\max_{\theta} \mathbb{E}\bigl[ \tfrac{\pi_\theta}{\pi_{\text{old}}} A^{\pi_{\text{old}}} \bigr] \quad \text{s.t.} \quad \mathbb{E}\bigl[\operatorname{KL}(\pi_{\text{old}} \| \pi_\theta)\bigr] \le \delta.
$$

KL 제약을 **2차 근사** (정리 3.3) 해 $\tfrac{1}{2} d\theta^\top F d\theta \le \delta$로 푼다. Conjugate gradient로 $F^{-1} g$ 계산.

### 3. Cross-Entropy Loss의 Hessian = Fisher — 정의 (B) 기반

$k$-클래스 분류 $p_\theta(y|x) = \operatorname{softmax}(\theta^\top x)_y$의 **음의 로그우도 Hessian**은 정확히 Fisher 정보행렬. 따라서 Newton 방법의 Hessian을 Fisher로 대체해도 **같은** 갱신. (단, 분류가 아닌 회귀에서는 일반적으로 다름 — "Generalized Gauss-Newton vs Fisher" 구분 필요.)

### 4. PAC-Bayes & Information Bottleneck

정의 (C)는 **KL = 정보손실**의 로컬 버전이므로, 모델 섭동 민감도 분석 (sharpness, PAC-Bayes bounds)에서 Fisher가 **지역 곡률**로 등장.

### 5. Empirical vs True Fisher의 함정

Kunstner et al. 2019: 딥러닝에서 **Empirical Fisher** $\hat F = \frac{1}{N}\sum \nabla \ell_n \nabla \ell_n^\top$는 최적화의 준-최적점이 아닌 곳에서 **True Fisher** (모델에 의존)와 크게 다를 수 있다. K-FAC 등은 이를 회피하기 위해 **모델 샘플링 버전** 사용.

---

## ⚖️ 가정과 한계

### (A) = (B) 에서 요구되는 것

- **동일한 지지집합**: $\theta$에 따라 $\{x : p_\theta(x) > 0\}$이 바뀌면 (예: $U(0,\theta)$) 미분·적분 교환 실패 → (A), (B) 불일치 또는 무의미.
- **스코어 평균 0**: $\mathbb{E}_\theta[s_i] = 0$이 성립해야 (A)가 공분산이 됨. 이는 정규화 조건의 1차 미분 결과이며, 적분·미분 교환이 안 되면 성립하지 않음.

### (B) = (C) 에서 요구되는 것

- **$\ell$의 $C^3$ 이상 smoothness**: Taylor 전개의 잉여항 제어.
- **균등 수렴**: $R_3 = o(\|\varepsilon\|^2)$가 **기대값 적분 후에도** 유효해야 함 — 일반적으로 Dominated Convergence로 보장.
- **$F$의 국소적 유계성**: $\mathbb{E}_\theta[|\partial_i \partial_j \ell|]$ 유한해야 함 (R7).

### KL의 비대칭성

- 정의 (C)는 $\operatorname{KL}(p_\theta \| p_{\theta+\varepsilon})$ 의 **2차** 전개. 3차 항은 비대칭이며, 이것이 **$\alpha = +1$ (e-connection)** 과 **$\alpha = -1$ (m-connection)** 의 구분으로 이어진다.

### 유한 차원 · 이산 확률공간

- $\mathcal{X}$가 이산이면 모든 적분을 합으로 바꾸어도 동일. 그러나 $\theta \in \mathbb{R}^\infty$ (비모수 모델) 경우 Fisher 행렬이 **작용소**로 일반화되어 더 정교한 분석 필요.

---

## 📌 핵심 정리

| 정의 | 수식 | 직관 | 사용처 |
|---|---|---|---|
| (A) 스코어 공분산 | $\mathbb{E}[\partial_i \ell \cdot \partial_j \ell]$ | 스코어의 분산 | NGD, K-FAC, Empirical Fisher |
| (B) 음의 Hessian | $-\mathbb{E}[\partial_i \partial_j \ell]$ | 로그우도 곡률 | Newton-like 최적화, PSD |
| (C) KL 2차 근사 | $\operatorname{KL}(p_\theta\|p_{\theta+d\theta}) \approx \tfrac{1}{2} d\theta^\top F d\theta$ | 분포거리의 곡률 | TRPO, Mirror Descent |

**동치성의 조건:** 정칙성 (R1)-(R7), 특히 적분·미분 교환.

**(A)=(B)의 핵심:** 정규화 조건 $\int p_\theta = 1$의 두 번 미분.

**(B)=(C)의 핵심:** $\log$ 함수의 Taylor 전개에서 1차 스코어 평균이 0이어서 소거.

**변환법칙:** $F$는 $(0,2)$-텐서 — 좌표 무관하게 "분포공간 위의 계량"으로 해석 가능. 이것이 **Fisher-Rao 계량** (다음 문서 03).

---

## 🤔 생각해볼 문제

1. **U(0, θ)의 반례**. $p_\theta(x) = \frac{1}{\theta}\mathbb{1}_{[0,\theta]}(x)$에서 (A), (B), (C) 각각이 왜 정의되지 않는지(혹은 어떤 엉뚱한 값이 나오는지) 분석하라. 지지집합이 $\theta$에 의존한다는 사실의 결과는?

2. **가우시안의 Fisher**. 다변수 $\mathcal{N}(\mu, \Sigma)$ 에서 $\theta = (\mu, \operatorname{vech}(\Sigma))$로 매개화할 때 Fisher 블록 구조를 유도하라. $\mu$와 $\Sigma$ 블록이 직교임을 보이고 ($F_{\mu\Sigma} = 0$), 이것이 자연경사의 independence를 어떻게 단순화하는지 설명하라.

3. **상수 부호 확인**. Newton 방법은 $\theta \leftarrow \theta - H^{-1} g$ (여기서 $H$는 Hessian). Fisher 기반 업데이트는 $\theta \leftarrow \theta - F^{-1} g$ (부호 주의!). 왜 부호가 맞는지 — 즉 $F$가 $-\text{Hessian}(\log p)$ 이지만 Loss $L = -\log p$ 의 Hessian은 $+F$라는 점을 확인.

4. **KL 비대칭 심화**. $\operatorname{KL}(p_{\theta+\varepsilon} \| p_\theta)$ 를 2차까지 전개하면 (C)와 같은 $\tfrac{1}{2} F$ 계수가 나온다. 하지만 **3차 항**은 둘이 다르다. 정확히 어떻게 다른지 계산하라. 이 차이가 다음 챕터의 $\alpha$-접속의 어떤 측면과 연결되는가?

5. **Empirical Fisher의 함정**. Kunstner et al. (2019)의 관찰 — 훈련 초기 단계나 under-fit 구간에서 Empirical Fisher가 True Fisher와 크게 다른 이유를 (B)=(A) 증명의 어느 단계가 깨지는지로 설명하라. 힌트: $\mathbb{E}_\theta[s_i] = 0$의 역할.

6. **무한차원 한계**. 신경망의 차원 $k$가 매우 큰 경우 $F \in \mathbb{R}^{k \times k}$ 저장·역변환 불가. K-FAC은 $F \approx A \otimes G$ (Kronecker 근사) 로 해결. 이 근사가 (A)의 어떤 구조적 가정을 하는 것인가? (힌트: 층별 독립성)

7. **지수족의 특별함**. $p_\theta(x) = h(x)\exp(\theta^\top T(x) - \psi(\theta))$ 에서 정의 (B)를 계산하면 $F_{ij}(\theta) = \partial_i \partial_j \psi(\theta)$로 단순해진다. 왜 $-\ell$의 Hessian이 cumulant 함수 $\psi$의 Hessian과 같은지 직접 계산하라. (Ch4의 핵심.)

---

<div align="center">

| [◀ 01. 통계다양체 정의](./01-statistical-manifold.md) | [📚 메인 README](../README.md) | [03. Fisher-Rao 계량 ▶](./03-fisher-rao-metric.md) |
|:---:|:---:|:---:|

</div>
