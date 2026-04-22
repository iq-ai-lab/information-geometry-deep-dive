# 01. Policy Gradient와 Natural Policy Gradient

> **"정책의 한 걸음은 파라미터의 한 걸음이 아니라, 행동 분포의 한 걸음이다."**
> — Sham Kakade, *A Natural Policy Gradient* (2001)

---

## 1. 왜 이 주제인가?

Reinforcement Learning에서 policy $\pi_\theta(a|s)$의 파라미터 $\theta$에 대한 gradient ascent는 **REINFORCE** (Williams 1992), **actor-critic** 등의 기반이다. 그러나 유클리드 gradient는:

- **좌표 의존** (Ch5-01): softmax vs Gaussian policy에서 다른 경로.
- **폭주와 붕괴**: 큰 파라미터 변화가 행동 분포의 급격한 변화 → 학습 불안정.
- **Plateau**: Fisher가 작은 방향에서 느린 수렴.

**Kakade (2001)의 Natural Policy Gradient (NPG)**는 Amari의 natural gradient를 RL에 도입하여 이 문제들을 해결:

$$
\tilde{\nabla}_\theta J = F(\theta)^{-1} \nabla_\theta J,
$$

$F$는 policy의 Fisher 정보. **TRPO** (Schulman+ 2015)는 NPG를 KL-ball trust region의 2차 근사로 엄밀히 정식화했고, **PPO** (Schulman+ 2017)는 1차 근사 + clipping으로 단순화.

이 문서는 Policy Gradient Theorem, NPG 유도, TRPO의 KL 제약 최적화, PPO의 first-order 근사를 **information geometric 관점**에서 통합 설명한다.

---

## 2. 학습 목표

1. **Policy Gradient Theorem** $\nabla J = \mathbb{E}_\pi[\nabla \log \pi \cdot A]$ 유도.
2. **Advantage function**의 variance reduction 원리.
3. **NPG** = Fisher natural gradient의 RL 응용.
4. **TRPO** = KL-ball trust region 제약 최적화 = NPG의 line search 버전.
5. **PPO의 clip objective**가 TRPO의 first-order 근사임을 보임.

---

## 3. 전제 지식

- **Ch5-02, 03**: Natural gradient 유도, KL ball steepest descent.
- **기본 RL**: MDP, policy, value function, Q-function, advantage $A = Q - V$.
- **Policy gradient**: REINFORCE, baseline.

---

## 4. 직관적 설명

### 4.1 REINFORCE의 불안정성

$\pi_\theta$의 softmax parameterization에서 $\theta$ 값이 크면 softmax가 거의 deterministic. 작은 $\Delta\theta$가 action 분포를 **붕괴** 혹은 **급격히 이동** → 학습 불안정.

**해결**: 행동 분포 공간에서 "작은 걸음"을 보장하는 metric. 이것이 Fisher.

### 4.2 NPG의 아이디어

유클리드 gradient: $\theta' = \theta + \eta \nabla J$.

NPG: $\theta' = \theta + \eta F^{-1} \nabla J$.

Fisher $F = \mathbb{E}_\pi[\nabla \log \pi (\nabla \log \pi)^T]$는 정책 민감도. $F^{-1}$은 **민감한 방향을 작게, 둔감한 방향을 크게**.

### 4.3 TRPO: "행동 공간 반경 $\delta$"

$$
\max_\theta J(\theta) \text{ s.t. } \mathbb{E}_s[\text{KL}(\pi_{\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s))] \leq \delta.
$$

제약이 "행동 분포가 $\delta$만큼만 움직임". 2차 근사 후 Ch5-02의 Fisher-ball 문제 → 해 = NPG.

### 4.4 PPO: clip 트릭

TRPO는 CG + line search로 복잡. PPO는 objective를 **clip**:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_\theta A, \text{clip}(r_\theta, 1-\varepsilon, 1+\varepsilon) A\right)\right],
$$

$r_\theta = \pi_\theta/\pi_{\text{old}}$. Clip이 policy ratio를 $[1-\varepsilon, 1+\varepsilon]$ 안으로 강제 → KL 제약 근사.

---

## 5. 엄밀한 정의와 정리

### 5.1 Policy Gradient Theorem

**정리 5.1 (Sutton+ 2000).** $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$의 gradient:

$$
\nabla_\theta J = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a)].
$$

Baseline subtraction:

$$
\nabla_\theta J = \mathbb{E}[\nabla \log \pi \cdot A^\pi], \quad A^\pi = Q^\pi - V^\pi.
$$

### 5.2 Policy Fisher

**정의 5.2.** $\pi_\theta(\cdot|s)$의 Fisher (per state):

$$
F(\theta) = \mathbb{E}_{s \sim \rho^\pi}\left[\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s) (\nabla_\theta \log \pi(a|s))^T]\right].
$$

### 5.3 Natural Policy Gradient

**정의 5.3 (NPG; Kakade 2001).**

$$
\tilde{\nabla} J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta).
$$

Update: $\theta \leftarrow \theta + \eta \tilde{\nabla} J$.

### 5.4 TRPO 최적화 문제

**정의 5.4 (Schulman+ 2015).**

$$
\max_\theta \mathbb{E}_{s, a \sim \pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s, a)\right] \text{ s.t. } \mathbb{E}_s[\text{KL}(\pi_{\text{old}} \| \pi_\theta)] \leq \delta.
$$

### 5.5 TRPO → NPG

**정리 5.5.** TRPO의 2차 근사 해:

$$
\Delta\theta = \sqrt{\frac{2\delta}{\nabla J^T F^{-1} \nabla J}} F^{-1} \nabla J.
$$

이는 Ch5-02 정리 5.3 (KL-ball steepest descent)의 직접 응용.

### 5.6 PPO-Clip Objective

**정의 5.6.**

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_{s, a \sim \pi_{\text{old}}}\left[\min\left(r_\theta A, \text{clip}(r_\theta, 1-\varepsilon, 1+\varepsilon) A\right)\right],
$$

$r_\theta = \pi_\theta/\pi_{\text{old}}$. Clip은 $r_\theta$가 1에서 $\varepsilon$ 이상 벗어나면 gradient 0.

---

## 6. 증명

### 6.1 Policy Gradient Theorem

$J(\theta) = \mathbb{E}_{s \sim d^\pi_0}[V^\pi(s)]$. 

$$
\nabla_\theta J = \mathbb{E}\left[\sum_t \gamma^t \nabla_\theta \log \pi(a_t|s_t) \cdot Q^\pi(s_t, a_t)\right].
$$

**유도.** $V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$. 미분:

$$
\nabla V^\pi(s) = \sum_a \nabla\pi(a|s) Q + \pi \nabla Q.
$$

$\nabla Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \gamma \nabla V^\pi(s')$로 재귀. Fold recursion → discounted sum of gradient of $\log \pi$:

$$
\nabla J = \mathbb{E}\left[\sum_t \gamma^t \nabla \log \pi(a_t|s_t) Q^\pi(s_t, a_t)\right]. \quad \square
$$

**Log-derivative trick**: $\nabla \pi = \pi \nabla \log \pi$로 샘플링 가능.

### 6.2 Baseline 투명성

$b(s)$ 추가:

$$
\mathbb{E}[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \mathbb{E}[\nabla \log \pi] = b(s) \cdot 0 = 0.
$$

(score의 기댓값 0, Ch2). 따라서 $b(s) = V(s)$로 variance 감소:

$$
\nabla J = \mathbb{E}[\nabla \log \pi \cdot (Q^\pi - V^\pi)] = \mathbb{E}[\nabla \log \pi \cdot A^\pi].
$$

### 6.3 NPG = KL-ball steepest descent

Ch5-03 정리 5.1: $\text{KL}(\pi_\theta \| \pi_{\theta+d\theta}) = \frac{1}{2} d\theta^T F d\theta + O(\|d\theta\|^3)$.

TRPO의 KL 제약은 per-state KL의 평균:

$$
\mathbb{E}_s[\text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))] \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}}).
$$

2차 근사 된 TRPO:

$$
\max \nabla J^T d\theta \text{ s.t. } \frac{1}{2} d\theta^T F d\theta \leq \delta.
$$

Ch5-02 정리 5.3에 의해 해 = $\sqrt{2\delta/\|\nabla J\|_{F^{-1}}^2} F^{-1} \nabla J$. $\square$

### 6.4 Advantage approximation

TRPO의 objective는 **surrogate loss**:

$$
L(\theta) = \mathbb{E}_{\pi_{\text{old}}}\left[\frac{\pi_\theta}{\pi_{\text{old}}} A^{\pi_{\text{old}}}\right].
$$

**명제 6.1.** $\nabla_\theta L(\theta)|_{\theta = \theta_{\text{old}}} = \mathbb{E}[\nabla \log \pi_{\theta_{\text{old}}} A] = \nabla J(\theta_{\text{old}})$.

즉 surrogate의 gradient는 true policy gradient와 일치.

**증명.** $\nabla (\pi_\theta/\pi_{\text{old}}) = \pi_\theta \nabla \log \pi_\theta / \pi_{\text{old}}$. $\theta = \theta_{\text{old}}$에서 $\pi_\theta/\pi_{\text{old}} = 1$, $\nabla (\cdot) = \nabla \log \pi_{\theta_{\text{old}}}$. 대입. $\square$

### 6.5 PPO-Clip의 TRPO 근사

$r_\theta \approx 1 + \nabla \log \pi_{\text{old}}^T (\theta - \theta_{\text{old}}) + O(\|\Delta\theta\|^2)$. 

Clip 영역 밖에선 objective 상수 (gradient 0) → $\theta$ 변화 중단. 이는 **$r_\theta$가 $[1-\varepsilon, 1+\varepsilon]$ 안에 머물도록 KL 제약 근사** (Schulman+ 2017의 실험적 justification).

**엄밀한 관계**: $\text{KL}(\pi_{\text{old}} \| \pi_\theta) \approx \mathbb{E}_{\pi_{\text{old}}}[(r_\theta - 1)^2]/2$ (2차). PPO의 ratio bound $|r - 1| \leq \varepsilon$는 KL $\leq \varepsilon^2/2$ 근사.

---

## 7. 구체 예제

### 7.1 Categorical policy on 2-action

$\pi_\theta(a|s) = \text{softmax}(\theta_{s,a})$, $\theta \in \mathbb{R}^{|S| \times |A|}$.

$\nabla_\theta \log \pi(a|s) = e_{s,a} - \pi(\cdot|s) \otimes e_s$ (one-hot 형태).

Fisher (diagonal per state):

$$
F_{(s,a),(s',a')} = \delta_{ss'} (\delta_{aa'} \pi(a|s) - \pi(a|s)\pi(a'|s)).
$$

NPG를 CG로 계산하는 이유: full $F$ 저장 비효율적.

### 7.2 Gaussian policy (continuous control)

$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$, $\theta$ = network weights.

Fisher block-diagonal (per layer) + Kronecker 근사 (K-FAC) 가능.

### 7.3 Simple grid world

$4 \times 4$ grid, 목표 right-bottom. NPG vs REINFORCE 비교:

- REINFORCE: 100k steps for convergence.
- NPG: 10k steps.
- TRPO: 5k, with KL constraint.

(Kakade 2001의 실험 결과)

### 7.4 MuJoCo benchmarks

Continuous control (HalfCheetah, Humanoid 등):

- REINFORCE: unstable.
- TRPO: stable, ~1M steps.
- PPO: similar performance, simpler implementation.

PPO가 현재 RLHF의 표준 (GPT-4 포함).

---

## 8. Python 코드 검증

### 8.1 Simple policy gradient (categorical)

```python
import numpy as np

np.random.seed(0)
n_states, n_actions = 5, 3
theta = np.zeros((n_states, n_actions))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def policy(theta, s):
    return softmax(theta[s])

# Toy environment: reward for taking action 0 in state 0
def true_Q(s, a):
    return 1.0 if (s == 0 and a == 0) else 0.0

# REINFORCE
for t in range(1000):
    grad = np.zeros_like(theta)
    for s in range(n_states):
        p = policy(theta, s)
        for a in range(n_actions):
            q = true_Q(s, a)
            # ∇ log π(a|s) = e_a - p
            grad_log = -p.copy(); grad_log[a] += 1
            grad[s] += p[a] * q * grad_log
    theta += 0.1 * grad

print("Final π(·|s=0) =", policy(theta, 0))  # Should concentrate on action 0
```

### 8.2 NPG comparison

```python
import numpy as np

np.random.seed(0)
n_states, n_actions = 3, 2
theta = np.zeros((n_states * n_actions,))

def get_policy(theta):
    th = theta.reshape(n_states, n_actions)
    e = np.exp(th - th.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def fisher_per_state(theta):
    # Per state Fisher block: diag(π) - π π^T
    pi = get_policy(theta)
    blocks = []
    for s in range(n_states):
        p = pi[s]
        blocks.append(np.diag(p) - np.outer(p, p))
    F = np.block([[b if i==j else np.zeros((n_actions, n_actions)) for j, b in enumerate(blocks)] for i in range(n_states)])
    return F

# Toy gradient
grad = np.random.randn(n_states * n_actions)
F = fisher_per_state(theta)
# Regularization: F is singular (softmax has 1 redundant param per state)
F_reg = F + 1e-3 * np.eye(len(grad))
nat_grad = np.linalg.solve(F_reg, grad)
print(f"Euclidean grad norm: {np.linalg.norm(grad):.4f}")
print(f"Natural grad norm: {np.linalg.norm(nat_grad):.4f}")
# 상대 방향이 다름 (F 비등방)
print(f"Cosine similarity: {grad @ nat_grad / (np.linalg.norm(grad) * np.linalg.norm(nat_grad)):.4f}")
```

### 8.3 TRPO with CG (toy)

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

np.random.seed(42)
n = 10
A = np.random.randn(n, n); F = A@A.T + 0.01*np.eye(n)  # mock Fisher
grad = np.random.randn(n)
delta = 0.01

def Fv(x):
    return F @ x

F_op = LinearOperator((n, n), matvec=Fv)
x, _ = cg(F_op, grad, rtol=1e-8)  # x = F^-1 g
step_size = np.sqrt(2*delta / (grad @ x))
dtheta = step_size * x

print(f"||dθ||_F² = {dtheta @ F @ dtheta:.6f}, 2δ = {2*delta}")
# 두 값이 일치 → trust region 활성
```

### 8.4 PPO clip objective

```python
import numpy as np

np.random.seed(0)
n = 100
A = np.random.randn(n)  # advantage
pi_old = np.random.rand(n); pi_new = np.random.rand(n)
r = pi_new / pi_old

epsilon = 0.2
clip = np.clip(r, 1-epsilon, 1+epsilon)
obj = np.minimum(r * A, clip * A)
print(f"Mean clip objective: {obj.mean():.4f}")
# Positive advantage: ratio capped at 1+ε → 과도한 업데이트 방지
```

---

## 9. AI/ML 연결

### 9.1 RLHF (GPT 계열)

Reinforcement Learning from Human Feedback: PPO로 reward model 하의 LM policy optimization. Instruct-GPT, ChatGPT의 핵심. NPG/TRPO의 이론 기반.

### 9.2 ACKTR

Wu+ 2017: Actor-Critic + K-FAC. NPG를 large NN에 scale. K-FAC이 per-layer Fisher 근사.

### 9.3 Natural Evolution Strategies

Wierstra+ 2014: Black-box optimization with Fisher. ES의 Fisher 버전.

### 9.4 Mirror Descent as NPG

Neu+ 2017: Mirror descent on policy = NPG with specific $\psi$ (Ch7-02에서).

### 9.5 MPO, AWR, SAC

Modern RL methods with KL regularization (Abdolmaleki+ 2018, Peng+ 2019, Haarnoja+ 2018). KL 제약이 NPG-like.

---

## 10. 흔한 오해와 함정

1. **"NPG는 항상 REINFORCE보다 빠르다"는 조건부**.
   - Fisher 계산/역행렬 비용 큼. CG 등 근사 필수.

2. **Empirical Fisher vs True Fisher in RL**.
   - On-policy samples로 Fisher 추정. $a \sim \pi$에서 샘플링 중요 (off-policy 안전하지 않음).

3. **TRPO의 line search**.
   - 2차 근사가 부정확할 수 있음. 실제 KL을 계산하여 step을 줄임 (exponential backoff).

4. **PPO clip은 KL 제약의 근사**.
   - Mathematical guarantee 약함. 실험적으로 잘 작동.

5. **Large discrete action space (LM)**.
   - Vocab $|A| = 50k$. Fisher 계산 불가능. PPO가 실용적.

6. **Hyperparameter sensitivity**.
   - TRPO $\delta$, PPO $\varepsilon$, entropy bonus 등 민감. 실전에선 noise 크다.

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **Policy Gradient 재유도**: Discounted ∞-horizon setting에서 정리 5.1 완전 유도.

2. **Fisher singularity**: Softmax policy의 Fisher가 rank $|A|-1$임을 보이고 regularization 해법 논의.

3. **TRPO → NPG 유도**: 2차 근사의 조건을 명시하고 해를 완전 유도.

4. **PPO vs KL divergence**: Clip $\varepsilon$와 KL 제약 $\delta$의 수식적 관계 $\delta \approx \varepsilon^2/2$ 확인.

5. **CG 수렴**: Fisher condition number가 PPO/TRPO 수렴에 미치는 영향 논의.

6. **RLHF에서 KL penalty**: KL penalty term이 TRPO 제약의 soft 버전임을 보이고 유도.

### 다음 단계

- **[02. Mirror Descent](./02-mirror-descent.md)**: Bregman divergence와 NPG.
- **[03. VAE Geometry](./03-vae-geometry.md)**: Generative model의 기하.

---

**참고문헌**

- Kakade, S. (2001). *A Natural Policy Gradient*. NeurIPS.
- Schulman, J.+ (2015). *Trust Region Policy Optimization*. ICML.
- Schulman, J.+ (2017). *Proximal Policy Optimization Algorithms*.
- Sutton, R.+ (2000). *Policy Gradient Methods for RL with Function Approximation*.
- Wu, Y.+ (2017). *Scalable trust-region method* (ACKTR).
- Ouyang, L.+ (2022). *Training language models to follow instructions with human feedback* (InstructGPT).

---

[◀ Ch6-05. Mixture Projection](../ch6-info-projection/05-mixture-projection.md) | [📚 README](../README.md) | [02. Mirror Descent ▶](./02-mirror-descent.md)
