# 05. 실전 Natural Gradient — K-FAC, Shampoo, TRPO

> **"$F^{-1}$을 정확히 계산하면 이론적으로 완벽하다. 그러나 현실에선 $10^9$ 파라미터 신경망에서 $F$조차 메모리에 안 들어간다."**

---

## 1. 왜 이 주제인가?

Ch5-02~04에서 natural gradient $\tilde{\nabla} L = F^{-1} \nabla L$의 이론적 우수성을 보았다:
- KL ball의 steepest descent
- Parameterization 불변
- Exp family에서 Newton's method와 같음

그러나 **현실의 벽**은 높다. 신경망이 $n = 10^7 \sim 10^{12}$ 파라미터를 갖는다면:
- **메모리**: $F$는 $n \times n$ 행렬 → $n = 10^7$이면 $10^{14}$ 엔트리, 수백 TB.
- **계산**: $F^{-1}$ 역행렬은 $O(n^3)$.
- **샘플링**: $F = \mathbb{E}_{y \sim p}[\nabla \log p \nabla \log p^T]$는 매 step마다 forward/backward 다수 회.

**해결의 길**: **구조적 근사**. Fisher는 완전 일반 행렬이 아니라 **층 구조, Kronecker 구조**를 가진다. 이를 이용해 $O(n)$ 메모리, $O(n \cdot \text{rank})$ 계산으로 줄이는 기법들 — **K-FAC, Shampoo, EKFAC, TRPO(CG)** 등. 이 문서는 이들의 수학적 원리와 실전 구현을 다룬다.

---

## 2. 학습 목표

1. **Gauss-Newton과 Fisher의 관계**를 명확히 이해.
2. **Kronecker-factored 근사 (K-FAC)**의 유도: $F_\ell \approx A_\ell \otimes G_\ell$.
3. **Shampoo**의 Kronecker 전처리자 유도.
4. **TRPO/PPO의 conjugate gradient**로 $F^{-1} v$ 근사 (Hessian-vector product 원리).
5. **Empirical Fisher vs True Fisher**의 차이와 위험 (Kunstner+ 2019).

---

## 3. 전제 지식

- **Ch5-02~04**: Natural gradient의 정의와 성질
- **Ch4-02**: Fisher = Hessian of $\psi$
- **Kronecker product**: $(A \otimes B)(C \otimes D) = AC \otimes BD$, $\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X)$
- **Conjugate gradient** (Nocedal & Wright Ch.5)

---

## 4. 직관적 설명

### 4.1 Fisher는 "Block" 구조를 가진다

Neural net의 파라미터는 층마다 나뉜다: $\theta = (W_1, W_2, \dots, W_L)$. Fisher의 block-diagonal 근사:

$$
F \approx \text{blkdiag}(F_1, F_2, \dots, F_L).
$$

**이유**: 서로 다른 층의 파라미터는 출력에 **곱셈적**으로 영향을 주므로 cross-covariance가 작다고 근사. 이것이 K-FAC의 출발점.

### 4.2 각 층의 Fisher는 Kronecker

층 $\ell$의 출력 $z = W a$ ( $a$: 입력 activation, $W \in \mathbb{R}^{p \times q}$, $z \in \mathbb{R}^p$). 손실의 출력에 대한 기울기 $g = \nabla_z L$. 그러면:

$$
\nabla_W L = g a^T.
$$

Fisher:

$$
F_\ell = \mathbb{E}[\text{vec}(ga^T)\text{vec}(ga^T)^T] = \mathbb{E}[(a \otimes g)(a \otimes g)^T] = \mathbb{E}[aa^T \otimes gg^T].
$$

**K-FAC 근사**: $\mathbb{E}[aa^T \otimes gg^T] \approx \mathbb{E}[aa^T] \otimes \mathbb{E}[gg^T] =: A \otimes G$. 이 근사는 **$a$와 $g$가 독립이라 가정**하는 것.

### 4.3 Kronecker의 힘

$F_\ell \approx A \otimes G$면:

$$
F_\ell^{-1} \approx A^{-1} \otimes G^{-1},
$$

$$
F_\ell^{-1} \text{vec}(\nabla_W L) = \text{vec}(G^{-1} (\nabla_W L) A^{-1}).
$$

저장은 $A \in \mathbb{R}^{q \times q}$, $G \in \mathbb{R}^{p \times p}$ — 총 $O(p^2 + q^2)$, 역행렬도 $O(p^3 + q^3)$, 곱도 $O(pq(p+q))$. **엄청난 절감**: 원래 $O((pq)^2)$, $O((pq)^3)$에서 drastic 감소.

---

## 5. 엄밀한 정의와 정리

### 5.1 Fisher vs Gauss-Newton

**정의 5.1.** 회귀/분류 문제에서 모델 출력 $f_\theta(x)$, 손실 $L(\theta) = \mathbb{E}[\ell(y, f_\theta(x))]$.

- **Fisher**: $F = \mathbb{E}_{x, y \sim p_\theta}[\nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^T]$. $y$는 **모델에서 샘플링**.
- **Empirical Fisher**: $\hat{F} = \frac{1}{N} \sum_i \nabla_\theta \log p(y_i|x_i;\theta) \nabla_\theta \log p(y_i|x_i;\theta)^T$. $y_i$는 **데이터**.
- **Gauss-Newton**: $G = \mathbb{E}_x[J_f^T H_\ell J_f]$, $J_f = \partial f/\partial\theta$, $H_\ell = \partial^2\ell/\partial f^2$.

**관계 (Martens 2014)**: 지수족 likelihood ($\ell = -\log p$, $p$ exp family)에서 $F = G$. 예: Gaussian regression ($\ell = \|y-f\|^2/2$), cross-entropy with softmax.

### 5.2 K-FAC 근사

**정의 5.2 (K-FAC; Martens & Grosse 2015).** 층 $\ell$의 MLP 블록 $z_\ell = W_\ell a_\ell$ (fully connected)에 대해:

$$
\boxed{F_\ell \approx A_\ell \otimes G_\ell, \quad A_\ell = \mathbb{E}[a_\ell a_\ell^T], \quad G_\ell = \mathbb{E}[g_\ell g_\ell^T],}
$$

$g_\ell = \nabla_{z_\ell} L$. 전체 Fisher를 block-diagonal로 근사:

$$
F \approx \text{blkdiag}(A_1 \otimes G_1, \dots, A_L \otimes G_L).
$$

### 5.3 K-FAC 업데이트 규칙

**정리 5.3.** K-FAC natural gradient 업데이트:

$$
\Delta W_\ell = -\eta G_\ell^{-1} (\nabla_{W_\ell} L) A_\ell^{-1}.
$$

(유도: $F_\ell^{-1} \text{vec}(\nabla W) = (A^{-1} \otimes G^{-1}) \text{vec}(\nabla W) = \text{vec}(G^{-1} (\nabla W) A^{-1})$)

### 5.4 Shampoo 업데이트

**정의 5.4 (Shampoo; Gupta+ 2018).** 텐서 파라미터 $W \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_k}$에 대해 각 축마다 preconditioner:

$$
L_i^{(t)} = L_i^{(t-1)} + G^{(t)}_{(i)} (G^{(t)}_{(i)})^T,
$$

$G^{(t)}_{(i)}$는 mode-$i$ matricization. 업데이트:

$$
W^{(t+1)} = W^{(t)} - \eta \cdot (L_1^{-1/(2k)}) \times_1 \dots \times_k (L_k^{-1/(2k)}) \cdot G^{(t)}.
$$

**Claim (Anil+ 2020)**: Shampoo는 **full-matrix AdaGrad의 최적 Kronecker 근사**이며, 많은 경우 Adam보다 빠름.

### 5.5 Conjugate Gradient로 $F^{-1}v$

**정리 5.5 (TRPO의 CG).** $Fx = v$를 CG로 풀기, 각 iteration에서 **Fisher-vector product** $Fv$만 필요. 이는 두 번의 backward로 계산:

$$
Fv = \nabla_\theta (\nabla_\theta L \cdot v).
$$

$k$ iteration으로 $\|Fx_k - v\|$가 기계 정밀도 근처까지 감소 (well-conditioned 시 $k \sim 10$).

---

## 6. 증명/유도

### 6.1 K-FAC 근사의 유도

신경망 층: $z = Wa$, $W \in \mathbb{R}^{p \times q}$, $a \in \mathbb{R}^q$, $z \in \mathbb{R}^p$.

$\nabla_W L = g a^T$, $g = \nabla_z L \in \mathbb{R}^p$. $\text{vec}(\nabla_W L) = a \otimes g$ (column-major). 

**Fisher**:

$$
F_\ell = \mathbb{E}[(a \otimes g)(a \otimes g)^T] = \mathbb{E}[(aa^T) \otimes (gg^T)].
$$

**K-FAC 근사**: 만약 $a$와 $g$가 독립이라면:

$$
\mathbb{E}[aa^T \otimes gg^T] = \mathbb{E}[aa^T] \otimes \mathbb{E}[gg^T] = A \otimes G.
$$

실제로 $a$와 $g$는 독립이 아니지만 (both 데이터와 $\theta$에 의존), 실험적으로 좋은 근사. $\square$

### 6.2 Kronecker 역행렬

**Lemma.** $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$.

**증명.** $(A \otimes B)(A^{-1} \otimes B^{-1}) = (AA^{-1}) \otimes (BB^{-1}) = I \otimes I = I$. $\square$

### 6.3 $\text{vec}$와 Kronecker 공식

**Lemma.** $\text{vec}(BXC) = (C^T \otimes B)\text{vec}(X)$.

따라서 $(A^{-1} \otimes G^{-1}) \text{vec}(\nabla W) = \text{vec}(G^{-1}(\nabla W)(A^{-1})^T)$. $A$는 대칭이므로 $A^{-T} = A^{-1}$:

$$
\text{vec}^{-1}(F^{-1} \text{vec}(\nabla W)) = G^{-1}(\nabla W)A^{-1}. \quad \square
$$

### 6.4 Convolution에서의 K-FAC

Grosse & Martens (2016): Conv 층에서

$$
z^{(x,y)} = \sum_{(u,v)} W_{u,v} a^{(x+u, y+v)}.
$$

Kronecker factor로 근사:

$$
A_{\text{conv}} = \frac{1}{|\text{loc}|} \sum_{(x,y)} \mathbb{E}[\bar{a}^{(x,y)} \bar{a}^{(x,y)T}],
$$

$\bar{a}$는 receptive field vector. **"spatial-temporal averaging"**으로 단일 $A$, 단일 $G$. 여전히 Kronecker 형태.

### 6.5 Shampoo as optimal Kronecker preconditioner

Anil+ (2020): full-matrix AdaGrad $G^{(t)} = \sum_s g_s g_s^T$에 대해:

$$
\min_{L_1, L_2} \|G - L_1 \otimes L_2\|_F.
$$

닫힌 해는 없지만, **Shampoo는 "adjoint trick"으로 이 문제를 각 축마다 근사적 푸는 최적화자**. 구체적으로:

$$
L_i = \sum_t G^{(t)}_{(i)} (G^{(t)}_{(i)})^T,
$$

이때 $L_i^{-1/(2k)}$의 Kronecker product는 전체 $G^{-1/2}$의 Kronecker 근사.

### 6.6 Hessian-vector product

$L(\theta)$의 gradient $g(\theta) = \nabla L$. 임의 벡터 $v$에 대해:

$$
Hv = \nabla (g \cdot v).
$$

PyTorch:

```python
loss.backward(create_graph=True)  # 1st grad with grad graph
grad = theta.grad
hv = torch.autograd.grad(grad @ v, theta)[0]  # 2nd backward
```

비용: 기본 backward의 2~3배 (∼ 5x forward).

Fisher-vector product도 유사: $Fv = \nabla_\theta (\text{KL}(p_\theta \| p_{\theta_{\text{old}}}) \cdot \text{things})$ — TRPO 구현 참조.

---

## 7. 구체 예제

### 7.1 작은 MLP에서 K-FAC 수치 검증

$W \in \mathbb{R}^{10 \times 5}$, batch 100.

```python
import numpy as np

np.random.seed(0)
p, q, N = 10, 5, 100
a = np.random.randn(N, q)
g = np.random.randn(N, p)

# 정확한 Fisher (층)
F_exact = np.zeros((p*q, p*q))
for i in range(N):
    vec = np.outer(g[i], a[i]).flatten()  # vec(g a^T)
    F_exact += np.outer(vec, vec) / N

# K-FAC 근사
A = (a.T @ a) / N  # q x q
G = (g.T @ g) / N  # p x p
F_kfac = np.kron(A, G)

# 차이
print(f"||F_exact - F_kfac||_F / ||F_exact||_F = {np.linalg.norm(F_exact - F_kfac, 'fro')/np.linalg.norm(F_exact, 'fro'):.4f}")

# 독립 가정이 잘 맞으면 차이 작음. a, g 독립 (여기선 서로 무관하게 생성)이라 매우 작음.
```

**예상**: $a, g$ 독립이므로 0.1 이하 상대 오차. 실제 NN에서는 의존성 있어 조금 더 큼.

### 7.2 K-FAC vs SGD on MNIST-like

```python
# PyTorch pseudocode (실제로 돌리려면 torch, nngeometry 등 필요)
"""
import torch
from torch import nn
# K-FAC: nngeometry or kfac-pytorch 라이브러리 사용 권장

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = KFACOptimizer(model, lr=0.01, damping=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()  # K-FAC 업데이트: ΔW = -η G^-1 (∇W) A^-1
"""
```

**관찰**: K-FAC은 SGD보다 **수 배 적은 iteration**으로 수렴 (batch당 계산은 더 큼). ResNet, Transformer에서 검증 (Grosse+ 2016, Pauloski+ 2020).

### 7.3 TRPO의 CG 예제

$F x = v$, $F = A A^T + \lambda I$, $A$ 임의 $n \times n$.

```python
import numpy as np
from scipy.sparse.linalg import cg

np.random.seed(1)
n = 100
A = np.random.randn(n, n)
F = A @ A.T + 0.01*np.eye(n)
v = np.random.randn(n)

# 직접
x_direct = np.linalg.solve(F, v)

# CG (matrix-vector product만 사용)
def Fv(x):
    return F @ x

x_cg, info = cg(np.array([[0]]), v)  # placeholder

# LinearOperator 사용이 깔끔
from scipy.sparse.linalg import LinearOperator
Fop = LinearOperator((n,n), matvec=Fv)
x_cg, info = cg(Fop, v, rtol=1e-8, maxiter=50)

print(f"CG iterations needed: info={info} (0=success)")
print(f"||x_direct - x_cg|| = {np.linalg.norm(x_direct - x_cg):.2e}")
```

**기대**: 50 iteration 내 수렴, 오차 `~1e-8`. NN에서는 condition number가 크면 damping $\lambda$ 필요.

### 7.4 Empirical vs True Fisher (Kunstner 2019의 함정)

Simple example: $p(y | x, \theta) = \mathcal{N}(x^T\theta, 1)$, data $y_i$에서 MLE 중이라 하자.

- **True Fisher**: $F = \mathbb{E}_{x, y\sim p_\theta}[xx^T] = \mathbb{E}_x[xx^T]$ (결과가 깔끔).
- **Empirical Fisher**: $\hat{F} = \frac{1}{N}\sum (y_i - x_i^T\theta)^2 x_i x_i^T$ — **잔차의 크기로 스케일링됨**.

$\theta$가 MLE 근처면 잔차 작음 → $\hat{F} \approx 0$ → $\hat{F}^{-1} \nabla L$이 엄청 큼 → 폭주. 이것이 Kunstner+ 2019가 경고한 경우. **해결**: true Fisher (model에서 y 샘플) 쓰거나 Gauss-Newton 사용.

---

## 8. Python 코드 검증

### 8.1 Kronecker 곱과 vec 공식

```python
import numpy as np

p, q = 4, 3
A = np.random.randn(q, q); A = A @ A.T + np.eye(q)
G = np.random.randn(p, p); G = G @ G.T + np.eye(p)
W = np.random.randn(p, q)

# 방법 1: 직접 Kronecker
F = np.kron(A, G)
grad_W = W  # 임의 gradient
vec_grad = grad_W.flatten(order='F')  # column-major vec
nat1 = np.linalg.solve(F, vec_grad)
nat1_W = nat1.reshape(p, q, order='F')

# 방법 2: Kronecker 공식 G^-1 (∇W) A^-1
nat2_W = np.linalg.solve(G, grad_W) @ np.linalg.inv(A)

# 비교
print(f"차이 norm: {np.linalg.norm(nat1_W - nat2_W):.2e}")
```

**기대**: `~1e-14`.

### 8.2 K-FAC vs SGD 수렴 비교 (toy)

```python
import numpy as np

# Linear regression, small toy
np.random.seed(0)
N, d = 500, 20
X = np.random.randn(N, d)
true_w = np.random.randn(d)
y = X @ true_w + 0.1*np.random.randn(N)

# SGD
w_sgd = np.zeros(d)
# K-FAC (여기선 layer 1개 = single matmul, A = X^T X / N, G = 1)
A = (X.T @ X) / N
A_inv = np.linalg.inv(A + 1e-3*np.eye(d))
w_kfac = np.zeros(d)

lr = 0.01
losses_sgd, losses_kfac = [], []
for t in range(100):
    # grad = -X^T (y - Xw)/N
    g_sgd = -(X.T @ (y - X @ w_sgd)) / N
    g_kfac = -(X.T @ (y - X @ w_kfac)) / N
    
    w_sgd  = w_sgd  - lr * g_sgd
    w_kfac = w_kfac - lr * (A_inv @ g_kfac)  # Natural gradient
    
    losses_sgd.append(0.5 * np.mean((y - X @ w_sgd)**2))
    losses_kfac.append(0.5 * np.mean((y - X @ w_kfac)**2))

print(f"After 100 steps: SGD loss = {losses_sgd[-1]:.6f}")
print(f"After 100 steps: K-FAC loss = {losses_kfac[-1]:.6f}")
# K-FAC (여기선 Newton과 같음) 훨씬 빠르게 수렴
```

**예상**: K-FAC은 수 iteration 내 수렴 (lin. regression에서 Newton과 동치), SGD는 100 step에도 멀리.

### 8.3 Fisher-vector product (Hessian-free)

```python
import torch

torch.manual_seed(0)
# Toy: negative log likelihood of Gaussian
theta = torch.randn(5, requires_grad=True)
X = torch.randn(100, 5)

def nll(theta):
    mu = X @ theta
    return 0.5 * ((X[:,0] - mu)**2).mean()  # toy loss

# 1차 grad
loss = nll(theta)
grad = torch.autograd.grad(loss, theta, create_graph=True)[0]

v = torch.randn_like(theta)
# Hv = d(grad · v)/dtheta
Hv = torch.autograd.grad(grad @ v, theta)[0]
print("Hv =", Hv)

# 정확한 H로 확인 (여기선 H = (1/N) X^T X)
H_exact = (X.T @ X) / 100
print("Hv exact =", H_exact @ v)
```

**기대**: 두 결과 소수점 일치.

---

## 9. AI/ML 연결

### 9.1 K-FAC in practice

- **Grosse & Martens (2016)**: Conv, RNN, CNN에서 K-FAC 확장.
- **Pauloski+ (2020)**: 대규모 분산 K-FAC (ResNet-50 ImageNet 5x speedup).
- **Osawa+ (2019)**: ImageNet Brand K-FAC.

### 9.2 Shampoo / Optimizer 대체

- **Google (2021)**: 대규모 학습에 Shampoo 도입, Adam 대비 30% 효율 증가.
- **DeepMind (2023)**: Distributed Shampoo. PaLM 계열에서 사용.

### 9.3 TRPO, PPO, ACKTR

- **TRPO (Schulman+ 2015)**: CG로 Fisher-vector product 풀기.
- **PPO (Schulman+ 2017)**: Soft KL penalty, 더 간단.
- **ACKTR (Wu+ 2017)**: Actor-critic + K-FAC.

### 9.4 VI와 EF / Fisher 근사

- **Bayesian Learning Rule (Khan & Rue 2023)**: VI를 natural gradient로 통합.
- **VADAM (Khan+ 2018)**: Adam을 VI의 natural gradient로 해석.

### 9.5 Diffusion models의 NG?

- **Diffusion score-matching**: $\nabla \log p_t(x)$ 추정. 2차 정보 (Fisher)는 아직 연구 단계.
- **EDM (Karras+ 2022)**: 경험적 2차 스케일링, 이론적 NG는 Ch7-05에서.

---

## 10. 흔한 오해와 함정

1. **"K-FAC이 항상 SGD보다 빠르다"는 거짓.**
   - Per-iteration 비용 큼 (역행렬 2~5배). 수렴이 빨라야 총 시간 이득. 작은 모델이나 쉬운 문제는 SGD+Adam이 더 낫기도.

2. **Damping 하이퍼파라미터 중요.**
   - $F + \lambda I$의 $\lambda$가 너무 작으면 불안정, 너무 크면 SGD로 수렴. Martens & Grosse (2015)의 adaptive damping 추천.

3. **Empirical Fisher의 위험.**
   - Kunstner+ 2019: $\theta$가 최적에 가까우면 잔차 → 0, empirical F → 0, 업데이트 폭주. 반드시 true Fisher (model 샘플) 또는 GN 사용.

4. **K-FAC의 독립 가정.**
   - $a$와 $g$가 실제로 독립 아님. 근사 오차 존재. EKFAC (George+ 2018)은 이 문제를 Eigenvalue correction으로 개선.

5. **Conv/Transformer에서 Kronecker 구조는 복잡.**
   - Attention layers, layer norm, residual connection 등에서 단순한 $A \otimes G$가 깨짐. Grosse+ 2016의 spatial-temporal averaging, Shampoo는 텐서 축마다 처리.

6. **CG의 수렴 보장.**
   - $F \succ 0$ 대칭이면 CG 보장. 하지만 empirical Fisher는 rank-deficient, $F + \lambda I$ damping 필수.

7. **Batch norm의 F.**
   - BN 파라미터는 scale/shift로 매우 민감. Separate treatment 필요 (Osawa+ 2019).

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **K-FAC 공식 유도**: $F_\ell \approx A \otimes G$에서 $\Delta W = -\eta G^{-1} (\nabla W) A^{-1}$를 $\text{vec}$와 Kronecker 공식으로 완전히 도출.

2. **Kronecker identity**: $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$를 두 방법 (정의, $\text{vec}$)으로 증명.

3. **EKFAC**: K-FAC의 eigenvalue correction 제안 (George+ 2018)을 읽고 K-FAC과의 차이를 정리.

4. **Linear regression 정확 해**: $L = \|y - X\theta\|^2/N$에서 Newton = Fisher natural gradient = K-FAC임을 보여라 ($A = X^T X/N$, $G = 1$).

5. **CG의 iteration 수**: Condition number $\kappa$에 따라 CG 수렴율 $\sqrt{(\kappa-1)/(\kappa+1)}$. $\kappa = 10^6$에서 $\varepsilon = 10^{-4}$ 오차까지 몇 iteration?

6. **Shampoo vs Full AdaGrad**: 메모리와 계산 비용 표 작성 ($n = 10^6, k = 2$).

### 다음 단계 (Chapter 6 예고)

- **[Ch6: Information Projection](../ch6-info-projection/README.md)**: e-projection, m-projection, EM의 이중 사영.
- **[Ch7: AI Applications](../ch7-ai-applications/README.md)**: Natural policy gradient, mirror descent, VAE, RHMC, diffusion.

---

**참고문헌**

- Martens, J. & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature* (K-FAC).
- Grosse, R. & Martens, J. (2016). *A Kronecker-factored Approximate Fisher Matrix for Convolution Layers*.
- George, T.+ (2018). *Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis* (EKFAC).
- Gupta, V., Koren, T., Singer, Y. (2018). *Shampoo: Preconditioned Stochastic Tensor Optimization*.
- Anil, R.+ (2020). *Scalable Second Order Optimization for Deep Learning* (Distributed Shampoo).
- Schulman, J.+ (2015). *Trust Region Policy Optimization* (TRPO).
- Wu, Y.+ (2017). *Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation* (ACKTR).
- Kunstner, F., Balles, L., Hennig, P. (2019). *Limitations of the Empirical Fisher Approximation*.
- Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*. JMLR.
- Khan, M.E., Rue, H. (2023). *The Bayesian Learning Rule*. JMLR.

---

[◀ 04. Parameterization 불변성](./04-parameterization-invariance.md) | [📚 README](../README.md) | [Ch6-01. e-projection & m-projection ▶](../ch6-info-projection/01-e-m-projection.md)
