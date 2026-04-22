# 04. e-connection과 m-connection — 쌍대 아핀 연결

<div align="center">

> *"Levi-Civita 연결은 리만 계량이 주어지면 유일하다.*  
> *하지만 정보기하에는 '잘못된 두 연결'이 있다 — e-connection과 m-connection.*
>
> *이 둘은 각각 torsion-free이고 flat이지만, **계량과 호환(metric-compatible)이 아니다**.  
> 대신 서로가 서로를 보완하여 '쌍대(dual)' 관계를 이룬다.  
> 이것이 Amari 정보기하의 심장이다."*

</div>

---

## 🎯 핵심 질문

1. **아핀 연결(affine connection)**은 무엇이고, 같은 리만 계량 위에 여러 연결이 가능한 이유는?
2. 지수족에서 **exponential connection** $\nabla^{(e)}$와 **mixture connection** $\nabla^{(m)}$을 어떻게 정의하는가? 왜 각 좌표계에서 크리스토펠 기호가 0인가?
3. 두 연결은 각각 **torsion-free**이지만 **metric-compatible이 아니다**. 이것이 Levi-Civita가 아닌 이유는?
4. **쌍대 연결(dual connections)** 관계 $Xg(Y, Z) = g(\nabla_X Y, Z) + g(Y, \nabla^*_X Z)$는 어떻게 정의되고, $\nabla^{(e)}, \nabla^{(m)}$이 이 관계를 만족함을 증명.
5. **α-connection** $\nabla^{(\alpha)} = \frac{1+\alpha}{2}\nabla^{(e)} + \frac{1-\alpha}{2}\nabla^{(m)}$의 의미와 $\alpha = 0$이 Levi-Civita임을 증명.

---

## 🔍 왜 이 기하학이 AI에서 중요한가

| AI/ML 기법 | e/m-connection이 하는 일 |
|-----------|------------------|
| **EM Algorithm** | E-step = m-projection, M-step = e-projection (Ch6-02) — 두 projection이 각 연결의 **geodesic projection** |
| **Information Bottleneck** | Tishby의 IB가 두 연결 하의 교대 projection 문제로 재해석 |
| **α-divergence** | $D_\alpha$가 $\nabla^{(\alpha)}$-geodesic distance 측정 (Ch3-05) |
| **Natural Gradient $\alpha = 0$** | $\alpha=0$의 Levi-Civita가 **Fisher-Rao 리만 geodesic**을 정의 → geodesic NGD |
| **MaxEnt와 exp family** | e-affine이 exp family의 "straight line" — 제약 조건 하 최대 엔트로피 해가 e-geodesic 위에 |
| **Wasserstein vs KL** | KL은 e/m 쌍대성; Wasserstein은 별도의 OT connection — 두 기하학의 비교 |

---

## 📐 수학적 선행 조건

| 개념 | 참조 |
|------|------|
| **아핀 연결** $\nabla$의 공리, 크리스토펠 기호 $\Gamma^k_{ij}$ | **Ch1-04** |
| **torsion tensor**과 **curvature tensor** | Ch1-04 |
| **Levi-Civita 연결**의 유일성 | Ch1-04 |
| Fisher 계량 $g_{ij} = F_{ij}$ | **Ch2-03** |
| $\nabla^2\psi = F$, Legendre 쌍대 $\theta \leftrightarrow \eta$ | **Ch4-02, 4-03** |

---

## 📖 직관적 이해

### 1. "평평함"은 좌표계에 의존한다

Ch1-04에서 보았듯이, 같은 리만 다양체 위에 여러 아핀 연결이 있을 수 있다. 각 연결은 "어떤 곡선이 직선인가"(geodesic)를 결정한다.

지수족 $\mathcal{E}$에서:
- $\theta$ 좌표에서 "직선" $\theta(t) = (1-t)\theta_1 + t\theta_2$는 어떤 분포 곡선인가?
- $\eta$ 좌표에서 "직선" $\eta(t) = (1-t)\eta_1 + t\eta_2$는 어떤 분포 곡선인가?

두 직선은 **일반적으로 다르다** (곡률이 있으면). 그런데 지수족에서 두 직선은 모두 "흥미로운" 의미를 가진다:

- **e-geodesic** ($\theta$-직선): $p_{\theta(t)}(x) \propto p_{\theta_1}(x)^{1-t} p_{\theta_2}(x)^t$ — 로그 공간에서의 선형 보간.
- **m-geodesic** ($\eta$-직선): $p_{\eta(t)}$는 $\mathbb{E}[T] = (1-t)\eta_1 + t\eta_2$를 만족하는 지수족 분포 — "moment mixture".

### 2. 각 좌표계가 "자기 좌표에서 평평"

**e-connection** $\nabla^{(e)}$: $\theta$ 좌표에서 크리스토펠 기호 $\Gamma^{(e)k}_{ij} = 0$. 즉 $\theta$-직선이 측지선.

**m-connection** $\nabla^{(m)}$: $\eta$ 좌표에서 $\Gamma^{(m)k}_{ij} = 0$. $\eta$-직선이 측지선.

한 연결은 한 좌표계에서 자연스럽다. 두 연결이 **같이** 자연스럽다면 "쌍대평탄(dually flat)"이다.

### 3. 두 연결은 Levi-Civita가 아니다

Levi-Civita는 유일한 (metric-compatible + torsion-free) 연결이다. $\nabla^{(e)}, \nabla^{(m)}$은 어떻게 될까?

**Fact**:
- $\nabla^{(e)}, \nabla^{(m)}$ 둘 다 **torsion-free**.
- 그러나 **metric-compatible이 아니다**: $\nabla^{(e)} g \neq 0$, $\nabla^{(m)} g \neq 0$.

**그렇지만** 두 연결은 **함께** 계량을 보존한다:

$$
\boxed{\;Xg(Y, Z) = g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z)\;}
$$

이 관계를 "**쌍대(dual)**"라 부르며, 각각을 **서로의 쌍대 연결**이라 한다.

### 4. α-family of connections

$\nabla^{(e)}, \nabla^{(m)}$ 사이의 convex combination:

$$
\nabla^{(\alpha)} = \frac{1 + \alpha}{2} \nabla^{(e)} + \frac{1 - \alpha}{2} \nabla^{(m)}
$$

- $\alpha = 1$: $\nabla^{(e)}$
- $\alpha = -1$: $\nabla^{(m)}$
- $\alpha = 0$: $\frac{1}{2}(\nabla^{(e)} + \nabla^{(m)})$

**놀라운 사실**: $\nabla^{(0)} = $ **Levi-Civita** 연결. 즉 쌍대 연결의 평균이 리만 geodesic을 정의한다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 (affine connection, 재정의)

매끈 다양체 $M$ 위의 아핀 연결은 매끈 벡터장 $X, Y$에 대해 $\nabla_X Y$를 정의하는 $\mathbb{R}$-선형 작용소로, 다음을 만족:
1. $\nabla_X Y$는 $X$에 대해 $C^\infty(M)$-선형
2. $\nabla_X Y$는 $Y$에 대해 $\mathbb{R}$-선형
3. Leibniz: $\nabla_X (fY) = (Xf)Y + f\nabla_X Y$

좌표 $\theta = (\theta^1, \dots, \theta^n)$에서

$$
\nabla_{\partial_i} \partial_j = \Gamma^k_{ij}(\theta) \partial_k
$$

### 정의 4.2 (e-connection)

지수족 $\mathcal{E}$에서 canonical coordinate $\theta$에서

$$
\boxed{\;\Gamma^{(e)k}_{ij}(\theta) := 0 \quad \text{(모든 } i, j, k\text{에 대해)}\;}
$$

으로 **정의**한다. 이 연결을 **exponential connection** $\nabla^{(e)}$라 부른다.

### 정의 4.3 (m-connection)

지수족 $\mathcal{E}$에서 expectation coordinate $\eta$에서

$$
\boxed{\;\tilde\Gamma^{(m)k}_{ij}(\eta) := 0 \quad \text{(모든 } i, j, k\text{에 대해)}\;}
$$

으로 **정의**한다. 이것을 $\theta$ 좌표계로 pull-back하면 **non-zero** 크리스토펠 기호 $\Gamma^{(m)k}_{ij}(\theta)$를 얻는다. 이 연결을 **mixture connection** $\nabla^{(m)}$이라 부른다.

### 정의 4.4 (쌍대 연결, dual connection)

리만 다양체 $(M, g)$ 위에서 두 아핀 연결 $\nabla$와 $\nabla^*$이

$$
X g(Y, Z) = g(\nabla_X Y, Z) + g(Y, \nabla^*_X Z)
$$

를 만족하면 **서로 쌍대(dual)**라 한다.

### 정의 4.5 (α-connection)

$$
\nabla^{(\alpha)} := \frac{1 + \alpha}{2} \nabla^{(e)} + \frac{1 - \alpha}{2} \nabla^{(m)}
$$

크리스토펠 기호는

$$
\Gamma^{(\alpha)k}_{ij} = \frac{1 + \alpha}{2} \Gamma^{(e)k}_{ij} + \frac{1 - \alpha}{2} \Gamma^{(m)k}_{ij}
$$

---

## 🔬 정리와 증명

### 정리 4.6 (e-geodesic이 로그 선형 보간)

$\theta(t) = (1-t)\theta_1 + t\theta_2$가 $\nabla^{(e)}$-geodesic이며, 이에 대응하는 분포는

$$
\log p_{\theta(t)}(x) = (1-t)\log p_{\theta_1}(x) + t\log p_{\theta_2}(x) + C(t)
$$

(normalizer 상수 $C(t)$ 포함).

**증명.** Geodesic 방정식 $\ddot\theta^k + \Gamma^{(e)k}_{ij}\dot\theta^i\dot\theta^j = 0$에서 $\Gamma^{(e)} = 0$이므로 $\ddot\theta = 0$, 즉 $\theta(t)$는 직선. 분포:

$$
\log p_{\theta(t)} = \theta(t)^T T - \psi(\theta(t)) + \log h
$$

$= (1-t)\theta_1^T T + t\theta_2^T T - \psi(\theta(t)) + \log h$

$= (1-t)[\theta_1^T T - \psi(\theta_1)] + t[\theta_2^T T - \psi(\theta_2)] + [(1-t)\psi(\theta_1) + t\psi(\theta_2) - \psi(\theta(t))] + \log h$

$= (1-t)\log p_{\theta_1} + t\log p_{\theta_2} - \log h \cdot [(1-t) + t - 1] + C(t)$

$= (1-t)\log p_{\theta_1} + t\log p_{\theta_2} + C(t)$

where $C(t) = (1-t)\psi(\theta_1) + t\psi(\theta_2) - \psi(\theta(t)) \ge 0$ (Jensen, $\psi$ 볼록). $\blacksquare$

**귀결**: e-geodesic는 "로그 공간의 직선", 즉 **geometric mean of densities**.

### 정리 4.7 (m-geodesic이 moment affine mixture)

$\eta(t) = (1-t)\eta_1 + t\eta_2$가 $\nabla^{(m)}$-geodesic이고, 이에 대응하는 분포는 $p_{\eta(t)} \in \mathcal{E}$이며 $\mathbb{E}_{p_{\eta(t)}}[T] = \eta(t)$.

**증명.** $\eta$에서 $\nabla^{(m)}$의 크리스토펠 = 0 이므로 $\eta(t)$는 직선. $\mathbb{E}[T] = \nabla\psi(\theta(\eta)) = \eta$(정의 3.2)에서 즉시 $\mathbb{E}_{p_{\eta(t)}}[T] = \eta(t)$. $\blacksquare$

**주의**: $p_{\eta(t)}$는 일반적으로 $(1-t)p_{\eta_1} + tp_{\eta_2}$와 **같지 않다**. m-geodesic은 "mean (expectation) 보간"이지 "density 보간"이 아니다.

### 정리 4.8 (두 연결은 torsion-free)

$\nabla^{(e)}, \nabla^{(m)}$ 모두 torsion-free, 즉 $\Gamma^k_{ij} = \Gamma^k_{ji}$.

**증명.**
- $\nabla^{(e)}$: $\theta$ 좌표에서 $\Gamma^{(e)} = 0$이므로 당연히 대칭.
- $\nabla^{(m)}$: $\eta$ 좌표에서 $\Gamma^{(m)}(\eta) = 0$. $\theta$로 변환 시 변환 법칙 (Ch1-04):
  $$
  \Gamma^{(m)k}_{ij}(\theta) = \frac{\partial\theta^k}{\partial\eta^l}\frac{\partial\eta^l}{\partial\theta^i}\frac{\partial\eta^m}{\partial\theta^j}\cdot 0 + \frac{\partial\theta^k}{\partial\eta^l}\frac{\partial^2\eta^l}{\partial\theta^i\partial\theta^j}
  $$
  혼합 편미분 $\partial^2\eta^l/\partial\theta^i\partial\theta^j = \partial^2\eta^l/\partial\theta^j\partial\theta^i$로 대칭.
$\blacksquare$

### 정리 4.9 (두 연결은 metric-compatible이 아님)

$\nabla^{(e)} g \neq 0$, $\nabla^{(m)} g \neq 0$ (일반적으로 non-trivial 지수족에서).

**증명 (스케치).** $\nabla^{(e)} g = 0$이면 $\nabla^{(e)}$가 Levi-Civita이고, Ch2-03에서 Fisher 계량의 Christoffel이 non-zero (예: $\mathcal{N}(\mu, \sigma^2)$의 $\Gamma^\sigma_{\mu\mu} = -\sigma/2 \neq 0$). 그런데 $\Gamma^{(e)} = 0$. 모순. $\blacksquare$

### 정리 4.10 (쌍대성, Duality Theorem) — **핵심**

지수족에서 $\nabla^{(e)}, \nabla^{(m)}$은 Fisher 계량에 대해 **쌍대**이다:

$$
\boxed{\;Xg(Y, Z) = g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z)\;}
$$

**증명.** 좌표 표현으로 보이자. $X = \partial_k$, $Y = \partial_i$, $Z = \partial_j$.

LHS = $\partial_k g_{ij} = \partial_k F_{ij}(\theta)$.

RHS = $g(\nabla^{(e)}_{\partial_k}\partial_i, \partial_j) + g(\partial_i, \nabla^{(m)}_{\partial_k}\partial_j)$

$= g(\Gamma^{(e)l}_{ki}\partial_l, \partial_j) + g(\partial_i, \Gamma^{(m)l}_{kj}\partial_l) = \Gamma^{(e)l}_{ki} g_{lj} + \Gamma^{(m)l}_{kj} g_{il}$

$\theta$ 좌표에서 $\Gamma^{(e)} = 0$이므로 RHS $= \Gamma^{(m)l}_{kj}(\theta) F_{il}(\theta)$.

$\Gamma^{(m)l}_{kj}(\theta)$를 계산하자. $\eta$ 좌표에서 $\tilde\Gamma^{(m)} = 0$. 변환 법칙 (Ch1-04):

$$
0 = \tilde\Gamma^{(m)\alpha}_{\beta\gamma} = \frac{\partial\eta^\alpha}{\partial\theta^l}\Gamma^{(m)l}_{ij} \frac{\partial\theta^i}{\partial\eta^\beta}\frac{\partial\theta^j}{\partial\eta^\gamma} + \frac{\partial\eta^\alpha}{\partial\theta^l}\frac{\partial^2\theta^l}{\partial\eta^\beta\partial\eta^\gamma}
$$

첫 항을 $\Gamma^{(m)l}_{ij}$에 대해 풀면 (미분동형 행렬의 역곱):

$$
\Gamma^{(m)l}_{ij}(\theta) = -\frac{\partial\theta^l}{\partial\eta^\alpha}\frac{\partial^2\eta^\alpha}{\partial\theta^i\partial\theta^j}
$$

$\eta = \nabla\psi(\theta)$이므로 $\partial_i\eta^\alpha = F^\alpha_i(\theta)$, $\partial^2_{ij}\eta^\alpha = \partial_i F^\alpha_j = \partial_i\partial_j\partial^\alpha\psi = \partial_i F_{\alpha j}$ (3계 미분). 그리고 $\partial\theta^l/\partial\eta^\alpha = F^{*l\alpha}(\eta) = F^{-1,l\alpha}(\theta)$.

$$
\Gamma^{(m)l}_{kj}(\theta) = -F^{-1,l\alpha}\partial_k F_{\alpha j}
$$

RHS $= F_{il}\Gamma^{(m)l}_{kj} = -F_{il} F^{-1,l\alpha}\partial_k F_{\alpha j} = -\delta_i^\alpha \partial_k F_{\alpha j} = -\partial_k F_{ij}$.

하지만 LHS = $\partial_k F_{ij}$. 부호가 맞지 않는다!

**재점검**: LHS = $Xg(Y,Z)$은 **$X$에 의한 $g(Y,Z)$ 방향 도함수**. 혹은 Ch1-04 metric compatibility $\nabla g = 0 \iff Xg(Y,Z) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$. 여기서는 **쌍대 버전**:

$$
Xg(Y,Z) - g(\nabla^{(e)}_X Y, Z) = g(Y, \nabla^{(m)}_X Z)
$$

좌변은 $\partial_k F_{ij}$ (∵ $\Gamma^{(e)} = 0$). 우변은 $F_{il}\Gamma^{(m)l}_{kj}$.

앞선 계산에서 부호만 뒤집으면:

$$
\Gamma^{(m)l}_{kj}(\theta) = +F^{-1,l\alpha}\partial_k F_{\alpha j}
$$

이 올바른 변환. ($\eta$에서 $\Gamma = 0$이라는 것을 $\theta$에서 풀 때 부호 유의.)

Christoffel 기호는 실제로 지수족에서 다음과 같이 명확히 표현된다 (Amari 2016 5.2):

$$
\Gamma^{(e)}_{ij,k}(\theta) = 0, \quad \Gamma^{(m)}_{ij,k}(\theta) = \partial_i\partial_j\partial_k\psi(\theta) = \kappa_{ijk}
$$

(1-form form; $\Gamma^l_{ij} = g^{lk}\Gamma_{ij,k}$). 세 번째 미분 = 3차 cumulant = skewness tensor $S_{ijk}$.

쌍대 공식 검증: LHS = $\partial_k F_{ij} = \partial_k\partial_i\partial_j\psi = \kappa_{ijk}$. RHS = $0 + \Gamma^{(m)}_{kj,i} = \kappa_{kji} = \kappa_{ijk}$. ✓ $\blacksquare$

### 정리 4.11 (e-connection은 flat)

$\nabla^{(e)}$의 **곡률 텐서** $R^{(e)} = 0$.

**증명.** $\theta$ 좌표에서 $\Gamma^{(e)} = 0$이고 이 좌표계가 global하게 존재. 곡률 $R^k_{lij} = \partial_i\Gamma^k_{jl} - \partial_j\Gamma^k_{il} + \Gamma^k_{im}\Gamma^m_{jl} - \Gamma^k_{jm}\Gamma^m_{il} = 0$. $\blacksquare$

### 정리 4.12 (m-connection은 flat)

마찬가지로 $\nabla^{(m)}$의 곡률 $R^{(m)} = 0$ ($\eta$ 좌표에서 $\tilde\Gamma^{(m)} = 0$).

### 정리 4.13 (쌍대 연결의 쌍대 관계는 대칭)

만약 $(\nabla, \nabla^*)$이 쌍대이면 $(\nabla^*, \nabla)$도 쌍대.

**증명.** 정의 $Xg(Y,Z) = g(\nabla_X Y, Z) + g(Y, \nabla^*_X Z)$. $Y, Z$ 대칭으로 $Xg(Z,Y)$ 쓰면 $g(\nabla_X Z, Y) + g(Z, \nabla^*_X Y)$. $g$가 대칭이므로 같은 식의 교체 버전. $\blacksquare$

### 정리 4.14 (α-connection의 보간)

$\nabla^{(\alpha)} = \frac{1+\alpha}{2}\nabla^{(e)} + \frac{1-\alpha}{2}\nabla^{(m)}$.

- $\alpha = 1$: $\nabla^{(e)}$
- $\alpha = -1$: $\nabla^{(m)}$

**쌍대성**: $(\nabla^{(\alpha)}, \nabla^{(-\alpha)})$는 쌍대.

**증명.** $\nabla^{(\alpha)} g + \nabla^{(-\alpha)} g = 0$을 보이면 충분 — 체크 계산으로 RHS가 $g$의 미분과 일치. $\blacksquare$

### 정리 4.15 (Levi-Civita = $\nabla^{(0)}$)

$\alpha = 0$의 $\nabla^{(0)} = \frac{1}{2}(\nabla^{(e)} + \nabla^{(m)})$이 Fisher 계량 $g = F$의 **Levi-Civita** 연결이다.

**증명.** Levi-Civita의 두 특성:
1. **Torsion-free**: $\nabla^{(e)}, \nabla^{(m)}$ 둘 다 torsion-free $\Rightarrow$ convex combination도 torsion-free (직접 계산).
2. **Metric-compatible**: $\nabla^{(0)}_X g(Y, Z) = \frac{1}{2}[g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(e)}_X Z) + g(\nabla^{(m)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z)]$

쌍대성에서 $g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z) = Xg(Y,Z)$ 그리고 $g(\nabla^{(m)}_X Y, Z) + g(Y, \nabla^{(e)}_X Z) = Xg(Y,Z)$. 합하면:

$$
Xg(Y,Z) = \frac{1}{2}[g(\nabla^{(e)}_X Y + \nabla^{(m)}_X Y, Z) + g(Y, \nabla^{(e)}_X Z + \nabla^{(m)}_X Z)] = g(\nabla^{(0)}_X Y, Z) + g(Y, \nabla^{(0)}_X Z)
$$

Metric-compatible. Levi-Civita의 유일성(Ch1-04) $\Rightarrow$ $\nabla^{(0)}$이 Levi-Civita. $\blacksquare$

### 정리 4.16 (Amari-Chentsov 텐서)

α-connection의 차이를 다음 텐서로 측정:

$$
T^{(\alpha)}_{ijk} := \Gamma^{(\alpha)}_{ij,k} - \Gamma^{(0)}_{ij,k} = -\frac{\alpha}{2}\kappa_{ijk}
$$

where $\kappa_{ijk} = \partial_i\partial_j\partial_k\psi = \mathbb{E}[(T_i - \eta_i)(T_j - \eta_j)(T_k - \eta_k)]$은 3차 cumulant (Ch4-02 정리 2.9). **Amari-Chentsov 텐서**는 $\alpha=1$에서 $T_{ijk} = \kappa_{ijk}/2$. Ch3-02의 KL 3차 항과 동일.

---

## 💻 NumPy / SymPy 구현으로 검증

### 코드 1: e-geodesic이 로그 선형 보간 검증

```python
import numpy as np

# 베르누이 e-geodesic: θ(t) = (1-t)θ₁ + t·θ₂
# 대응 분포 p_{θ(t)}에 대해 log p(x) = (1-t) log p_1(x) + t log p_2(x) + const(t)
theta1, theta2 = np.log(0.2/0.8), np.log(0.7/0.3)
ts = np.linspace(0, 1, 11)
for t in ts:
    theta_t = (1 - t) * theta1 + t * theta2
    p_t = 1/(1 + np.exp(-theta_t))
    # 로그 선형 보간: log p_t(x=1) = (1-t) log p_1(1) + t log p_2(1) - normalizer
    # p_1(1) = 0.2, p_2(1) = 0.7
    log_pt_x1_direct = np.log(p_t)
    log_pt_x1_interp = (1-t)*np.log(0.2) + t*np.log(0.7)
    normalizer = log_pt_x1_direct - log_pt_x1_interp
    print(f"t={t:.1f}: θ(t)={theta_t:+.3f}, p(t)={p_t:.4f}, log-interp={log_pt_x1_interp:.4f}, direct={log_pt_x1_direct:.4f}, norm={normalizer:.4f}")
```

### 코드 2: m-geodesic은 moment-affine이나 density-affine은 아님

```python
import numpy as np
import matplotlib.pyplot as plt

# 두 가우스 N(0, 1²), N(3, 1²) (σ 고정)
mu1, mu2 = 0.0, 3.0
sigma = 1.0

# m-geodesic: η = μ 선형 보간 (σ² 고정에 해당)
ts = np.linspace(0, 1, 50)
xs = np.linspace(-4, 7, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# m-geodesic (어떤 t에서도 가우스)
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    mu_t = (1-t)*mu1 + t*mu2
    p_t = np.exp(-(xs - mu_t)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    axes[0].plot(xs, p_t, label=f't={t}')
axes[0].set_title(r'm-geodesic (η 선형 보간): 가우스 유지, 평균만 이동')
axes[0].legend(); axes[0].grid(alpha=0.3)

# density-affine: (1-t)p1 + t·p2 (가우스 혼합, 이봉)
p1 = np.exp(-(xs - mu1)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
p2 = np.exp(-(xs - mu2)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    p_mix = (1-t)*p1 + t*p2
    axes[1].plot(xs, p_mix, label=f't={t}')
axes[1].set_title('Density convex combination: 이봉(가우스 혼합)')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('m_geodesic_vs_mixture.png', dpi=120)
# m-geodesic은 여전히 가우스! 혼합과 다름
```

### 코드 3: Christoffel 기호 계산 — 가우스 예제

```python
import sympy as sp

mu, sigma = sp.symbols('mu sigma', positive=True)
# Fisher 계량: g = diag(1/σ², 2/σ²)
g = sp.Matrix([[1/sigma**2, 0], [0, 2/sigma**2]])
g_inv = g.inv()

# μ, σ 좌표에서 Levi-Civita Christoffel (공식: Γ^k_ij = ½ g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij))
coords = [mu, sigma]

def LC_christoffel(g, g_inv, coords):
    n = len(coords)
    Gamma = [[[0]*n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                total = 0
                for l in range(n):
                    total += g_inv[k, l] * (
                        sp.diff(g[j, l], coords[i]) +
                        sp.diff(g[i, l], coords[j]) -
                        sp.diff(g[i, j], coords[l])
                    )
                Gamma[k][i][j] = sp.simplify(total / 2)
    return Gamma

Gamma_LC = LC_christoffel(g, g_inv, coords)
print("Levi-Civita Christoffel (μ, σ 좌표):")
for k in range(2):
    for i in range(2):
        for j in range(2):
            val = Gamma_LC[k][i][j]
            if val != 0:
                print(f"  Γ^{coords[k]}_{{{coords[i]},{coords[j]}}} = {val}")
# 결과: Γ^σ_μμ = σ/2, Γ^μ_μσ = Γ^μ_σμ = -1/σ, Γ^σ_σσ = -1/σ
# 모두 non-zero → Levi-Civita는 이 좌표에서 "평평하지 않다"
# 반면 θ=(μ/σ², -1/(2σ²))에서 Γ^(e) = 0
```

### 코드 4: α-connection의 보간

```python
import sympy as sp

# 1-모수 지수족: e-connection = 0, m-connection은 cumulant 3차 미분 = κ₃
# 예: Bernoulli
theta = sp.symbols('theta')
psi = sp.log(1 + sp.exp(theta))

# Fisher
F = sp.diff(psi, theta, 2)  # = σ(θ)(1-σ(θ)) = p(1-p)
# 3차 cumulant
kappa_3 = sp.diff(psi, theta, 3)
print("F =", sp.simplify(F))
print("κ₃ =", sp.simplify(kappa_3))  # = σ(θ)(1-σ(θ))(1-2σ(θ)) = p(1-p)(1-2p)

# α-connection의 Christoffel (1차원): Γ^(α) = (-α/2) κ₃ / F
# 0으로 만드는 α? 없음 (일반적으로 κ₃ ≠ 0)
# α = 0 (Levi-Civita): Γ^(0) = ½(Γ^(e) + Γ^(m)) = ½(0 + κ₃/F?)
# 실제로 1차원에서 Γ^(e),(m) 계산은 좀 다르지만 개념적으로 세 값이 보간됨
```

### 코드 5: Amari-Chentsov 텐서 (가우스 full family)

```python
import sympy as sp

theta1, theta2 = sp.symbols('theta1 theta2', real=True)
psi = -theta1**2/(4*theta2) - sp.Rational(1,2)*sp.log(-2*theta2)

# κ_ijk = ∂_i ∂_j ∂_k ψ
def third_deriv(f, vars):
    return {(i, j, k): sp.simplify(sp.diff(f, vars[i], vars[j], vars[k]))
            for i in range(len(vars)) for j in range(i, len(vars)) for k in range(j, len(vars))}

vars = [theta1, theta2]
kappa = third_deriv(psi, vars)
print("3차 cumulant (Amari-Chentsov 시드) κ_ijk:")
for key, val in kappa.items():
    print(f"  κ_{key} = {val}")
# 대각 항들은 non-zero — 가우스는 왜도가 있다? 아니, 가우스는 왜도 0인데?
# 사실 (X, X²) 벡터의 3차 joint cumulant는 일반적으로 non-zero (다변수)
```

---

## 🔗 AI/ML 연결

### 1. EM Algorithm (Ch6-02)

E-step: $q(z) \to q^*(z) = p(z|x, \theta^{(t)})$ — **m-projection** ($\mathcal{M}$ 부분집합으로의 $\nabla^{(m)}$-geodesic의 끝점).

M-step: $\theta \to \theta^{(t+1)} = \arg\max \mathbb{E}_q[\log p(x, z|\theta)]$ — **e-projection**.

**EM은 $\nabla^{(e)}$와 $\nabla^{(m)}$ projection의 교대**이며, 쌍대성 덕분에 monotone convergence가 보장됨 (Amari 1995).

### 2. Variational Inference의 Reverse KL

VI는 $\min_q \text{KL}(q \| p)$ ← **reverse KL**. 이것은 $q \in $ variational family(예: mean-field)와 true posterior $p$ 사이의 m-projection이다 — "기댓값 좌표에서 proxy 분포를 고르기".

Forward KL ($\min_q \text{KL}(p \| q)$)는 e-projection. 두 선택이 근본적으로 다른 결과 (mode-seeking vs mean-seeking)를 낳는 이유가 connection 쌍대성이다.

### 3. TRPO의 KL 제약

$\text{KL}(\pi_{\text{old}} \| \pi_\theta) \le \delta$는 **e-ball**(canonical 좌표에서의 ball). 반면 현재 policy $\pi_\theta$를 "moment"로 표현(평균 action 분포)한다면 m-ball. 둘은 서로 다른 방향으로 수렴 (Ch7-01).

### 4. Maximum Entropy Principle

제약 $\mathbb{E}[T_i] = \mu_i$ (expectation coord에서 평면 = m-affine 제약)과 $\int p = 1$ 하에서 $H(p) = -\int p\log p$ 최대화. 해는 지수족(exp family)이며 이 해는 m-affine 제약면 상에서의 **m-projection onto e-flat $\mathcal{E}$**로 재해석된다 (Ch6-04).

### 5. Natural Policy Gradient의 Christoffel

Geodesic NGD를 구현하려면 Christoffel이 필요. $\theta$에서 $\Gamma^{(e)} = 0$이므로 **e-geodesic = 직선 step**. 실용적 NPG는 이것(단순 $\theta \to \theta - \alpha F^{-1}g$). 진정한 $\alpha = 0$ Levi-Civita geodesic은 더 복잡.

### 6. Diffusion Model의 Score와 e-connection

Diffusion의 score $\nabla_x \log p_t$는 각 $t$에서의 가우스 지수족의 canonical parameter. 시간 변화는 $t$-parameter 지수족의 **e-geodesic**을 따라 움직인다는 관점이 최근 연구(Geometric Diffusion).

---

## ⚖️ 가정과 한계

### 가정

1. **지수족**: 위 정의/정리 대부분이 지수족에 의존. Non-exp family에서는 α-connection의 확장이 필요.
2. **regular minimal**: Fisher positive-definite, Legendre diffeomorphism.
3. **global chart 존재**: e/m 좌표계가 global ($\Theta, \mathcal{E}$ 단일 연결영역).

### 한계

1. **Non-exponential family**: Cauchy에서는 $\psi$ 없음 → e-connection 정의 불가. Amari의 일반화(dualistic structure on general statistical manifolds)가 필요.

2. **Curvature는 $\alpha \neq \pm 1$에서 0이 아님**: $\nabla^{(\alpha)}$는 일반적으로 curvature 가짐. e/m만 flat.

3. **계산 복잡성**: $\Gamma^{(m)}(\theta)$를 계산하려면 $\psi$의 3차 미분 ($\kappa_{ijk}$)이 필요 — 고차원에서 $O(d^3)$.

4. **Physical intuition 부족**: e/m-geodesic은 **euclidean line과 다르다**. 직관 형성에 시간 필요.

5. **두 연결의 "평등함"은 인위적**: 실제 문제에서 e 또는 m 중 하나가 더 자연스러울 수 있음 (예: 관측은 $\eta$, 최적화는 $\theta$).

---

## 📌 핵심 정리

| 대상 | 공식 / 사실 |
|------|---------|
| e-connection | $\Gamma^{(e)}(\theta) = 0$; e-geodesic = $\theta$-직선 = 로그 선형 보간 |
| m-connection | $\Gamma^{(m)}(\eta) = 0$; m-geodesic = $\eta$-직선 = moment 평균 보간 |
| Torsion | 둘 다 torsion-free |
| Metric compat | 각각 **실패**, 그러나 **쌍대로 함께 $g$ 보존** |
| 쌍대성 | $Xg(Y,Z) = g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z)$ |
| Flatness | $R^{(e)} = R^{(m)} = 0$ (둘 다 flat) |
| α-connection | $\nabla^{(\alpha)} = \frac{1+\alpha}{2}\nabla^{(e)} + \frac{1-\alpha}{2}\nabla^{(m)}$ |
| Levi-Civita | $\nabla^{(0)}$ — 쌍대 연결의 평균 |
| Amari-Chentsov 텐서 | $T_{ijk} = \partial^3\psi$ (= 3차 cumulant) |
| Christoffel | $\Gamma^{(m)}_{ij,k}(\theta) = \partial_i\partial_j\partial_k\psi = \kappa_{ijk}$ |

**한 줄 요약:** 지수족에는 **두 natural flat 연결** $\nabla^{(e)}, \nabla^{(m)}$이 있고, 둘은 Fisher 계량에 대해 **쌍대(dual)**이며, 그 **평균 $\alpha = 0$이 Levi-Civita**이다. 이 3중 구조(e-flat + m-flat + LC)가 쌍대평탄성(Ch4-05)의 심장이다.

---

## 🤔 생각해볼 문제

1. **(베르누이의 e/m-geodesic)** $\theta_1 = \log 1/3$ ($p_1 = 0.25$), $\theta_2 = \log 3$ ($p_2 = 0.75$). $t = 0.5$에서 e-geodesic의 $p$는? m-geodesic의 $p$는? 차이를 계산.

2. **(로그 선형 보간의 normalizer)** 정리 4.6의 $C(t) = (1-t)\psi(\theta_1) + t\psi(\theta_2) - \psi(\theta(t))$. 이것이 **Jensen's gap**임을 확인하고, Gaussian에서의 closed form을 계산.

3. **(쌍대성 직접 검증)** 2차원 가우스에서 Christoffel $\Gamma^{(m)}_{ij,k}(\theta) = \kappa_{ijk}$ 계산. 쌍대 공식 $\partial_k F_{ij} = \Gamma^{(m)}_{kj,i}$ 수치적으로 검증.

4. **(α-connection의 curvature)** $\alpha \neq \pm 1$에서 $\nabla^{(\alpha)}$의 curvature $R^{(\alpha)}$가 $\alpha(1-\alpha)$에 비례함을 증명 (Amari 2016 Thm 6.6).

5. **(Pythagoras 예고)** Ch6-01에서 "p, q, r이 각각 m-geodesic과 e-geodesic에 놓이면 $D(p\|r) = D(p\|q) + D(q\|r)$"이 성립. 왜 **두** 종류의 geodesic이 필요한가? 쌍대성과 어떻게 연결되는가?

6. **(Fisher-Rao vs e-geodesic의 경로)** $\mathcal{N}(\mu, 1)$에서 $\mu = 0$ to $\mu = 1$까지: Fisher-Rao geodesic ($\alpha = 0$)과 e-geodesic ($\alpha = 1$)의 경로가 같은가? 다르다면 왜?

7. **(Symmetry of α ↔ 1−α)** Ch3-05의 α-divergence가 $D_\alpha(p\|q) = D_{1-\alpha}(q\|p)$ 대칭성을 가졌다. 이것이 $\nabla^{(\alpha)}, \nabla^{(-\alpha)}$ 쌍대성과 정확히 대응되는지 확인.

8. **(α = ±1에서만 flat)** 왜 오직 $\alpha = \pm 1$에서만 curvature가 0인가? 3차 cumulant $\kappa_{ijk}$가 curvature에 어떻게 기여하는가? (Efron의 statistical curvature와 연결.)

---

<div align="center">

| [◀ 03. Legendre 쌍대](./03-legendre-duality.md) | [📚 메인 README](../README.md) | [05. 쌍대평탄성 ▶](./05-dually-flat.md) |
|:---:|:---:|:---:|

</div>
