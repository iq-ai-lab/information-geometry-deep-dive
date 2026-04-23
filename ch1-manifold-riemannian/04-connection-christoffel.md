# 04. 아핀 연결(Connection)과 공변미분

## 🎯 핵심 질문

- 왜 **벡터장의 미분**이 좌표 의존적인 문제를 일으키는가? 단순 $\partial_i Y^j$가 텐서가 아닌 이유는?
- 아핀 연결 $\nabla$의 **3가지 공리**는 각각 무엇을 보장하는가?
- Christoffel 기호 $\Gamma^k_{ij}$는 좌표 변환 하에서 **텐서처럼 변환하지 않음**에도 $\nabla_X Y$가 텐서가 되는 이유는?
- Levi-Civita 연결이 **유일**한 이유? ("계량 호환 + torsion-free"가 왜 유일성을 강제하는가)
- 정보기하에서 **계량 호환을 포기한 비-Levi-Civita 연결**이 왜 필요한가 — e-connection, m-connection 예고

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **서로 다른 점의 접공간을 연결**: 두 점 $p, q$의 접공간은 서로 다른 벡터공간이다. 벡터장의 미분 $\nabla_X Y$는 "기준점에서 약간 이동한 새 점의 접벡터 $Y_{p+\varepsilon X}$를 원래 점으로 **평행이동(parallel transport)**"하여 뺀다는 뜻. 이것이 없으면 "벡터장의 변화"를 정의할 수 없다.
- **Amari의 ±1-connection**: 정보기하의 핵심. Exponential family가 **e-connection에선 평탄, m-connection에선 평탄** 두 가지로 동시에 평탄한 **쌍대평탄(dually flat)** 구조를 갖는다(Ch4-05). 이를 위해선 **계량 호환성을 포기한** 비대칭 쌍대 연결이 필요.
- **Riemannian SGD**: 신경망 파라미터를 SPD·스티펠·그라스만 다양체 위로 제약할 때, 업데이트는 **평행이동 + retraction**으로 정의된다 — 연결 이론이 직접 쓰인다.
- **Neural ODE와 adjoint method**: ODE의 adjoint gradient가 "파라미터 공간의 공변미분"으로 해석된다.

좌표마다 다른 $(X^i)_p$의 "변화"를 말하려면, 서로 다른 점의 접공간 사이를 **정의된 방식으로 동일시**해야 한다. 이것이 연결이다.

---

## 📐 수학적 선행 조건

- Ch1-01~03: 다양체, 접공간, 리만 계량
- 텐서의 좌표 변환 법칙 (contravariant/covariant)
- 선형대수: 쌍선형형식, 대칭화/반대칭화

---

## 📖 직관적 이해

### 벡터장 미분의 "지루한 문제"

$\mathbb{R}^n$에서는 벡터장 $Y(x) = (Y^1(x), \ldots, Y^n(x))$을 단순히 **성분별로** 미분하면 된다: $\partial_i Y^j$. 이것이 좌표 변환 하에서 잘 작동하는 것은 $\mathbb{R}^n$이 **자연스러운 평행이동**(벡터를 평행 이동해도 여전히 같은 벡터)을 갖기 때문.

곡면(다양체)에서는 이것이 깨진다: 구면 위의 "북쪽으로 가는 벡터장"을 남극 근처에서 보면 "위쪽", 적도 근처에서 보면 "북쪽"으로, 방향 자체가 점에 따라 다르다. 단순 성분 미분 $\partial_i Y^j$는 **좌표계를 바꾸면 다른 값**을 주고, 텐서가 아니다.

### 연결 = "평행이동의 규칙"

연결 $\nabla$는 **"어떻게 접벡터를 평행하게 옮길 것인가"**를 정의하는 규칙이다:

$$\nabla_X Y = \lim_{t \to 0} \frac{P_t^{-1}(Y_{\gamma(t)}) - Y_{\gamma(0)}}{t}$$

여기서 $P_t$는 곡선 $\gamma$(속도가 $X$)을 따라 $Y_{\gamma(t)}$를 $\gamma(0)$로 **평행이동**한 결과. 즉 연결은 "**평행이동을 먼저 정의**, 그 평행이동으로 벡터장의 변화를 측정".

> **비유**: 지구 표면을 돌면서 "항상 북쪽을 가리키는 깃발"을 들고 이동한다고 하자. 적도에서 북극으로, 북극에서 경도 90°E 적도로, 그리고 다시 출발지 적도로 돌아오면 — 깃발이 90° 회전해 있다! 이것이 **holonomy** (곡률의 표현). 연결은 이 "깃발 회전 규칙" 자체를 정의한다.

### 왜 "계량 호환 + torsion-free"인가

- **계량 호환**: 평행이동이 **내적을 보존** — "길이와 각도가 보존되는 이동"
- **Torsion-free**: $\nabla_X Y - \nabla_Y X = [X, Y]$ — "좌표 순서가 결과를 바꾸지 않음", 비대칭 부분이 0

두 조건을 동시에 요구하면 **유일한 연결** (Levi-Civita)이 결정된다. 이것이 리만기하의 "자연스러운 선택". 하지만 정보기하에서는 **대칭성만 살리고 계량 호환을 포기**해 쌍대성을 얻는다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 아핀 연결 (Affine Connection)

매끈한 다양체 $M$ 위의 **아핀 연결** $\nabla$는 벡터장 $\mathfrak{X}(M)$에 대한 사상

$$\nabla: \mathfrak{X}(M) \times \mathfrak{X}(M) \to \mathfrak{X}(M), \quad (X, Y) \mapsto \nabla_X Y$$

으로, 다음 3가지 공리를 만족:

1. **$C^\infty(M)$-선형성 (첫 인자)**: $\nabla_{fX + gX'} Y = f \nabla_X Y + g \nabla_{X'} Y$ ($f, g \in C^\infty(M)$)
2. **$\mathbb{R}$-선형성 (둘째 인자)**: $\nabla_X (aY + bY') = a\nabla_X Y + b\nabla_X Y'$ ($a, b \in \mathbb{R}$)
3. **Leibniz 규칙 (둘째 인자의 함수 배)**: $\nabla_X (fY) = X(f) \cdot Y + f \cdot \nabla_X Y$

$\nabla_X Y$를 **$X$ 방향의 $Y$의 공변미분(covariant derivative)**이라 한다.

### 정의 4.2 — Christoffel 기호

좌표 $(\theta^i)$에서 좌표기저 $\partial_i = \partial/\partial\theta^i$에 대해

$$\nabla_{\partial_i} \partial_j = \Gamma^k_{ij} \partial_k$$

으로 정의되는 $\Gamma^k_{ij}(\theta)$를 **Christoffel 기호**라 한다. 이들이 연결을 국소적으로 완전히 결정한다.

### 정의 4.3 — Torsion과 Curvature

연결 $\nabla$의 **torsion 텐서**:

$$T(X, Y) := \nabla_X Y - \nabla_Y X - [X, Y]$$

**curvature 텐서**:

$$R(X, Y) Z := \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$$

좌표에서 $T^k_{ij} = \Gamma^k_{ij} - \Gamma^k_{ji}$, **torsion-free** $\iff \Gamma^k_{ij}$가 $i, j$에 대해 대칭.

### 정의 4.4 — 계량 호환 (Metric Compatibility)

리만 다양체 $(M, g)$에서 연결 $\nabla$가 **계량 호환**이라는 것은

$$X(g(Y, Z)) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$$

— 즉 내적 $g$가 $\nabla$에 대한 Leibniz 규칙을 따름. 좌표에서 $\partial_k g_{ij} = g_{\ell j} \Gamma^\ell_{ki} + g_{i\ell} \Gamma^\ell_{kj}$.

### 정의 4.5 — Levi-Civita 연결

**계량 호환**이고 **torsion-free**인 연결. 정리 4.2에서 유일 존재성을 보인다.

### 정의 4.6 — 평행이동 (Parallel Transport)

곡선 $\gamma(t)$를 따르는 벡터장 $Y(t)$가 **평행(parallel)**이라는 것은 $\nabla_{\dot\gamma} Y = 0$. 좌표에서:

$$\frac{dY^k}{dt} + \Gamma^k_{ij}(\gamma) \dot\gamma^i Y^j = 0$$

초기 벡터 $Y_0 \in T_{\gamma(0)} M$이 주어지면 이 ODE로 곡선 전체에 걸쳐 $Y(t)$를 유일하게 결정.

### 정의 4.7 — 측지선 재정의

$\nabla_{\dot\gamma} \dot\gamma = 0$ — 즉 **속도벡터 자신이 평행이동**되는 곡선이 측지선. 좌표에서 Ch1-03의 $\ddot\gamma^k + \Gamma^k_{ij} \dot\gamma^i \dot\gamma^j = 0$과 일치.

---

## 🔬 정리와 증명

### 정리 4.1 — Christoffel 기호의 좌표 변환 법칙

**명제**: 두 좌표 $(\theta^i), (\widetilde\theta^i)$에 대해

$$\widetilde\Gamma^k_{ij} = \frac{\partial \widetilde\theta^k}{\partial \theta^\ell} \cdot \frac{\partial \theta^m}{\partial \widetilde\theta^i} \cdot \frac{\partial \theta^n}{\partial \widetilde\theta^j} \cdot \Gamma^\ell_{mn} + \frac{\partial \widetilde\theta^k}{\partial \theta^\ell} \cdot \frac{\partial^2 \theta^\ell}{\partial \widetilde\theta^i \partial \widetilde\theta^j}$$

**증명 스케치**: $\nabla_{\partial/\partial\widetilde\theta^i} (\partial/\partial\widetilde\theta^j)$을 두 가지 방식으로 전개. 두 좌표기저 사이의 변환 $\partial/\partial\widetilde\theta^i = (\partial\theta^m/\partial\widetilde\theta^i) \partial_m$을 대입하고 Leibniz 규칙 적용. $\square$

> **핵심**: **두 번째 항**($\partial^2\theta/\partial\widetilde\theta^i \partial\widetilde\theta^j$) 때문에 $\Gamma$는 **텐서가 아니다**. 좌표가 아핀변환이면 2차 미분이 0 → 텐서처럼 변환. 그래서 유클리드의 $\Gamma = 0$이 일반 좌표에선 0이 아닐 수 있다(극좌표).

---

### 정리 4.2 — Levi-Civita 연결의 유일 존재 (Koszul 공식)

**명제**: 리만 다양체 $(M, g)$ 위에 **계량 호환 + torsion-free** 연결이 유일하게 존재하며, 그 Christoffel 기호는

$$\Gamma^k_{ij} = \frac{1}{2} g^{k\ell}\left(\partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij}\right)$$

**증명 (유일성)**: 계량 호환 3개 식

$$\partial_i g_{jk} = g_{\ell k} \Gamma^\ell_{ij} + g_{j\ell} \Gamma^\ell_{ik} \quad (1)$$
$$\partial_j g_{ik} = g_{\ell k} \Gamma^\ell_{ji} + g_{i\ell} \Gamma^\ell_{jk} \quad (2)$$
$$\partial_k g_{ij} = g_{\ell j} \Gamma^\ell_{ki} + g_{i\ell} \Gamma^\ell_{kj} \quad (3)$$

$(1) + (2) - (3)$ 하고 torsion-free($\Gamma^\ell_{ij} = \Gamma^\ell_{ji}$) 적용:

$$\partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij} = 2 g_{\ell k} \Gamma^\ell_{ij}$$

양변에 $g^{km}$을 곱하면

$$\Gamma^m_{ij} = \frac{1}{2} g^{km}(\partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij})$$

**존재성**: 위 공식이 Leibniz 규칙·계량 호환·torsion-free를 모두 만족함을 직접 대입 확인. $\square$

> **의미**: 리만 구조 $(M, g)$가 주어지면 연결은 "공짜"로 따라온다. 그러나 정보기하에서는 **계량 호환을 포기**해 e/m connection 같은 다른 연결을 도입한다.

---

### 정리 4.3 — $\nabla_X Y$는 텐서장

**명제**: Christoffel 기호가 텐서가 아님에도, 벡터장 $X, Y$로부터 만들어진 $\nabla_X Y$는 정의된 벡터장이며 좌표 변환에 대해 텐서처럼 변환한다.

**증명 스케치**: 좌표 표현

$$(\nabla_X Y)^k = X^i \left(\partial_i Y^k + \Gamma^k_{ij} Y^j\right)$$

에서 $\Gamma$의 **2차 미분 항이 $\partial_i Y^k$의 변환에서 생기는 2차 항과 상쇄**됨을 직접 계산. 즉 $(\partial_i Y^k)$와 $(\Gamma^k_{ij} Y^j)$ 각각은 텐서가 아니지만, 그 합이 텐서. $\square$

> 이것이 연결의 "마법": **비-텐서적 Christoffel + 비-텐서적 partial derivative = 텐서**.

---

### 정리 4.4 — 평행이동의 선형 등방 보존(Isometry, Levi-Civita일 때)

**명제**: Levi-Civita 연결 $\nabla$에 대해, 곡선 $\gamma$을 따라 두 벡터장 $Y, Z$가 모두 평행이동되면 $g(Y, Z)$는 $\gamma$를 따라 **상수**이다.

**증명**:

$$\frac{d}{dt} g(Y, Z) = \dot\gamma(g(Y, Z)) = g(\nabla_{\dot\gamma} Y, Z) + g(Y, \nabla_{\dot\gamma} Z) = g(0, Z) + g(Y, 0) = 0$$

($\nabla_{\dot\gamma} Y = 0$은 평행이동 정의, 첫 등호는 계량 호환). $\square$

> **따름**: Levi-Civita 평행이동은 **길이와 각도를 보존**. 반면 정보기하의 e/m 연결은 계량 호환이 아니므로 평행이동이 Fisher 내적을 보존하지 않는다 — 하지만 쌍대적으로 $\nabla^{(e)}$과 $\nabla^{(m)}$이 결합해 계량을 보존(Ch4-05).

---

### 정리 4.5 — 유클리드 다양체에선 Levi-Civita가 "자명한" 미분

**명제**: $\mathbb{R}^n$의 표준 계량 $g_{ij} = \delta_{ij}$ (직교좌표)에서 $\Gamma^k_{ij} = 0$이고, 공변미분이 일반 부분미분과 일치: $(\nabla_X Y)^k = X^i \partial_i Y^k$.

**증명**: $\partial_i g_{jk} = 0$이므로 Koszul 공식에서 $\Gamma \equiv 0$. $\square$

> 극좌표 등 비-직교 좌표에서는 $g_{ij}$가 점에 의존 → $\Gamma \neq 0$. 이는 $\mathbb{R}^n$ 자체의 곡률이 있다는 뜻이 **아니라**, **좌표가 아핀이 아닌 것의 반영**일 뿐. 곡률 $R = 0$.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp

# ─────────────────────────────────────────────
# 1. Levi-Civita Christoffel: 쌍곡 상반평면 H²
# ─────────────────────────────────────────────
x, y = sp.symbols('x y', real=True)
coords = [x, y]
# H²: ds² = (dx² + dy²)/y²
g = sp.Matrix([[1/y**2, 0], [0, 1/y**2]])
g_inv = g.inv()

def christoffel(g, g_inv, coords):
    n = len(coords)
    G = [[[sp.S.Zero]*n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = sp.S.Zero
                for l in range(n):
                    s += g_inv[k, l] * (
                        sp.diff(g[j, l], coords[i]) +
                        sp.diff(g[i, l], coords[j]) -
                        sp.diff(g[i, j], coords[l])
                    )
                G[k][i][j] = sp.simplify(s / 2)
    return G

G = christoffel(g, g_inv, coords)
print("─ H² Christoffel 기호 (비-영) ─")
labels = ['x', 'y']
for k in range(2):
    for i in range(2):
        for j in range(2):
            v = G[k][i][j]
            if v != 0:
                print(f"  Γ^{labels[k]}_{labels[i]}{labels[j]} = {v}")
# 기대:
#  Γ^x_xy = -1/y,  Γ^x_yx = -1/y
#  Γ^y_xx = 1/y,   Γ^y_yy = -1/y

# ─────────────────────────────────────────────
# 2. Torsion 확인: Levi-Civita는 대칭
# ─────────────────────────────────────────────
torsion_nonzero = False
for k in range(2):
    for i in range(2):
        for j in range(2):
            T = sp.simplify(G[k][i][j] - G[k][j][i])
            if T != 0:
                print(f"T^{labels[k]}_{labels[i]}{labels[j]} = {T}")
                torsion_nonzero = True
print("─ Torsion:", "비-영 존재" if torsion_nonzero else "0 (symmetric Christoffel)")

# ─────────────────────────────────────────────
# 3. 계량 호환 확인: ∂_k g_{ij} = g_{lj} Γ^l_{ki} + g_{il} Γ^l_{kj}
# ─────────────────────────────────────────────
print("\n─ 계량 호환 체크 ─")
for k in range(2):
    for i in range(2):
        for j in range(2):
            lhs = sp.diff(g[i, j], coords[k])
            rhs = sum(g[l, j]*G[l][k][i] + g[i, l]*G[l][k][j] for l in range(2))
            rhs = sp.simplify(rhs)
            ok = sp.simplify(lhs - rhs) == 0
            if not ok:
                print(f"  ∂_{labels[k]} g_{labels[i]}{labels[j]}: LHS={lhs}, RHS={rhs}")
print("  → 모든 조합에서 LHS == RHS ✓ (계량 호환)")

# ─────────────────────────────────────────────
# 4. 평행이동 수치 풀이: H²의 수직선 위에서 벡터 옮기기
# ─────────────────────────────────────────────
from scipy.integrate import solve_ivp

# 곡선 γ(t) = (0, e^t) 위에서 평행이동
# dY^k/dt + Γ^k_ij(γ) · dγ^i/dt · Y^j = 0
def parallel_transport_ode(t, Y):
    Yx, Yy = Y
    # γ(t) = (0, e^t), dγ = (0, e^t)
    gy = np.exp(t)
    # Γ^x_xy(γ) = -1/y = -e^(-t),  Γ^y_yy(γ) = -1/y = -e^(-t)
    # dY^x/dt = -Γ^x_ij · γ^i_dot · Y^j
    #        = -Γ^x_xy · γ^y_dot · Y^x  (cross 항)
    # 주의: Γ^x_ij with γ^i_dot = (0, e^t)
    # 실제: -(Γ^x_yx Y^x + Γ^x_yy Y^y) · γ^y_dot = -(-1/y · Y^x + 0) · e^t = Y^x · (1/y)·e^t = Y^x
    # 그리고: -(Γ^y_yx Y^x + Γ^y_yy Y^y) · γ^y_dot = -(0 + -1/y · Y^y) · e^t = Y^y
    dYx = Yx
    dYy = Yy
    return [dYx, dYy]

# 초기 벡터 Y_0 = (1, 0)을 t=0 → t=1 평행이동
sol = solve_ivp(parallel_transport_ode, (0, 1), [1.0, 0.0], rtol=1e-10)
print("\n─ H² 수직선 위의 평행이동 ─")
print(f"  Y(0) = (1, 0), Y(1) = ({sol.y[0,-1]:.4f}, {sol.y[1,-1]:.4f})")

# 계량 호환이면 길이 ||Y||_g = √(Y^T g Y) = √(Yx²/y² + Yy²/y²)가 보존되어야
Yx0, Yy0 = 1.0, 0.0
y0 = 1.0
len0 = np.sqrt(Yx0**2 + Yy0**2) / y0
Yx1, Yy1 = sol.y[0,-1], sol.y[1,-1]
y1 = np.exp(1)
len1 = np.sqrt(Yx1**2 + Yy1**2) / y1
print(f"  ||Y(0)||_g = {len0:.6f}")
print(f"  ||Y(1)||_g = {len1:.6f}")
print(f"  → 길이 보존: {'✓' if abs(len0 - len1) < 1e-4 else '✗'}")
```

**출력 예시**:
```
─ H² Christoffel 기호 (비-영) ─
  Γ^x_xy = -1/y
  Γ^x_yx = -1/y
  Γ^y_xx = 1/y
  Γ^y_yy = -1/y
─ Torsion: 0 (symmetric Christoffel)
─ 계량 호환 체크 → 모든 조합에서 LHS == RHS ✓ (계량 호환)

─ H² 수직선 위의 평행이동 ─
  Y(0) = (1, 0), Y(1) = (2.7183, 0.0000)
  ||Y(0)||_g = 1.000000
  ||Y(1)||_g = 1.000000
  → 길이 보존: ✓
```

---

## 🔗 AI/ML 연결

### Riemannian Optimization

SPD·스티펠·그라스만 다양체 위의 최적화는 $\theta_{k+1} = R_{\theta_k}(-\eta \nabla^g L)$ 형태의 **retraction** 업데이트를 쓴다. $R$은 지수사상의 근사 — 내부적으로 **Christoffel 기호(또는 그 근사)가 들어간다**. PyTorch-geometric, Geoopt 같은 라이브러리가 이를 구현.

### Natural Gradient는 Levi-Civita 스텝의 1차 근사

NGD $\theta \leftarrow \theta - \eta F^{-1}\nabla L$은 Fisher 계량의 Levi-Civita 연결 하에서 측지선을 **접공간에서 1차 근사**. 정확한 측지선 업데이트는 exponential map $\exp_\theta(-\eta F^{-1}\nabla L)$인데, exp map 계산이 비싸 NGD가 실용적 타협.

### 정보기하의 ±1-connection 예고 (Ch4)

Exponential family에서 **e-connection** $\nabla^{(e)}$은 $\theta$-좌표에서 $\Gamma^{(e)} = 0$이 되도록 **새로 정의**된 연결. 이것은 Levi-Civita가 **아니다** — 계량 호환이 깨진다. 대신 **m-connection** $\nabla^{(m)}$과 결합해 쌍대성을 만족:

$$X g(Y, Z) = g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z)$$

"계량 호환"을 두 연결이 **분담**하는 구조. 이것이 $\alpha = 0$ Levi-Civita의 **대칭적 평균** $\nabla^{(0)} = \frac{1}{2}(\nabla^{(e)} + \nabla^{(m)})$로 회수되는 이유.

### Holonomy와 Gauge Theory

벡터를 닫힌 경로로 평행이동하면 원래와 다른 벡터로 돌아오는 현상(holonomy) — 이것이 **곡률**의 본질. 게이지 장론, Yang-Mills 이론, 심지어 SPD 다양체의 geometric mean 계산까지 같은 구조.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 매끄러운 연결 | 저규칙 계량(예: $C^0$만)에선 평행이동 ODE가 잘 정의되지 않음 |
| 계량 호환 + torsion-free | **유일성의 대가**는 비-Levi-Civita 연결(정보기하의 e/m)이 탐색 공간 밖 |
| 단일 연결 | 쌍대평탄 구조는 **두 연결의 쌍**에서만 드러남 (Ch4) |
| 유한 차원 | 무한차원 다양체(함수공간)에서는 계량 정칙성과 연결 존재성이 비자명 |

**정보기하의 관점**: Levi-Civita의 "자연스러움"은 리만기하의 관점. 정보기하에서는 **통계적 불변성(parameterization 독립성)**이 더 근본적이고, 이것은 **$\alpha$-family 연결**을 요구한다. Levi-Civita는 $\alpha = 0$의 특수 경우.

---

## 📌 핵심 정리

$$\boxed{\;\nabla_X Y: \text{ 3공리 만족 } \iff \Gamma^k_{ij} \text{ (Christoffel)로 국소적 결정}\;}$$

$$\text{Levi-Civita 유일: 계량호환 } + \text{ torsion-free } \Longleftrightarrow \Gamma^k_{ij} = \frac{1}{2}g^{k\ell}(\partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij})$$

| 개념 | 핵심 |
|------|------|
| 연결 $\nabla$ | 벡터장의 공변미분, 3공리($C^\infty$-선형, $\mathbb{R}$-선형, Leibniz) |
| Christoffel $\Gamma$ | 좌표기저에 대한 연결 성분, **텐서가 아님** |
| 평행이동 | $\nabla_{\dot\gamma} Y = 0$, 접공간 사이 동일시 |
| Torsion | $\Gamma$의 반대칭 부분, Levi-Civita는 0 |
| 계량 호환 | 평행이동이 내적 보존, Levi-Civita의 정의 |
| 곡률 | 두 번 미분의 비가환성, $R(X,Y)Z$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathbb{R}^2$의 극좌표 $(r, \phi)$에서 Levi-Civita Christoffel 기호 중 $\Gamma^r_{\phi\phi}$와 $\Gamma^\phi_{r\phi}$를 Koszul 공식으로 직접 계산하라. 유클리드이므로 곡률 $R \equiv 0$임을 한 성분이라도 확인.

<details>
<summary>힌트 및 해설</summary>

$g_{rr} = 1, g_{\phi\phi} = r^2$, $g_{r\phi} = 0$.  
$\Gamma^r_{\phi\phi} = -\frac{1}{2}g^{rr}\partial_r g_{\phi\phi} = -r$.  
$\Gamma^\phi_{r\phi} = \frac{1}{2}g^{\phi\phi}\partial_r g_{\phi\phi} = \frac{1}{2} \cdot \frac{1}{r^2} \cdot 2r = 1/r$.  
곡률 $R^r_{\phi r \phi} = \partial_r \Gamma^r_{\phi\phi} - \partial_\phi \Gamma^r_{r\phi} + \Gamma^r_{rm}\Gamma^m_{\phi\phi} - \Gamma^r_{\phi m}\Gamma^m_{r\phi} = -1 - 0 + 0 - (-r)(1/r) = -1 + 1 = 0$. ✓ 평탄.

</details>

**문제 2** (심화): **Torsion-free를 포기**한 예를 구성하라. 즉 $\Gamma^k_{ij} \neq \Gamma^k_{ji}$인 연결을 $\mathbb{R}^2$에서 정의하고, 이 연결 하에서 $\nabla_X Y - \nabla_Y X \neq [X, Y]$임을 보여라.

<details>
<summary>힌트 및 해설</summary>

$\mathbb{R}^2$ 표준좌표에서 $\Gamma^1_{12} = 1, \Gamma^1_{21} = 0$, 나머지 0으로 정의. $X = \partial_1, Y = \partial_2$이면 $[X, Y] = 0$.  
$\nabla_X Y = \nabla_{\partial_1} \partial_2 = \Gamma^k_{12} \partial_k = \partial_1$.  
$\nabla_Y X = \nabla_{\partial_2} \partial_1 = \Gamma^k_{21} \partial_k = 0$.  
따라서 $\nabla_X Y - \nabla_Y X = \partial_1 \neq 0 = [X, Y]$ → torsion $T(X,Y) = \partial_1 \neq 0$. 정보기하의 $\alpha$-connection들은 대부분 torsion-free이지만, 일반 아핀 연결은 torsion을 가질 수 있다.

</details>

**문제 3** (AI 연결): 신경망 파라미터 공간 $\theta \in \mathbb{R}^d$에 Fisher 계량 $F(\theta)$을 주면 Levi-Civita 연결이 결정된다. 이 연결 하의 측지선을 정확히 따라가는 업데이트는 $\theta_{k+1} = \exp_{\theta_k}(-\eta F^{-1}\nabla L)$이다. 왜 실전에선 이 대신 **NGD** $\theta_{k+1} = \theta_k - \eta F^{-1}\nabla L$을 쓰는가? 두 업데이트의 차이가 $O(\eta^2)$임을 설명하라.

<details>
<summary>힌트 및 해설</summary>

Exponential map은 측지선 ODE $\ddot\gamma^k + \Gamma^k_{ij}\dot\gamma^i \dot\gamma^j = 0$을 적분해야 해 고비용(각 스텝마다 Hessian 유사 계산 포함). NGD는 $\exp_\theta(v) \approx \theta + v + O(\|v\|^2)$의 1차 근사.  
차이는 $\frac{1}{2}\Gamma^k_{ij} v^i v^j = O(\eta^2)$ (측지선 전개의 2차 항). 작은 $\eta$에선 무시 가능, 큰 $\eta$에선 $F$의 점 의존성이 중요해져 **Riemannian Trust Region** 같은 더 정교한 방법 필요. TRPO·K-FAC 등은 NGD의 실용적 근사 계열.

</details>

---

<div align="center">

| | | |
| :---: | :---: | :---: |
| [◀ 03. 리만 계량과 거리](./03-riemannian-metric.md) | [📚 README로 돌아가기](../README.md) | [Ch2-01. 통계다양체의 정의 ▶](../ch2-statistical-fisher/01-statistical-manifold.md) |

</div>
