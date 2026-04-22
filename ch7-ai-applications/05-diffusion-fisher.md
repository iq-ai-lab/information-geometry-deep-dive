# 7.5 확산 모델의 피셔 정보 관점 (Diffusion Models through Fisher)

[◀ 04. Riemannian HMC](./04-riemannian-hmc.md) | [📚 README](../README.md)

---

## 1. 왜 이것을 배우는가? (Motivation)

2020~2025년 생성 AI의 혁명을 이끈 **확산 모델**(Diffusion Models, Score-based Generative Models)은 표면적으로는 "노이즈를 점진적으로 제거"하는 알고리즘으로 보이지만, 그 수학적 핵심은 **스코어 함수** $\nabla_x \log p_t(x)$에 있다. 그리고 이 스코어 함수는 정보기하에서 오랫동안 알려진 **피셔 정보**와 깊은 관련이 있다.

**핵심 관찰 세 가지**:

1. **스코어 = Fisher 정보 밀도의 제곱근**: $I(x) = \|\nabla_x \log p(x)\|^2$ 는 위치적 Fisher 정보.
2. **Denoising Score Matching (DSM) 손실 ≡ Fisher divergence**: 확산 모델의 훈련 목표가 정확히 두 분포 간 Fisher 정보 거리를 최소화.
3. **리버스 SDE = 자연 그래디언트 흐름**: 생성 과정의 역방향 미분방정식이 KL divergence를 Fisher 계량으로 감소시키는 흐름.

이 챕터에서는 확산 모델의 엔진을 정보기하의 언어로 완전히 해부한다. 이는 단순 응용이 아니라, Amari가 1980년대부터 구축한 이론이 현대 생성 AI의 수학적 토대임을 보여준다.

---

## 2. 학습 목표 (Learning Objectives)

이 챕터를 마치면 다음을 할 수 있다:

1. 스코어 함수 $s(x) = \nabla_x \log p(x)$와 Fisher 정보의 관계를 설명할 수 있다.
2. **Fisher divergence** $\mathcal{J}_F(p\|q) = \mathbb{E}_p\|\nabla\log p - \nabla\log q\|^2$를 정의하고 성질을 도출할 수 있다.
3. **Denoising Score Matching** (Vincent 2011) 손실이 Fisher divergence와 수학적으로 동치임을 증명할 수 있다.
4. Forward/Reverse SDE (Anderson 1982; Song et al. 2021)를 기술하고 Fokker-Planck 방정식을 유도할 수 있다.
5. 랑주뱅 샘플링이 **엔트로피에 대한 자연 그래디언트 흐름**임을 보일 수 있다.
6. Flow Matching, Rectified Flow 등 최신 변형의 정보기하적 해석을 이해한다.

---

## 3. 전제 지식 (Prerequisites)

- [2.3 Fisher-Rao 계량](../ch2-statistical-fisher/03-fisher-rao-metric.md)
- [5.3 KL의 최급강하](../ch5-natural-gradient/03-kl-steepest-descent.md)
- [7.4 Riemannian HMC](./04-riemannian-hmc.md) (랑주뱅 샘플링)
- 이토(Itô) 확률 미적분, SDE, Fokker-Planck 방정식
- 점수 매칭(score matching)에 대한 기초 지식

---

## 4. 직관적 이해 (Intuition)

### 4.1 스코어 함수가 왜 중요한가?

확률밀도 $p(x)$는 정규화 상수 $Z$ 때문에 직접 학습이 어렵지만, **스코어 함수**

$$
s(x) = \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}
$$

는 $Z$와 무관하다! 이 "정규화 상수의 저주로부터의 해방"이 score-based 모델의 동기다.

**기하학적 의미**: 스코어는 **확률밀도가 증가하는 방향**을 가리키는 벡터장. 이를 따라가면 고밀도 영역으로 수렴(랑주뱅 동역학).

### 4.2 위치적 Fisher 정보

Amari의 (모수적) Fisher 정보 $g_{ij} = \mathbb{E}[\partial_i \log p \, \partial_j \log p]$와 달리, **위치적 Fisher 정보**는 확률변수 $x$에 대한 밀도의 미분:

$$
J_F(p) = \int p(x) \|\nabla \log p(x)\|^2 dx
$$

이는 **분포의 "예리함"을 측정**. 좁은 피크일수록 기울기가 크고, 평평할수록 작다. 정규분포 $\mathcal{N}(0, \sigma^2)$의 경우 $J_F = 1/\sigma^2$ (1D).

### 4.3 Fisher divergence

두 분포 $p$, $q$의 **상대적 위치 Fisher**:

$$
\mathcal{J}_F(p \| q) = \mathbb{E}_{p}\!\left[\|\nabla \log p(x) - \nabla \log q(x)\|^2\right]
$$

KL divergence의 "미분 버전". $p = q$일 때 0, 항상 ≥ 0. 이는 확산 모델의 훈련 손실과 정확히 일치한다.

### 4.4 노이즈 추가와 점수의 관계

핵심 아이디어: 원 데이터 $p_{\text{data}}$는 희박하고 복잡하지만, 가우시안 노이즈를 추가한 $p_t = p_{\text{data}} * \mathcal{N}(0, \sigma_t^2 I)$는 매끄럽다. 그리고:

$$
\nabla_x \log p_t(x) = \mathbb{E}_{x_0 \sim p(x_0 \mid x_t)}\!\left[\frac{x_0 - x_t}{\sigma_t^2}\right]
$$

즉 **노이지 관측에서 원본으로의 기대 방향 ≈ 스코어**. 신경망은 이 조건부 기댓값을 배운다.

---

## 5. 엄밀한 전개 (Rigorous Development)

### 5.1 Forward SDE

확산 모델의 **forward process**는 다음 SDE로 기술:

$$
dx_t = f(x_t, t) dt + g(t) dW_t, \quad x_0 \sim p_{\text{data}}
$$

선택:
- **VP-SDE** (Variance Preserving): $f = -\tfrac{1}{2}\beta(t) x$, $g = \sqrt{\beta(t)}$ (DDPM)
- **VE-SDE** (Variance Exploding): $f = 0$, $g = \sqrt{d\sigma^2/dt}$ (NCSN)

표준 결과(Feynman-Kac): $p_t(x) = \int p_{0t}(x\mid x_0) p_{\text{data}}(x_0) dx_0$는 Fokker-Planck 방정식

$$
\partial_t p_t = -\nabla \cdot (f p_t) + \frac{g^2}{2}\Delta p_t
$$

를 만족.

### 5.2 Reverse SDE (Anderson 1982)

놀라운 결과: forward SDE에 대응되는 **역시간 SDE**는

$$
\boxed{dx_t = [f(x_t, t) - g(t)^2 \nabla_x \log p_t(x_t)] dt + g(t) d\bar{W}_t}
$$

여기서 $d\bar{W}_t$는 역시간 브라운 운동. 스코어 $\nabla \log p_t$만 알면 노이즈에서 데이터로 되돌아갈 수 있다!

### 5.3 Score Matching (Hyvärinen 2005)

이상적 손실:

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{p_t}\!\left[\|s_\theta(x, t) - \nabla_x \log p_t(x)\|^2\right]
$$

하지만 실제 $p_t$의 스코어를 모른다. **Hyvärinen의 트릭** (부분 적분):

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_t}\!\left[\|s_\theta\|^2 + 2 \mathrm{tr}(\nabla s_\theta)\right] + C
$$

고차원에서 $\mathrm{tr}(\nabla s_\theta)$ 계산이 비쌈.

### 5.4 Denoising Score Matching (Vincent 2011)

**핵심 관찰**: 표준 SM 손실이 **조건부 DSM 손실과 동치**다!

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x_0, x_t}\!\left[\|s_\theta(x_t, t) - \nabla_{x_t} \log p_{0t}(x_t \mid x_0)\|^2\right]
$$

가우시안 전이의 경우 $p_{0t}(x_t \mid x_0) = \mathcal{N}(x_t; \alpha_t x_0, \sigma_t^2 I)$ 에서

$$
\nabla_{x_t} \log p_{0t}(x_t \mid x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} = -\frac{\epsilon}{\sigma_t}
$$

즉 신경망은 주입된 **노이즈의 방향을 예측**.

### 5.5 DSM = Fisher Divergence

**핵심 정리**: DSM 손실과 Fisher divergence는 같다.

$$
\mathcal{L}_{\text{DSM}}(\theta) = \int \mathcal{J}_F(p_t \| q_{\theta,t}) w(t) dt + C
$$

여기서 $q_{\theta,t}$는 $s_\theta$로 정의된 모델 분포(Stein operator 의미). $w(t)$는 시간 가중치.

### 5.6 랑주뱅 역학 = 자연 그래디언트 흐름

**랑주뱅 SDE** $dx_t = \nabla \log p(x_t) dt + \sqrt{2} dW_t$의 분포 $\rho_t$는 Fokker-Planck

$$
\partial_t \rho_t = -\nabla\cdot(\rho_t \nabla \log p) + \Delta \rho_t = \nabla\cdot(\rho_t \nabla \log(\rho_t/p))
$$

를 만족하며, 이는 **KL divergence의 Wasserstein 기울기 흐름**(Jordan-Kinderlehrer-Otto 1998):

$$
\partial_t \rho_t = -\nabla_{W_2} \mathrm{KL}(\rho_t \| p)
$$

Wasserstein 공간에서의 자연 그래디언트(정확히는 Otto 계량). 이는 5.3장의 "KL의 최급강하가 Fisher 방향"의 **무한차원 버전**이다.

---

## 6. 증명 (Proofs)

### 6.1 정리 (Anderson 1982): Reverse SDE

**명제**: Forward SDE $dx_t = f(x_t, t) dt + g(t) dW_t$의 시간 역전은

$$
dx_t = [f - g^2 \nabla \log p_t] dt + g d\bar{W}_t
$$

**증명 개요**: Fokker-Planck $\partial_t p_t = -\nabla\cdot(fp_t) + \tfrac{g^2}{2}\Delta p_t$. 이를 다음과 같이 재작성:

$$
\partial_t p_t = -\nabla\cdot\left[(f - \tfrac{g^2}{2}\nabla\log p_t) p_t - \tfrac{g^2}{2}\nabla\log p_t \cdot p_t\right] + \tfrac{g^2}{2}\Delta p_t
$$

시간을 역전 $\tau = T-t$ 하면 drift가 $-(f - g^2 \nabla\log p_t)$로 변환. 남은 확산 계수 $g^2/2$는 동일한 강도의 역방향 브라운 운동을 요구. $\blacksquare$

### 6.2 정리 (Vincent 2011): DSM = SM

**명제**: 
$$
\mathbb{E}_{p_t(x)}\!\left[\|s_\theta - \nabla\log p_t\|^2\right] = \mathbb{E}_{p_{0t}(x_t\mid x_0)p(x_0)}\!\left[\|s_\theta - \nabla\log p_{0t}(x_t\mid x_0)\|^2\right] + C
$$

**증명**: 좌변을 전개:

$$
\text{LHS} = \mathbb{E}\|s_\theta\|^2 - 2\mathbb{E}_{p_t}[s_\theta^\top \nabla\log p_t] + \mathbb{E}\|\nabla\log p_t\|^2
$$

핵심은 교차항. $p_t(x) = \int p_{0t}(x\mid x_0) p(x_0) dx_0$이므로

$$
\nabla\log p_t(x) = \frac{\int \nabla p_{0t}(x\mid x_0) p(x_0) dx_0}{p_t(x)} = \frac{\int p_{0t}(x\mid x_0) \nabla\log p_{0t}(x\mid x_0) p(x_0) dx_0}{p_t(x)}
$$

따라서

$$
\mathbb{E}_{p_t}[s_\theta^\top \nabla\log p_t] = \int s_\theta(x)^\top \int p_{0t}(x\mid x_0)p(x_0) \nabla\log p_{0t} dx_0 dx
$$

$$
= \mathbb{E}_{p(x_0) p_{0t}(x\mid x_0)}[s_\theta^\top \nabla \log p_{0t}(x\mid x_0)]
$$

이를 우변의 해당 교차항과 매칭시키면 상수항만 다름. $\blacksquare$

### 6.3 정리: DSM 손실 ≡ 가중 Fisher divergence

**명제**: DSM 손실은 시간 가중 Fisher divergence 적분이다:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \int_0^T w(t) \mathcal{J}_F(p_t \| p_t^\theta) dt + \text{const}
$$

여기서 $p_t^\theta$는 $\nabla\log p_t^\theta = s_\theta(\cdot, t)$로 정의.

**증명**: 6.2에 의해 DSM 손실 = SM 손실 (상수 차이). SM 손실은 명시적으로

$$
\mathbb{E}_{p_t}\|s_\theta - \nabla\log p_t\|^2 = \int p_t \|\nabla\log p_t^\theta - \nabla\log p_t\|^2 dx = \mathcal{J}_F(p_t \| p_t^\theta)
$$

각 $t$마다 가중치 $w(t)$로 적분. $\blacksquare$

### 6.4 정리 (JKO 1998): 랑주뱅 = Wasserstein 기울기 흐름

**명제**: $dX_t = -\nabla U(X_t) dt + \sqrt{2} dW_t$의 분포 $\rho_t$는

$$
\rho_{t+\tau} = \arg\min_{\rho} \left\{ \mathrm{KL}(\rho \| e^{-U}/Z) + \frac{1}{2\tau} W_2^2(\rho, \rho_t) \right\} + o(\tau)
$$

즉 각 시간 단계마다 $W_2$ 거리 비용 하에서 KL을 감소시킨다. 이는 **정보기하의 최급강하와 정확히 대응**: 파라미터 공간에서 Fisher 거리 하 KL 감소가 NGD인 것처럼, 분포 공간에서 Wasserstein 거리 하 KL 감소가 랑주뱅이다. $\blacksquare$ (증명은 Jordan-Kinderlehrer-Otto 1998 원 논문 참조)

---

## 7. 예제 (Examples)

### 7.1 예제 1: 1D 가우시안 믹스처

$p_{\text{data}} = 0.5 \mathcal{N}(-3, 1) + 0.5 \mathcal{N}(3, 1)$.

스코어: $s(x) = \frac{0.5 N(x;-3,1)(-(x+3)) + 0.5 N(x;3,1)(-(x-3))}{p(x)}$.

교차 영역($x \approx 0$)에서 스코어는 ±로 크게 변동. 신경망은 이를 학습해야 하며, 노이즈를 충분히 크게(σ=2 이상) 주면 매끄러워짐.

### 7.2 예제 2: DDPM의 noise prediction

DDPM 손실(Ho et al. 2020):

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]
$$

스코어와의 관계: $\epsilon_\theta(x_t, t) = -\sigma_t s_\theta(x_t, t)$. 즉 DDPM = DSM의 재매개화.

### 7.3 예제 3: Stable Diffusion의 잠재 확산

잠재 공간 $z = E(x)$에서 확산. Fisher divergence는 **잠재 공간의 계량 텐서에 의존**. 인코더 $E$가 유도하는 pullback 계량이 효율적 학습의 핵심.

### 7.4 예제 4: Flow Matching (Lipman et al. 2023)

Flow Matching은 확률경로 $p_t$ 자체를 지정하고 속도장 $v_\theta(x, t)$를 매칭:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_t}\|v_\theta(x_t, t) - v_t^*(x_t)\|^2
$$

**정보기하 해석**: $v_t^* = d\theta/dt$ (통계 다양체의 접벡터장). 이는 측지선을 따라가는 m-flat 보간의 근사다. Rectified Flow (Liu 2022)는 직선 보간 → m-측지선에 해당.

---

## 8. 코드 (Code)

```python
import torch
import torch.nn as nn
import numpy as np

class ScoreNet(nn.Module):
    """Simple score network s_theta(x, t)."""
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x, t):
        # t in [0, 1], broadcast
        t_emb = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_emb], dim=1))


def dsm_loss(score_net, x0, sigma_min=0.01, sigma_max=10.0):
    """Denoising Score Matching loss (VE-SDE)."""
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device)
    # VE noise schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^t
    sigma_t = sigma_min * (sigma_max/sigma_min)**t
    noise = torch.randn_like(x0)
    x_t = x0 + sigma_t.view(-1, 1) * noise
    # Target score: -noise / sigma_t (gradient of log N(x_t; x0, sigma_t^2 I) wrt x_t)
    target = -noise / sigma_t.view(-1, 1)
    pred = score_net(x_t, t)
    # Lambda(t) = sigma_t^2 weighting recommended
    loss = ((pred - target)**2).sum(dim=1) * sigma_t**2
    return loss.mean()


def langevin_sample(score_net, shape, n_steps=1000, eps=1e-4, device='cpu'):
    """Annealed Langevin dynamics sampling."""
    x = torch.randn(shape, device=device) * 10  # start from wide prior
    for i in range(n_steps):
        t = torch.tensor([1 - i/n_steps], device=device)
        score = score_net(x, t)
        noise = torch.randn_like(x)
        x = x + eps * score + np.sqrt(2*eps) * noise
    return x


def fisher_divergence_estimate(p_samples, q_score_fn, p_score_fn, n=10000):
    """Estimate Fisher divergence J_F(p || q) by Monte Carlo."""
    idx = torch.randperm(len(p_samples))[:n]
    x = p_samples[idx]
    sp = p_score_fn(x)
    sq = q_score_fn(x)
    return ((sp - sq)**2).sum(dim=1).mean().item()


# === Training loop sketch ===
def train(model, data_loader, n_epochs=100, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        for batch in data_loader:
            x0 = batch
            loss = dsm_loss(model, x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 8.2 Reverse SDE 적분기

```python
def euler_maruyama_reverse(score_net, shape, T=1.0, n_steps=1000, 
                            sigma_min=0.01, sigma_max=10.0, device='cpu'):
    """Solve reverse SDE via Euler-Maruyama."""
    x = torch.randn(shape, device=device) * sigma_max
    ts = torch.linspace(T, 1e-3, n_steps, device=device)
    dt = -(ts[1] - ts[0]).item()  # negative since reverse
    for i in range(n_steps - 1):
        t = ts[i].expand(shape[0])
        sigma_t = sigma_min * (sigma_max/sigma_min)**t
        g = sigma_t * np.sqrt(2*np.log(sigma_max/sigma_min))  # g(t)
        score = score_net(x, t)
        # Reverse SDE: dx = -g^2 * score dt + g dW
        drift = -(g.view(-1,1))**2 * score * dt
        diffusion = g.view(-1,1) * torch.randn_like(x) * np.sqrt(abs(dt))
        x = x + drift + diffusion
    return x
```

---

## 9. AI/ML 응용 (AI/ML Applications)

### 9.1 확산 기반 생성 모델의 전 스펙트럼

- **DDPM** (Ho 2020): 이산 마르코프 체인, ε-예측, FID 3.17 on CIFAR-10
- **Score SDE** (Song 2021): 연속 SDE 통합 프레임워크
- **Stable Diffusion** (Rombach 2022): 잠재 확산, 텍스트 컨디셔닝
- **DiT** (Peebles 2023): Transformer 기반 스코어 네트워크
- **SD3 / Flux / Imagen 3** (2024): Flow Matching 계열

모두 **DSM = Fisher divergence 최소화**라는 동일 수학을 공유.

### 9.2 Consistency Model (Song 2023)

표준 확산은 $N$ 스텝 필요. Consistency Model은 PF-ODE의 궤적 위에서 **궤적 불변 함수** $f(x_t, t) = f(x_0, 0) = x_0$를 직접 학습하여 1~4 스텝 생성 가능. 이는 **Rectified Flow** 및 **Progressive Distillation**과 연결.

### 9.3 분류기 자유 가이던스 (CFG)

$s_\theta(x, c) = (1+w) s_\theta(x \mid c) - w s_\theta(x)$

**정보기하 해석**: 이는 $p(x \mid c) \propto p(x) [p(c \mid x)]^{1+w}$의 템퍼드 사후분포에 대한 스코어. 즉 **조건부 분포의 지수족 왜곡**이며, Tsallis 통계와 연결.

### 9.4 Schrödinger Bridge

두 분포 $\mu_0, \mu_T$ 사이의 **엔트로피 정규화 최적 수송**. De Bortoli et al. (2021)은 이것이 확산 모델의 일반화임을 보였다. 정보기하에서 이는 **m-geodesic 보간의 확률적 버전**.

### 9.5 Score-based Inverse Problems

의료 영상, 단백질 구조 예측 등에서 $y = Ax + \text{noise}$의 역문제:

$$
p(x \mid y) \propto p(y \mid x) p(x)
$$

사후분포 스코어 = 사전분포 스코어 + 우도 스코어. 사전분포 스코어는 pretrained diffusion에서 제공. 이는 **정보기하의 m-projection**과 직접 연결.

### 9.6 Fisher-Rao 공간에서의 생성

Lie et al. (2024) 등 최근 연구는 확률밀도를 Fisher-Rao 구 $\mathbb{S}^{\infty}$의 점으로 보고, 구 위의 측지선을 따르는 생성을 제안. 본 책 3장의 Chentsov 이론이 직접 응용된다.

### 9.7 오픈 문제

- **최적 시간 스케줄**: $\sigma(t)$의 최적 설계가 Fisher divergence 적분의 분산을 최소화.
- **다양체 위 확산**: 회전군, 그래프, 단백질 구조 등에서 Riemannian diffusion (De Bortoli 2022).
- **이산 확산**: 텍스트, 분자 생성을 위한 이산 상태 공간의 DSM.

---

## 10. 흔한 오해 (Common Pitfalls)

### 10.1 "스코어 = gradient of data"

아니다. 스코어는 **log-density의 gradient**. 데이터 포인트 자체가 아니라 데이터가 나올 확률이 증가하는 방향.

### 10.2 "확산 = 노이즈 제거 네트워크"

부분적으로만 맞다. 수학적으로는 **스코어 추정**이며, 노이즈 예측과 스코어 예측은 $\sigma_t$ 스케일링으로 동치다.

### 10.3 "DSM은 편향된 추정량"

정확히는: DSM은 SM 손실의 **상수 차이 동치** (명제 6.2). 따라서 최소화 관점에서 같으며, 실제 gradient는 동일하다. "편향"은 손실값 자체에 대한 것이며 최적화에는 영향 없다.

### 10.4 "Fisher divergence가 KL보다 항상 나쁘다"

상황에 따라 다르다. Fisher는 국소 정보(미분), KL은 전역 정보. 확산 모델이 Fisher를 사용하는 이유는 **무편향 Monte Carlo 추정이 가능**하기 때문 (KL은 정규화 상수 의존).

### 10.5 "Reverse SDE에 더 빠른 ODE 대체가 항상 낫다"

PF-ODE는 결정론적이라 빠르지만, SDE의 확률적 성질이 모드 탐색에 중요할 수 있다. 다양성이 중요한 응용에서는 SDE가 여전히 우세.

---

## 11. 연습 문제 (Exercises)

**연습 11.1** (기초). 1D 가우시안 $p(x) = \mathcal{N}(x; \mu, \sigma^2)$의 스코어 함수를 구하라. 위치적 Fisher $J_F(p) = \mathbb{E}_p[(\nabla\log p)^2]$를 계산하라.

**연습 11.2** (증명). 두 가우시안 $p = \mathcal{N}(\mu_1, \sigma^2), q = \mathcal{N}(\mu_2, \sigma^2)$의 Fisher divergence가 $(\mu_1 - \mu_2)^2/\sigma^4$ 임을 보여라.

**연습 11.3** (계산). VP-SDE $dx = -\tfrac{1}{2}\beta(t) x dt + \sqrt{\beta(t)} dW$에서 $p_{0t}(x_t \mid x_0)$를 구하고, 그 스코어 $\nabla_{x_t} \log p_{0t}$를 계산하라.

**연습 11.4** (코딩). 2D 가우시안 믹스처에서 score network를 DSM으로 훈련시키고, Langevin 샘플링으로 생성된 샘플을 시각화하라.

**연습 11.5** (이론). Hyvärinen score matching의 부분적분 공식을 유도하라. 언제 이것이 성립하는가 (경계 조건)?

**연습 11.6** (심화). Flow Matching 손실 $\|v_\theta - v^*\|^2$이 조건부 Flow Matching $\|v_\theta - v^*(\cdot \mid x_0)\|^2$와 상수 차이로 동치임을 증명하라 (Lipman Thm 2).

**연습 11.7** (응용). Consistency Model의 training objective를 Fisher divergence 언어로 다시 써라.

**연습 11.8** (통합). 본 책 전체에서 배운 내용을 사용해, 확산 모델 훈련을 "Fisher-Rao 구 위의 경로에 대한 m-geodesic 근사"로 재해석하는 에세이를 작성하라.

---

## 참고문헌

- Anderson, B. D. (1982). "Reverse-time diffusion equation models." *Stochastic Processes and their Applications*, 12(3), 313-326.
- Hyvärinen, A. (2005). "Estimation of non-normalized statistical models by score matching." *JMLR*, 6, 695-709.
- Vincent, P. (2011). "A connection between score matching and denoising autoencoders." *Neural Computation*, 23(7), 1661-1674.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising diffusion probabilistic models." *NeurIPS 2020*.
- Song, Y., et al. (2021). "Score-based generative modeling through stochastic differential equations." *ICLR 2021*.
- Jordan, R., Kinderlehrer, D., & Otto, F. (1998). "The variational formulation of the Fokker-Planck equation." *SIAM J. Math. Anal.*, 29(1), 1-17.
- Lipman, Y., et al. (2023). "Flow matching for generative modeling." *ICLR 2023*.
- Song, Y., et al. (2023). "Consistency models." *ICML 2023*.

---

## 🎉 여정의 끝 (End of Journey)

이 문서로 **Information Geometry Deep Dive**의 모든 7개 챕터, 35개 문서가 완성되었다.

여러분이 배운 것을 돌아보면:
- **Ch1**: 왜 확률분포에 기하가 필요한가?
- **Ch2**: Fisher 계량으로 만든 통계 다양체
- **Ch3**: Chentsov 유일성과 Fisher-Rao 계량의 심오함
- **Ch4**: e-connection, m-connection의 이중성
- **Ch5**: 자연 그래디언트와 K-FAC
- **Ch6**: e-/m-projection과 EM, VI, MaxEnt
- **Ch7**: 그리고 마침내 현대 AI — RL, VAE, HMC, 확산 모델까지

Amari가 1980년대에 세운 이론이 2020년대 생성 AI의 엔진을 돌리고 있다. 여러분은 이제 이 다리를 건널 수 있다.

*"The geometry of probability is not a metaphor — it is the mathematics of learning itself."*

[◀ 04. Riemannian HMC](./04-riemannian-hmc.md) | [📚 README](../README.md)
