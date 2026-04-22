# 03. VAE의 Information Geometry 해석

> **"VAE는 encoder와 decoder 사이의 '정보 거리'를 재는 기계다. 그 거리가 ELBO이며, 그 기하가 information projection이다."**

---

## 1. 왜 이 주제인가?

Variational Autoencoder (VAE; Kingma & Welling 2014)는 현대 생성 모델의 근간이다. ELBO를 재매개변수화 (reparameterization) 트릭과 함께 SGD로 최적화하여:

- **Generative capability**: $z \sim p(z)$, $x \sim p_\theta(x|z)$.
- **Inference capability**: $q_\phi(z|x)$ encoder.

표준 유도는 Jensen 부등식. 하지만 **information geometric 관점**에서 VAE는:

- **m-projection**: encoder $q_\phi(z|x)$가 true posterior $p(z|x)$의 근사.
- **Rate-distortion theory**: $\beta$-VAE의 $\beta$ trade-off가 정보 이론적 양의 balancing.
- **Posterior collapse**: KL → 0 정확히 **degenerate projection**.

이 문서는 VAE를 Ch6 information projection의 직접 응용으로 재해석하고, **$\beta$-VAE, InfoVAE, VQ-VAE**의 기하학적 차이를 정리한다. Diffusion (Ch7-05) 이전의 마지막 latent variable generative model.

---

## 2. 학습 목표

1. VAE의 **ELBO 분해** $\log p(x) \geq \text{ELBO}$ 재유도.
2. ELBO = **negative reconstruction loss − KL(encoder || prior)**의 해석.
3. **Reparameterization trick**과 backprop through stochasticity.
4. **$\beta$-VAE**의 rate-distortion 해석.
5. **Posterior collapse**의 기하학적 이해.
6. **InfoVAE, VQ-VAE**의 variational family extension.

---

## 3. 전제 지식

- **Ch6-02**: EM 알고리즘의 information geometry.
- **Ch6-03**: Variational Inference, Mean-field.
- **신경망, SGD, autograd**.

---

## 4. 직관적 설명

### 4.1 VAE의 두 network

- **Encoder $q_\phi(z|x)$**: $x$ → $z$ 분포 (보통 $\mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$).
- **Decoder $p_\theta(x|z)$**: $z$ → $x$ 분포 (보통 $\mathcal{N}(f_\theta(z), \sigma^2)$ or Bernoulli).
- **Prior $p(z)$**: 보통 $\mathcal{N}(0, I)$.

### 4.2 Evidence 분해

$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] + \text{KL}(q_\phi(z|x) \| p_\theta(z|x)).
$$

첫 항 = ELBO, 둘째 항 = "variational gap".

ELBO를 $q_\phi, p_\theta$ 모두에 대해 최대화:

- Encoder 입장: $q_\phi \to p_\theta(z|x)$ (amortized inference).
- Decoder 입장: reconstruction likelihood 최대화.

### 4.3 기하학적 관점

**Joint distribution space**에서:

- Data: $p_{\text{data}}(x) q_\phi(z|x)$ — "empirical joint".
- Model: $p(z) p_\theta(x|z)$ — generative joint.

ELBO 최대화 = 두 joint의 KL 최소화 = **m-projection**이 generative family에 있고 **e-projection**이 amortized inference family에 있는 동시 최적화.

### 4.4 Rate-Distortion View

$\beta$-VAE: $L = -\mathbb{E}_q[\log p(x|z)] + \beta \cdot \text{KL}(q \| p)$.

- Reconstruction = **distortion** $D$.
- KL = **rate** $R$ (정보 이론적 "encoding cost").

$\beta$: rate-distortion tradeoff. Higher $\beta$ → disentangled but worse reconstruction.

---

## 5. 엄밀한 정의와 정리

### 5.1 ELBO

**정의 5.1.** VAE objective:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z)).
$$

### 5.2 Evidence 분해

**정리 5.2.**

$$
\boxed{\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z|x) \| p_\theta(z|x)).}
$$

따라서 $\mathcal{L} \leq \log p_\theta(x)$ with equality iff $q_\phi = p_\theta(z|\cdot)$ exactly.

### 5.3 $\beta$-VAE

**정의 5.3 (Higgins+ 2017).**

$$
\mathcal{L}_\beta = \mathbb{E}_q[\log p(x|z)] - \beta \cdot \text{KL}(q(z|x) \| p(z)).
$$

$\beta = 1$ → standard VAE. $\beta > 1$ → more disentanglement, less reconstruction.

### 5.4 Rate-Distortion 분해 (Alemi+ 2018)

**정리 5.4.** ELBO per datum:

$$
-\mathcal{L} = D + R,
$$

$D = -\mathbb{E}_q[\log p(x|z)]$ (distortion), $R = \text{KL}(q(z|x)\|p(z))$ (rate).

### 5.5 Posterior Collapse

**정의 5.5.** $\text{KL}(q_\phi(z|x) \| p(z)) \to 0$ for some (or all) $x$. 의미: $z$가 $x$ 정보를 담지 못함.

**명제 5.6.** Strong decoder (예: autoregressive $p(x|z) = \prod p(x_t|x_{<t})$)에서 posterior collapse 발생 경향.

---

## 6. 증명

### 6.1 ELBO 분해

$$
\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \int q_\phi(z|x) \frac{p_\theta(x, z)}{q_\phi(z|x)} dz.
$$

Jensen 부등식:

$$
\log p_\theta(x) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right] = \mathcal{L}.
$$

Gap:

$$
\log p_\theta(x) - \mathcal{L} = \log p_\theta(x) - \mathbb{E}_q[\log p_\theta(x)] + \mathbb{E}_q[\log p_\theta(x)] - \mathbb{E}_q\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right]
$$

$$
= 0 + \mathbb{E}_q\left[\log \frac{p_\theta(x) q_\phi(z|x)}{p_\theta(x,z)}\right] = \mathbb{E}_q\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right] = \text{KL}(q_\phi \| p_\theta(z|x)). \quad \square
$$

### 6.2 Reparameterization Trick

$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]$를 직접 계산하기 어려움. Reparameterization:

$z = g_\phi(\epsilon, x)$, $\epsilon \sim p(\epsilon)$ (독립).

$$
\mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_\epsilon[f(g_\phi(\epsilon, x))].
$$

이제 $\nabla_\phi$가 expectation 안으로: $\nabla_\phi = \mathbb{E}_\epsilon[\nabla_\phi f(g_\phi(\epsilon, x))]$ (SGD로 샘플 가능).

**예 (Gaussian)**: $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$. $z = \mu_\phi + \sigma_\phi \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. $\square$

### 6.3 Rate-Distortion 분해

$$
-\mathcal{L} = -\mathbb{E}_q[\log p(x|z)] + \text{KL}(q(z|x)\|p(z)).
$$

첫 항은 per-sample reconstruction log-likelihood의 음수 = **distortion**.

둘째 항은 encoder가 $x$에 대해 "얼마나 많은 정보"를 encode하는지 = **rate** (Shannon 의미에서).

Average over $x$:

$$
-\mathbb{E}_{p_{\text{data}}}[\mathcal{L}] = D + R.
$$

이 tradeoff는 $\beta$ 조절로 명시적. $\square$

### 6.4 Posterior Collapse의 기하

$q_\phi(z|x) = p(z)$ (prior)로 붕괴하면:

- KL = 0 (rate 최소).
- Distortion = $-\mathbb{E}_{p(z)}[\log p(x|z)]$ — $x$-independent, maximum distortion.

Strong decoder가 $z$ 없이도 $p(x)$를 잘 예측하면 ($\log p(x|z) \approx \log p(x)$), encoder가 $z$에 정보 안 넣는게 optimal (KL penalty 피함).

**해결**: $\beta$ reduction (Bowman+ 2016), KL annealing, free bits (Kingma+ 2016), vector quantization (VQ-VAE).

---

## 7. 구체 예제

### 7.1 Gaussian VAE on MNIST

- Encoder: CNN → $\mu(x), \log\sigma(x)$.
- Decoder: Deconv → $\text{Bernoulli}$ logits per pixel.
- Prior: $\mathcal{N}(0, I)$.

Loss per batch:

$$
-\mathcal{L} = \text{BCE}(x, \hat{x}) + \frac{1}{2}\sum(\mu^2 + \sigma^2 - 1 - \log\sigma^2).
$$

두 번째 항은 analytic KL between $\mathcal{N}(\mu, \sigma^2)$ and $\mathcal{N}(0, 1)$.

### 7.2 ELBO vs Log-likelihood Gap

학습 후 Importance Weighted ELBO (IWAE, Burda+ 2016)로 더 tight bound:

$$
\mathcal{L}_K = \mathbb{E}_{z_1, \dots, z_K \sim q}\left[\log \frac{1}{K}\sum_k \frac{p(x, z_k)}{q(z_k|x)}\right] \geq \mathcal{L}.
$$

$K \to \infty$면 $\mathcal{L}_K \to \log p(x)$.

### 7.3 $\beta$-VAE로 disentanglement

$\beta = 4$ on dSprites: latent dimensions이 인간-해석 가능 factor (position, scale, rotation) 학습. Chen+ 2018 "Isolating Sources of Disentanglement"로 더 정교한 분해.

### 7.4 VQ-VAE (van den Oord+ 2017)

Continuous $z$ 대신 **discrete codebook** $\{e_k\}_{k=1}^K$. Encoder $z_e(x) \to$ nearest codebook $e_{k^*}$. Straight-through estimator로 backprop.

장점: Posterior collapse 없음 (discrete), high-fidelity 생성 (DALL-E 2, Stable Diffusion의 latent space).

### 7.5 Normalizing Flow Decoder

Decoder를 invertible flow로 → exact likelihood 계산 가능. ELBO vs log-likelihood gap 없음. Glow, Real NVP.

---

## 8. Python 코드 검증

### 8.1 Minimal VAE (PyTorch)

```python
"""
# Pseudocode - 실제로는 torch 필요

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden=400, latent=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.fc2 = nn.Linear(latent, hidden)
        self.fc3 = nn.Linear(hidden, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def elbo_loss(x_hat, x, mu, logvar):
    # Reconstruction (per-pixel Bernoulli)
    BCE = F.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
    # KL analytic for Gaussian
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training loop
# for epoch in range(epochs):
#     for x, _ in dataloader:
#         x_hat, mu, logvar = model(x)
#         loss = elbo_loss(x_hat, x, mu, logvar)
#         optimizer.zero_grad(); loss.backward(); optimizer.step()
"""
print("VAE PyTorch implementation sketched above.")
```

### 8.2 Analytic KL between two Gaussians

```python
import numpy as np

def kl_gaussians(mu1, var1, mu2, var2):
    # KL(N(mu1, var1) || N(mu2, var2))
    return 0.5 * (np.log(var2/var1) + (var1 + (mu1-mu2)**2)/var2 - 1)

# q = N(0.5, 0.8²), p = N(0, 1)
kl = kl_gaussians(0.5, 0.64, 0.0, 1.0)
print(f"KL(q||p) = {kl:.6f}")

# VAE 구현에서 사용: 
# KL = 0.5 * sum(mu² + σ² - 1 - log σ²)
mu, sigma = 0.5, 0.8
kl_vae_form = 0.5 * (mu**2 + sigma**2 - 1 - np.log(sigma**2))
print(f"KL (VAE form) = {kl_vae_form:.6f}")
assert abs(kl - kl_vae_form) < 1e-10
```

### 8.3 ELBO decomposition on toy 1D data

```python
import numpy as np
from scipy.stats import norm
# Data: x ~ N(μ=2, σ=0.5)
# Model: z ~ N(0, 1), x|z ~ N(z + 1, σ_x²) (decoder = identity + offset)
# True posterior p(z|x) can be computed analytically

np.random.seed(0)
X = np.random.normal(2, 0.5, 1000)

# True posterior (for fixed decoder params θ_d=(offset=1, σ_x=0.5))
# p(z|x) = N((x-1)/σ_x² · σ_post², σ_post²) with σ_post² = σ_x²/(1+σ_x²)
offset, sigma_x = 1.0, 0.5
sigma_post2 = sigma_x**2 / (1 + sigma_x**2)
mu_post = lambda x: (x - offset) / (1 + sigma_x**2)

# Encoder: q(z|x) = N(μ_φ(x), σ_φ²)
# 학습되지 않은 q로 ELBO 계산
mu_q = lambda x: 0.5*(x - 1)  # 임의 encoder (not optimal)
sigma_q2 = 0.5

# ELBO = E_q[log p(x|z)] - KL(q||prior)
def elbo_per_x(x):
    # E_q[log p(x|z)] where p(x|z) = N(z+offset, σ_x²)
    # z ~ q(z|x) = N(mu_q(x), sigma_q²)
    # log p(x|z) = -0.5*(x - z - offset)²/σ_x² - 0.5*log(2πσ_x²)
    # E_q[(x-z-offset)²] = (x - μ_q(x) - offset)² + sigma_q²
    E_log_px_z = -0.5*((x - mu_q(x) - offset)**2 + sigma_q2)/sigma_x**2 - 0.5*np.log(2*np.pi*sigma_x**2)
    # KL(N(μ_q, σ_q²)||N(0,1))
    KLD = 0.5*(mu_q(x)**2 + sigma_q2 - 1 - np.log(sigma_q2))
    return E_log_px_z - KLD

# log p(x) analytic (marginal)
# x = z + offset + ε, z ~ N(0,1), ε ~ N(0, σ_x²)
# x ~ N(offset, 1 + σ_x²)
log_px = norm.logpdf(X, offset, np.sqrt(1 + sigma_x**2))

# ELBO per sample
elbo = np.array([elbo_per_x(x) for x in X])

# Gap = log p(x) - ELBO = KL(q || p(z|x))
kl_post = log_px - elbo
print(f"Mean log p(x) = {log_px.mean():.4f}")
print(f"Mean ELBO    = {elbo.mean():.4f}")
print(f"Mean gap (should be >= 0) = {kl_post.mean():.4f}")
```

**기대**: gap = KL between encoder and true posterior, > 0 (encoder is suboptimal).

---

## 9. AI/ML 연결

### 9.1 Diffusion models as hierarchical VAE

Diffusion (Ho+ 2020) = deep hierarchical VAE with $T$ latent levels. Forward process = fixed encoder, reverse = learned decoder. Ch7-05에서 상세.

### 9.2 Stable Diffusion's latent space

VQ-VAE (or VAE) 먼저 학습 → latent space에서 diffusion. 4x-8x 작은 dimension → 빠른 생성.

### 9.3 GAN vs VAE

GAN: implicit density, sharper samples, mode collapse.
VAE: explicit likelihood, blurrier, full support.
Hybrid: AAE (Adversarial Autoencoder), VAEGAN.

### 9.4 Latent Dirichlet Allocation as VAE

LDA의 amortized inference = topic-VAE. NLP에서 document representation.

### 9.5 World Models (Ha & Schmidhuber 2018)

RL agent의 environment model = VAE + RNN. Latent imagination rollouts for planning.

---

## 10. 흔한 오해와 함정

1. **ELBO 낮은 것 ≠ 항상 나쁨**.
   - Tight decoder + bad prior fit이어도 ELBO 낮을 수 있음. Sample quality와 ELBO 다를 수 있음.

2. **Reparameterization trick의 일반성**.
   - Gaussian: $z = \mu + \sigma \epsilon$. 
   - Discrete: Gumbel-softmax (continuous relaxation).
   - Complex: Normalizing flows.

3. **KL annealing 필수**.
   - $\beta$를 0에서 1로 서서히 증가. 초기 KL penalty 없이 encoder 학습 후 regularize.

4. **Posterior collapse ≠ identifiability issue**.
   - Pure ML optimization 문제. 아키텍처/loss 수정으로 완화.

5. **Amortization gap**.
   - Encoder가 network 형태라 모든 $x$에 대해 최적 $q$ 못 줌. Cremer+ 2018.

6. **ELBO 비교는 동일 모델에서만**.
   - Decoder 아키텍처 다르면 ELBO 직접 비교 의미 없음.

---

## 11. 연습문제 및 다음 단계

### 연습문제

1. **ELBO 재유도**: Jensen 없이, KL 비음성만으로 ELBO 분해를 증명.

2. **Gaussian KL 공식**: $\text{KL}(\mathcal{N}(\mu, \Sigma) \| \mathcal{N}(0, I))$ 유도. Diagonal covariance 특별 케이스 명시.

3. **IWAE vs ELBO**: IWAE bound의 tightness가 sample size $K$에 따라 개선됨을 증명.

4. **$\beta$-VAE의 rate-distortion**: $\beta$ 변화 시 $(D, R)$ 점이 rate-distortion curve 위에서 어떻게 움직이는지 분석.

5. **Posterior collapse 예제**: PixelCNN decoder + VAE에서 collapse 실험적 관찰.

6. **VQ-VAE**: Straight-through estimator의 gradient가 biased임을 보이고 실전 trick 정리.

### 다음 단계

- **[04. Riemannian HMC](./04-riemannian-hmc.md)**: Fisher를 mass matrix로.
- **[05. Diffusion Fisher](./05-diffusion-fisher.md)**: Score matching과 Fisher divergence.

---

**참고문헌**

- Kingma, D., Welling, M. (2014). *Auto-Encoding Variational Bayes*.
- Rezende, D.+ (2014). *Stochastic Backpropagation and Approximate Inference*.
- Higgins, I.+ (2017). *β-VAE: Learning Basic Visual Concepts*.
- Alemi, A.+ (2018). *Fixing a Broken ELBO*.
- Bowman, S.+ (2016). *Generating Sentences from a Continuous Space* (KL annealing).
- van den Oord, A.+ (2017). *Neural Discrete Representation Learning* (VQ-VAE).
- Burda, Y.+ (2016). *Importance Weighted Autoencoders* (IWAE).
- Cremer, C.+ (2018). *Inference Suboptimality in VAEs*.

---

[◀ 02. Mirror Descent](./02-mirror-descent.md) | [📚 README](../README.md) | [04. Riemannian HMC ▶](./04-riemannian-hmc.md)
