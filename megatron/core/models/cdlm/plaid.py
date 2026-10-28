# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Plaid continuous diffusion components (Gulrajani & Hashimoto, 2023).

Ports of the backbone-independent parts of https://github.com/igul222/plaid (`lib/models.py`,
`train.py`): learned embeddings, learned noise schedule, the VDM likelihood bound and its
variance-reduction tricks. Nothing in this module depends on Megatron, so it can be unit-tested
on CPU against the reference implementation.

Conventions follow Plaid: gamma = log(sigma^2 / alpha^2), alpha^2 = sigmoid(-gamma),
sigma^2 = sigmoid(gamma); all schedule arithmetic is done in fp64.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, autograd, nn


class Fp32Island(nn.Module):
    """Module whose tensors follow device moves but ignore dtype casts.

    Megatron wraps the model in `Float16Module`, which calls `module.bfloat16()`. Plaid's noise
    schedule, gamma bounds and embedding matrix need full-precision values in the forward pass,
    so modules deriving from this class keep their dtype (the optimizer already supports fp32
    params next to bf16 ones).
    """

    def _apply(self, fn, recurse=True):
        probe = fn(torch.zeros((), dtype=torch.float32))
        if probe.dtype != torch.float32:
            device = probe.device
            fn = lambda t: t.to(device)
        return super()._apply(fn, recurse)


def _mark_aux(param: nn.Parameter) -> nn.Parameter:
    # Picked up by the optimizer override in pretrain_cdlm.py (Plaid trains these with lr 1e-2).
    param.cdlm_aux = True
    return param


class EmbeddingMatrix(Fp32Island):
    """Learned token embeddings, renormalised to unit L2 norm on every forward.

    The matrix is stored flattened: Megatron assigns weight decay 0 to 1-D params, which is the
    weight decay Plaid uses for the embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        matrix = torch.randn(vocab_size, embed_dim)
        matrix /= matrix.norm(p=2, dim=1, keepdim=True)
        self.weight = _mark_aux(nn.Parameter(matrix.view(-1)))

    def forward(self) -> Tensor:
        matrix = self.weight.view(self.vocab_size, self.embed_dim)
        return matrix / (torch.linalg.norm(matrix, dim=1, keepdim=True) + 1e-8)


class NoiseSchedule(Fp32Island):
    """Monotone MLP t -> gamma_tilde(t), normalised to [0, 1] on [0, 1].

    Weights are stored flattened (weight decay 0, as in Plaid); W1 is Plaid's [1024, 1] matrix and
    W2 its [1, 1024] matrix.
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.W1 = _mark_aux(nn.Parameter(torch.randn(hidden_size)))
        self.b1 = _mark_aux(nn.Parameter(torch.randn(hidden_size)))
        self.W2 = _mark_aux(nn.Parameter(torch.randn(hidden_size)))

    def forward(self, t: Tensor) -> Tensor:
        """t: [n] -> gamma normalised to [0, 1], fp64."""
        W1 = F.softplus(self.W1.double())
        W2 = 0.01 * F.softplus(self.W2.double())
        b1 = self.b1.double()

        def gamma_tilde(t):
            h = t[:, None].double() - 0.5
            h = torch.tanh(h * W1[None, :] + b1[None, :])
            return h @ W2

        ends = gamma_tilde(torch.tensor([0.0, 1.0], dtype=torch.float64, device=t.device))
        return (gamma_tilde(t) - ends[0]) / (ends[1] - ends[0])


class GammaBounds(Fp32Island):
    """Learned endpoints gamma_0 = gamma(t=0) and gamma_1 = gamma(t=1)."""

    def __init__(self, gamma_0: float, gamma_1: float):
        super().__init__()
        self.gamma_0 = _mark_aux(nn.Parameter(torch.tensor([float(gamma_0)])))
        self.gamma_1 = _mark_aux(nn.Parameter(torch.tensor([float(gamma_1)])))

    def forward(self) -> Tuple[Tensor, Tensor]:
        return self.gamma_0.double(), self.gamma_1.double()


class OutputBias(Fp32Island):
    """Zero-initialised logit bias, kept in fp32 like the rest of Plaid's output head."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size))

    def forward(self) -> Tensor:
        return self.weight


def gamma_fourier_embed(gamma: Tensor, dim: int = 64) -> Tensor:
    """Sinusoidal embedding of gamma: [n] -> [n, dim] (fp64)."""
    freqs = torch.linspace(-5.0, 5.0, dim // 2, dtype=torch.float64, device=gamma.device).exp()
    g = freqs[None, :] * gamma[:, None].double()
    return torch.cat([g.sin(), g.cos()], dim=1)


def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    """KL(p || q) between diagonal Gaussians."""
    return (
        torch.log(sigma_q) - torch.log(sigma_p)
        + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
        - 0.5
    )


@dataclass
class TLayout:
    """Diffusion times and self-conditioning mask for a batch.

    The first `reconst_bs` entries have t = 0 and are used for the reconstruction loss; the rest
    follow a low-discrepancy grid on (0, 1).
    """

    t: Tensor  # [B], fp64
    selfcond_mask: Tensor  # [B], fp32 in {0, 1}
    reconst_bs: int
    # Normalisers of Plaid's estimators (expected counts, see train.py:334-345).
    avg_reconst_bs: float
    avg_selfcond: float


def make_train_layout(
    batch_size: int, reconst_bs: int, rng: np.random.Generator, selfcond: bool
) -> TLayout:
    """Plaid train-mode layout (train.py:179-200) over the full local batch of one step."""
    assert 1 <= reconst_bs <= batch_size - 1
    n = batch_size - reconst_bs
    t = np.zeros(batch_size)
    t[reconst_bs:] = (np.arange(n) + rng.uniform()) / n
    selfcond_mask = np.zeros(batch_size, dtype=np.float32)
    if selfcond:
        selfcond_mask[int(rng.integers(4)) :: 4] = 1.0
    return TLayout(
        t=torch.from_numpy(t),
        selfcond_mask=torch.from_numpy(selfcond_mask),
        reconst_bs=reconst_bs,
        avg_reconst_bs=float(reconst_bs),
        avg_selfcond=0.25 if selfcond else 1.0,
    )


def make_eval_layout(batch_size: int, rng: np.random.Generator, selfcond: bool) -> TLayout:
    """Plaid eval-mode layout (train.py:170-200): ~1/8 reconstruction samples, all self-cond."""
    reconst_bs = batch_size // 8 + int(rng.binomial(1, (batch_size % 8) / 8.0))
    n = batch_size - reconst_bs
    t = np.zeros(batch_size)
    if n > 0:
        t[reconst_bs:] = (np.arange(n) + rng.uniform()) / n
    selfcond_mask = np.full(batch_size, 1.0 if selfcond else 0.0, dtype=np.float32)
    return TLayout(
        t=torch.from_numpy(t),
        selfcond_mask=torch.from_numpy(selfcond_mask),
        reconst_bs=reconst_bs,
        avg_reconst_bs=batch_size / 8.0,
        avg_selfcond=1.0,
    )


class ReconstAllocator(Fp32Island):
    """Adaptive number of t = 0 (reconstruction) samples per step (train.py:138-168).

    Keeps bias-corrected EMAs of the first two moments of the reconstruction and diffusion losses
    and allocates samples proportionally to their standard deviations. The EMAs are buffers, so
    they are checkpointed with the model.
    """

    def __init__(self, ema: float):
        super().__init__()
        self.ema = ema
        # [bias, reconst, reconst_sqr, diffusion, diffusion_sqr]
        self.register_buffer('stats', torch.full((5,), 1e-8, dtype=torch.float64))

    def reconst_bs(self, batch_size: int) -> int:
        _, r, r_sqr, d, d_sqr = self.stats / self.stats[0]  # bias-corrected moments
        r_std = (r_sqr - r**2).clamp(min=0).sqrt()
        d_std = (d_sqr - d**2).clamp(min=0).sqrt()
        reconst_bs = batch_size * (r_std / (1e-8 + r_std + d_std))
        return int(reconst_bs.round().clamp(1, batch_size - 1))

    @torch.no_grad()
    def update(
        self, reconst: Tensor, diffusion: Tensor, avg_reconst_bs: float, avg_diffusion_bs: float
    ):
        """reconst: losses of the t = 0 samples; diffusion: losses of the t > 0 samples."""
        target = torch.stack(
            [
                torch.ones((), dtype=torch.float64, device=self.stats.device),
                reconst.double().sum() / avg_reconst_bs,
                reconst.double().pow(2).sum() / avg_reconst_bs,
                diffusion.double().sum() / avg_diffusion_bs,
                diffusion.double().pow(2).sum() / avg_diffusion_bs,
            ]
        )
        self.stats.lerp_(target, 1 - self.ema)


class ScheduleGradHook:
    """Plaid's schedule variance reduction (train.py:219-233).

    The noise schedule receives gradients of E[loss^2] (via grad * 2 * loss) while every other
    parameter receives gradients of E[loss]. `loss` must be set after the forward pass.
    """

    def __init__(self, reconst_bs: int):
        self.reconst_bs = reconst_bs
        self.loss: Optional[Tensor] = None

    def attach(self, tensor: Tensor):
        if not tensor.requires_grad:
            return

        def grad_hook(grad):
            handle.remove()
            new_grad = torch.clone(grad.detach())
            new_grad[self.reconst_bs :] *= 2.0 * self.loss[self.reconst_bs :].detach()
            return new_grad

        handle = tensor.register_hook(grad_hook)


@dataclass
class PlaidTerms:
    """Per-sample loss terms of one Plaid forward pass (nats per token)."""

    reconst: Tensor  # [reconst_bs]: -log p(x | z_0)
    diffusion: Tensor  # [B]: continuous-time diffusion loss (rows < reconst_bs unused)
    prior: Tensor  # [B]: KL(q(z_1 | x) || N(0, I))
    gamma_0: Tensor
    gamma_1: Tensor
    reconst_bs: int


# denoiser(z, gamma, x_selfcond, selfcond_mask, embedding, bias_scale, targets) -> (ce, x_reconst)
#   z: [B, S, D] fp32 noisy latents, gamma: [B] fp64, x_selfcond: [B, S, D],
#   selfcond_mask: [B] or None (= all ones), embedding: [V, D] unit-norm rows,
#   targets: [B, S] token ids or None. Returns per-token CE [B, S] (None without targets) and
#   x_reconst = softmax(logits) @ E: [B, S, D].
Denoiser = Callable[..., Tuple[Optional[Tensor], Tensor]]


def plaid_forward(
    x: Tensor,
    layout: TLayout,
    loss_mask: Tensor,
    embedding_matrix: EmbeddingMatrix,
    noise_schedule: NoiseSchedule,
    gamma_bounds: GammaBounds,
    denoiser: Denoiser,
    bias_scale: float,
    generator: Optional[torch.Generator] = None,
) -> PlaidTerms:
    """One Plaid forward pass (train.py:177-322) over a batch whose t = 0 rows come first.

    Args:
        x: [B, S] token ids.
        layout: times and self-conditioning mask for these B rows.
        loss_mask: [B, S], tokens excluded from the per-token means are 0.
    """
    device = x.device
    t = layout.t.to(device)
    selfcond_mask = layout.selfcond_mask.to(device)
    reconst_bs = int(t.eq(0).sum())
    assert bool(t[:reconst_bs].eq(0).all()), "t = 0 rows must come first"

    embedding = embedding_matrix()

    # gamma'(t) needs autograd even in eval, where the caller runs under no_grad.
    grad_enabled = torch.is_grad_enabled()
    with torch.enable_grad():
        t = t.detach().double().requires_grad_(True)
        # Don't propagate grads for the reconstruction rows.
        gamma = torch.cat(
            [noise_schedule(t[:reconst_bs]).detach(), noise_schedule(t[reconst_bs:])]
        )
        gamma_prime = autograd.grad(
            gamma.sum(), [t], create_graph=grad_enabled, allow_unused=True
        )[0]
        if gamma_prime is None:
            gamma_prime = torch.zeros_like(t)
    hook = ScheduleGradHook(reconst_bs)
    if not grad_enabled:
        gamma, gamma_prime = gamma.detach(), gamma_prime.detach()
    gamma = gamma.clone()
    hook.attach(gamma)
    hook.attach(gamma_prime)
    gamma_0, gamma_1 = gamma_bounds()
    gamma = gamma_0 + (gamma_1 - gamma_0) * gamma
    gamma_prime = (gamma_1 - gamma_0) * gamma_prime

    selfcond_d = selfcond_mask.double()
    gamma = torch.lerp(gamma, gamma.detach(), selfcond_d)
    gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), selfcond_d)

    alpha_squared = torch.sigmoid(-gamma)
    sigma_squared = torch.sigmoid(gamma)
    alpha = alpha_squared.sqrt()
    sigma = sigma_squared.sqrt()
    snr_prime = -(-gamma).exp() * gamma_prime  # SNR = exp(-gamma)
    alpha_1 = torch.sigmoid(-gamma_1).sqrt()
    sigma_1 = torch.sigmoid(gamma_1).sqrt()

    # Construct z with reparameterisation gradients.
    x_embed = embedding[x]
    x_embed = torch.lerp(x_embed, x_embed.detach(), selfcond_mask.float()[:, None, None])
    noise = torch.randn(
        x_embed.shape, dtype=torch.float32, device=device, generator=generator
    )
    z = noise * sigma.float()[:, None, None] + alpha.float()[:, None, None] * x_embed

    # Self-conditioning pass.
    x_selfcond = torch.zeros_like(z)
    selfcond_idx = selfcond_mask.nonzero()[:, 0]
    if len(selfcond_idx) > 0:
        with torch.no_grad():
            z_sc = z[selfcond_idx]
            _, x_reconst_sc = denoiser(
                z_sc, gamma[selfcond_idx], torch.zeros_like(z_sc), None, embedding, bias_scale, None
            )
            x_selfcond[selfcond_idx] = x_reconst_sc.float()

    # Main pass.
    ce, x_reconst = denoiser(z, gamma, x_selfcond, selfcond_mask, embedding, bias_scale, x)

    mask = loss_mask.double()
    num_tokens = mask.sum(dim=1).clamp(min=1)

    def seq_mean(v):  # [B, S] -> [B]
        return (v.double() * mask).sum(dim=1) / num_tokens
    reconst = (ce[:reconst_bs].double() * mask[:reconst_bs]).sum(dim=1) / num_tokens[:reconst_bs]

    alpha_1_masked = torch.lerp(alpha_1.expand_as(selfcond_d), alpha_1.detach(), selfcond_d)
    sigma_1_masked = torch.lerp(sigma_1.expand_as(selfcond_d), sigma_1.detach(), selfcond_d)
    prior = gaussian_kl(
        alpha_1_masked[:, None, None] * x_embed,
        sigma_1_masked[:, None, None],
        torch.tensor(0.0, dtype=torch.float64, device=device),
        torch.tensor(1.0, dtype=torch.float64, device=device),
    ).sum(dim=2)
    prior = seq_mean(prior)

    diffusion = (x_embed - x_reconst).pow(2).double()  # [B, S, D]
    diffusion = ((diffusion * mask[:, :, None]).sum(dim=1) / num_tokens[:, None]).sum(dim=1)
    diffusion = -0.5 * (snr_prime * diffusion)
    hook.loss = diffusion

    return PlaidTerms(
        reconst=reconst,
        diffusion=diffusion,
        prior=prior,
        gamma_0=gamma_0.detach(),
        gamma_1=gamma_1.detach(),
        reconst_bs=reconst_bs,
    )


def plaid_losses(
    terms: PlaidTerms,
    selfcond_mask: Tensor,
    avg_reconst_bs: float,
    avg_diffusion_bs: float,
    total_bs: float,
    avg_selfcond: float,
):
    """Combine per-sample terms into Plaid's training loss and NLL bound (train.py:334-345).

    The normalisers refer to the whole batch the estimators are defined over (the local batch of
    a step in training), so summing these per-micro-batch values over micro-batches reproduces
    Plaid's estimators.

    Returns:
        (loss, nll_reconst, nll_diffusion, nll_prior): the training loss and the three parts of
        the NLL bound (which is computed on self-conditioned rows only).
    """
    reconst_bs = terms.reconst_bs
    selfcond_mask = selfcond_mask.to(terms.prior.device).double()
    reconst = terms.reconst.sum() / avg_reconst_bs
    diffusion = terms.diffusion[reconst_bs:].sum() / avg_diffusion_bs
    prior = terms.prior.sum() / total_bs
    loss = reconst + diffusion + prior

    nll_reconst = (terms.reconst * selfcond_mask[:reconst_bs]).sum() / (
        avg_reconst_bs * avg_selfcond
    )
    nll_diffusion = (terms.diffusion[reconst_bs:] * selfcond_mask[reconst_bs:]).sum() / (
        avg_diffusion_bs * avg_selfcond
    )
    return loss, nll_reconst.detach(), nll_diffusion.detach(), prior.detach()
