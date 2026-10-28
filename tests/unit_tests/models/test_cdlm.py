# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Tests for the Plaid continuous diffusion LM (megatron/core/models/cdlm).

The CPU tests check the port against a transcription of the reference implementation
(https://github.com/igul222/plaid, lib/models.py and train.py) on a toy position-wise backbone:
losses, NLL bound and all parameter gradients must match.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import autograd, nn

from megatron.core.models.cdlm.cdlm_model import STAT_NAMES, CDLMConfig, CDLMModel
from megatron.core.models.cdlm.plaid import (
    EmbeddingMatrix,
    Fp32Island,
    GammaBounds,
    NoiseSchedule,
    OutputBias,
    ReconstAllocator,
    TLayout,
    make_eval_layout,
    make_train_layout,
    plaid_forward,
    plaid_losses,
)

VOCAB, EMBED_DIM, HIDDEN, SEQ, BATCH = 11, 4, 8, 6, 8


# --------------------------------------------------------------------------------------------
# Reference: Plaid, transcribed with explicit fp64 where Plaid relies on the fp64 default dtype,
# without muP / apex / flash-attn, and with a seeded generator for the noise.


class RefNoiseSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(1024, 1))
        self.b1 = nn.Parameter(torch.randn(1024))
        self.W2 = nn.Parameter(torch.randn(1, 1024))

    def forward(self, t):
        W1 = F.softplus(self.W1.double())
        W2 = 0.01 * F.softplus(self.W2.double())

        def gamma_tilde(t):
            h = t[:, None] - 0.5
            h = (h @ W1.T) + self.b1[None, :].double()
            h = torch.tanh(h)
            h = (h @ W2.T)[:, 0]
            return h

        gamma_tilde_0 = gamma_tilde(torch.tensor([0.0], dtype=torch.float64))
        gamma_tilde_1 = gamma_tilde(torch.tensor([1.0], dtype=torch.float64))
        gamma_tilde_t = gamma_tilde(t)
        return (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)


class Backbone(nn.Module):
    """Toy position-wise stand-in for the transformer blocks + output norm."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class RefDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_linear = nn.Linear(EMBED_DIM, HIDDEN, bias=False)
        self.selfcond_linear = nn.Linear(EMBED_DIM, HIDDEN, bias=False)
        self.gamma_linear = nn.Linear(64, HIDDEN, bias=False)
        self.backbone = Backbone()
        self.output_linear = nn.Linear(HIDDEN, VOCAB)
        self.embed_dim = EMBED_DIM

    def forward(self, z, gamma, embedding_matrix, bias_scale, x_selfcond, selfcond_mask=None):
        if selfcond_mask is None:
            selfcond_mask = torch.ones(z.shape[0], dtype=torch.float64)
        alpha_squared = torch.sigmoid(-gamma)[:, None, None]
        sigma_squared = torch.sigmoid(gamma)[:, None, None]
        alpha = alpha_squared.sqrt()
        z_variance = (alpha_squared / self.embed_dim) + sigma_squared
        x = z / z_variance.sqrt().float()
        x = self.input_linear(x)
        x = x + self.selfcond_linear(x_selfcond * float(np.sqrt(self.embed_dim)))
        gamma_embed = torch.linspace(-5.0, 5.0, 64 // 2, dtype=torch.float64)
        gamma_embed = gamma_embed.exp()[None, :] * gamma[:, None]
        gamma_embed = torch.cat([gamma_embed.sin(), gamma_embed.cos()], dim=1)
        gamma_embed = self.gamma_linear(gamma_embed.float())[:, None, :]
        x = x + gamma_embed
        x = self.backbone(x)
        W = torch.cat(
            [self.output_linear.weight.T, embedding_matrix.T, embedding_matrix.T.detach()], dim=0
        )
        z_scaled_for_bias = bias_scale * (alpha / sigma_squared).float() * z
        x = torch.cat(
            [
                x,
                z_scaled_for_bias * (1 - selfcond_mask.float()[:, None, None]),
                z_scaled_for_bias * selfcond_mask.float()[:, None, None],
            ],
            dim=2,
        )
        logits = torch.addmm(
            self.output_linear.bias.view(1, VOCAB),
            x.view(-1, HIDDEN + 2 * self.embed_dim),
            W.view(HIDDEN + 2 * self.embed_dim, VOCAB),
        ).view(x.shape[0], x.shape[1], VOCAB)
        x_reconst = F.softmax(logits, dim=2)
        x_reconst = x_reconst @ torch.cat([embedding_matrix, embedding_matrix.detach()], dim=1)
        x_reconst = torch.lerp(
            x_reconst[:, :, : self.embed_dim],
            x_reconst[:, :, self.embed_dim :],
            selfcond_mask.float()[:, None, None],
        )
        return logits, x_reconst


def ref_gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    return sigma_q.log() - sigma_p.log() + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2) - 0.5


def ref_plaid_step(x, t, selfcond_mask, reconst_bs, embedding_matrix_param, noise_schedule,
                   gamma_bounds, model, bias_scale, generator):
    """train.py:144-345 in train mode with accum_total = 1 and a given t / selfcond layout."""
    batch_size = x.shape[0]
    avg_reconst_bs = float(reconst_bs)
    avg_selfcond_mask = 0.25
    norm = torch.linalg.norm(embedding_matrix_param, dim=1, keepdim=True)
    embedding_matrix = embedding_matrix_param / (norm + 1e-8)
    t = t.clone().requires_grad_(True)
    with torch.enable_grad():
        gamma = torch.cat(
            [noise_schedule(t[:reconst_bs]).detach(), noise_schedule(t[reconst_bs:])]
        )
        gamma_prime = autograd.grad(gamma.sum(), [t], create_graph=True)[0]

    def set_grad_hook(tensor):
        if tensor.requires_grad:

            def grad_hook(grad):
                handle.remove()
                new_grad = torch.clone(grad.detach())
                new_grad[reconst_bs:] *= 2.0 * (grad_hook_loss[reconst_bs:].detach())
                return new_grad

            handle = tensor.register_hook(grad_hook)

    gamma = gamma.clone()
    set_grad_hook(gamma)
    set_grad_hook(gamma_prime)
    gamma_0, gamma_1 = gamma_bounds[0].double(), gamma_bounds[1].double()
    gamma = gamma_0 + (gamma_1 - gamma_0) * gamma
    gamma_prime = (gamma_1 - gamma_0) * gamma_prime
    gamma = torch.lerp(gamma, gamma.detach(), selfcond_mask)
    gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), selfcond_mask)
    alpha_squared = torch.sigmoid(-gamma)
    sigma_squared = torch.sigmoid(gamma)
    alpha = alpha_squared.sqrt()
    sigma = sigma_squared.sqrt()
    snr_prime = -(-gamma).exp() * gamma_prime
    alpha_1 = torch.sigmoid(-gamma_1).sqrt()
    sigma_1 = torch.sigmoid(gamma_1).sqrt()
    x_embed = embedding_matrix[x]
    x_embed = torch.lerp(x_embed, x_embed.detach(), selfcond_mask.float()[:, None, None])
    z = torch.randn([x.shape[0], x.shape[1], EMBED_DIM], dtype=torch.float32, generator=generator)
    z = z * sigma.float()[:, None, None]
    z = z + alpha.float()[:, None, None] * x_embed
    selfcond_idx = selfcond_mask.nonzero()[:, 0]
    x_selfcond = torch.zeros_like(z)
    with torch.no_grad():
        z_selfcond = z[selfcond_idx]
        _, x_reconst = model(
            z_selfcond, gamma[selfcond_idx], embedding_matrix, bias_scale,
            torch.zeros_like(z_selfcond),
        )
        x_selfcond[selfcond_idx] = x_reconst
    logits, x_reconst = model(
        z, gamma, embedding_matrix, bias_scale, x_selfcond, selfcond_mask=selfcond_mask
    )
    logprobs = F.log_softmax(logits[:reconst_bs], dim=2)
    reconst_loss = -logprobs.gather(2, x[:reconst_bs, :, None])[:, :, 0]
    reconst_loss = reconst_loss.mean(dim=1).double()
    alpha_1_masked = torch.lerp(alpha_1, alpha_1.detach(), selfcond_mask)[:, None, None]
    sigma_1_masked = torch.lerp(sigma_1, sigma_1.detach(), selfcond_mask)[:, None, None]
    prior_loss = ref_gaussian_kl(
        (alpha_1_masked * x_embed), sigma_1_masked,
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64),
    ).sum(dim=2).mean()
    diffusion_loss = (x_embed - x_reconst).pow(2)
    diffusion_loss = diffusion_loss.mean(dim=1).double().sum(dim=1)
    diffusion_loss = -0.5 * (snr_prime * diffusion_loss)
    grad_hook_loss = diffusion_loss
    loss = reconst_loss.sum() / avg_reconst_bs
    loss += diffusion_loss[reconst_bs:].sum() / (batch_size - avg_reconst_bs)
    loss += prior_loss
    nll = (reconst_loss * selfcond_mask[:reconst_bs]).sum() / (avg_reconst_bs * avg_selfcond_mask)
    nll += (diffusion_loss[reconst_bs:] * selfcond_mask[reconst_bs:]).sum() / (
        (batch_size - avg_reconst_bs) * avg_selfcond_mask
    )
    nll += prior_loss
    return loss, nll


# --------------------------------------------------------------------------------------------
# Port: CDLMModel's denoiser with the toy backbone in place of the mcore TransformerBlock.


class ToyCDLM(nn.Module):
    _denoise = CDLMModel._denoise
    _head = CDLMModel._head
    _head_chunk = CDLMModel._head_chunk
    _train_layout = CDLMModel._train_layout

    def __init__(self, ref: RefDiffusionModel, head_chunk_tokens: int):
        super().__init__()
        self.cdlm_config = CDLMConfig(embed_dim=EMBED_DIM, head_chunk_tokens=head_chunk_tokens)
        self.input_linear = ref.input_linear
        self.selfcond_linear = ref.selfcond_linear
        self.gamma_linear = ref.gamma_linear
        self.backbone = ref.backbone
        self.output_weight = ref.output_linear.weight
        self.output_bias = OutputBias(VOCAB)
        self.output_bias.weight = ref.output_linear.bias
        self.reconst_allocator = ReconstAllocator(0.997)
        self._step_iteration = None

    def rotary_pos_emb(self, seq_len):
        return None

    def decoder(self, hidden_states, attention_mask, rotary_pos_emb):
        return self.backbone(hidden_states)

    def _dp_rank_and_group(self):
        return 0, None


def _randomize(module):
    for p in module.parameters():
        nn.init.normal_(p, std=0.5)


@pytest.mark.parametrize("head_chunk_tokens", [5, 4096])
def test_plaid_forward_matches_reference(head_chunk_tokens):
    torch.manual_seed(0)
    ref_model = RefDiffusionModel()
    _randomize(ref_model)
    ref_schedule = RefNoiseSchedule()
    ref_bounds = [nn.Parameter(torch.tensor(-3.0)), nn.Parameter(torch.tensor(6.0))]
    embedding = EmbeddingMatrix(VOCAB, EMBED_DIM)
    ref_embedding = nn.Parameter(embedding.weight.detach().view(VOCAB, EMBED_DIM).clone())

    schedule = NoiseSchedule()
    with torch.no_grad():
        schedule.W1.copy_(ref_schedule.W1.view(-1))
        schedule.b1.copy_(ref_schedule.b1)
        schedule.W2.copy_(ref_schedule.W2.view(-1))
    bounds = GammaBounds(-3.0, 6.0)
    model = ToyCDLM(ref_model, head_chunk_tokens)

    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    reconst_bs = 3
    t = torch.zeros(BATCH, dtype=torch.float64)
    t[reconst_bs:] = (torch.arange(BATCH - reconst_bs, dtype=torch.float64) + 0.37) / (
        BATCH - reconst_bs
    )
    selfcond_mask = torch.zeros(BATCH, dtype=torch.float64)
    selfcond_mask[1::4] = 1
    bias_scale = 0.6

    ref_loss, ref_nll = ref_plaid_step(
        x, t, selfcond_mask, reconst_bs, ref_embedding, ref_schedule, ref_bounds, ref_model,
        bias_scale, torch.Generator().manual_seed(42),
    )
    ref_loss.backward()
    ref_grads = {
        'embedding': ref_embedding.grad.view(-1),
        'W1': ref_schedule.W1.grad.view(-1),
        'b1': ref_schedule.b1.grad,
        'W2': ref_schedule.W2.grad.view(-1),
        'gamma_0': ref_bounds[0].grad.view(1),
        'gamma_1': ref_bounds[1].grad.view(1),
    }
    ref_model_grads = {n: p.grad.clone() for n, p in ref_model.named_parameters()}
    ref_model.zero_grad()

    layout = TLayout(
        t=t, selfcond_mask=selfcond_mask.float(), reconst_bs=reconst_bs,
        avg_reconst_bs=float(reconst_bs), avg_selfcond=0.25,
    )
    terms = plaid_forward(
        x, layout, torch.ones(BATCH, SEQ), embedding, schedule, bounds, model._denoise,
        bias_scale, generator=torch.Generator().manual_seed(42),
    )
    loss, nll_r, nll_d, nll_p = plaid_losses(
        terms, layout.selfcond_mask, avg_reconst_bs=reconst_bs,
        avg_diffusion_bs=BATCH - reconst_bs, total_bs=BATCH, avg_selfcond=0.25,
    )
    loss.backward()

    tol = dict(rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(loss, ref_loss, **tol)
    torch.testing.assert_close(nll_r + nll_d + nll_p, ref_nll.detach(), **tol)
    grads = {
        'embedding': embedding.weight.grad,
        'W1': schedule.W1.grad,
        'b1': schedule.b1.grad,
        'W2': schedule.W2.grad,
        'gamma_0': bounds.gamma_0.grad,
        'gamma_1': bounds.gamma_1.grad,
    }
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], **tol, msg=name)
    for name, p in ref_model.named_parameters():
        torch.testing.assert_close(p.grad, ref_model_grads[name], **tol, msg=name)


def test_eval_forward_without_grad():
    torch.manual_seed(0)
    model = ToyCDLM(RefDiffusionModel(), 4096)
    layout = make_eval_layout(BATCH, np.random.default_rng(0), selfcond=True)
    with torch.no_grad():
        terms = plaid_forward(
            torch.randint(0, VOCAB, (BATCH, SEQ)), layout, torch.ones(BATCH, SEQ),
            EmbeddingMatrix(VOCAB, EMBED_DIM), NoiseSchedule(), GammaBounds(-3, 6),
            model._denoise, 1.0,
        )
        losses = plaid_losses(
            terms, layout.selfcond_mask, layout.avg_reconst_bs, BATCH - layout.avg_reconst_bs,
            BATCH, layout.avg_selfcond,
        )
    assert all(torch.isfinite(v) for v in losses)
    assert terms.reconst_bs == 1


def test_noise_schedule_monotone_and_derivative():
    torch.manual_seed(0)
    schedule = NoiseSchedule()
    t = torch.linspace(0, 1, 101, dtype=torch.float64, requires_grad=True)
    gamma = schedule(t)
    assert torch.allclose(gamma[[0, -1]], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert bool((gamma[1:] > gamma[:-1]).all())
    gamma_prime = autograd.grad(gamma.sum(), [t])[0]
    eps = 1e-6
    t_inner = t.detach()[1:-1]
    finite_diff = (schedule(t_inner + eps) - schedule(t_inner - eps)) / (2 * eps)
    torch.testing.assert_close(gamma_prime[1:-1], finite_diff.detach(), rtol=1e-5, atol=1e-7)


def test_fp32_island_ignores_dtype_casts():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.embedding = EmbeddingMatrix(5, 3)
            self.allocator = ReconstAllocator(0.9)

    model = Model().bfloat16()
    assert model.linear.weight.dtype == torch.bfloat16
    assert model.embedding.weight.dtype == torch.float32
    assert model.allocator.stats.dtype == torch.float64
    assert getattr(model.embedding.weight, 'cdlm_aux', False)
    assert isinstance(model.embedding, Fp32Island)


def test_reconst_allocator():
    allocator = ReconstAllocator(0.9)
    assert allocator.reconst_bs(64) == 1  # no statistics yet
    for _ in range(200):
        allocator.update(
            torch.tensor([1.0, 9.0]), torch.tensor([5.0, 5.1, 4.9, 5.0]), 2.0, 4.0
        )
    # Reconstruction losses vary much more than diffusion losses.
    assert 50 <= allocator.reconst_bs(64) <= 63


def test_train_layout_partitions_step():
    model = ToyCDLM(RefDiffusionModel(), 4096)
    num_microbatches, micro_batch_size = 4, 4
    ts, masks = [], []
    for _ in range(num_microbatches):
        layout, step, _ = model._train_layout(micro_batch_size, 7, num_microbatches)
        assert layout.reconst_bs == int(layout.t.eq(0).sum())
        ts.append(layout.t)
        masks.append(layout.selfcond_mask)
    torch.testing.assert_close(torch.cat(ts), step.t)
    torch.testing.assert_close(torch.cat(masks), step.selfcond_mask)
    assert step.reconst_bs == 1
    assert float(step.selfcond_mask.sum()) == 4
    # Same layout on a fresh pass over the same iteration (e.g. all ranks / restarts).
    model._step_iteration = None
    layout, step2, idx = model._train_layout(micro_batch_size, 7, num_microbatches)
    assert idx == 0
    torch.testing.assert_close(step2.t, step.t)


def test_make_train_layout():
    layout = make_train_layout(16, 5, np.random.default_rng(1), selfcond=True)
    assert bool(layout.t[:5].eq(0).all()) and bool((layout.t[5:] > 0).all())
    assert bool((layout.t < 1).all())
    assert float(layout.selfcond_mask.sum()) == 4
    assert layout.avg_selfcond == 0.25


# --------------------------------------------------------------------------------------------
# GPU: the real model with a Transformer Engine backbone.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestCDLMModel:
    def setup_method(self, method):
        from megatron.core.models.cdlm.cdlm_layer_specs import get_cdlm_layer_spec
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True,
            bf16=True, params_dtype=torch.bfloat16,
        )
        self.model = CDLMModel(
            config=config,
            transformer_layer_spec=get_cdlm_layer_spec(use_transformer_engine=True),
            vocab_size=128,
            max_sequence_length=33,
            cdlm_config=CDLMConfig(head_chunk_tokens=40, bias_warmup_steps=10),
        ).cuda()

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    def test_forward_backward(self):
        from megatron.core.transformer.module import Float16Module

        model = Float16Module(self.model.config, self.model)
        assert self.model.noise_schedule.W1.dtype == torch.float32
        tokens = torch.randint(0, 128, (4, 33), device='cuda')
        loss_mask = torch.ones(4, 33, device='cuda')
        positions = torch.arange(33, device='cuda')[None].expand(4, -1)
        for microbatch in range(2):
            loss, stats = model(
                tokens, positions, None, labels=tokens, loss_mask=loss_mask,
                iteration=3, num_microbatches=2,
            )
            assert torch.isfinite(loss)
            (loss / 2).backward()
        assert stats.shape == (len(STAT_NAMES),)
        for name, p in self.model.named_parameters():
            if getattr(p, 'cdlm_aux', False):
                assert p.grad is not None and torch.isfinite(p.grad).all(), name

        model.eval()
        with torch.no_grad():
            loss, stats = model(tokens, positions, None, labels=tokens, loss_mask=loss_mask)
        assert torch.isfinite(stats).all()
