# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Continuous diffusion language model (Plaid) on top of the mcore transformer block."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from megatron.core import parallel_state
from megatron.core.models.cdlm.plaid import (
    EmbeddingMatrix,
    GammaBounds,
    NoiseSchedule,
    OutputBias,
    ReconstAllocator,
    TLayout,
    gamma_fourier_embed,
    make_eval_layout,
    make_train_layout,
    plaid_forward,
    plaid_losses,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

# Order of the statistics returned by CDLMModel.forward.
STAT_NAMES = (
    'nll',
    'nll_reconst',
    'nll_diffusion',
    'nll_prior',
    'denominator',
    'gamma_0',
    'gamma_1',
    'reconst_bs',
)


@dataclass
class CDLMConfig:
    """Plaid hyper-parameters (defaults are Plaid's, train.py:38-58)."""

    embed_dim: int = 16
    gamma_0: float = -3.0
    gamma_1: float = 6.0
    selfcond: bool = True
    bias_warmup_steps: int = 5000
    reconst_bs_ema: float = 0.997
    # Probability of truncating a micro-batch to a random length (as MegaDLMs' discrete DLM).
    varilen_prob: float = 0.01
    # Tokens per chunk of the vocabulary-sized output head (bounds its activation memory).
    head_chunk_tokens: int = 4096
    seed: int = 1234


class CDLMModel(LanguageModule):
    """Plaid continuous diffusion LM with a bidirectional mcore transformer backbone.

    Tokens are mapped to unit-norm `embed_dim`-dimensional embeddings, noised with a learned
    Gaussian schedule and denoised by the transformer, which predicts token logits; the model is
    trained on the VDM likelihood bound (see `plaid.py`). `forward` returns the training loss and
    reporting statistics instead of logits.

    Only TP = PP = CP = 1 is supported: self-conditioning runs the backbone twice per step and the
    output head needs the full vocabulary.

    Args:
        config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Layer spec with non-causal self-attention.
        vocab_size (int): Vocabulary size.
        max_sequence_length (int): Maximum sequence length.
        cdlm_config (CDLMConfig): Diffusion hyper-parameters.
        pg_collection (ProcessGroupCollection): Model communication process groups.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        cdlm_config: CDLMConfig,
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)
        assert config.tensor_model_parallel_size == 1, "CDLM supports TP=1 only"
        assert config.pipeline_model_parallel_size == 1, "CDLM supports PP=1 only"
        assert config.context_parallel_size == 1, "CDLM supports CP=1 only"

        self.cdlm_config = cdlm_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = True
        self.post_process = True
        self.share_embeddings_and_output_weights = False
        self.vp_stage = vp_stage
        self.model_type = ModelType.encoder_or_decoder
        hidden_size = config.hidden_size
        embed_dim = cdlm_config.embed_dim

        # Diffusion process (fp32 islands, trained with their own learning rate).
        self.embedding_matrix = EmbeddingMatrix(vocab_size, embed_dim)
        self.noise_schedule = NoiseSchedule()
        self.gamma_bounds = GammaBounds(cdlm_config.gamma_0, cdlm_config.gamma_1)
        self.reconst_allocator = ReconstAllocator(cdlm_config.reconst_bs_ema)

        # Input: noisy latent, self-conditioning estimate and noise level (Plaid models.py:136-140).
        self.input_linear = nn.Linear(embed_dim, hidden_size, bias=False)
        self.selfcond_linear = nn.Linear(embed_dim, hidden_size, bias=False)
        self.gamma_linear = nn.Linear(64, hidden_size, bias=False)
        nn.init.zeros_(self.selfcond_linear.weight)
        nn.init.zeros_(self.gamma_linear.weight)

        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            use_cpu_initialization=config.use_cpu_initialization,
            cp_group=self.pg_collection.cp,
        )

        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        # Output head, zero-initialised as in Plaid. A plain parameter rather than a
        # ColumnParallelLinear: it is applied chunk-wise under activation checkpointing.
        self.output_weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))
        self.output_weight.is_embedding_or_output_parameter = True
        self.output_bias = OutputBias(vocab_size)

        # Per-step state.
        self._step_iteration = None
        self._step_layout: Optional[TLayout] = None
        self._microbatch_idx = 0
        self._generator = None
        self._eval_rng = None

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """See megatron.core.models.gpt.GPTModel.set_input_tensor."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for cdlm'
        self.decoder.set_input_tensor(input_tensor[0])

    def _dp_rank_and_group(self):
        if not torch.distributed.is_initialized():
            return 0, None
        return (
            parallel_state.get_data_parallel_rank(),
            parallel_state.get_data_parallel_group(),
        )

    def _noise_generator(self, device) -> torch.Generator:
        # Same stream within a data-parallel rank, different streams across ranks.
        if self._generator is None or self._generator.device != device:
            dp_rank, _ = self._dp_rank_and_group()
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.cdlm_config.seed + dp_rank)
        return self._generator

    def _train_layout(self, micro_batch_size: int, iteration: int, num_microbatches: int):
        """Layout of this micro-batch's rows within the step's local batch."""
        local_batch_size = micro_batch_size * num_microbatches
        assert local_batch_size >= 2, "CDLM needs at least 2 samples per rank and step"
        if self._step_iteration != iteration:
            dp_rank, dp_group = self._dp_rank_and_group()
            if dp_group is not None:
                # Plaid synchronises the allocator statistics once per step.
                torch.distributed.all_reduce(self.reconst_allocator.stats, group=dp_group)
                self.reconst_allocator.stats /= torch.distributed.get_world_size(dp_group)
            reconst_bs = self.reconst_allocator.reconst_bs(local_batch_size)
            rng = np.random.default_rng((self.cdlm_config.seed, iteration, dp_rank))
            self._step_layout = make_train_layout(
                local_batch_size, reconst_bs, rng, self.cdlm_config.selfcond
            )
            self._step_iteration = iteration
            self._microbatch_idx = 0
        step = self._step_layout
        microbatch_idx = self._microbatch_idx
        self._microbatch_idx = (microbatch_idx + 1) % num_microbatches
        rows = slice(microbatch_idx * micro_batch_size, (microbatch_idx + 1) * micro_batch_size)
        t = step.t[rows]
        layout = TLayout(
            t=t,
            selfcond_mask=step.selfcond_mask[rows],
            reconst_bs=int(t.eq(0).sum()),
            avg_reconst_bs=step.avg_reconst_bs,
            avg_selfcond=step.avg_selfcond,
        )
        return layout, step, microbatch_idx

    def _maybe_truncate(self, x, mask, iteration, microbatch_idx):
        # Like MegaDLMs' varilen data: the decision is shared by all data-parallel ranks.
        rng = np.random.default_rng((self.cdlm_config.seed, iteration, microbatch_idx, 1))
        if rng.uniform() < self.cdlm_config.varilen_prob:
            length = int(rng.integers(low=2, high=x.shape[1] + 1))
            x, mask = x[:, :length], mask[:, :length]
        return x, mask

    def _head_chunk(self, h, z_bias, embedding, targets):
        logits = F.linear(h, self.output_weight).float() + self.output_bias() + z_bias @ embedding.t()
        log_probs = F.log_softmax(logits, dim=-1)
        ce = None if targets is None else -log_probs.gather(-1, targets[:, None])[:, 0]
        x_reconst = log_probs.exp() @ embedding
        return ce, x_reconst

    def _head(self, h, z_bias, embedding, targets):
        """Logits with Plaid's output prior, CE and x_reconst, chunked over tokens."""
        n, s, _ = h.shape
        h = h.reshape(n * s, -1)
        z_bias = z_bias.reshape(n * s, -1)
        targets = None if targets is None else targets.reshape(-1)
        chunk = self.cdlm_config.head_chunk_tokens
        ces, x_reconsts = [], []
        for i in range(0, n * s, chunk):
            args = (
                h[i : i + chunk],
                z_bias[i : i + chunk],
                embedding,
                None if targets is None else targets[i : i + chunk],
            )
            if torch.is_grad_enabled():
                ce, x_reconst = checkpoint(self._head_chunk, *args, use_reentrant=False)
            else:
                ce, x_reconst = self._head_chunk(*args)
            ces.append(ce)
            x_reconsts.append(x_reconst)
        ce = None if targets is None else torch.cat(ces).view(n, s)
        return ce, torch.cat(x_reconsts).view(n, s, -1)

    def _denoise(self, z, gamma, x_selfcond, selfcond_mask, embedding, bias_scale, targets):
        """Plaid's DiffusionModel.forward (models.py:159-221) with the mcore backbone."""
        batch_size, seq_len, embed_dim = z.shape
        dtype = self.input_linear.weight.dtype

        # Rescale the input to unit variance.
        alpha_squared = torch.sigmoid(-gamma)
        sigma_squared = torch.sigmoid(gamma)
        z_variance = (alpha_squared / embed_dim) + sigma_squared
        h = self.input_linear((z / z_variance.sqrt().float()[:, None, None]).to(dtype))
        h = h + self.selfcond_linear((x_selfcond * float(np.sqrt(embed_dim))).to(dtype))
        h = h + self.gamma_linear(gamma_fourier_embed(gamma).float().to(dtype))[:, None, :]

        h = h.transpose(0, 1).contiguous()  # [s, b, h]
        rotary_pos_emb = self.rotary_pos_emb(seq_len)
        h = self.decoder(hidden_states=h, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
        h = h.transpose(0, 1)  # [b, s, h]

        z_bias = (bias_scale * alpha_squared.sqrt() / sigma_squared).float()[:, None, None] * z

        # Self-conditioned rows see detached embeddings in the output prior and in x_reconst.
        if selfcond_mask is None:
            detached_rows = torch.ones(batch_size, dtype=torch.bool, device=z.device)
        else:
            detached_rows = selfcond_mask.to(z.device) > 0
        ce = None
        if targets is not None:
            ce = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=z.device)
        x_reconst = torch.zeros_like(z)
        for detached in (False, True):
            idx = (detached_rows == detached).nonzero()[:, 0]
            if len(idx) == 0:
                continue
            emb = embedding.detach() if detached else embedding
            ce_rows, x_reconst_rows = self._head(
                h[idx], z_bias[idx], emb, None if targets is None else targets[idx]
            )
            if ce is not None:
                ce[idx] = ce_rows
            x_reconst[idx] = x_reconst_rows
        return ce, x_reconst

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor],
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        iteration: int = 0,
        num_microbatches: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """Plaid training/evaluation forward pass.

        Args:
            input_ids (Tensor): [b, s + 1] tokens from the GPT dataset. As in MegaDLMs' noshift
                mode, the model operates on `input_ids[:, 1:]` (== `labels[:, :-1]`).
            position_ids, attention_mask: unused (attention is bidirectional, RoPE positions
                are implicit).
            labels (Tensor): only checked for presence; sampling is not implemented.
            loss_mask (Tensor): [b, s + 1] loss mask aligned with `labels`.
            iteration (int): current training iteration.
            num_microbatches (int): micro-batches per step on this rank.

        Returns:
            (loss, stats): `loss` is a scalar whose mean over the step's micro-batches is Plaid's
            training loss on this rank. `stats` holds `STAT_NAMES`: the NLL bound (nats/token)
            and its parts as numerators over `denominator`, then gamma_0, gamma_1 and reconst_bs.
        """
        if labels is None:
            raise NotImplementedError("CDLMModel does not support sampling yet")
        x = input_ids[:, 1:]
        if loss_mask is None:
            mask = torch.ones_like(x, dtype=torch.float32)
        else:
            mask = loss_mask[:, :-1].float()

        if self.training:
            layout, step, microbatch_idx = self._train_layout(
                x.shape[0], iteration, num_microbatches
            )
            x, mask = self._maybe_truncate(x, mask, iteration, microbatch_idx)
            local_batch_size = len(step.t)
            avg_reconst_bs = step.avg_reconst_bs
            avg_diffusion_bs = local_batch_size - avg_reconst_bs
            warmup = self.cdlm_config.bias_warmup_steps
            bias_scale = min(1.0, (iteration + 1e-8) / (warmup + 1e-8))
            denominator = 1.0 / num_microbatches
        else:
            if self._eval_rng is None:
                dp_rank, _ = self._dp_rank_and_group()
                self._eval_rng = np.random.default_rng((self.cdlm_config.seed, dp_rank, 7))
            layout = make_eval_layout(x.shape[0], self._eval_rng, self.cdlm_config.selfcond)
            local_batch_size = x.shape[0]
            avg_reconst_bs = layout.avg_reconst_bs
            avg_diffusion_bs = local_batch_size - avg_reconst_bs
            bias_scale = 1.0
            denominator = 1.0

        terms = plaid_forward(
            x,
            layout,
            mask,
            self.embedding_matrix,
            self.noise_schedule,
            self.gamma_bounds,
            self._denoise,
            bias_scale,
            generator=self._noise_generator(x.device),
        )
        loss, nll_reconst, nll_diffusion, nll_prior = plaid_losses(
            terms,
            layout.selfcond_mask,
            avg_reconst_bs=avg_reconst_bs,
            avg_diffusion_bs=avg_diffusion_bs,
            total_bs=local_batch_size,
            avg_selfcond=layout.avg_selfcond,
        )

        if self.training:
            # Per micro-batch update with per-micro-batch normalisers, as Plaid does per
            # gradient-accumulation step.
            self.reconst_allocator.update(
                terms.reconst,
                terms.diffusion[terms.reconst_bs :],
                avg_reconst_bs / num_microbatches,
                x.shape[0] - avg_reconst_bs / num_microbatches,
            )
            # DDP's overlapped grad reduction needs a gradient for every parameter; a micro-batch
            # of only self-conditioned rows (possible with micro-batch size 1) would leave the
            # diffusion-process parameters out of the graph.
            diffusion_params = (
                *self.embedding_matrix.parameters(),
                *self.noise_schedule.parameters(),
                *self.gamma_bounds.parameters(),
            )
            loss = loss + 0.0 * sum(p.sum() for p in diffusion_params)
            loss = loss * num_microbatches

        stats = torch.stack(
            [
                nll_reconst + nll_diffusion + nll_prior,
                nll_reconst,
                nll_diffusion,
                nll_prior,
                torch.tensor(denominator, dtype=torch.float64, device=x.device),
                terms.gamma_0.view(()),
                terms.gamma_1.view(()),
                torch.tensor(
                    float(step.reconst_bs if self.training else layout.reconst_bs),
                    dtype=torch.float64,
                    device=x.device,
                ),
            ]
        ).float()
        return loss.float(), stats

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = (), metadata=None):
        """Skip LanguageModule's embedding/output-layer tying logic: CDLMModel has neither."""
        return MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)
