# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Pretrain a continuous diffusion LM (Plaid, Gulrajani & Hashimoto 2023).

Uses the GPT data pipeline unchanged. The reported `lm loss` is Plaid's NLL bound in nats/token,
comparable to the AR cross-entropy and to the masked-diffusion ELBO.
"""

import copy
from functools import partial

import torch

import megatron.training.training as training_module
from megatron.core.enums import ModelType
from megatron.core.models.cdlm import CDLMConfig, CDLMModel
from megatron.core.models.cdlm.cdlm_layer_specs import get_cdlm_layer_spec
from megatron.core.models.cdlm.cdlm_model import STAT_NAMES
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, get_timers, inprocess_restart, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from model_provider import model_provider
from pretrain_gpt import (
    SPIKY_LOSS_FACTOR,
    get_batch,
    has_nvidia_modelopt,
    stimer,
    train_valid_test_datasets_provider,
)

if has_nvidia_modelopt:
    from megatron.post_training.arguments import add_modelopt_args


def cdlm_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build a CDLMModel; same signature as gpt_builders.gpt_builder."""
    print_rank_0('building CDLM model ...')
    assert pre_process and post_process, "CDLM does not support pipeline parallelism"
    assert args.position_embedding_type == 'rope', "CDLM requires --position-embedding-type rope"
    assert args.num_experts is None, "CDLM does not support MoE yet"
    if config is None:
        config = core_transformer_config_from_args(args)
    transformer_layer_spec = get_cdlm_layer_spec(
        use_transformer_engine=args.transformer_impl == "transformer_engine",
        qk_layernorm=args.qk_layernorm,
        qk_l2_norm=args.qk_l2_norm,
        normalization=args.normalization,
        use_kitchen=config.use_kitchen,
    )
    cdlm_config = CDLMConfig(
        embed_dim=args.cdlm_embed_dim,
        gamma_0=args.cdlm_gamma_0,
        gamma_1=args.cdlm_gamma_1,
        selfcond=args.cdlm_selfcond,
        bias_warmup_steps=args.cdlm_bias_warmup_steps,
        reconst_bs_ema=args.cdlm_reconst_bs_ema,
        varilen_prob=args.cdlm_varilen_prob,
        head_chunk_tokens=args.cdlm_head_chunk_tokens,
        seed=args.seed,
    )
    return CDLMModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        cdlm_config=cdlm_config,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )


def loss_func(output_tensor, model=None):
    """Loss function.

    Args:
        output_tensor: (loss, stats) as returned by CDLMModel.forward.

    Returns:
        the loss for this micro-batch (the scheduler averages it over micro-batches) and a dict of
        reporting metrics; `(numerator, denominator)` pairs are reduced across micro-batches and
        data-parallel ranks.
    """
    args = get_args()
    loss, stats = output_tensor
    stats = dict(zip(STAT_NAMES, stats.detach()))

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    denominator = stats['denominator']
    report = {
        'lm loss': torch.stack([stats['nll'], denominator]),
        'nll reconst': torch.stack([stats['nll_reconst'], denominator]),
        'nll diffusion': torch.stack([stats['nll_diffusion'], denominator]),
        'nll prior': torch.stack([stats['nll_prior'], denominator]),
        'gamma_0': stats['gamma_0'].view(1),
        'gamma_1': stats['gamma_1'].view(1),
        'reconst_bs': stats['reconst_bs'].view(1),
    }
    return loss, report


def forward_step(data_iterator, model: CDLMModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (CDLMModel): The CDLM model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            None,
            labels=labels,
            loss_mask=loss_mask,
            iteration=getattr(args, 'curr_iteration', 0),
            num_microbatches=get_num_microbatches(),
        )

    return output_tensor, partial(loss_func, model=model)


def install_cdlm_optimizer_overrides():
    """Train the diffusion-process parameters (tagged `cdlm_aux`) with their own learning rate.

    Plaid uses lr 1e-2 for the embedding matrix, noise schedule and gamma bounds. Their weight
    decay is already 0: they are stored as 1-D parameters.
    """
    get_megatron_optimizer_config = training_module.get_megatron_optimizer_config

    def get_cdlm_optimizer_config(args):
        config, config_overrides = get_megatron_optimizer_config(args)
        assert args.decoupled_lr is None, "--decoupled-lr is not supported with CDLM"
        aux_config = copy.deepcopy(config)
        aux_config.lr = args.cdlm_aux_lr
        # Same schedule shape as the main learning rate.
        aux_config.min_lr = args.cdlm_aux_lr * (args.min_lr / args.lr) if args.lr else 0.0
        config_overrides = dict(config_overrides or {})
        config_overrides[ParamKey(attr="cdlm_aux")] = aux_config
        return config, config_overrides

    training_module.get_megatron_optimizer_config = get_cdlm_optimizer_config


def add_cdlm_args(parser):
    """CDLM (Plaid) arguments."""
    group = parser.add_argument_group(title='cdlm')
    group.add_argument('--cdlm-embed-dim', type=int, default=16,
                       help='Dimension of the diffused token embeddings.')
    group.add_argument('--cdlm-gamma-0', type=float, default=-3.0,
                       help='Initial log-SNR bound gamma(t=0).')
    group.add_argument('--cdlm-gamma-1', type=float, default=6.0,
                       help='Initial log-SNR bound gamma(t=1).')
    group.add_argument('--cdlm-selfcond', action='store_true', default=True,
                       help='Train with self-conditioning on 25%% of the samples.')
    group.add_argument('--no-cdlm-selfcond', action='store_false', dest='cdlm_selfcond',
                       help='Disable self-conditioning.')
    group.add_argument('--cdlm-bias-warmup-steps', type=int, default=5000,
                       help='Iterations over which the output prior is linearly warmed up.')
    group.add_argument('--cdlm-reconst-bs-ema', type=float, default=0.997,
                       help='EMA decay of the loss statistics that set the number of '
                       'reconstruction (t=0) samples.')
    group.add_argument('--cdlm-aux-lr', type=float, default=1e-2,
                       help='Learning rate of the embedding matrix, noise schedule and gamma '
                       'bounds.')
    group.add_argument('--cdlm-varilen-prob', type=float, default=0.01,
                       help='Probability of truncating a micro-batch to a random length.')
    group.add_argument('--cdlm-head-chunk-tokens', type=int, default=4096,
                       help='Tokens per chunk of the checkpointed vocabulary-sized output head.')
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    install_cdlm_optimizer_overrides()

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, cdlm_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_cdlm_args,
        store=store,
    )
