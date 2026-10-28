# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Transformer layer specs for diffusion LMs: GPT layers with bidirectional self-attention."""

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec


def get_cdlm_layer_spec(
    use_transformer_engine: bool,
    qk_layernorm: bool = False,
    qk_l2_norm: bool = False,
    normalization: str = 'LayerNorm',
    use_kitchen: bool = False,
) -> ModuleSpec:
    """Dense GPT layer spec whose self-attention attends to every position.

    TE gets `no_mask` (TE turns it into `padding` for packed sequences); the local implementation
    gets `padding`, which with `attention_mask=None` is an unmasked softmax.
    """
    if use_transformer_engine:
        spec = get_gpt_layer_with_transformer_engine_spec(
            qk_layernorm=qk_layernorm, qk_l2_norm=qk_l2_norm, use_kitchen=use_kitchen
        )
        attn_mask_type = AttnMaskType.no_mask
    else:
        spec = get_gpt_layer_local_spec(
            qk_layernorm=qk_layernorm, normalization=normalization, use_kitchen=use_kitchen
        )
        attn_mask_type = AttnMaskType.padding
    spec.submodules.self_attention.params["attn_mask_type"] = attn_mask_type
    return spec
