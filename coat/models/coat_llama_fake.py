# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
MXFP4 fake-quantized Llama model.

This module mirrors the reference Llama implementation but injects MXFP4
fake quantization into the MLP (gate/up/down projections). Attention stays
identical to the upstream implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from typing import Optional, Union, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fake_quant_ops.quant.operators import quant_dequant_tensor_with_backward

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    Cache,
    DynamicCache,
    LlamaAttention,
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaModel,
    LlamaForCausalLM,
)



class FakeQuantLinear(nn.Linear):
    """Linear layer that fake-quantizes inputs and weights before matmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LlamaConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

        # Quantization formats from config with MXFP4 defaults.
        self.forward_format = getattr(config, "fabit", "mxfp4_e2m1")
        self.backward_format = getattr(config, "babit", "mxfp4_e2m1")
        self.backward_quantize = bool(getattr(config, "backward_quantize", False))
        self.minus_exp = getattr(config, "minus_exp", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = quant_dequant_tensor_with_backward(
            x,
            forward_format=self.forward_format,
            minus_exp=self.minus_exp,
            backward_quantize=self.backward_quantize,
            backward_format=self.backward_format,
        ).to(torch.bfloat16)

        w_q = quant_dequant_tensor_with_backward(
            self.weight,
            forward_format=self.forward_format,
            minus_exp=self.minus_exp,
            backward_quantize=self.backward_quantize,
            backward_format=self.backward_format,
        ).to(torch.bfloat16)

        return F.linear(x_q, w_q, self.bias)


class LlamaMLPFake(nn.Module):
    """MLP with fake-quantized gate/up/down projections."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = FakeQuantLinear(
            self.hidden_size, self.intermediate_size, config, bias=config.mlp_bias
        )
        self.up_proj = FakeQuantLinear(
            self.hidden_size, self.intermediate_size, config, bias=config.mlp_bias
        )
        self.down_proj = FakeQuantLinear(
            self.intermediate_size, self.hidden_size, config, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaDecoderLayerFake(nn.Module):
    """Decoder layer with fake-quantized MLP; attention left unquantized."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Keep attention identical to HF implementation (no fake quantization on attn path).
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLPFake(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Any:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModelFake(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerFake(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()


class LlamaForCausalLMFake(LlamaForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModelFake(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()


__all__ = [
    "FakeQuantLinear",
    "LlamaMLPFake",
    "LlamaDecoderLayerFake",
    "LlamaModelFake",
    "LlamaForCausalLMFake",
]
