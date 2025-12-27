# coding=utf-8
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

"""
CoatLlamaFake: Llama model with fake quantization using fake_quant_ops.
This mirrors the fp8 CoatLlama architecture but replaces real FP8 kernels
with simulated quant/dequant operators so the forward graph stays BF16.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import logging as hf_logging

# Project root so fake_quant_ops can be imported the same way as CoatOLMoFake
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

# Fake quant ops
try:
	from fake_quant_ops.quant.operators import quant_dequant_qkv, quant_dequant_tensor_with_backward
except ImportError:
	logging.warning("fake_quant_ops not found. Fake quantization will fail at runtime.")
	quant_dequant_tensor_with_backward = None
	quant_dequant_qkv = None

# Reuse attention logic without linear layers from the real CoatLlama implementation
from .coat_llama import (
	COAT_LLAMA_ATTENTION_CLASSES as COAT_LLAMA_BASE_ATTENTION_CLASSES,
	LlamaAttentionWithoutLinear,
	LlamaFlashAttention2WithoutLinear,
	LlamaSdpaAttentionWithoutLinear,
)

from ..activation.liger.cross_entropy import LigerForCausalLMLoss
from ..utils._fp8_quantization_config import QuantizationConfig

logger = hf_logging.get_logger(__name__)

__all__ = [
	"CoatLlamaFakeConfig",
	"CoatLlamaFakeBeforeAttentionResidual",
	"CoatLlamaFakeAfterAttentionResidual",
	"CoatLlamaFakeMLPResidual",
	"CoatLlamaFakeDecoderLayer",
	"CoatLlamaFakePreTrainedModel",
	"CoatLlamaFakeModel",
	"CoatLlamaFakeForCausalLM",
]


class CoatLlamaFakeConfig(LlamaConfig):
	model_type = "fp8_llama_fake"


def apply_fake_quant(x: torch.Tensor, qargs: QuantizationConfig, minus_exp: Optional[int] = None) -> torch.Tensor:
	"""Apply fake quant + dequant; fall back to identity if ops are unavailable."""
	if quant_dequant_tensor_with_backward is None:
		return x

	forward_format = getattr(qargs, "fabit", "E4M3")
	backward_format = getattr(qargs, "babit", "E5M2")
	backward_quantize = getattr(qargs, "backward_quantize", True)

	x_quant = quant_dequant_tensor_with_backward(
		x,
		forward_format=forward_format,
		backward_quantize=backward_quantize,
		backward_format=backward_format,
		minus_exp=minus_exp,
	)
	return x_quant.to(x.dtype)


class CoatLlamaFakeBeforeAttentionResidual(nn.Module):
	"""RMSNorm -> fake quant -> Q/K/V projections."""

	def __init__(self, config: CoatLlamaFakeConfig, qargs: QuantizationConfig, layer_idx: Optional[int] = None):
		super().__init__()
		self.config = config
		self.qargs = qargs
		self.layer_idx = layer_idx

		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads

		self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
		self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
		self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

		self.minus_exp = getattr(qargs, "minus_exp", None)

	def forward(self, residual_x: torch.Tensor, _: Optional[torch.Tensor], __: Optional[torch.Tensor], rmsnorm_weight):
		x = F.rms_norm(residual_x, rmsnorm_weight.shape, rmsnorm_weight, eps=self.config.rms_norm_eps)
		x_quant = apply_fake_quant(x, self.qargs, self.minus_exp)

		query_states = self.q_proj(x_quant)
		key_states = self.k_proj(x_quant)
		value_states = self.v_proj(x_quant)

		if getattr(self.qargs, "attn_quantize", False) and quant_dequant_qkv is not None:
			qkv_forward_format = getattr(self.qargs, "attn_quantize_forward_bit", "bf16")
			qkv_backward_format = getattr(self.qargs, "attn_quantize_backward_bit", None) or getattr(
				self.qargs, "babit", None
			)
			use_backward_quant = getattr(self.qargs, "backward_quantize", False)

			query_states, key_states, value_states = quant_dequant_qkv(
				query_states,
				key_states,
				value_states,
				forward_format=qkv_forward_format,
				backward_quantize=use_backward_quant,
				backward_format=qkv_backward_format,
				minus_exp=self.minus_exp,
			)
			dtype = x.dtype
			query_states = query_states.to(dtype)
			key_states = key_states.to(dtype)
			value_states = value_states.to(dtype)

		return residual_x, query_states, key_states, value_states


class CoatLlamaFakeAfterAttentionResidual(nn.Module):
	"""Fake-quantized attention output projection + residual add."""

	def __init__(self, config: CoatLlamaFakeConfig, qargs: QuantizationConfig, layer_idx: Optional[int] = None):
		super().__init__()
		self.config = config
		self.qargs = qargs
		self.layer_idx = layer_idx

		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

		self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
		self.minus_exp = getattr(qargs, "minus_exp", None)

	def forward(self, residual_x: torch.Tensor, attn_x: torch.Tensor):
		attn_quant = apply_fake_quant(attn_x, self.qargs, self.minus_exp)
		out = self.o_proj(attn_quant)
		return residual_x + out, None, None


class CoatLlamaFakeMLPResidual(nn.Module):
	"""RMSNorm -> fake quant -> Gate/Up -> Silu -> fake quant -> Down -> residual."""

	def __init__(
		self,
		config: CoatLlamaFakeConfig,
		qargs: QuantizationConfig,
		layer_idx: Optional[int] = None,
		hidden_size: Optional[int] = None,
	):
		super().__init__()
		self.config = config
		self.qargs = qargs
		self.layer_idx = layer_idx
		self.hidden_size = config.hidden_size
		self.intermediate_size = config.intermediate_size

		self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
		self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
		self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

		self.act_fn = F.silu
		self.minus_exp = getattr(qargs, "minus_exp", None)

	def forward(self, residual_x: torch.Tensor, _: Optional[torch.Tensor], __: Optional[torch.Tensor], rmsnorm_weight):
		x_norm = F.rms_norm(residual_x, rmsnorm_weight.shape, rmsnorm_weight, eps=self.config.rms_norm_eps)
		x_quant = apply_fake_quant(x_norm, self.qargs, self.minus_exp)

		gate = self.gate_proj(x_quant)
		up = self.up_proj(x_quant)
		act_out = self.act_fn(gate) * up

		act_quant = apply_fake_quant(act_out, self.qargs, self.minus_exp)
		down = self.down_proj(act_quant)
		return residual_x + down, None, None


COAT_LLAMA_FAKE_ATTENTION_CLASSES = COAT_LLAMA_BASE_ATTENTION_CLASSES


def _identity_quantize_input(hidden_states: torch.Tensor):
	return hidden_states, None, None


def _identity_quantize_output(
	hidden_states: torch.Tensor, quant_hidden_states: Optional[torch.Tensor], scale_hidden_states: Optional[torch.Tensor]
):
	return hidden_states


class CoatLlamaFakeDecoderLayer(nn.Module):
	def __init__(self, config: CoatLlamaFakeConfig, layer_idx: int):
		super().__init__()
		self.layer_idx = layer_idx
		self.hidden_size = config.hidden_size

		self.self_attn = COAT_LLAMA_FAKE_ATTENTION_CLASSES[config._attn_implementation](
			config=config, layer_idx=layer_idx
		)

		self.qargs = QuantizationConfig(**config.coat_fp8_args)
		self.BeforeAttention = CoatLlamaFakeBeforeAttentionResidual(config, self.qargs, layer_idx)
		self.AfterAttention = CoatLlamaFakeAfterAttentionResidual(config, self.qargs, layer_idx)
		self.MLPResidual = CoatLlamaFakeMLPResidual(config, self.qargs, layer_idx, self.hidden_size)

		self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(
		self,
		hidden_states: torch.Tensor,
		quant_hidden_states: Optional[torch.Tensor],
		scale_hidden_states: Optional[torch.Tensor],
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: Optional[bool] = False,
		use_cache: Optional[bool] = False,
		cache_position: Optional[torch.LongTensor] = None,
		position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
		**kwargs,
	) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

		residual, query_states, key_states, value_states = self.BeforeAttention(
			hidden_states, quant_hidden_states, scale_hidden_states, self.input_layernorm.weight
		)

		hidden_states, self_attn_weights, present_key_value = self.self_attn(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=use_cache,
			cache_position=cache_position,
			position_embeddings=position_embeddings,
			**kwargs,
		)

		hidden_states, quant_hidden_states, scale_hidden_states = self.AfterAttention(residual, hidden_states)
		hidden_states, quant_hidden_states, scale_hidden_states = self.MLPResidual(
			hidden_states, quant_hidden_states, scale_hidden_states, self.post_attention_layernorm.weight
		)

		outputs = ((hidden_states, quant_hidden_states, scale_hidden_states),)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)

		return outputs


class CoatLlamaFakePreTrainedModel(PreTrainedModel):
	config_class = CoatLlamaFakeConfig
	base_model_prefix = "model"
	supports_gradient_checkpointing = True
	_no_split_modules = ["LlamaDecoderLayer"]
	_skip_keys_device_placement = ["past_key_values"]
	_supports_flash_attn_2 = True
	_supports_sdpa = True
	_supports_cache_class = True
	_supports_quantized_cache = True
	_supports_static_cache = True

	def _init_weights(self, module):
		std = self.config.initializer_range
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()


class CoatLlamaFakeModel(CoatLlamaFakePreTrainedModel):
	"""Coat Transformer decoder with fake quantization."""

	def __init__(self, config: CoatLlamaFakeConfig):
		super().__init__(config)
		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size

		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
		self.layers = nn.ModuleList(
			[CoatLlamaFakeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
		)
		self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.rotary_emb = LlamaRotaryEmbedding(config=config)
		self.gradient_checkpointing = False

		# Identity hooks to mimic the fp8 API surface
		self.quantize_input_before_block = _identity_quantize_input
		self.quantize_output_after_block = _identity_quantize_output

		self.post_init()

	def get_input_embeddings(self):
		return self.embed_tokens

	def set_input_embeddings(self, value):
		self.embed_tokens = value

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
	) -> Union[Tuple, BaseModelOutputWithPast]:

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		if input_ids is None and inputs_embeds is None:
			raise ValueError("You must specify either input_ids or inputs_embeds")

		if self.gradient_checkpointing and self.training and use_cache:
			logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
			use_cache = False

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		return_legacy_cache = False
		if use_cache and not isinstance(past_key_values, Cache):
			return_legacy_cache = True
			if past_key_values is None:
				past_key_values = DynamicCache()
			else:
				past_key_values = DynamicCache.from_legacy_cache(past_key_values)
				logger.warning_once(
					"Passing `past_key_values` as tuple of tuples is deprecated. It will be removed in v4.47."
				)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		causal_mask = self._update_causal_mask(
			attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
		)

		hidden_states = inputs_embeds

		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None

		hidden_states, quant_hidden_states, scale_hidden_states = self.quantize_input_before_block(hidden_states)

		for decoder_layer in self.layers:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			if self.gradient_checkpointing and self.training:
				layer_outputs = self._gradient_checkpointing_func(
					decoder_layer.__call__,
					hidden_states,
					quant_hidden_states,
					scale_hidden_states,
					causal_mask,
					position_ids,
					past_key_values,
					output_attentions,
					use_cache,
					cache_position,
					position_embeddings,
				)
			else:
				layer_outputs = decoder_layer(
					hidden_states,
					quant_hidden_states,
					scale_hidden_states,
					attention_mask=causal_mask,
					position_ids=position_ids,
					past_key_value=past_key_values,
					output_attentions=output_attentions,
					use_cache=use_cache,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
				)

			hidden_states, quant_hidden_states, scale_hidden_states = layer_outputs[0]

			if use_cache:
				next_decoder_cache = layer_outputs[2 if output_attentions else 1]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

		hidden_states = self.quantize_output_after_block(hidden_states, quant_hidden_states, scale_hidden_states)
		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		next_cache = next_decoder_cache if use_cache else None
		if return_legacy_cache:
			next_cache = next_cache.to_legacy_cache()

		if not return_dict:
			return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)

	_update_causal_mask = LlamaModel._update_causal_mask


class CoatLlamaFakeForCausalLM(CoatLlamaFakePreTrainedModel, GenerationMixin):
	_tied_weights_keys = ["lm_head.weight"]

	def __init__(self, config: CoatLlamaFakeConfig):
		super().__init__(config)
		self.model = CoatLlamaFakeModel(config)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		self.post_init()

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def set_input_embeddings(self, value):
		self.model.embed_tokens = value

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def set_decoder(self, decoder):
		self.model = decoder

	def get_decoder(self):
		return self.model

	@property
	@lru_cache
	def loss_function(self):
		return LigerForCausalLMLoss

	forward = LlamaForCausalLM.forward
	prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation


def make_state_dict_compatible(state_dict: dict[str, torch.Tensor]):
	compatible_state_dict = {}

	for key, value in state_dict.items():
		if "self_attn.q_proj" in key:
			new_key = key.replace("self_attn.q_proj", "BeforeAttention.q_proj")
		elif "self_attn.k_proj" in key:
			new_key = key.replace("self_attn.k_proj", "BeforeAttention.k_proj")
		elif "self_attn.v_proj" in key:
			new_key = key.replace("self_attn.v_proj", "BeforeAttention.v_proj")
		elif "self_attn.o_proj" in key:
			new_key = key.replace("self_attn.o_proj", "AfterAttention.o_proj")
		elif "mlp.gate_proj" in key:
			new_key = key.replace("mlp.gate_proj", "MLPResidual.gate_proj")
		elif "mlp.up_proj" in key:
			new_key = key.replace("mlp.up_proj", "MLPResidual.up_proj")
		elif "mlp.down_proj" in key:
			new_key = key.replace("mlp.down_proj", "MLPResidual.down_proj")
		else:
			new_key = key

		compatible_state_dict[new_key] = value

	return compatible_state_dict


AutoConfig.register("fp8_llama_fake", CoatLlamaFakeConfig)
AutoModel.register(CoatLlamaFakeConfig, CoatLlamaFakeModel)
AutoModelForCausalLM.register(CoatLlamaFakeConfig, CoatLlamaFakeForCausalLM)
