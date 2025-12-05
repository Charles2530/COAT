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
CoatOLMoFake: OLMo model with fake quantization using fake_quant_ops.
This module provides simulated quantization for research and evaluation purposes.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path to import fake_quant_ops
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from olmo.config import ModelConfig, QuantActivationConfig
from olmo.model import OLMo, OLMoBlock, OLMoSequentialBlock, BufferCache
from olmo.exceptions import OLMoConfigurationError

log = logging.getLogger(__name__)


class FakeQuantizedLinear(nn.Module):
    """
    A Linear layer wrapper with fake quantization.
    Uses fake_quant_ops for simulated quantization.
    """
    
    def __init__(
        self, 
        linear: nn.Linear, 
        quant_format: str = 'fp8_e5m2',
        quantize_input: bool = True,
        quantize_weight: bool = True,
        quantize_output: bool = True
    ):
        super().__init__()
        self.linear = linear
        self.quant_format = quant_format
        self.quantize_input = quantize_input
        self.quantize_weight = quantize_weight
        self.quantize_output = quantize_output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input
        if self.quantize_input:
            from fake_quant_ops.quant.mxfp import quant_dequant_tensor
            x = quant_dequant_tensor(x, self.quant_format)
        
        # Quantize weight
        if self.quantize_weight:
            from fake_quant_ops.quant.mxfp import quant_dequant_tensor
            weight = quant_dequant_tensor(self.linear.weight, self.quant_format)
            # Use quantized weight for computation
            output = F.linear(x, weight, self.linear.bias)
        else:
            output = self.linear(x)
        
        # Quantize output
        if self.quantize_output:
            from fake_quant_ops.quant.mxfp import quant_dequant_tensor
            output = quant_dequant_tensor(output, self.quant_format)
        
        return output


class CoatOLMoFakeSequentialBlock(OLMoSequentialBlock):
    """
    OLMo Sequential Block with fake quantization.
    Applies fake quantization to QKV, attention output, and MLP layers.
    """
    
    def __init__(self, layer_id: int, config: ModelConfig, qargs: QuantActivationConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.qargs = qargs
        
        # Determine quantization format
        if hasattr(qargs, 'fabit') and qargs.fabit:
            if qargs.fabit == 'E4M3':
                self.quant_format = 'fp8_e4m3'
            elif qargs.fabit == 'E5M2':
                self.quant_format = 'fp8_e5m2'
            else:
                self.quant_format = 'fp8_e5m2'  # default
        else:
            self.quant_format = 'fp8_e5m2'  # default
        
        # Wrap linear layers with fake quantization
        # Replace att_proj
        original_att_proj = self.att_proj
        self.att_proj = FakeQuantizedLinear(
            original_att_proj,
            quant_format=self.quant_format,
            quantize_input=True,
            quantize_weight=True,
            quantize_output=True
        )
        
        # Replace ff_proj
        original_ff_proj = self.ff_proj
        self.ff_proj = FakeQuantizedLinear(
            original_ff_proj,
            quant_format=self.quant_format,
            quantize_input=True,
            quantize_weight=True,
            quantize_output=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass with fake quantization on QKV tensors.
        """
        # Apply norm before attention
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                h = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                h = self.attn_norm(x)
        else:
            h = x
        
        # Get QKV projections (already wrapped with fake quantization)
        qkv = self.att_proj(h)
        
        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        
        q, k, v = qkv.split(self.fused_dims, dim=-1)
        
        # Apply additional fake quantization to QKV if enabled
        if getattr(self.qargs, 'quant_qkv', False):
            from fake_quant_ops.quant.mxfp import quant_dequant_qkv
            qkv_format = "fp8_e4m3" if getattr(self.qargs, 'qkvbit', None) == "mxfp8e4m3" else "fp8_e5m2"
            q, k, v = quant_dequant_qkv(q, k, v, qkv_format)
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        
        # Get attention scores
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(
                self.attention,
                q, k, v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )
        else:
            att, cache = self.attention(
                q, k, v,
                attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )
        
        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)
        
        # Add attention scores
        x = x + self.dropout(att)
        
        # MLP (already wrapped with fake quantization)
        og_x = x
        
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)
            else:
                x = self.ff_norm(x)
        
        x = self.ff_proj(x)
        
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        
        x = self.ff_out(x)
        
        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)
            else:
                x = self.ff_norm(x)
        
        x = og_x + self.dropout(x)
        
        return x, cache


class CoatOLMoFake(OLMo):
    """
    OLMo model with fake quantization using fake_quant_ops.
    
    This class wraps the standard OLMo model and applies fake quantization
    to activations and weights for research and evaluation purposes.
    Unlike real quantization, fake quantization does not accelerate training
    but allows studying quantization effects.
    """
    
    def __init__(self, config: ModelConfig, qargs: QuantActivationConfig, init_params: bool = True):
        # Initialize base OLMo model
        super().__init__(config, init_params=init_params)
        self.qargs = qargs
        
        # Get cache from base model
        cache = self._OLMo__cache if hasattr(self, '_OLMo__cache') else BufferCache()
        
        # Replace blocks with fake quantized blocks
        if hasattr(self.transformer, 'blocks'):
            blocks = []
            for i, block in enumerate(self.transformer.blocks):
                if isinstance(block, OLMoSequentialBlock):
                    fake_block = CoatOLMoFakeSequentialBlock(
                        i, config, qargs, cache
                    )
                    # Copy weights from original block
                    fake_block.load_state_dict(block.state_dict(), strict=False)
                    blocks.append(fake_block)
                else:
                    blocks.append(block)
            self.transformer.blocks = nn.ModuleList(blocks)
        elif hasattr(self.transformer, 'block_groups'):
            # Handle block groups
            for block_group in self.transformer.block_groups:
                for i, block in enumerate(block_group):
                    if isinstance(block, OLMoSequentialBlock):
                        fake_block = CoatOLMoFakeSequentialBlock(
                            block.layer_id, config, qargs, cache
                        )
                        fake_block.load_state_dict(block.state_dict(), strict=False)
                        block_group[i] = fake_block
        
        log.info("Initialized CoatOLMoFake with fake quantization using fake_quant_ops")
        log.info(f"Quantization format: {getattr(qargs, 'fabit', 'fp8_e5m2')}")
        if getattr(qargs, 'quant_qkv', False):
            log.info(f"QKV fake quantization enabled: {getattr(qargs, 'qkvbit', 'fp8_e5m2')}")

