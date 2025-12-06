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
The basic idea is to add quantization operators at the input of all layers (including attn and linear)
and convert back to torch.bfloat16.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import torch
import torch.nn as nn

# Add project root to sys.path to import fake_quant_ops
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from olmo.config import ModelConfig, QuantActivationConfig
from olmo.model import OLMo, OLMoBlock, OLMoSequentialBlock, BufferCache
from olmo.exceptions import OLMoConfigurationError

log = logging.getLogger(__name__)


class CoatOLMoFakeSequentialBlock(OLMoSequentialBlock):
    """
    OLMo Sequential Block with fake quantization.
    Applies fake quantization to all linear layer inputs (att_proj, attn_out, ff_proj, ff_out).
    The basic idea is to add quantization operators at the input of all layers and convert back to torch.bfloat16.
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
        Forward pass with fake quantization on all linear layer inputs.
        Quantization is applied at the input of:
        1. att_proj (after norm)
        2. attn_out (after attention)
        3. ff_proj (after norm)
        4. ff_out (after activation)
        All quantized tensors are converted back to torch.bfloat16.
        """
        from fake_quant_ops.quant.mxfp import quant_dequant_tensor, quant_dequant_qkv
        
        # ========== Attention path ==========
        # Apply norm before attention
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                h = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                h = self.attn_norm(x)
        else:
            h = x
        
        # Quantize input to att_proj and convert back to bfloat16
        h_quant = quant_dequant_tensor(h, self.quant_format)
        h_quant = h_quant.to(torch.bfloat16)
        
        # Get QKV projections
        qkv = self.att_proj(h_quant)
        
        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        
        q, k, v = qkv.split(self.fused_dims, dim=-1)
        
        # Apply additional fake quantization to QKV if enabled
        # Note: quant_dequant_qkv already converts to bfloat16 internally
        if getattr(self.qargs, 'quant_qkv', False):
            qkv_format = "fp8_e4m3" if getattr(self.qargs, 'qkvbit', None) == "mxfp8e4m3" else "fp8_e5m2"
            q, k, v = quant_dequant_qkv(q, k, v, qkv_format)
            # Ensure bfloat16 dtype (quant_dequant_qkv already does this, but keep for consistency with coat_olmo.py)
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        
        # Compute attention scores manually to allow quantization before attn_out
        # Note: We cannot use self.attention() directly as it already applies attn_out
        B, T, C = q.size()
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        if self.config.rope:
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        # Re-assemble all head outputs side-by-side
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        cache = present
        
        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)
        
        # Quantize input to attn_out and convert back to bfloat16
        att_quant = quant_dequant_tensor(att, self.quant_format)
        att_quant = att_quant.to(torch.bfloat16)
        
        # Attention output projection
        att_out = self.attn_out(att_quant)
        
        # Add attention scores
        x = x + self.dropout(att_out)
        
        # ========== MLP path ==========
        og_x = x
        
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)
            else:
                x = self.ff_norm(x)
        
        # Quantize input to ff_proj and convert back to bfloat16
        x_quant = quant_dequant_tensor(x, self.quant_format)
        x_quant = x_quant.to(torch.bfloat16)
        
        # Feed-forward projection
        x = self.ff_proj(x_quant)
        
        # Activation function
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        
        # Quantize input to ff_out and convert back to bfloat16
        x_quant = quant_dequant_tensor(x, self.quant_format)
        x_quant = x_quant.to(torch.bfloat16)
        
        # Output projection
        x = self.ff_out(x_quant)
        
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
    to activations for research and evaluation purposes.
    Unlike real quantization, fake quantization does not accelerate training
    but allows studying quantization effects.
    
    The basic idea is to add quantization operators at the input of all layers
    (including attn and linear) and convert back to torch.bfloat16.
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
            # Handle block groups - replace blocks within each group
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
