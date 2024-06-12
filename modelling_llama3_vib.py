# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
	PyTorch LLaMA model.
	Modified in Bonsai to allow for "virtually" instantiating sub-models and also for computing the
	module-level statistics used for initiating priors

"""
import time
import math
from typing import List, Optional, Tuple, Union
import gc
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import time
from transformers.activations import ACT2FN
from utils.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
	BaseModelOutputWithPast,
	CausalLMOutputWithPast,
	QuestionAnsweringModelOutput,
	SequenceClassifierOutputWithPast,
	TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	is_flash_attn_2_available,
	is_flash_attn_greater_or_equal_2_10,
	logging,
	replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from vib_layer_lay import InformationBottleneck

if is_flash_attn_2_available():
	from flash_attn import flash_attn_func, flash_attn_varlen_func
	from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
	seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
	indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
	max_seqlen_in_batch = seqlens_in_batch.max().item()
	cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
	return (
		indices,
		cu_seqlens,
		max_seqlen_in_batch,
	)



class LlamaRMSNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-6):
		"""
		LlamaRMSNorm is equivalent to T5LayerNorm
		"""
		super().__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.variance_epsilon = eps
		self.mean_variance = None

	def forward(self, hidden_states,hidden_z=None):
		input_dtype = hidden_states.dtype
		if hidden_z is not None:
			remaining_index = torch.where(~hidden_z.eq(0))[0]
			compressed_hidden_states = torch.index_select( hidden_states, dim=-1, index=remaining_index)
			compressed_weight = self.weight[remaining_index]
			normalized_shape = len(remaining_index)
			variance = compressed_hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
			compressed_hidden_states = compressed_hidden_states * torch.rsqrt(variance + self.variance_epsilon)
			if compressed_weight.dtype in [torch.float16, torch.bfloat16]:
				compressed_hidden_statess = compressed_hidden_states.to(compressed_weight.dtype)
			normed_states= compressed_weight * compressed_hidden_states
			output = hidden_states.clone()
			output[:, :, remaining_index] = normed_states
			return output
		else:
			variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
			hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
		
		return self.weight * hidden_states.to(input_dtype)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

class LlamaRotaryEmbedding(nn.Module):
	def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
		super().__init__()
		self.scaling_factor = scaling_factor
		self.dim = dim
		self.max_position_embeddings = max_position_embeddings
		self.base = base
		inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
		self.register_buffer("inv_freq", inv_freq, persistent=False)
		# For BC we register cos and sin cached
		self.max_seq_len_cached = max_position_embeddings

	@torch.no_grad()
	def forward(self, x, position_ids):
		# x: [bs, num_attention_heads, seq_len, head_size]
		inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
		position_ids_expanded = position_ids[:, None, :].float()
		# Force float32 since bfloat16 loses precision on long contexts
		# See https://github.com/huggingface/transformers/pull/29285
		device_type = x.device.type
		device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
		with torch.autocast(device_type=device_type, enabled=False):
			freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
			emb = torch.cat((freqs, freqs), dim=-1)
			cos = emb.cos()
			sin = emb.sin()		
		return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
	
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
	"""LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

	def forward(self, x, position_ids):
		# difference to the original RoPE: a scaling factor is aplied to the position ids
		position_ids = position_ids.float() / self.scaling_factor
		cos, sin = super().forward(x, position_ids)
		return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
	"""LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

	def forward(self, x, position_ids):
		# difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
		seq_len = torch.max(position_ids) + 1
		if seq_len > self.max_position_embeddings:
			base = self.base * (
				(self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
			) ** (self.dim / (self.dim - 2))
			inv_freq = 1.0 / (
				base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
			)
			self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

		cos, sin = super().forward(x, position_ids)
		return cos, sin

def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
	cos = cos.unsqueeze(unsqueeze_dim)
	sin = sin.unsqueeze(unsqueeze_dim)
	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	# del cos,sin
	# gc.collect()
	# torch.cuda.empty_cache()
	return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
	"""
	This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
	num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
	"""
	batch, num_key_value_heads, slen, head_dim = hidden_states.shape
	if n_rep == 1:
		return hidden_states
	hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
	return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
	
class LlamaAttention(nn.Module):
	"""Multi-headed attention from 'Attention Is All You Need' paper"""

	def __init__(self, config: LlamaConfig,layer_idx):
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx
		if layer_idx is None:
			logger.warning_once(
				f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
				"lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
				"when creating this class."
			)

		self.attention_dropout = config.attention_dropout
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.hidden_size // self.num_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.max_position_embeddings = config.max_position_embeddings
		self.rope_theta = config.rope_theta
		self.is_causal = True
		if (self.head_dim * self.num_heads) != self.hidden_size:
			raise ValueError(
				f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
				f" and `num_heads`: {self.num_heads})."
			)
		if self.config.vib_layers == True:
			self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
			self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
			self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
			self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
			self.ib_1= InformationBottleneck(self.num_key_value_heads,kl_mult=config.att_mul)
			
		else:
			self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
			self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
			self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
			self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
			
		self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings,base=self.rope_theta)

	def _init_rope(self):
		if self.config.rope_scaling is None:
			self.rotary_emb = LlamaRotaryEmbedding(
				self.head_dim,
				max_position_embeddings=self.max_position_embeddings,
				base=self.rope_theta,
			)
		else:
			scaling_type = self.config.rope_scaling["type"]
			scaling_factor = self.config.rope_scaling["factor"]
			if scaling_type == "linear":
				self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
					self.head_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factor=scaling_factor,
					base=self.rope_theta,
				)
			elif scaling_type == "dynamic":
				self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
					self.head_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factor=scaling_factor,
					base=self.rope_theta,
				)
			else:
				raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
		

	def get_num_rem_weights(self,hidden_mask_size):
		mask_1= self.ib_1.get_mask_hard()
		num_states_rem_1 =torch.sum((mask_1 == 1).int()) 
		return ((self.head_dim*num_states_rem_1*2 * hidden_mask_size) *  (1+ self.num_key_value_groups)) #no bias

	def get_sparsity(self, hidden_mask_sparse):
		att_sp= (self.head_dim * (self.ib_1.sparse/1e3) * 2 * (hidden_mask_sparse/1e3)) * (1+ self.num_key_value_groups)#no bias,
		return att_sp #no bias

	def get_kld_loss(self,hidden_mask,kl_fac):
		kl= (self.ib_1.kld*kl_fac + hidden_mask.kld*kl_fac)
		return kl

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Tuple[torch.Tensor]] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		cache_position:Optional[torch.LongTensor] = None,
		hidden_mask=None,
		
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		
		bsz, q_len, _ = hidden_states.size()
		query_states = self.q_proj(hidden_states)
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

		cos, sin = self.rotary_emb(value_states, position_ids)
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

		if past_key_value is not None:
			# sin and cos are specific to RoPE models; cache_position needed for the static cache
			cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
			key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		key_states = repeat_kv(key_states, self.num_key_value_groups)
		value_states = repeat_kv(value_states, self.num_key_value_groups)
		attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

		if attention_mask is not None:  # no matter the length, we just slice it
			causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
			attn_weights = attn_weights + causal_mask

		# upcast attention to fp32
		attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
		attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
		attn_output = torch.matmul(attn_weights, value_states)

		if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
			raise ValueError(
				f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
				f" {attn_output.size()}"
			)
		if self.config.vib_layers==True:
			attn_output = attn_output.permute(0,2,3,1).reshape(bsz* q_len* self.head_dim, self.num_key_value_groups,self.num_key_value_heads).reshape(bsz* q_len* self.head_dim*self.num_key_value_groups,-1) #bsz,q_len ,self.num_key_value_heads, self.num_key_value_groups,self.head_dim
			attn_output= self.ib_1(attn_output).to(attn_output.dtype)
			attn_output = attn_output.reshape(bsz,q_len,self.head_dim,self.num_key_value_heads* self.num_key_value_groups).permute(0,3,1,2) #bsz,self.num_heads,q_len, self.head_dim

					
		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
		attn_output = self.o_proj(attn_output)
		if self.config.vib_layers==True: 
			attn_output= attn_output.reshape(bsz* q_len,self.hidden_size)
			attn_output = hidden_mask(attn_output).to(attn_output.dtype)
			attn_output= attn_output.reshape(bsz, q_len, self.hidden_size)

		if not output_attentions:
			attn_weights = None
		
		return attn_output, attn_weights, past_key_value

class LlamaSdpaAttention(LlamaAttention):
	"""
	Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
	`LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
	SDPA API.
	"""

	# Adapted from LlamaAttention.forward
	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		cache_position: Optional[torch.LongTensor] = None,
		hidden_mask= None
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		if output_attentions:
			# TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
			logger.warning_once(
				"LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
				'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
			)
			return super().forward(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=output_attentions,
				use_cache=use_cache,
				cache_position=cache_position,
			)

		bsz, q_len, last_dim = hidden_states.size()
		query_states = self.q_proj(hidden_states)
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		kv_seq_len = key_states.shape[-2]
		cos, sin = self.rotary_emb(value_states, position_ids)
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
		if past_key_value is not None:
			# sin and cos are specific to RoPE models; cache_position needed for the static cache
			cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
			key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		key_states = repeat_kv(key_states, self.num_key_value_groups)
		value_states = repeat_kv(value_states, self.num_key_value_groups)

		causal_mask = attention_mask
		if attention_mask is not None:
			causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

		# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
		# Reference: https://github.com/pytorch/pytorch/issues/112577.
		if query_states.device.type == "cuda" and causal_mask is not None:
			query_states = query_states.contiguous()
			key_states = key_states.contiguous()
			value_states = value_states.contiguous()

		# We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
		# in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
		is_causal = True if causal_mask is None and q_len > 1 else False

		attn_output = torch.nn.functional.scaled_dot_product_attention(
			query_states,
			key_states,
			value_states,
			attn_mask=causal_mask,
			dropout_p=self.attention_dropout if self.training else 0.0,
			is_causal=is_causal,
		)
		if self.config.vib_layers==True:
			attn_output = attn_output.permute(0,2,3,1).reshape(bsz* q_len* self.head_dim, self.num_key_value_groups,self.num_key_value_heads).reshape(bsz* q_len* self.head_dim*self.num_key_value_groups,-1) #bsz,q_len ,self.num_key_value_heads, self.num_key_value_groups,self.head_dim
			attn_output= self.ib_1(attn_output).to(attn_output.dtype)
			attn_output = attn_output.reshape(bsz,q_len,self.head_dim,self.num_key_value_heads* self.num_key_value_groups).permute(0,3,1,2)  #bsz,self.num_heads,q_len, self.head_dim

		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.view(bsz, q_len, self.hidden_size)

		attn_output = self.o_proj(attn_output)
		if self.config.vib_layers==True: 
			attn_output= attn_output.reshape(bsz* q_len,self.hidden_size)
			attn_output = hidden_mask(attn_output).to(attn_output.dtype)
			attn_output= attn_output.reshape(bsz, q_len, self.hidden_size)
			
		return attn_output, None, past_key_value

LLAMA_ATTENTION_CLASSES = {
	#"eager": LlamaAttention,
	#"flash_attention_2": LlamaFlashAttention2,
	"sdpa": LlamaSdpaAttention,
}

import numpy as np

class LlamaMLP(nn.Module):
	def __init__(
		self,
		config,		
	):
		super().__init__()
		self.config = config
		self.hidden_size = config.hidden_size
		self.intermediate_size = config.intermediate_size
		if config.vib_layers==True:
			self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
			self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
			self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
			self.ib_1= InformationBottleneck(self.intermediate_size,kl_mult=config.inter_mul)			
		else:
			self.gate_proj = nn.Linear(self.hidden_size,self.intermediate_size, bias=False)
			self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
			self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
			
		self.act_fn = ACT2FN[config.hidden_act]
		
		
	def get_num_rem_weights(self,hidden_mask_size):
		mask_1 = self.ib_1.get_mask_hard()
		num_states_rem_1 = torch.sum((mask_1 == 1).int())
		rem= ((2*num_states_rem_1 * hidden_mask_size) + (hidden_mask_size * num_states_rem_1)) #no bias
		return rem
	
	

	def get_sparsity(self,hidden_mask_sparse):
		att_sp= (3*(self.ib_1.sparse/1e3) * (hidden_mask_sparse/1e3)) 		
		return att_sp

	def get_kld_loss(self,hidden_mask,kl_fac):
		kl=((self.ib_1.kld*kl_fac) + (hidden_mask.kld*kl_fac))		
		return kl

	def forward(self, x,hidden_mask):
		bsz,seq,_ = x.size()   
		if self.config.vib_layers==True: 
			intermed_result= (self.act_fn(self.gate_proj(x)) * self.up_proj(x)).reshape(bsz*seq,-1)
			intermed_result = self.ib_1(intermed_result).to(intermed_result.dtype)			
			intermed_result= intermed_result.reshape(bsz,seq,-1)
		else:
			intermed_result = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
		intermed_result=self.down_proj(intermed_result)
		
		dim =intermed_result.size(2)
		if self.config.vib_layers==True: #only during finetuning
			intermed_result= intermed_result.reshape(bsz*seq,dim)
			intermed_result = hidden_mask(intermed_result).to(intermed_result.dtype) 
			intermed_result= intermed_result.reshape(bsz,seq,dim)			
		return intermed_result


class LlamaDecoderLayer(nn.Module):
	def __init__(self, config: LlamaConfig, layer_idx:int = -1):
		super().__init__()
		self.hidden_size = config.hidden_size
		self.config=config
		self.intermediate_size = config.intermediate_size
		#self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
		self.config._attn_implementation = 'sdpa'
		#print("\n att=",config._attn_implementation,self.config._attn_implementation )
		self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
		self.mlp = LlamaMLP(config)
		self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		

	def get_kld_loss(self,hidden_mask,kl_fac):
		return self.self_attn.get_kld_loss(hidden_mask,kl_fac) + self.mlp.get_kld_loss(hidden_mask,kl_fac)

	def get_pars(self):
		inter_pars = 3*(self.intermediate_size * self.hidden_size) 
		atten_pars= ((2* self.hidden_size * self.hidden_size) + (self.hidden_size * 2 * self.self_attn.num_key_value_heads * self.self_attn.head_dim))   
		layer_norm=  2* self.hidden_size  
		return (inter_pars+atten_pars+layer_norm)
		
	
	def get_num_rem_weights(self,hidden_mask_size): 
		if self.mlp is not None:
			mlp_out_rem = self.mlp.get_num_rem_weights(hidden_mask_size)
		else:
			mlp_out_rem=0
		
		if self.self_attn is not None:
			att_out= self.self_attn.get_num_rem_weights(hidden_mask_size)
		else:
			att_out= 0
		layer_norm= 2 * hidden_mask_size
		return att_out, (mlp_out_rem + layer_norm) 

	def get_sparsity(self,hidden_mask_sparse): 
		if self.mlp is not None:
			out_sparse= self.mlp.get_sparsity(hidden_mask_sparse) 
		else:
			out_sparse = 0.0
		if self.self_attn is not None:
			attn_sparse=  self.self_attn.get_sparsity(hidden_mask_sparse)
		else:
			attn_sparse=0.0
		layer_norm= 2 * hidden_mask_sparse.detach()/1e6
		return attn_sparse,(out_sparse +layer_norm) 

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Tuple[torch.Tensor]] = None,
		output_attentions: Optional[bool] = False,
		use_cache: Optional[bool] = False,
		cache_position: Optional[torch.LongTensor] = None,
		hidden_mask= None
	) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
		"""
		Args:
			hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
			attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
				`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
			output_attentions (`bool`, *optional*):
				Whether or not to return the attentions tensors of all attention layers. See `attentions` under
				returned tensors for more detail.
			use_cache (`bool`, *optional*):
				If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
				(see `past_key_values`).
			past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
		"""
		hidden_z= hidden_mask.get_mask_hard() if self.config.vib_layers==True else None #self.mlp.ib_2.get_mask_hard() if self.config.vib_layers==True else None
		residual = hidden_states
		if self.config.vib_layers==True: 
			hidden_states = self.input_layernorm(hidden_states,hidden_z)			
		else:
			#start_nor= time.time()
			hidden_states = self.input_layernorm(hidden_states)
			#lay_time= time.time() - start_nor
		
		# Self Attention
		#st_att= time.time()
		if self.self_attn is not None: #changed - added
			hidden_states, self_attn_weights, present_key_value = self.self_attn(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=output_attentions,
				use_cache=use_cache,
				cache_position=cache_position,
				hidden_mask=hidden_mask,
			)
			#att_time= time.time() - st_att 
			hidden_states = residual + hidden_states
		
			# Fully Connected
			residual = hidden_states
			
			if self.config.vib_layers==True:				
				hidden_states = self.post_attention_layernorm(hidden_states,hidden_z)				
			else:
				#start_norm= time.time()
				hidden_states = self.post_attention_layernorm(hidden_states)
				#lay_time += (time.time() - start_norm)
		
		#st_mlp= time.time()
		hidden_states = self.mlp(hidden_states, hidden_mask=hidden_mask)
		#mlp_time= time.time() - st_mlp
		hidden_states = residual + hidden_states		
		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)		

		return outputs #, lay_time, att_time, mlp_time


LLAMA_START_DOCSTRING = r"""
	This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
	library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
	etc.)

	This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
	Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
	and behavior.

	Parameters:
		config ([`LlamaConfig`]):
			Model configuration class with all the parameters of the model. Initializing with a config file does not
			load the weights associated with the model, only the configuration. Check out the
			[`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
	"The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
	LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
	config_class = LlamaConfig
	base_model_prefix = "model"
	supports_gradient_checkpointing = True
	_no_split_modules = ["LlamaDecoderLayer"]
	_keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

	def _init_weights(self, module):
		std = self.config.initializer_range
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=std)
			module.weight.data.half()
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
		elif isinstance(module, InformationBottleneck):
			module.post_z_mu.data.normal_(mean=1, std=0.01)
			module.post_z_logD.data.normal_(mean=-9, std=0.01)
			module.post_z_mu.data= module.post_z_mu.data.half()
			module.post_z_logD.data= module.post_z_logD.data.half()


	def _set_gradient_checkpointing(self, module, value=False):
		if isinstance(module, LlamaModel):
			module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
	Args:
		input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
			Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
			it.

			Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.

			[What are input IDs?](../glossary#input-ids)
		attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
			Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.

			[What are attention masks?](../glossary#attention-mask)

			Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.

			If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
			`past_key_values`).

			If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
			and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
			information on the default strategy.

			- 1 indicates the head is **not masked**,
			- 0 indicates the head is **masked**.
		position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
			config.n_positions - 1]`.

			[What are position IDs?](../glossary#position-ids)
		past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
			Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
			`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
			`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

			Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
			blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

			If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
			don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
			`decoder_input_ids` of shape `(batch_size, sequence_length)`.
		inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
			is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
			model's internal embedding lookup matrix.
		use_cache (`bool`, *optional*):
			If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
			`past_key_values`).
		output_attentions (`bool`, *optional*):
			Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
			tensors for more detail.
		output_hidden_states (`bool`, *optional*):
			Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
			more detail.
		return_dict (`bool`, *optional*):
			Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
	"The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
	LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
	"""
	Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

	Args:
		config: LlamaConfig
	"""

	def __init__(self, config: LlamaConfig):
		super().__init__(config)
		self.padding_idx = config.pad_token_id
		self.config._attn_implementation ="sdpa"
		self.vocab_size = config.vocab_size
		if config.vib_layers == True:
			self.hidden_mask= InformationBottleneck(config.hidden_size)
		else:
			self.hidden_mask=None
		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
		self.layers = nn.ModuleList(
			[LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
		)
		self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.gradient_checkpointing = False

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.embed_tokens

	def set_input_embeddings(self, value):
		self.embed_tokens = value

	
	def get_num_rem_weights(self,mlp_list):
		mask = self.hidden_mask.get_mask_hard().int()
		num_states_rem = torch.sum((mask == 1).int())
		em_rem= (num_states_rem * self.config.vocab_size)+ num_states_rem #taking laynorm
		
		return em_rem 

	def get_pars(self): 
		emb_pars= (self.config.vocab_size* self.config.hidden_size) 
		emb_pars+= self.norm.weight.size(0)
		return emb_pars

	def get_sparsity(self,mlp_list):
		em_sp=((self.hidden_mask.sparse/1e6) * self.config.vocab_size)
		em_sp += self.hidden_mask.sparse/1e6		
		return em_sp 

	@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutputWithPast]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)

		if self.gradient_checkpointing and self.training and use_cache:
			logger.warning_once(
				"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
			)
			use_cache = False

		if inputs_embeds is None:
			st_em=time.time()
			inputs_embeds = self.embed_tokens(input_ids)
			emb_time = time.time() - st_em
		
		if self.config.vib_layers is True: 		
			bsz,seq,dim= inputs_embeds.size()
			inputs_embeds =inputs_embeds.reshape(bsz*seq,dim)
			inputs_embeds=self.hidden_mask(inputs_embeds).to(inputs_embeds.dtype)
			inputs_embeds =inputs_embeds.reshape(bsz,seq,dim)			

		return_legacy_cache = False
		if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
			return_legacy_cache = True
			past_key_values = DynamicCache.from_legacy_cache(past_key_values)

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
		if self.config.vib_layers is True: 
			bsz,seq,dim= inputs_embeds.size()
			inputs_embeds=self.hidden_mask(inputs_embeds.reshape(bsz*seq,dim)).reshape(bsz,seq,dim)
		# embed positions
		hidden_states = inputs_embeds
		
		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None
		
		# att_time=0
		# mlp_time=0
		# norm_time=0
		for decoder_layer in self.layers:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			if self.gradient_checkpointing and self.training:
				layer_outputs = self._gradient_checkpointing_func(
					decoder_layer.__call__,
					hidden_states,
					causal_mask,
					position_ids,
					past_key_values,
					output_attentions,
					use_cache,
					cache_position,
				)
			else:
				layer_outputs = decoder_layer(
					hidden_states,
					attention_mask=causal_mask,
					position_ids=position_ids,
					past_key_value=past_key_values,
					output_attentions=output_attentions,
					use_cache=use_cache,
					cache_position=cache_position,
					hidden_mask=self.hidden_mask 
				)
				
				# att_time += att_t
				# mlp_time += mlp_t 
				# norm_time += lay_t
				
			hidden_states = layer_outputs[0]

			if use_cache:
				next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

			if output_attentions:
				all_self_attns += (layer_outputs[1],)			
			
		hidden_z=self.hidden_mask.get_mask_hard().int() if self.config.vib_layers == True else None
		if self.config.vib_layers == True: #during finetuning
			hidden_states = self.norm(hidden_states,hidden_z)			
		else:
			#st_norm =time.time()
			hidden_states = self.norm(hidden_states)
			#norm_time += (time.time() - st_norm)
		#print(f"total norm time {norm_time} att time {att_time} mlp time {mlp_time}")		

		# add hidden states from the last decoder layer
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

	def _update_causal_mask(
		self,
		attention_mask: torch.Tensor,
		input_tensor: torch.Tensor,
		cache_position: torch.Tensor,
		past_key_values: Cache,
		output_attentions: bool,
	):
		# TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
		# KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
		# (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
		# `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

		if self.config._attn_implementation == "flash_attention_2":
			if attention_mask is not None and 0.0 in attention_mask:
				return attention_mask
			return None

		# For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
		# order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
		# to infer the attention mask.
		past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
		using_static_cache = isinstance(past_key_values, StaticCache)

		# When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
		if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
			if AttentionMaskConverter._ignore_causal_mask_sdpa(
				attention_mask,
				inputs_embeds=input_tensor,
				past_key_values_length=past_seen_tokens,
				#is_training=self.training,
			):
				return None

		dtype, device = input_tensor.dtype, input_tensor.device
		min_dtype = torch.finfo(dtype).min
		sequence_length = input_tensor.shape[1]
		if using_static_cache:
			target_length = past_key_values.get_max_length()
		else:
			target_length = (
				attention_mask.shape[-1]
				if isinstance(attention_mask, torch.Tensor)
				else past_seen_tokens + sequence_length + 1
			)

		if attention_mask is not None and attention_mask.dim() == 4:
			# in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
			if attention_mask.max() != 0:
				raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
			causal_mask = attention_mask
		else:
			causal_mask = torch.full(
				(sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
			)
			if sequence_length != 1:
				causal_mask = torch.triu(causal_mask, diagonal=1)
			causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
			causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
			if attention_mask is not None:
				causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
				mask_length = attention_mask.shape[-1]
				padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
				padding_mask = padding_mask == 0
				causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
					padding_mask, min_dtype
				)
		if (
			self.config._attn_implementation == "sdpa"
			and attention_mask is not None
			and attention_mask.device.type == "cuda"
			and not output_attentions
		):
			# Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
			# using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
			# Details: https://github.com/pytorch/pytorch/issues/110213
			causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

		return causal_mask

class VIBLlamaForCausalLM(LlamaPreTrainedModel):
	_tied_weights_keys = ["lm_head.weight"]

	def __init__(self, config):
		super().__init__(config)
		super().__init__(config)
		self.model = LlamaModel(config)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		#for sparsity
		if config.vib_layers is True:
			self.lambda_1 = torch.nn.Parameter(torch.zeros(1)) 
			self.lambda_2 = torch.nn.Parameter(torch.zeros(1))
		else:
			self.lambda_1=None 
			self.lambda_2=None 
		
		self.config = config
		self.sparsity=0.0
		self.start_sparsity=0.0
		self.target_sparsity=0.0
		self.spar_loss_1=0.0
		self.spar_loss_2=0.0
		self.lagrangian_warmup=20
		self.emb_mult=1
		self.prunable_parameters=self.get_prunable_pars()

		# Initialize weights and apply final processing
		self.post_init()

	def set_lagpars(self):
		torch.nn.init.zeros_(self.lambda_1)
		torch.nn.init.zeros_(self.lambda_2)
		
	def prune_heads(self, heads):
		len_heads = len(heads)
		if len_heads == 0:
			return

		_, index = find_pruneable_heads_and_indices(heads, config.num_attention_heads, 128, set())
		
		# Prune linear layers
		if len(index) == 0:
			self.self.query = None
			self.self.key = None
			self.self.value = None
			self.output = None
		else:
			self.self.query = prune_linear_layer(self.self.query, index)
			self.self.key = prune_linear_layer(self.self.key, index)
			self.self.value = prune_linear_layer(self.self.value, index)
			self.output.dense = prune_linear_layer(
				self.output.dense, index, dim=1)

		# Update hyper params and store pruned heads
		self.num_heads = self.num_heads -  len(index)

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

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def set_input_embeddings(self, value):
		self.model.embed_tokens = value

	def get_prunable_pars(self):
		pars=0
		for layer in self.model.layers:
			pars += int(layer.get_pars())
			
		# if self.emb_mult!=0:
		# 	pars+= int(self.model.get_pars())
			
		return pars


	def use_masking(self,mask,emb=0.0): #changed
		if self.emb_mult ==1: #first time
			self.emb_mult=emb
			self.prunable_parameters= self.get_prunable_pars()
			
		if self.config.vib_layers==True:
			self.model.hidden_mask.masking= mask
			self.model.hidden_mask.sample_in_training= False if self.config.finetune == True else True
			for layer in self.model.layers:
				if layer.self_attn is not None: 
					layer.self_attn.ib_1.masking = mask
					layer.self_attn.ib_1.sample_in_training= False if self.config.finetune == True else True
					
				if layer.mlp is not None:
					layer.mlp.ib_1.masking =mask
					layer.mlp.ib_1.sample_in_training= False if self.config.finetune == True else True
					

	def get_kld_loss(self,emb_mult=25,layer_mult=1000,kl_fac=1e-5):
		loss = 0
		self.emb_mult=emb_mult
		for layer in self.model.layers:
			loss += layer.get_kld_loss(self.model.hidden_mask,kl_fac)
			
		if self.emb_mult != 0:
			loss+= (self.model.hidden_mask.kld*emb_mult*kl_fac) #embedding loss weighted more
			
		return loss
	
	def get_num_rem_weights(self):
		num_states_rem = 0
		hidden_mask_size= torch.sum((self.model.hidden_mask.get_mask_hard().int() ==1))#.item()
		for (i, layer) in enumerate(self.model.layers):
			att_sparsity, ffn_sparsity = layer.get_num_rem_weights(hidden_mask_size) 
			num_states_rem += (int(att_sparsity  + ffn_sparsity)) 		
		
		# if self.emb_mult != 0:
		# 	num_states_rem += self.model.get_num_rem_weights(hidden_mask_size) #mlp_scores)
		return num_states_rem

	def get_sparsity(self):
		rem_sum = 0.0
		hidden_mask_sparse =self.model.hidden_mask.sparse
		for (i, layer) in enumerate(self.model.layers):
			att_sparsity, ffn_sparsity = layer.get_sparsity(hidden_mask_sparse) 
			rem_sum += (att_sparsity + ffn_sparsity )     
				
		# if self.emb_mult != 0:
		# 	rem_sum+= self.model.get_sparsity(hidden_mask_sparse) #mlp_scores) 			
		remaining=rem_sum / (self.prunable_parameters/1e6)		
		self.sparsity=1-remaining 
		return self.sparsity

	def set_lagrangian_warmup_steps(self, lagrangian_warmup):
		self.lagrangian_warmup = lagrangian_warmup

	def set_start_and_target_sparsity(self, target_sparsity):
		rem_pars= self.get_num_rem_weights()
		self.start_sparsity=(self.prunable_parameters-rem_pars)/self.prunable_parameters
		self.target_sparsity=target_sparsity

	def get_target_sparsity(self,pruned_steps):
		target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps/self.lagrangian_warmup) + self.start_sparsity 
		return target_sparsity


	def lagrangian_regularization(self, pruned_steps): 
		expected_sparsity = self.get_sparsity()
		if self.lagrangian_warmup > 0:
			target_sparsity = self.get_target_sparsity(pruned_steps)
		else:
			target_sparsity=target_sparsity

		lagrangian_loss = (  (self.lambda_1[0] * (expected_sparsity - target_sparsity))+ (self.lambda_2[0] * ((expected_sparsity - target_sparsity) ** 2))) 
		spar_loss_1=self.lambda_1.detach().item() * (expected_sparsity.detach().cpu() - target_sparsity)
		spar_loss_2=self.lambda_2.detach().item() * ((expected_sparsity.detach().cpu() - target_sparsity) ** 2)
		return lagrangian_loss, expected_sparsity, target_sparsity,spar_loss_1,spar_loss_2

	def calculate_model_size(self): 
		remaining_model_size = self.get_num_rem_weights()		
		pruned_size = (self.prunable_parameters - remaining_model_size)
		hidden_mask_size= self.model.hidden_mask.get_mask_hard()
		embedded_dims=torch.sum((hidden_mask_size ==1).int()).item()  #number of embedding hidden_states left
		embed_rem= (self.config.vocab_size * embedded_dims) 
		intermediate_dims=[]
		attention_head_dims=[]		
		att_head_rem=0
		atten_out_rem=0
		inter_rem=0
		out_rem=0
		tot_att_head_rem=0
		tot_atten_out_rem=0
		tot_inter_rem=0
		tot_out_rem=0
		for (i, layer) in enumerate(self.model.layers):
			if layer.mlp is not None:
				mask_1 = torch.sum((layer.mlp.ib_1.get_mask_hard()==1).int()).item()
				intermediate_dims.append((mask_1))
				inter_rem+= (2*embedded_dims*mask_1) #no bias
				out_rem+= (embedded_dims* mask_1)#no bias
				tot_inter_rem+= (2* self.config.hidden_size * self.config.intermediate_size)
				tot_out_rem+= (self.config.hidden_size * self.config.intermediate_size)
			
			if layer.self_attn is not None:
				mask_1= torch.sum((layer.self_attn.ib_1.get_mask_hard() == 1).int()).item()
				num_states_rem_self= mask_1
				attention_head_dims.append(num_states_rem_self)				
				att_head_rem+= (embedded_dims* num_states_rem_self*layer.self_attn.head_dim*2) + (embedded_dims* num_states_rem_self*layer.self_attn.num_key_value_groups* layer.self_attn.head_dim)
				atten_out_rem+= (layer.self_attn.num_key_value_groups * num_states_rem_self*layer.self_attn.head_dim*embedded_dims)
				tot_att_head_rem += ((self.config.hidden_size * self.config.hidden_size) + (self.config.hidden_size * self.config.num_key_value_heads * layer.self_attn.head_dim * 2))
				tot_atten_out_rem += (self.config.hidden_size * self.config.hidden_size)			

		results = {}
		results["embedded_dims"] = embedded_dims
		results["attention_head_dim1"] = embedded_dims
		results["attention_head_dims"] = attention_head_dims
		results["attention_output_dims"] = embedded_dims
		results["intermediate_dim1"] = embedded_dims
		results["intermediate_dims"] = intermediate_dims
		results["output_layer_dims"] = embedded_dims
		results["pruned_params (Mil)"] = pruned_size/1e6
		results["remaining_params (Mil)"] = remaining_model_size/1e6
		results["embedding pars "] = embed_rem  / ((self.config.vocab_size * self.config.hidden_size))
		results["attention head pars "] =  att_head_rem / tot_att_head_rem
		results["atten_out pars "] =  atten_out_rem / tot_atten_out_rem
		results["intermediate pars"] = inter_rem / tot_inter_rem
		results["out pars "] =  out_rem / tot_out_rem      
		results["actual_model_sparsity"] = round((pruned_size / self.prunable_parameters),5)
		return results
	@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
	@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
	) -> Union[Tuple, CausalLMOutputWithPast]:
		r"""
		Args:
			labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
				Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
				config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
				(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

		Returns:

		Example:

		```python
		>>> from transformers import AutoTokenizer, LlamaForCausalLM

		>>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
		>>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

		>>> prompt = "Hey, are you consciours? Can you talk to me?"
		>>> inputs = tokenizer(prompt, return_tensors="pt")

		>>> # Generate
		>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
		>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		"Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
		```"""

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		#print("\n modeliiiiiiii")
		# decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			cache_position=cache_position,
		)

		hidden_states = outputs[0]
		logits = self.lm_head(hidden_states)
		logits = logits.float()
		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			# Flatten the tokens
			loss_fct = CrossEntropyLoss()
			shift_logits = shift_logits.view(-1, self.config.vocab_size)
			shift_labels = shift_labels.view(-1)
			# Enable model parallelism
			shift_labels = shift_labels.to(shift_logits.device)
			loss = loss_fct(shift_logits, shift_labels)
		if not return_dict:
			output = (logits,) + outputs[1:]
			return (loss,) + output if loss is not None else output

		return CausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	def prepare_inputs_for_generation(
		self,
		input_ids,
		past_key_values=None,
		attention_mask=None,
		inputs_embeds=None,
		cache_position=None,
		use_cache=True,
		**kwargs,
		):
		past_length = 0
		if past_key_values is not None:
			if isinstance(past_key_values, Cache):
				past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
				max_cache_length = (
					torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
					if past_key_values.get_max_length() is not None
					else None
				)
				cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
			# TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
			else:
				cache_length = past_length = past_key_values[0][0].shape[2]
				max_cache_length = None

			# Keep only the unprocessed tokens:
			# 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
			# some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
			if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
				input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
			# 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
			# input_ids based on the past_length.
			elif past_length < input_ids.shape[1]:
				input_ids = input_ids[:, past_length:]
			# 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

			# If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
			if (
				max_cache_length is not None
				and attention_mask is not None
				and cache_length + input_ids.shape[1] > max_cache_length
			):
				attention_mask = attention_mask[:, -max_cache_length:]

		position_ids = kwargs.get("position_ids", None)
		if attention_mask is not None and position_ids is None:
			# create position_ids on the fly for batch generation
			position_ids = attention_mask.long().cumsum(-1) - 1
			position_ids.masked_fill_(attention_mask == 0, 1)
			if past_key_values:
				position_ids = position_ids[:, -input_ids.shape[1] :]

		# if `inputs_embeds` are passed, we only want to use them in the 1st generation step
		if inputs_embeds is not None and past_key_values is None:
			model_inputs = {"inputs_embeds": inputs_embeds}
		else:
			# The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
			# recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
			# TODO: use `next_tokens` directly instead.
			model_inputs = {"input_ids": input_ids.contiguous()}

		input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
		if cache_position is None:
			cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
		elif use_cache:
			cache_position = cache_position[-input_length:]

		model_inputs.update(
			{
				"position_ids": position_ids,
				"cache_position": cache_position,
				"past_key_values": past_key_values,
				"use_cache": use_cache,
				"attention_mask": attention_mask,
			}
		)
		return model_inputs


@add_start_docstrings(
	"""
	The LLaMa Model transformer with a sequence classification head on top (linear layer).

	[`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
	(e.g. GPT-2) do.

	Since it does classification on the last token, it requires to know the position of the last token. If a
	`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
	no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
	padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
	each row of the batch).
	""",
	LLAMA_START_DOCSTRING,
)

class LlamaForSequenceClassification(LlamaPreTrainedModel):
	_keys_to_ignore_on_load_missing = [r"lm_head.weight","lambda_1","lambda_2"]

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.model = LlamaModel(config)
		self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
		# params for sparsity
		self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
		self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))
		self.config = config
		self.sparsity=0.0
		self.start_sparsity=0.0
		self.target_sparsity=0.0
		self.spar_loss_1=0.0
		self.spar_loss_2=0.0
		self.lagrangian_warmup=20
		self.emb_mult=1
		self.prunable_parameters=self.get_prunable_pars()


		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def set_input_embeddings(self, value):
		self.model.embed_tokens = value

	
	@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, SequenceClassifierOutputWithPast]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		transformer_outputs = self.model(
			input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states = transformer_outputs[0]
		logits = self.score(hidden_states)

		if input_ids is not None:
			batch_size = input_ids.shape[0]
		else:
			batch_size = inputs_embeds.shape[0]

		if self.config.pad_token_id is None and batch_size != 1:
			raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
		if self.config.pad_token_id is None:
			sequence_lengths = -1
		else:
			if input_ids is not None:
				sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
			else:
				sequence_lengths = -1

		pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

		loss = None
		if labels is not None:
			labels = labels.to(logits.device)
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(pooled_logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(pooled_logits, labels)
		if not return_dict:
			output = (pooled_logits,) + transformer_outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutputWithPast(
			loss=loss,
			logits=pooled_logits,
			past_key_values=transformer_outputs.past_key_values,
			hidden_states=transformer_outputs.hidden_states,
			attentions=transformer_outputs.attentions,
		)
	

