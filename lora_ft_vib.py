#!/usr/bin/env python
# coding=utf-8
"""
	Code here heavily borrows from https://github.com/locuslab/wanda/tree/main
"""
from transformers.pytorch_utils import  find_pruneable_heads_and_indices, prune_linear_layer
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List 
import pdb
import pickle as pkl
import gc
import time
import json

# Uncomment this out if running the Eleuther evaluation harness
from lm_evaluation_harness_new.lm_eval import evaluator
from torch import nn
import datasets
from datasets import load_from_disk
import torch
from data import get_loaders 
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

import transformers
from transformers import (
	CONFIG_MAPPING,
	MODEL_FOR_CAUSAL_LM_MAPPING,
	AutoConfig,
	AutoModelForCausalLM,
	AutoTokenizer,
	HfArgumentParser,
	Trainer,
	TrainingArguments,
	default_data_collator,
	is_torch_tpu_available,
	set_seed,
	#BitsAndBytesConfig
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft import (
	LoraConfig,
	get_peft_model,
	get_peft_model_state_dict,
	prepare_model_for_kbit_training,
	set_peft_model_state_dict,
	PeftConfig,
	PeftModel
)
from evaluate_ppl import evaluate_ppl 
from trainer_vib import VIBCustomTrainer
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.29.0.dev0")
from importlib.metadata import version
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('datasets', version('datasets'))
#print('BitsAndBytesConfig', version('BitsAndBytesConfig'))
print('# of gpus: ', torch.cuda.device_count())
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
	"""

	model_name_or_path: Optional[str] = field(
		default=None,
		metadata={
			"help": (
				"The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
			)
		},
	)
	
	lora_r: Optional[int] = field(
		default=64,
		metadata={"help": "parameter lora_r"},
	)
	lora_alpha_ratio: Optional[float] = field(
		default=2.0,
		metadata={"help": "parameter lora_alpha"},
	)
	lora_dropout: Optional[float] = field(
		default=0.05,
		metadata={"help": "parameter lora_dropout"},
	)

	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	
	full_ft: bool = field(
		default=False, metadata={"help": "Whether to perform full fine-tuning on the model"}
	)
	
	
	ctx_length: Optional[int] = field(
		default=2048,
		metadata={"help": "context length"},
	)



@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	dataset_name: Optional[str] = field(
		default="wikitext", metadata={"help": "The name of the dataset to use (via the datasets library)."}
	)
	dataset_config_name: Optional[str] = field(
		default="wikitext-2-raw-v1", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of training examples to this "
				"value if set."
			)
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of evaluation examples to this "
				"value if set."
			)
		},
	)
	streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
	block_size: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"Optional input sequence length after tokenization. "
				"The training dataset will be truncated in block of this size for training. "
				"Default to the model max input length for single sentence inputs (take into account special tokens)."
			)
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
	)

	
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	keep_linebreaks: bool = field(
		default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
	)
	def __post_init__(self):
		if self.streaming:
			require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

		if self.dataset_name is None and self.train_file is None and self.validation_file is None:
			raise ValueError("Need either a dataset name or a training/validation file.")
		
@dataclass
class AdditionalArguments():
	vib_learning_rate: float = field(
		default=0.05, metadata={"help": "vib leanring rate"}
	)
	kl_factor: float = field(
		default=1e-6, metadata={"help": "kl multi to vib loss"}
	)
	reg_learning_rate: float = field(
		default=0.1, metadata={"help": "reg leanring rate"}
	)
	target_sparsity: float = field(
		default=0.0, metadata={"help": "sparsity target"}
	)
	emb_mult: Optional[int] = field(
		default=30,
		metadata={
			"help": "emb multiplier"
		},
	)

	bsz: Optional[int] = field(		default=1,	metadata={		"help": "batch size"},	)
	prune_method: Optional[str] = field(default=None, metadata={"help": "method to prune- VIB or None for finetuning"})
	save_loc: Optional[str] = field(default=None, metadata={"help": "path to save pruned/finetuned models"})
	mask_loc: Optional[str] = field(default=None, metadata={"help": "path to pruning masks"})
	distill: bool = field(default=False, metadata={"help": "distill or not"})
	lagrangian_warmup_epochs: float = field(default=0.2,	metadata={"help": "warmup epochs for the lagrangian in pruning"},)
	distill_loss_alpha: float = field(default=0.9, metadata={"help": "layer loss alpha"}	)
	distill_ce_loss_alpha: float = field(		default=0.01, metadata={"help": "logit distil loss alpha"}	)
	distill_temp : float = field(		default=2.0, metadata={"help": "disitl temp"}	)
	layer_distill_version: Optional[int] = field(		default=2,	metadata={		"help": "layer disitil version"},	)
	att_mul: Optional[float] = field(default=1.0,	metadata={"help": "vib multiplier for attention heads"},	)
	inter_mul:Optional[float] = field(default=1.0,	metadata={"help": "vib multiplier for intermediate layers"},	)
	epoch_f: Optional[float] = field(default=1.0,	metadata={"help": "total epochs to finetune for"},	)
	do_zero_eval: bool = field(default=False, metadata={"help": "Eleuther eval or not"})
	finetune: bool = field(default=False, metadata={"help": "finetune the pruned model"})
	write_out: bool = field(default=False, metadata={"help": "write output of the zero shot tasks to file"})

class CustomTrainer(Trainer):
	def set_distill_info(self, teacher_model, kl_weight=1.0, hidden_mse_weight=1.0, distill_temp=2.0):
		if teacher_model is not None:
			teacher_model.eval()
		self.kl_weight = kl_weight
		self.kl_fnct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
		self.mse_fnct = torch.nn.MSELoss()
		self.teacher_model = teacher_model
		self.distill_temp=distill_temp

	def compute_loss(self, model, inputs, return_outputs=False):
		#inputs['output_hidden_states'] = True
		if self.teacher_model is not None:
			with torch.no_grad():
				teacher_out = self.teacher_model(**inputs)
				teacher_logits = torch.nn.functional.log_softmax(teacher_out["logits"] , dim=-1)
				#teacher_hidden = teacher_out['hidden_states']

		outputs = model(**inputs)
		# Save past state if it exists
		if self.args.past_index >= 0:
			self._past = outputs[self.args.past_index]

		# calculate the classic distillation loss
		if self.teacher_model is not None:
			student_logits = torch.nn.functional.log_softmax(outputs["logits"] , dim=-1)
			kl_loss = self.kl_fnct(student_logits, teacher_logits)
		#print("\n kl loss=", self.kl_weight*kl_loss)
		
		loss = outputs['loss']
		#print("\n loss and kl", loss.detach().cpu(), (self.kl_weight*kl_loss).detach().cpu())
		loss = (loss + (self.kl_weight*kl_loss)) if self.teacher_model is not None else loss #+ (self.hidden_mse_weight*hidden_loss)
		
		return (loss, outputs) if return_outputs else loss

# Prune the model according to the masks
def prune_model(model, tokenizer, prune_info_path):

	def get_param_count(model, exclude=[]):
		return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

	def get_mask_hard(mu,logD, threshold=0):
		logalpha = logD.data - torch.log(mu.data.pow(2) + 1e-8)
		hard_mask = (logalpha < threshold) #.float() 
		#print("\n hidden mask non_zero",torch.sum(hard_mask.int() != 0))
		hard_mask_ind = torch.where(hard_mask.float() != 0.0)[0]
		return hard_mask,hard_mask_ind

	def get_mask_weighted(mu,logD, threshold=0):
		logalpha = logD.data - torch.log(mu.data.pow(2) + 1e-8)
		mask = (logalpha < threshold) * mu.data
		mask_ind = torch.where(mask.float() != 0.0)[0] 
		return mask,mask_ind
	
	def get_mask_weighted_adapted(mu,logD, threshold=0):
		logalpha = logD.data - torch.log(mu.data.pow(2) + 1e-8)
		mask = (logalpha < threshold) * mu.data
		mask_ind = torch.where(mask.float() != 0.0)[0] #torch.LongTensor(mask.nonzero().squeeze()).tolist())-issue with nonzero().squeeze() in places with 1 head left
		tensor_dims=128 # 64 or 128 or 256
		if len(mask_ind) % tensor_dims == 0: 
			return mask,mask_ind
		n= len(mask_ind)
		shift_thres=  n+ (tensor_dims - n % tensor_dims) if (tensor_dims - n % tensor_dims) <= (n % tensor_dims) else n-(n % tensor_dims) 
		#change threshold 
		sr_logalpha=torch.sort(logalpha, stable=True)
		threshold=sr_logalpha[1][shift_thres-1]
		mask = (logalpha < threshold) * mu.data
		mask_ind = sr_logalpha[1][:(shift_thres)]
		return mask,mask_ind

	def get_mask_weighted_att(mu,logD, threshold=0):
		logalpha = logD.data - torch.log(mu.data.pow(2) + 1e-8)
		mask = (logalpha < threshold) * mu.data
		mask_ind = torch.where(mask.float() != 0.0)[0] 
		if len(mask_ind) != 0: #all pruned
			return mask,mask_ind		
		#change threshold 
		shift_thres=1
		sr_logalpha=torch.sort(logalpha, stable=True)
		threshold=sr_logalpha[1][shift_thres]
		mask = (logalpha < threshold) * mu.data 
		if len(torch.where(mask.float() != 0.0)[0]) == 0:
			shift_thres += 1
			mask = (logalpha < sr_logalpha[1][shift_thres]) * mu.data 
		mask_ind = torch.where(mask.float() != 0.0)[0]
		return mask,mask_ind
	
	def prune_layer_norm(layernorm, index,in_dtype):
		w=layernorm.weight.index_select(0, index).clone().detach()
		i_device= layernorm.weight.device
		layernorm.weight = None 
		layernorm.weight=torch.nn.parameter.Parameter(w.contiguous()).to(i_device)
		return layernorm

	in_dytpe= model.model.embed_tokens.weight.dtype
	mask_info_loc = prune_info_path
	original_param_count_full_model = get_param_count(model,exclude=[])
	original_param_count = get_param_count(model,exclude=['embed','head'])
	if os.path.exists(mask_info_loc):
		with open(mask_info_loc, 'rb') as handle:
			mask_info = pkl.load(handle)
		
		hidden_mask,hidden_mask_ind = get_mask_weighted(mask_info['hidden_mask_z_mu'], mask_info['hidden_mask_z_logD']) #can be changed by get_mask_hard to get binary masks
		logger.info("hidden_mask ind = %d",len(hidden_mask_ind))
		
		hidden_mask,hidden_mask_ind = get_mask_weighted_adapted(mask_info['hidden_mask_z_mu'], mask_info['hidden_mask_z_logD']) 
		logger.info("Adjusted hidden_mask ind = %d",len(hidden_mask_ind))
		
		#prune embeds
		model.model.embed_tokens.weight.data= model.model.embed_tokens.weight.data.mul(hidden_mask).to(in_dytpe) #mult with ib pars
		model.model.embed_tokens.weight = torch.nn.parameter.Parameter(model.model.embed_tokens.weight.index_select(1, hidden_mask_ind.cuda()).clone())
		model.model.embed_tokens.embedding_dim = hidden_mask_ind.shape[0]

		prune_heads_ind={}
		tot_att_pars=0
		tot_mlp_pars=0
		for (i, layer) in enumerate(model.model.layers):
			inter_mask,inter_mask_ind = get_mask_weighted(mask_info[f'inter_layer{i}_z_mu'], mask_info[f'inter_layer{i}_z_logD'])#can be changed by get_mask_hard to get binary masks
			logger.info("intermediate_mask ind = %d",len(inter_mask_ind))
			inter_mask,inter_mask_ind = get_mask_weighted_adapted(mask_info[f'inter_layer{i}_z_mu'], mask_info[f'inter_layer{i}_z_logD'])
			logger.info("Adjusted intermediate_mask ind = %d",len(inter_mask_ind))
			attn_mask, attn_mask_ind = get_mask_weighted(mask_info[f'attn_layer{i}_z_mu'], mask_info[f'attn_layer{i}_z_logD'])#can be changed by get_mask_hard to get binary masks
			logger.info("att_mask ind = %d",len(attn_mask_ind))
			attn_mask, attn_mask_ind = get_mask_weighted_att(mask_info[f'attn_layer{i}_z_mu'], mask_info[f'attn_layer{i}_z_logD'])
			logger.info("Adjusted att_mask ind = %d",len(attn_mask_ind))
			#intermediate and FFN
			#logger.info("intermediate mask ind = %d",len(inter_mask_ind))
			#logger.info("attn_mask ind = %d",len(attn_mask_ind))
			model.model.layers[i].mlp.down_proj.weight.data = model.model.layers[i].mlp.down_proj.weight.data.mul(inter_mask).to(in_dytpe)
			model.model.layers[i].mlp.down_proj.weight.data = model.model.layers[i].mlp.down_proj.weight.data.transpose(0, 1).mul(hidden_mask).transpose(0, 1).to(in_dytpe)
			tot_mlp_pars += prune_mlp(inter_mask_ind,hidden_mask_ind, i,model,in_dytpe)
			model.model.layers[i].mlp.intermediate_size = len(inter_mask_ind)
			#attention
			#multiply vib layer to output layer
			if model.config.num_attention_heads != model.config.num_key_value_heads: #for GQA-based models
				atten_self= torch.repeat_interleave(attn_mask,(model.model.layers[i].self_attn.num_key_value_groups* model.model.layers[i].self_attn.head_dim))
			else: #for MHA-based models
				atten_self= torch.repeat_interleave(attn_mask, model.model.layers[i].self_attn.head_dim) 
			index= torch.where(attn_mask == 0)[0]
			#print("\n index length=", index)
			model.model.layers[i].self_attn.o_proj.weight.data= model.model.layers[i].self_attn.o_proj.weight.data.mul(atten_self).to(in_dytpe)
			model.model.layers[i].self_attn.o_proj.weight.data = model.model.layers[i].self_attn.o_proj.weight.data.transpose(0, 1).mul(hidden_mask).transpose(0, 1).to(in_dytpe)
			prune_heads_ind[i] =index.tolist() 

			if model.config.num_attention_heads != model.config.num_key_value_heads: #for GQA-based models
				ind_q= torch.where(torch.repeat_interleave(attn_mask,model.model.layers[i].self_attn.num_key_value_groups) == 0)[0]
				prune_heads_q= ind_q.tolist()
				_,rem_kv= find_pruneable_heads_and_indices(prune_heads_ind[i], model.config.num_key_value_heads, (model.config.hidden_size // model.config.num_attention_heads), set())
				_,rem_q= find_pruneable_heads_and_indices(prune_heads_q, model.config.num_attention_heads, (model.config.hidden_size // model.config.num_attention_heads), set())	
				prune_heads(model,rem_kv,rem_q,i)
				# if model.model.layers[i].self_attn is not None:
				model.model.layers[i].self_attn.num_key_value_heads = len(torch.where(attn_mask != 0)[0])
				model.model.layers[i].self_attn.num_heads = model.model.layers[i].self_attn.num_heads - len(ind_q)
				model.model.layers[i].self_attn.hidden_size = model.model.layers[i].self_attn.num_heads * model.model.layers[i].self_attn.head_dim	
			else:
				_,rem= find_pruneable_heads_and_indices(prune_heads_ind[i], model.config.num_attention_heads, (model.config.hidden_size // model.config.num_attention_heads), set())
				prune_heads(model,rem,rem,i)
				#if model.model.layers[i].self_attn is not None:
				model.model.layers[i].self_attn.num_heads = len(torch.where(attn_mask != 0)[0])
				model.model.layers[i].self_attn.num_key_value_heads = len(torch.where(attn_mask != 0)[0])
				model.model.layers[i].self_attn.hidden_size = model.model.layers[i].self_attn.num_heads * model.model.layers[i].self_attn.head_dim				
			#print("\n  heads=",model.model.layers[i].self_attn.num_heads,model.model.layers[i].self_attn.num_key_value_heads)
			
			#if model.model.layers[i].self_attn is not None:
			tot_att_pars += prune_attn(hidden_mask_ind,i,model,in_dytpe)
			
			#print("\n after att heads=",model.model.layers[i].self_attn.num_heads,model.model.layers[i].self_attn.num_key_value_heads)	

			#prune layer norms
			model.model.layers[i].input_layernorm = (prune_layer_norm(model.model.layers[i].input_layernorm,hidden_mask_ind.cuda(),in_dytpe ))
			model.model.layers[i].post_attention_layernorm = (prune_layer_norm(model.model.layers[i].post_attention_layernorm,hidden_mask_ind.cuda(),in_dytpe))
		model.model.norm = (prune_layer_norm(model.model.norm,hidden_mask_ind.cuda(),in_dytpe))
		model.lm_head = (prune_linear_layer( model.lm_head, hidden_mask_ind, dim=1) ).half()
		model.lm_head.weight.data=  model.lm_head.weight.data.to(in_dytpe) 
		gc.collect()
		torch.cuda.empty_cache() 
		print(f" Total att pars {tot_att_pars} Total mlp pars {tot_mlp_pars}")

	else:
		print("\n mask path not found..exiting")	
	final_param_count = get_param_count(model,exclude=['embed', 'head'])
	#final_param_count_full_model = get_param_count(model,exclude=[])
	#print(model)
	model.eval()
	
	#print('Final model sparsity is : {:.3f} '.format(1.0 - final_param_count_full_model/original_param_count_full_model))
	print('Final model sparsity (except embed and last lm_head) : {:.3f} '.format(1.0 - final_param_count/original_param_count))

	gc.collect()
	torch.cuda.empty_cache() 

def prune_mlp(inter_i,hidden_i, layer,model,in_dytpe,full=False):
	if len(hidden_i)==0: 
		model.model.layers[i].mlp =None
	else:
		model.model.layers[layer].mlp.gate_proj = (prune_linear_layer( model.model.layers[layer].mlp.gate_proj, inter_i, dim=0)).half()                 
		new_gate = (prune_linear_layer( model.model.layers[layer].mlp.gate_proj, hidden_i, dim=1)).half()#reducing both the dimensions
		model.model.layers[layer].mlp.gate_proj= new_gate
		model.model.layers[layer].mlp.up_proj = (prune_linear_layer( model.model.layers[layer].mlp.up_proj, inter_i, dim=0)).half()                 
		new_up = (prune_linear_layer( model.model.layers[layer].mlp.up_proj, hidden_i, dim=1)).half() #reducing both the dimensions  
		model.model.layers[layer].mlp.up_proj = new_up
		model.model.layers[layer].mlp.down_proj = (prune_linear_layer( model.model.layers[layer].mlp.down_proj, hidden_i, dim=0)).half()
		new_down = (prune_linear_layer( model.model.layers[layer].mlp.down_proj, inter_i, dim=1)).half()
		model.model.layers[layer].mlp.down_proj = new_down
	gc.collect()
	torch.cuda.empty_cache()
	tot= sum([p.numel() for p in model.model.layers[layer].mlp.parameters() ])
	return tot

def prune_attn(hidden_i, layer,model,in_dytpe,full=False):
	new_q = (prune_linear_layer(model.model.layers[layer].self_attn.q_proj , hidden_i, dim=1)).half()
	model.model.layers[layer].self_attn.q_proj = new_q
	new_k = (prune_linear_layer(model.model.layers[layer].self_attn.k_proj , hidden_i, dim=1)).half()
	model.model.layers[layer].self_attn.k_proj = new_k
	new_v = (prune_linear_layer(model.model.layers[layer].self_attn.v_proj , hidden_i, dim=1)).half()
	model.model.layers[layer].self_attn.v_proj = new_v
	new_o = (prune_linear_layer(model.model.layers[layer].self_attn.o_proj , hidden_i, dim=0)).half()
	model.model.layers[layer].self_attn.o_proj = new_o
	gc.collect()
	torch.cuda.empty_cache() 
	tot= sum([p.numel() for p in model.model.layers[layer].self_attn.parameters() ])
	return tot

def prune_heads(model,rem_index,rem_q_ind,i,full=False):	
	if len(rem_index)==0: 
		model.model.layers[i].self_attn =None
	else:
		model.model.layers[i].self_attn.q_proj = (prune_linear_layer(model.model.layers[i].self_attn.q_proj , rem_q_ind, dim=0)).half()
		model.model.layers[i].self_attn.k_proj = (prune_linear_layer(model.model.layers[i].self_attn.k_proj , rem_index, dim=0)).half()
		model.model.layers[i].self_attn.v_proj = (prune_linear_layer(model.model.layers[i].self_attn.v_proj , rem_index, dim=0)).half()
		model.model.layers[i].self_attn.o_proj = (prune_linear_layer(model.model.layers[i].self_attn.o_proj , rem_q_ind, dim=1)).half()
	gc.collect()
	torch.cuda.empty_cache() 


def get_param_count(model, exclude=[]):
	return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		 model_args, data_args, training_args, additional_args = parser.parse_json_file(
			json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

	# Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
	# information sent is the one passed as arguments along with your Python/PyTorch versions.
	#send_example_telemetry("run_clm", model_args, data_args)
	
	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	if training_args.should_log:
		# The default of training_args.log_level is passive, so we set log level at info here to have that default.
		transformers.utils.logging.set_verbosity_info()

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation arguments {training_args}")
	logger.info(f"Data arguments {data_args}")
	logger.info(f"Model arguments {model_args}")
	logger.info(f"Additional arguments {additional_args}")

	# Detecting last checkpoint.
	last_checkpoint = None
	if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		#print(last_checkpoint)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
			)
		elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
			)

	# Set seed before initializing model.
	set_seed(training_args.seed)
	torch.manual_seed(training_args.seed)
	
	# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
	# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
	# (the dataset will be downloaded automatically from the datasets Hub).
	#
	# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
	# 'text' is found. You can easily tweak this behavior (see below).
	#
	# In distributed training, the load_dataset function guarantee that only one local process can concurrently
	# download the dataset.
	
	config_kwargs = {
		"revision": model_args.model_revision,
		"use_auth_token": None 
	}
	config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs,trust_remote_code=True) 
	config.vib_layers = True if additional_args.prune_method is not None else False
	config.finetune= False
	config.att_mul= additional_args.att_mul
	config.inter_mul= additional_args.inter_mul
	config.output_attentions = False
	config.output_hidden_states= False
	config.use_cache= False
	tokenizer_kwargs = {
		"use_fast": False,
		"revision": model_args.model_revision,
		"use_auth_token": None # if model_args.use_auth_token else None,
	}
	tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,use_fast=False, trust_remote_code=True)
	if 'mistral' in model_args.model_name_or_path:
		if additional_args.prune_method is None:
			from transformers import AutoModelForCausalLM as VIBLlamaForCausalLM
		else:
			from modelling_mistral_vib import MistralForCausalLM as VIBLlamaForCausalLM
	elif 'llama3' in model_args.model_name_or_path:
		if additional_args.prune_method is None:
			from transformers import AutoModelForCausalLM as VIBLlamaForCausalLM
		else:
			from modelling_llama3_vib import VIBLlamaForCausalLM
	else:#for llama1 and 2
		if additional_args.prune_method is None:
			from transformers import AutoModelForCausalLM as VIBLlamaForCausalLM
		else:
			from modelling_llama_vib import VIBLlamaForCausalLM
		
	if additional_args.prune_method is None:
		if 'mistral' in model_args.model_name_or_path:
			config.max_position_embeddings = 8192 
		
		model = VIBLlamaForCausalLM.from_pretrained(model_args.model_name_or_path, config= config, torch_dtype=torch.float16,low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True) #do not use bfloat16 as it leads to inconsistencies in loss calculation
	else:
		model = VIBLlamaForCausalLM.from_pretrained(model_args.model_name_or_path,config= config,torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
	
	model.seqlen = model.config.max_position_embeddings
	#print("\n max positions=", model.config.max_position_embeddings)
	print("\n is tokenizer fast=", tokenizer.is_fast)
	os.makedirs(additional_args.save_loc, exist_ok=True)
	out_file = open(os.path.join(additional_args.save_loc, "output.log"), 'w')
	print("Num params = : ", get_param_count(model,exclude=[]))
	
	
	if training_args.do_eval and additional_args.prune_method is None:
		# if additional_args.do_zero_eval is True:
		# 	results = evaluator.simple_evaluate(
		# 			model="hf-causal-experimental",
		# 			#model_args=f"pretrained={additional_args.save_loc},peft={additional_args.save_loc}", #"pretrained={}".format(model_args.model_name_or_path),
		# 			model_args=  f"pretrained={model_args.model_name_or_path}",
		# 			tasks=["hellaswag","winogrande","openbookqa","arc_easy","arc_challenge", "piqa", "boolq","rte"],
		# 			num_fewshot=0,
		# 			no_cache=True,
		# 			pretrained_model=model,
		# 			#tokenizer=tokenizer,
		# 			decontamination_ngrams_path=None,
		# 			check_integrity=False,
		# 			description_dict={},
		# 		)
		# 	print(results)
		logger.info("*** Evaluate ***")
		model.eval()
		st_t= time.time()
		og_ppl, og_runtime = evaluate_ppl('wikitext2', model, tokenizer, model.seqlen) 
		og_tot_time= (time.time()-st_t)
		out_str = "Original perplexity on wikitext = {:.3f} per batch infer time {:.3f} total infer time= {:.3f}".format(og_ppl,og_runtime,og_tot_time)		
		print(out_str)
		
	gc.collect()
	torch.cuda.empty_cache()
	if additional_args.mask_loc is not None and additional_args.prune_method is None:
		print("\n pruning model........")
		prune_model(model, tokenizer, additional_args.mask_loc)
		gc.collect()
		torch.cuda.empty_cache()
		model.eval()		
		
		st_t1= time.time()
		before_train_ppl, final_runtime = evaluate_ppl('wikitext2', model, tokenizer, model.seqlen) 
		fin_tot_time= (time.time()-st_t1)
		speedup = og_runtime / final_runtime
		speed_test=og_tot_time/fin_tot_time
		out_str = "[SpeedUp for a batch={:.3f}] SpeedUp on test set={:.3f}| W/o finetuning perplexity on wikitext = {:.3f} infer time= {:.3f}".format(speedup,speed_test, before_train_ppl,fin_tot_time)
		print(out_str)
						
		if additional_args.do_zero_eval is True:
			#To evaluate on peft lora-based model uncomment below
			# lm_head_state_dict = torch.load(f'{additional_args.save_loc}/lm_head_state_dict.pth')
			# model.lm_head.load_state_dict(lm_head_state_dict)
			# adap_config= PeftConfig.from_pretrained(additional_args.save_loc)
			# model = PeftModel.from_pretrained(model, config= adap_config, model_id=additional_args.save_loc, is_trainable=False  )
			results = evaluator.simple_evaluate(
					model="hf",#-causal-experimental",
					model_args=  f"pretrained={model_args.model_name_or_path}",
					tasks=["hellaswag","winogrande","openbookqa","arc_easy","arc_challenge", "piqa", "boolq","rte"],
					num_fewshot=0,
					#no_cache=True,
					# pretrained_model=model,
					#decontamination_ngrams_path=None,
					check_integrity=False,
					log_samples=additional_args.write_out,
					#description_dict={},
				)
			print("\n Zero shot results W/o finetuning", results)
			# if additional_args.write_out:
			with open(os.path.join(additional_args.save_loc, "results.json"), 'w') as fw:
				json.dump(results, fw, indent=4)
			if additional_args.finetune is not True:
				exit()

	if "c4" in data_args.dataset_name:
		print("\n getting c4 dataset")
		raw_datasets={}
		raw_datasets['train'] =  load_dataset('c4', data_files='train/c4-train.00000-of-01024.json',split='train', cache_dir= 'c4') #changed
		if additional_args.prune_method is not None:
			raw_datasets['validation']= load_dataset('c4', data_files='validation/c4-validation.00000-of-00008.json',split='train',cache_dir= 'c4')		
	elif 'wikitext' in data_args.dataset_name:
		raw_datasets={}
		if additional_args.prune_method is not None:
			try:
				raw_datasets['validation'] = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
			except:
				raw_datasets['validation'] = load_from_disk(os.path.join('../wikitext2rawv1','test'))
		try:
			raw_datasets['train']=load_dataset('../wikitext2rawv1',  data_dir='train')
		except:
			raw_datasets['train'] =load_from_disk(os.path.join('../wikitext2rawv1','train'))
		
	
	print("\n loaded dataset")
	if training_args.do_train:
		column_names = list(raw_datasets["train"].features)
	else:
		column_names = list(raw_datasets["validation"].features)
	text_column_name = "text" if "text" in column_names else column_names[0]

	# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
	tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
	def tokenize_function(examples):
		with CaptureLogger(tok_logger) as cl:
			output = tokenizer(examples[text_column_name])
		# clm input could be much much longer than block_size
		if "Token indices sequence length is longer than the" in cl.out:
			tok_logger.warning(
				"^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
				" before being passed to the model."
			)
		return output
	tokenized_datasets={}
	with training_args.main_process_first(desc="dataset map tokenization"):
		if not data_args.streaming:
			tokenized_datasets["train"] = raw_datasets["train"].map(
											tokenize_function,
											batched=True,
											num_proc=data_args.preprocessing_num_workers,
											remove_columns=column_names,
											load_from_cache_file=not data_args.overwrite_cache,
											desc="Running tokenizer on dataset",
										)
			if additional_args.prune_method is not None:
				tokenized_datasets["validation"] =raw_datasets["validation"].map(
											tokenize_function,
											batched=True,
											num_proc=data_args.preprocessing_num_workers,
											remove_columns=column_names,
											load_from_cache_file=not data_args.overwrite_cache,
											desc="Running tokenizer on dataset",
											) 
										
		else:
			tokenized_datasets = raw_datasets.map(
				tokenize_function,
				batched=True,
				remove_columns=column_names,
			)

	if data_args.block_size is None:
		block_size = tokenizer.model_max_length
		if block_size > 1024:
			logger.warning(
				"The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
				" of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
				" override this default with `--block_size xxx`."
			)
			block_size = 1024
	else:
		if data_args.block_size > tokenizer.model_max_length:
			logger.warning(
				f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
				f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
			)
		block_size = min(data_args.block_size, tokenizer.model_max_length)

	# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
	def group_texts(examples):
		# Concatenate all texts.
		concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
		total_length = len(concatenated_examples[list(examples.keys())[0]])
		# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
		# customize this part to your needs.
		if total_length >= block_size:
			total_length = (total_length // block_size) * block_size
			#print("\n total len blcoked=", total_length)
		# Split by chunks of max_len.
		result = {
			k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
			for k, t in concatenated_examples.items()
		}

		result["labels"] = result["input_ids"].copy()
		return result

	# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
	# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
	# to preprocess. 
	#
	# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
	# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
	lm_datasets={}
	with training_args.main_process_first(desc="grouping texts together"):
		if not data_args.streaming:
			lm_datasets["train"] = tokenized_datasets["train"].map(
										group_texts,
										batched=True,
										num_proc=data_args.preprocessing_num_workers,
										load_from_cache_file=not data_args.overwrite_cache,
										desc=f"Grouping texts in chunks of {block_size}",
									) 
			if additional_args.prune_method is not None:
				lm_datasets["validation"] = tokenized_datasets["validation"].map(
											group_texts,
											batched=True,
											num_proc=data_args.preprocessing_num_workers,
											load_from_cache_file=not data_args.overwrite_cache,
											desc=f"Grouping texts in chunks of {block_size}",
										)
			
		else:
			lm_datasets = tokenized_datasets.map(
				group_texts,
				batched=True,
			)
	print("\n grouped examples")
	############################################################################################
	model = prepare_model_for_kbit_training(model)  #changed - commented
	if model_args.full_ft:
		for k, v in model.named_parameters():
			v.requires_grad = False if 'embed' in k  else True #changed
	else:
		if 'llama3' in model_args.model_name_or_path:
			target_modules = ["q_proj","v_proj",  "up_proj"] #fewer models can be finetuned as llama3 takes more space
		else:
			target_modules = ["q_proj","v_proj","k_proj", "o_proj", "up_proj", "gate_proj", "down_proj"] 
		if additional_args.prune_method is None:
			print("\n Target modules finetuned", target_modules)
			config = LoraConfig(
				r=model_args.lora_r,
				lora_alpha=int(model_args.lora_r*model_args.lora_alpha_ratio),
				target_modules=target_modules,
				lora_dropout=model_args.lora_dropout,
				bias="none",
				task_type="CAUSAL_LM",
			)
			model = get_peft_model(model, config)
	#print(model)
	############################################################################################

	# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
	# on a small vocab and want a smaller embedding size, remove this test.
	embedding_size = model.get_input_embeddings().weight.shape[0]
	if len(tokenizer) > embedding_size:
		model.resize_token_embeddings(len(tokenizer))

	logger.info("len of dataset %d",len(lm_datasets["train"]))
	if training_args.do_train:
		if "train" not in tokenized_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = lm_datasets["train"]
		if data_args.max_train_samples is not None:
			max_train_samples = min(len(train_dataset), data_args.max_train_samples)
			train_dataset = train_dataset.select(range(max_train_samples))

	if training_args.do_eval and additional_args.prune_method is not None:
		if "validation" not in tokenized_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = lm_datasets["validation"]
		if data_args.max_eval_samples is not None:
			max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
			eval_dataset = eval_dataset.select(range(max_eval_samples))

	def preprocess_logits_for_metrics(logits, labels):
		if isinstance(logits, tuple):
			# Depending on the model and config, logits may contain extra tensors,
			# like past_key_values, but logits always come first
			logits = logits[0]
		return logits.argmax(dim=-1)
	del raw_datasets,tokenized_datasets,lm_datasets
	gc.collect()
	torch.cuda.empty_cache()
################################################################################################################
	batch_size = 128 if additional_args.prune_method is None else 1
	training_args.gradient_accumulation_steps = 1 if additional_args.prune_method is not None else batch_size // training_args.per_device_train_batch_size #8 #changed from 1 as taken by us before #batch_size // training_args.per_device_train_batch_size
	training_args.warmup_steps = 5
	training_args.fp16 = True
	training_args.logging_steps = 10 #10
	training_args.optim = "adamw_torch"
	training_args.save_strategy = "steps" 
	training_args.num_train_epochs = additional_args.epoch_f if additional_args.prune_method is None else 1.0
	training_args.eval_steps = 1000 if training_args.gradient_accumulation_steps==1 else 10
	training_args.save_steps = 5000 if training_args.gradient_accumulation_steps==1 else 50
	training_args.save_total_limit = 2
	training_args.group_by_length = False
	training_args.log_level = 'info'
	norms = ["layernorm" ,"norm"]
	if not model_args.full_ft:
		for k, v in model.named_parameters():
			if "lm_head" in k:
				v.requires_grad = True if additional_args.prune_method is None else False
			
################################################################################################################
	# Initialize our Trainer
	if additional_args.prune_method is not None:
		teacher_model= None
		model.use_masking(mask=False,emb=additional_args.emb_mult) #pruning
		trainer = VIBCustomTrainer(
			model=model,
			args=training_args,
			additional_args=additional_args,
			train_dataset=train_dataset if training_args.do_train else None,
			eval_dataset=eval_dataset if training_args.do_eval else None,
			tokenizer=tokenizer,
			#teacher_model= teacher_model,
			# Data collator will default to DataCollatorWithPadding, so we change it.
			data_collator=default_data_collator,
			compute_metrics=None,
			preprocess_logits_for_metrics=preprocess_logits_for_metrics
			if training_args.do_eval and not is_torch_tpu_available()
			else None,
		)
	else:
		trainer = CustomTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval and additional_args.prune_method is not None else None,
		tokenizer=tokenizer,
		# Data collator will default to DataCollatorWithPadding, so we change it.
		data_collator=default_data_collator,
		compute_metrics=None,
		preprocess_logits_for_metrics=preprocess_logits_for_metrics
		if training_args.do_eval and not is_torch_tpu_available()
		else None,
		)
		if additional_args.distill is not False:
			teacher_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True	)
			teacher_model.seqlen = teacher_model.config.max_position_embeddings
		else:
			teacher_model= None
		trainer.set_distill_info(teacher_model, kl_weight=additional_args.distill_ce_loss_alpha, hidden_mse_weight=additional_args.distill_ce_loss_alpha,distill_temp=additional_args.distill_temp)
		
	############## code imported from alpaca-lora ###################
	model.config.use_cache = False

	if not model_args.full_ft and additional_args.prune_method is None:
		old_state_dict = model.state_dict
		model.state_dict = (
			lambda self, *_, **__: get_peft_model_state_dict(
				self, old_state_dict()
			)
		).__get__(model, type(model))

# 	if torch.__version__ >= "2" and sys.platform != "win32":
# 		model = torch.compile(model)
	############## code imported from alpaca-lora ###################
	# Training
	if training_args.do_train:		
		if additional_args.prune_method is None:
			checkpoint = None
			if last_checkpoint is not None:
				checkpoint = last_checkpoint
			train_result = trainer.train(resume_from_checkpoint=checkpoint)
		else:
			start_time= time.time()
			train_result = trainer.train()
			train_time= time.time() - start_time
			print("\n Train time=", train_time)
		# trainer.save_model()  # Saves the tokenizer too for easy upload
		
# 		#############################################################
		if additional_args.prune_method is None: #saving finetuned adapters
			model.save_pretrained(f"{training_args.output_dir}/new") 
			tokenizer.save_pretrained(f"{training_args.output_dir}/new")
			torch.save(trainer.model.state_dict(), f"{training_args.output_dir}/new/adapter_model.bin")
			torch.save(model.lm_head.state_dict(), f"{training_args.output_dir}/new/lm_head_state_dict.pth")
# 		#############################################################
		metrics = train_result.metrics
		max_train_samples = (
			data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))
		print(f"Evaluating: {metrics}")
		#trainer.log_metrics("train", metrics)
		#trainer.save_metrics("train", metrics)
		trainer.save_state()
	
	if training_args.do_eleuther_eval:
		results = evaluator.simple_evaluate(
			model="hf-causal-experimental",
			model_args="pretrained={}".format(model_args.model_name_or_path),
			tasks=["winogrande", "boolq", "arc_challenge", "arc_easy", "hellaswag", "mmlu", "gsm8k"],
			num_fewshot=0,
			no_cache=True,
			pretrained_model=model,
		)
		print(results)
	
	del trainer
	if additional_args.prune_method is None:
		if additional_args.distill is not False:
			del teacher_model
		del training_args
		# Evaluation		
		gc.collect()
		torch.cuda.empty_cache()
		print("*** Evaluate after finetuning pruned model***")
		model.eval()
		final_ppl, _ = evaluate_ppl('wikitext2', model, tokenizer, model.seqlen)
		out_str = "Finetuned perplexity on wikitext= {:.3f}".format(final_ppl)
		print(out_str)
				
	#kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
	# if data_args.dataset_name is not None:
	# 	kwargs["dataset_tags"] = data_args.dataset_name
	# 	if data_args.dataset_config_name is not None:
	# 		kwargs["dataset_args"] = data_args.dataset_config_name
	# 		kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
	# 	else:
	# 		kwargs["dataset"] = data_args.dataset_name

	# if training_args.push_to_hub: #changed
	# 	trainer.push_to_hub(**kwargs)
	# else:
	# 	trainer.create_model_card(**kwargs)


# def _mp_fn(index): #changed
# 	# For xla_spawn (TPUs)
# 	main()


if __name__ == "__main__":
	os.environ["WANDB_DISABLED"] = "true"
	main()