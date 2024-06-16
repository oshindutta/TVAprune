import math
import os
import sys
import gc
import time
#import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_linear_schedule_with_warmup 
from torch.optim import AdamW
from transformers import Trainer
from transformers.trainer_utils import TrainOutput 
from transformers.utils import logging
#from torch.profiler import profile, record_function, ProfilerActivity
import pickle as pkl
from evaluate_ppl import evaluate_ppl

logger = logging.get_logger(__name__)
class Eval_Counter():
	def __init__(self):
		self.epoch = 0
		self.global_step = 0
		self.best_eval_score = 1000
		self.near_sparsity_eval_times = 0
		self.level_best_score = {0.85: 0, 0.8: 0, 0.7: 0,
								 0.6: 0, 0.75: 0, 0.9: 0, 0.95: 0, 0.65: 0}

	def round_nearest(self, x, a):
		return round(round(x / a) * a, -int(math.floor(math.log10(a))))

	def update(self, epoch, global_step, eval_score):
		best_so_far = False
		if eval_score < self.best_eval_score: #changed for perplexity
			self.epoch = epoch
			self.global_step = global_step
			self.best_eval_score = eval_score
			best_so_far = True
		return best_so_far

	def clear(self):
		self.eval_score = 0

class VIBCustomTrainer(Trainer):
	def __init__(
			self,
			model,
			args,
			additional_args,
			tokenizer,
			train_dataset= None,
			eval_dataset= None,
			data_collator=None,
			compute_metrics=None,
			preprocess_logits_for_metrics= None,
			**kwargs,

		):

		Trainer.__init__(self, model,args,data_collator,train_dataset,eval_dataset,tokenizer,compute_metrics, **kwargs)

		self.start_prune = False
		self.tokenizer=tokenizer
		self.lagrangian_warmup_steps=0
		self.lagrangian_optimizer = None
		self.kl_optimizer=None
		self.additional_args=additional_args
		self.target_sparsity=additional_args.target_sparsity
		self.args=args
		self.eval_counter = Eval_Counter()
		self.start_saving_best = True if additional_args.prune_method is None else False
		self.prune_method=additional_args.prune_method
		self.model=model #.cuda()
		self.optimizer =None
		self.lr_scheduler=None
		self.global_step=0
		self.model.config.output_attentions = False
		self.eval_dataset= eval_dataset
		log_level = args.get_process_log_level()
		logging.set_verbosity(log_level)
		logger.setLevel(log_level)
		
	def calculate_average(self,numbers):
		total = sum(numbers)
		count = len(numbers)
		average = total / count
		return float(average)

	def create_optimizer_and_scheduler(self,num_training_steps: int, build_l0_optimizer:bool=False):
		def log_params(param_groups, des):
			for i, grouped_parameters in enumerate(param_groups):
				logger.info(f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

		if self.optimizer is None:
			no_decay = ["layernorm" ,"norm"] #"bias",
			
			self.ib_params, self.ib_param_names, self.ib_layer_params, self.ib_layer_param_names = [], [], [], []
			
			for name, param in self.model.named_parameters():
				if 'z_mu' in name or 'z_logD' in name: #'z_mu' in name or 'z_logD' in name:
					param.requires_grad=True if self.prune_method is not None else False
					self.ib_params.append(param)
					self.ib_param_names.append(name)
				elif 'lambda' in name:
					param.requires_grad= True if self.prune_method is not None else False
				else:
					param.requires_grad= False
			
			self.main_model_params = []
			if self.main_model_params!=[]:
				self.optimizer = AdamW(
					self.main_model_params,
					betas=(0.9, 0.999), 
					eps=1e-8, 
				)
			if build_l0_optimizer: #only during pruning
				kl_params=[{
						"params": self.ib_params,
						"weight_decay": 0.0, #self.additional_args.vib_decay,
						"lr": self.additional_args.vib_learning_rate
							},
						]
				self.kl_optimizer = AdamW(
					kl_params,
					betas=(0.9, 0.999), 
					eps=1e-8, 
				)
				log_params(kl_params, "kl params")
		if build_l0_optimizer : #only during pruning
			self.model.set_lagpars() #initializing lag parameters to zeros
			lagrangian_params = [{
				"params": [self.model.lambda_1, self.model.lambda_2], 
				"weight_decay": 0.0,
				"lr": -self.additional_args.reg_learning_rate
			}]
			log_params(lagrangian_params, "l0 reg lagrangian params")
			print("\n init lag pars=",self.model.lambda_1.detach().item(),self.model.lambda_2.detach().item() )
			self.lagrangian_optimizer = AdamW(lagrangian_params,
													betas=(0.9, 0.999),
													eps=1e-8) 			

		if self.lr_scheduler is None:
			if build_l0_optimizer :
				if self.kl_optimizer is not None:					
					self.lr_scheduler =get_linear_schedule_with_warmup(self.kl_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps ) 
			else:
				if  self.optimizer is not None:
					self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps )
			

	def train(self):		
		train_dataloader = self.get_train_dataloader()
		num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
		num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1) #! 12272
		if self.prune_method is not None:
			lagrangian_warmup_steps = int(np.ceil(self.additional_args.lagrangian_warmup_epochs * num_update_steps_per_epoch)) #! 
			self.model.use_masking(mask=False,emb=self.additional_args.emb_mult)
			self.model.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
			self.model.set_start_and_target_sparsity(self.target_sparsity)
			logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

		
		self.t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
		total_train_batch_size = (  self.additional_args.bsz   )
		logger.info(" ***** Running training *****")
		logger.info("  Num examples = %d ", len(train_dataloader))
		logger.info("  Num Epochs = %d", self.args.num_train_epochs)
		logger.info("  Instantaneous batch size per device = %d",  self.additional_args.bsz)
		logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
		logger.info("  Gradient Accumulation steps = %d",  self.args.gradient_accumulation_steps)
		logger.info("  Total optimization steps = %d", self.t_total)
		
		self.global_step = 0
		self.epoch = 0
		self.total_flos = 0
		epochs_trained = 0
		tr_loss = torch.tensor(0.0)
		lag_loss = torch.tensor(0.0)	

		self.model.zero_grad()
		if self.optimizer is not None:
			self.optimizer.zero_grad()
		if self.kl_optimizer is not None:
			self.kl_optimizer.zero_grad()
		if self.lagrangian_optimizer is not None:
			self.lagrangian_optimizer.zero_grad()
		
		metrics={}
		st_time=time.time()
		expected_sparsity=0.0
		best_so_far=False		
		for epoch in range(epochs_trained, int(np.ceil(self.args.num_train_epochs))): 
			epoch_start = time.time()
			nsamples = len(train_dataloader) 			
			self.eval_counter.clear()
			nlls=[]
			for step, inputs in enumerate(train_dataloader): 
				if self.global_step==0 and self.prune_method is not None: 
					self.start_prune = True
					self.optimizer = None
					self.kl_optimizer=None
					self.lr_scheduler = None
					lr_steps = self.t_total 
					# reset the optimizer
					self.create_optimizer_and_scheduler(lr_steps, self.start_prune)

					logger.info("Starting pruning!")
					self.model.use_masking(mask=False,emb=self.additional_args.emb_mult)

				elif self.start_prune is not False:
					self.model.use_masking(mask=False,emb=self.additional_args.emb_mult) #so that kld is not 0.0 for proper backprop and changing mask to false after previous eval
				
				loss_terms = self.training_step(self.model, step,inputs)
				tr_loss = loss_terms["loss"]
				nlls.append(tr_loss)			
				
				if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
					len(train_dataloader) <= self.args.gradient_accumulation_steps
					and (step + 1) == len(train_dataloader)):
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm) 
					if self.start_prune is False: 
						self.optimizer.step()
						if self.lr_scheduler is not None:
							self.lr_scheduler.step() 
					else: #during pruning
						if self.optimizer is not None:
							self.optimizer.step() 
						if self.kl_optimizer is not None:
							self.kl_optimizer.step()
						if self.lr_scheduler is not None:
							self.lr_scheduler.step() 							
						if self.lagrangian_optimizer is not None:                        
							self.lagrangian_optimizer.step()
				  
					self.model.zero_grad()
					if self.optimizer is not None:
						self.optimizer.zero_grad()
					if self.kl_optimizer is not None:
						self.kl_optimizer.zero_grad()
					if self.lagrangian_optimizer is not None:
						self.lagrangian_optimizer.zero_grad()

					self.epoch = epoch + (step + 1) / nsamples                    
					ppl = torch.exp(tr_loss) #torch.stack(nlls).mean())
				if (self.global_step+1) % 100 == 0: 
					logger.info(f" step {(step//self.args.gradient_accumulation_steps)}/{(nsamples//self.args.gradient_accumulation_steps)}")
					logger.info("train ppl= %.2f", ppl.item())
					if self.prune_method is not None:
						logger.info(" expected spar= %.4f",loss_terms["expected_sparsity"])
						logger.info(" target sparsity warmed= %.4f",loss_terms["target_sparsity"])
						expected_sparsity = loss_terms["expected_sparsity"]
						logger.info("predic loss %.3f", tr_loss)
						logger.info("kld loss %.3f", loss_terms["kld_loss"].detach())
						logger.info("lagrangian loss %.3f", loss_terms["lagrangian_loss"])
						hidden_m_s=torch.sum(self.model.model.hidden_mask.get_mask_hard().int()!= 0)
						logger.info(" hidden size= %d", hidden_m_s.data)
						# pruned_model_size_info = self.model.calculate_model_size()									
						# metrics.update(pruned_model_size_info)
						# logger.info(f"Evaluating: {metrics}")						
						
				if self.prune_method is not None:
					if self.additional_args.target_sparsity > 0.70: #very high sparsity
						epsilon= 0.008
						max_epsilon = 0.006
						if (expected_sparsity - self.additional_args.target_sparsity >= -epsilon) and (expected_sparsity - self.additional_args.target_sparsity <= max_epsilon): #self.additional_args.sparsity_epsilon):
							self.start_saving_best = True
						else:
							self.start_saving_best = False
					else:
						epsilon = 0.008
						max_epsilon= 0.001
						if (expected_sparsity - self.additional_args.target_sparsity >= -epsilon) and (expected_sparsity - self.additional_args.target_sparsity <= max_epsilon): #self.additional_args.sparsity_epsilon):
							self.start_saving_best = True
							
						else:
							self.start_saving_best = False
						
					if self.start_saving_best and self.prune_method:
						if self.global_step % 50 == 0:
							self.eval_score = self.evaluate()
							best_so_far = self.eval_counter.update( self.epoch, self.global_step, self.eval_score)
							if best_so_far:
								pruned_model_size_info = self.model.calculate_model_size()
								metrics['acc']=self.eval_score
								metrics.update(pruned_model_size_info)
								logger.info(f"Evaluating: {metrics}")
								best_dir = os.path.join(self.additional_args.save_loc, "best")
								if not os.path.exists(best_dir):
									os.makedirs(best_dir)
								logger.info('Gathering statistics for saving masks')
								   
								mask_info = self.get_all_masks() 
								# Save the mask info -can be modifeid to save for epoch
								save_loc = os.path.join(best_dir, f'mask_info_{self.eval_score}.pkl') 
								with open(save_loc, 'wb') as handle:
									pkl.dump(mask_info, handle)      							
				self.global_step += 1
			ppl = torch.exp(torch.stack(nlls).mean())
			logger.info("Mean train ppl= %.2f", ppl)
			epoch_end = time.time()							
			logger.info( f"Epoch {epoch} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")		   
			if self.t_total > 0 and self.global_step >= self.t_total:
				break

		if self.args.past_index and hasattr(self, "_past"):
			# Clean the state at the end of training
			delattr(self, "_past")
		return TrainOutput(self.global_step, tr_loss.item() / self.global_step, metrics=metrics)	
	
	def get_all_masks(self):
		mask_info={}
		post_z_mu=  self.model.model.hidden_mask.post_z_mu.data
		post_z_logD=  self.model.model.hidden_mask.post_z_logD.data
		mask_info['hidden_mask_z_mu']= post_z_mu
		mask_info['hidden_mask_z_logD']= post_z_logD
		#intermediate weights
		for (i, layer) in enumerate(self.model.model.layers):
			mask_info[f'inter_layer{i}_z_mu']=layer.mlp.ib_1.post_z_mu.data
			mask_info[f'inter_layer{i}_z_logD']=layer.mlp.ib_1.post_z_logD.data
			mask_info[f'attn_layer{i}_z_mu']=layer.self_attn.ib_1.post_z_mu.data
			mask_info[f'attn_layer{i}_z_logD']=layer.self_attn.ib_1.post_z_logD.data
		return mask_info

	def evaluate(self):		
		logger.info("*** Evaluate ***")
		self.model.eval()
		self.model.use_masking(mask=True,emb=self.additional_args.emb_mult)
		eval_dataloader= self.get_eval_dataloader(self.eval_dataset)
		len_data=len(eval_dataloader)
		logger.info("number of examples %d", len(eval_dataloader))
		nlls=[]
		for step, inputs in enumerate(eval_dataloader):
			with torch.no_grad():
				outputs = self.model(**inputs).loss
				nlls.append(outputs)
			
		ppl = torch.exp(torch.stack(nlls).mean())
		logger.info("Eval perplexity = %.3f",ppl)
		return ppl
	
	def calculate_distillation_loss(self, teacher_outputs_log,teacher_outputs_states, student_outputs):
		layer_loss = self.calculate_layer_distillation_loss(teacher_outputs_states, student_outputs)
		distill_loss = layer_loss	
		ce_distill_loss = F.kl_div(
			input=F.log_softmax(
				student_outputs[1] / self.additional_args.distill_temp, dim=-1), 
			target=F.softmax(
				teacher_outputs_log / self.additional_args.distill_temp, dim=-1),
			reduction="batchmean") * (self.additional_args.distill_temp ** 2)

		loss = self.additional_args.distill_ce_loss_alpha * ce_distill_loss
		if distill_loss is not None:
			loss += self.additional_args.distill_loss_alpha * distill_loss	

		return distill_loss, ce_distill_loss, loss

	
	def training_step(self, model: torch.nn.Module, i: int, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
		model.train() 		
		distill_loss = None
		distill_ce_loss = None
		if self.additional_args.distill is True:
			teacher_reps_loc= os.path.join('../wiki_teacher_saved',f'teacher_reps_batch{i}.pkl')
			if os.path.exists(teacher_reps_loc):
				#print('Successfully loaded teacher reps')
				with open(teacher_reps_loc, 'rb') as handle:
					teacher_outputs = pkl.load(handle)
			else:
				logger.info("path to teacher not found")
				exit()
			student_outputs = model(**inputs, ) #! get the two outputs
			distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss( teacher_outputs[f'logits_bs{i}'],teacher_outputs[f'layer_hidden_states_{i}'], student_outputs)			
		else:
			# Forward pass through the model
			outputs = model(**inputs)      
			if self.args.past_index >= 0:
				self._past = outputs[self.args.past_index]     			
			loss = outputs['loss']			
		lagrangian_loss = None
		ppl_loss = loss.detach() / 1 #not shared data anymore		
		
		if self.start_prune and self.prune_method is not None:
			lagrangian_loss, expected_sparsity, target_sparsity_warmed,spar_loss_1,spar_loss_2 =  model.lagrangian_regularization((self.global_step/self.args.gradient_accumulation_steps))			
			loss += lagrangian_loss
			kld_loss = model.get_kld_loss(self.additional_args.emb_mult,1,self.additional_args.kl_factor)
			loss = loss + kld_loss
			
		else: 
			kld_loss=0.0
			
		if math.isnan(loss.detach()):
			logger.info("check loss, as it is NaN")
			logger.info("Printing all losses")
			logger.info("kld loss=",kld_loss)
			logger.info("lag loss=", lagrangian_loss.detach())
			#logger.info(" distill loss= %.3f",distill_loss)
			#logger.info(" ce_loss= %.3f", distill_ce_loss)
			logger.info(" tot loss= %.3f", loss)
			return
		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps
		loss.backward()
		return {"loss": ppl_loss,
				"lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
				"distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
				"distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None,
				"kld_loss":kld_loss,
				"expected_sparsity":expected_sparsity.detach() if self.start_prune else None,
				"target_sparsity": target_sparsity_warmed if self.start_prune else None}

	
