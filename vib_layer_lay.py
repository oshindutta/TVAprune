""" Adapted for Transformer-like network. Original code found in VIBnet- https://github.com/zhuchen03/VIBNet """
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np


def reparameterize(mu, logvar, batch_size, sampling=True):
	# output dim: batch_size * dim
	if sampling:
		std = logvar.mul(0.5).exp_()
		if mu.get_device() != -1:
			eps = torch.FloatTensor(batch_size, std.size(0)).cuda(mu.get_device()).normal_()
		else:
			eps = torch.FloatTensor(batch_size, std.size(0)).normal_()
		eps = Variable(eps)
		return mu.view(1, -1) + eps * std.view(1, -1)
	else:
		return mu.view(1, -1)

class InformationBottleneck(nn.Module):
	def __init__(self, dim, mask_thresh=0, init_mag=9, init_var=0.01, kl_mult=1, sample_in_training=True, sample_in_testing=False, masking=True):
		super(InformationBottleneck, self).__init__()
		self.post_z_mu = nn.Parameter(torch.Tensor(dim))
		self.post_z_logD = nn.Parameter(torch.Tensor(dim))
		self.epsilon = 1e-8
		self.dim = dim
		self.sample_in_training = sample_in_training
		self.sample_in_testing = sample_in_testing
		self.masking = masking
		
		
		# initialization
		self.post_z_mu.data.normal_(1, init_var)
		self.post_z_logD.data.normal_(-init_mag, init_var)
		self.mask_thresh = mask_thresh
		self.kl_mult=kl_mult
		self.kld=0.		
		#self.flag=0 #to indicate initialization of self.mask
		

	def adapt_shape(self, src_shape, x_shape): #changed
		new_shape = src_shape   
		if len(src_shape)<len(x_shape): 
			new_shape=list(new_shape)
			new_shape = [1 for i in range(len(x_shape)-len(src_shape))] + new_shape 			
		return new_shape

	def get_logalpha(self):
		return self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

	def get_mask_hard(self, threshold=0):
		logalpha = self.get_logalpha()		
		hard_mask = (logalpha < threshold).float() 
		return hard_mask
	
	def get_sparse(self):
		beta_mul= 10 #atleast 5-for mistral, 10-llama for stable expected and actual sparsity measurement
		soft_mask = torch.sigmoid(-beta_mul*(self.post_z_logD - torch.log(self.post_z_mu.pow(2) + self.epsilon))) 
		return soft_mask 

	def get_mask_weighted(self, threshold=0):
		logalpha = self.get_logalpha()
		if self.flag==0:
			self.mask= self.get_mask_fine().cuda() 
			self.flag+=1
		mask = (logalpha < threshold).float() * self.post_z_mu.data  
		return mask	

	def forward(self, x):
		# 4 modes: sampling, hard mask, weighted mask, use mean value
		if self.masking: # if masking=True, apply mask directly				
			mask = self.get_mask_weighted(self.mask_thresh) 
			new_shape = self.adapt_shape(mask.size(), x.size())
			self.sparse=self.get_sparse().sum()
			return x * Variable(mask.view(new_shape))
			
		bsize = x.size(0)		
		if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
			z_scale = reparameterize(self.post_z_mu, self.post_z_logD, bsize,sampling=True)
		
		self.kld = self.kl_closed_form(x) if self.sample_in_training else 0
		new_shape = self.adapt_shape(z_scale.size(), x.size())
		self.sparse= self.get_sparse().sum() if self.sample_in_training else 0
		return x* z_scale.view(new_shape)

	def kl_closed_form(self, x):
		new_shape = self.adapt_shape(self.post_z_mu.size(), x.size())
		h_D = torch.exp(self.post_z_logD.view(new_shape))
		h_mu = self.post_z_mu.view(new_shape)
		if x.sum().eq(0).item():
			print("\n all input 0ed")
			KLD= torch.Tensor([0.0])
			return KLD
		
		KLD = torch.sum(torch.log(1 + h_mu.pow(2)/(h_D + self.epsilon) ),dtype=torch.float32) 		
		return KLD * 0.5 * self.kl_mult
