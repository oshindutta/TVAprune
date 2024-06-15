#!/bin/bash

python lora_ft_vib.py --model_name_or_path ${1} \
	--num_train_epochs 1 \
	--block_size 512 \
	--save_loc ${2} \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 8 \
	--do_train \
	--do_eval \
	--seed 42 \
	--overwrite_output_dir \
	--output_dir  ${2} \
	--dataset_name ${7} \
	--vib_learning_rate ${3} \
	--reg_learning_rate 0.1 \
	--target_sparsity ${4} \
	--prune_method 'vib' \
	--kl_factor 1e-6 \
	--lagrangian_warmup_epochs ${5} \
	--max_eval_samples 128 \
	--max_train_samples 4000 \
	--att_mul ${6} \
	--inter_mul 1 


