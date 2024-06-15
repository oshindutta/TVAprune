#!/bin/bash

python lora_ft_vib.py --model_name_or_path ${1} \
	--save_loc ${2} \
	--num_train_epochs 1 \
	--block_size 512 \
	--lora_r 128 \
	--learning_rate 1e-4 \
	--lora_alpha_ratio 4 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 8 \
	--do_train \
	--do_eval \
	--seed 42 \
	--output_dir ${2} \
	--overwrite_output_dir \
	--dataset_name "wikitext2" \
	--max_eval_samples 128 \
	--distill True \
	--distill_ce_loss_alpha 0.01 \
	--max_train_samples 30000 \
	--epoch_f 2.0 \
	--mask_loc ${3}




