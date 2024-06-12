###2 hrs to prune llama1,2; 2 to finetune for llama2 on 1 ep on wiki5k , 3 hrs for mistral,llama3; 7 hrs to finetune on c4 for ep1 , 14 for 2eps
cd tva_prune/


##to prune LLaMA-7B
python lora_ft_vib.py --model_name_or_path "../llama-2-7bhf" \
	--num_train_epochs 1 \
	--block_size 512 \
	--save_loc llama2_saves_tva/c4 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 8 \
	--do_train \
	--do_eval \
	--seed 42 \
	--overwrite_output_dir \
	--output_dir  llama2_saves_tva/c4 \
	--dataset_name "c4" \
	--vib_learning_rate 0.05 \
	--reg_learning_rate 0.1 \
	--target_sparsity 0.50 \
	--prune_method 'vib' \
	--kl_factor 1e-6 \
	--lagrangian_warmup_epochs 0.1 \
	--max_eval_samples 128 \
	--max_train_samples 4000 \
	--att_mul 64 \
	--inter_mul 1 

##To prune Mistral-7B
python lora_ft_vib.py --model_name_or_path "../mistral_7b" \
	--num_train_epochs 1 \
	--block_size 512 \
	--save_loc mistral_saves_tva/c4 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 8 \
	--do_train \
	--do_eval \
	--seed 42 \
	--overwrite_output_dir \
	--output_dir  mistral_saves_tva/c4 \
	--dataset_name "c4" \
	--vib_learning_rate 0.1 \
	--reg_learning_rate 0.1 \
	--target_sparsity 0.50 \
	--prune_method 'vib' \
	--kl_factor 1e-6 \
	--lagrangian_warmup_epochs 0.1 \
	--max_eval_samples 128 \
	--max_train_samples 4000 \
	--att_mul 256 \
	--inter_mul 1 \
	--emb_mult 30 

##To prune LLaMA-3-8B
python lora_ft_vib.py --model_name_or_path "../llama3_8b/llama3_8b" \
	--num_train_epochs 1 \
	--block_size 400 \
	--save_loc llama3_saves_tva/c4 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 8 \
	--do_train \
	--do_eval \
	--seed 42 \
	--overwrite_output_dir \
	--output_dir  llama3_saves_tva/c4 \
	--dataset_name "c4" \
	--vib_learning_rate 0.1 \
	--reg_learning_rate 0.1 \
	--target_sparsity 0.50 \
	--prune_method 'vib' \
	--kl_factor 1e-6 \
	--lagrangian_warmup_epochs 0.1 \
	--max_eval_samples 128 \
	--max_train_samples 4000 \
	--att_mul 512 \
	--inter_mul 1 \
	--emb_mult 30

#for finetuning	a model after pruning with learned VIB masks,dimension_adaptation,weight_updation
python lora_ft_vib.py --model_name_or_path "../mistral_7b" \
	--save_loc llama2_saves/c4/model_att256_int1_bt128_wiki \
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
	--output_dir lora_ft \
	--overwrite_output_dir \
	--dataset_name "wikitext2" \
	--max_eval_samples 128 \
	--distill True \
	--distill_ce_loss_alpha 0.01 \
	--max_train_samples 30000 \
	--epoch_f 2.0 \
	--mask_loc 'mistral_saves_tva/c4/best/mask_info_13.952906608581543.pkl'

# #for evaluating on eleuther
python lora_ft_vib.py --model_name_or_path "../mistral_7b" \
	--save_loc mistral_saves_tva/c4/model_att256_int1_bt128/new \
	--overwrite_output_dir \
	--mask_loc 'mistral_saves_tva/c4/best/mask_info_18.891157150268555.pkl' \
	--output_dir lora_ft \
	--do_zero_eval True 


