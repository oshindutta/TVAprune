# **TVAprune: Token Dependency-aware Variational Adapted pruning**
## Introduction 
- [x] Includes pruning of Grouped-Query Attention (GQA) based models
- [x] Post prune instant weight update to recover performance
- [x] Post-prune dimension adjustment to make weight matrix dimensions conform to dimensions used by GPU for better paralellism and hence faster inference
 
#### Why TVAprune for pruning LLMs:
- [x] **Structured Pruning**: Suitable to deploy compressed dense models on devices
- [x] **Efficient Compression**: Better performance without finetuning model parameters than other structured pruning methods (LLM-pruner, Bonsai, FLAP)
- [x] **Faster Inference**: Pruned models infer faster than other methods
- [x] **Low Resource Compression**: Requires only 1 GPU (tested on NVIDIA A100(40GB))

#### Supported LLMs:
- [x] [Llama-2 Hugging Face](https://huggingface.co/meta-llama)
- [x] [Llama-3 Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [x] [Mistral Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1)


## Table of Contents
- [Installation](#installation)
- [Evaluation of our pruned models](#Evaluation-of-our-pruned-models)
- [Example of Pruning](#example-of-pruning)
- [Finetuning with LoRA](#fine)

## Installation
```
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirement.txt
```
#### Package versions tested on:
- torch 2.2.1
- transformers 4.40.2
- accelerate 0.30.1
- datasets 2.19.1

#### To evaluate on [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
```
cd lm-evaluation-harness
pip install -e .
```
## Evaluation of our pruned models
Our pruning masks to prune Mistral-7B and LLaMA-3-7B are in mistral_saves_tva and llama3_saves_tva respectively. 
The speedup may differ slightly depending on the machine.
```
python lora_ft_vib.py --model_name_or_path [PATH TO UNPRUNED MODEL] \
	--do_eval \
	--overwrite_output_dir \
	--save_loc [PATH TO SAVE RESULTS] \
	--mask_loc [PATH TO MASK LOCATION] \
	--output_dir [PATH TO SAVE MODELS] \
	--do_zero_eval True
```
``--write_out True`` can write out into a file the loglikelihood results and examples of zero-shot tasks
``--mask_loc``can be assigned 'mistral_saves_tva/mask_info_18.891157150268555.pkl' to denote path to our pruning mask for Mistral-7B

## Example of Pruning
Pruning with TVAprune to replicate our model in Table 1
```
UNPRUNED_MODEL=[PATH TO MODEL]
MASK_SAVE=[PATH TO SAVE MASKS]
VIB_LR=0.05 #can be changed to 0.1 for target sparsity>0.5
TARGET_SPARSITY=0.2 
LAGRANGIAN_WARMUP=0.1 #can be changed to 0.2 for target sparsity>0.6
ATT_MUL=256 #can be changed to 512 to pruned more attention weights for target sparsity>0.6
bash script/tva_prune.sh $UNPRUNED_MODEL $MASK_SAVE $VIB_LR $TARGET_SPARSITY $LAGRANGIAN_WARMUP $ATT_MUL
```

### Finetuning with [LoRA](https://github.com/microsoft/LoRA)
Speed-up over un-pruned model is seen at the start of finetuning.
```
UNPRUNED_MODEL=[PATH TO MODEL]
PATH_MASK=[PATH TO SAVED MASK]
SAVE_MODEL=[PATH TO SAVE MODEL]
Bash script/tva_fine.sh $UNPRUNED_MODEL $SAVE_MODEL $PATH_MASK
```


