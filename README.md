## Introduction 
**TVAprune: Token Dependency-aware Variational Adapted pruning**
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
- [Evaluation of our models](#Evaluation of our pruned models)
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
## To evaluate our Mistral and LLaMA-3 pruned models:
Our pruning masks to prune Mistral-7B and LLaMA-3-7B are in mistral_saves_tva and llama3_saves_tva respectively. 
The speedup may differ slightly depending on the machine.
```
python lora_ft_vib.py --model_name_or_path [PATH TO UNPRUNED MODEL] \
	--do_eval \
	--overwrite_output_dir \
	--mask_loc [PATH TO PRUNING MASK] \
	--do_zero_eval True
```
- ``mask_loc``can be assigned 'mistral_saves_tva/mask_info_18.891157150268555.pkl'
## Example of Pruning

Pruning with TVAprune to replicate our model in Table 1
```
bash script/llama_prune.sh
```

### Finetuning with [LoRA](https://github.com/microsoft/LoRA)

Speed-up over un-pruned model is seen at the start of finetuning. Our obatined VIB mask stored in /best/ is used below for finetuning. 

Arguments:
- ``lora_r`` and ``lora_alpha`` control the LoRA configuration
- ``distill_ce_loss_alpha``: multplier of logit-based distillation loss
- ``mask_loc``: uploading a saved mask
- ``dataset_name``: we finetune on wikitext2 , but previous techniques have finetuned on c4 have higher train samples. It may be changed to c4 to get better perplexity.

