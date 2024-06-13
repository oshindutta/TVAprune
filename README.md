## Introduction 
**TVAprune: Token Dependency-aware Variational Adapted pruning**

#### Package versions tested on:
- torch 2.2.1
- transformers 4.40.2
- accelerate 0.30.1
- datasets 2.19.1

#### Why TVAprune for pruning LLMs:
- [x] **Structured Pruning**: Suitable to deploy compressed dense models on devices
- [x] **Efficient Compression**: Better performance without finetuning model parameters than other structured pruning methods (LLM-pruner, Bonsai)
- [x] **Faster Inference**: Pruned models infer faster than other methods
- [x] **Low Resource Compression**: Requires only 1 GPU (tested on NVIDIA A6000-48GB and NVIDIA A100(40GB))

#### Supported LLMs:
- [x] [Llama-2 Hugging Face](https://huggingface.co/meta-llama)
- [x] [Llama-3 Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [x] [Mistral Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1)


## Table of Contents
- [Installation](#installation)
- [Example of Pruning](#example-of-pruning)
- [Step-by-step Instructions](#step-by-step-instructions)

## Installation
```
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirement.txt
```
## To evaluate our Mistral and LLaMA-3 pruned models:
```
python lora_ft_vib.py --model_name_or_path [PATH TO UNPRUNED MODEL] \
	--do_eval \
	--overwrite_output_dir \
	--mask_loc [PATH TO PRUNING MASK] \
	--do_zero_eval True
```
## Example of Pruning

Pruning with TVAprune to replicate our model in Table 1
```
bash script/llama_prune.sh
```

## Step-by-step Instructions  
### Pruning

Arguments:
- ``vib_learning_rate``: is the set learning rate of VIB parameters
- ``target_sparsity``: is the amount of sparsity or weights to be removed. 0.5 indicates removal of 50% of the model parameters.
- ``lagrangian_warmup_epochs``: indicates how slowly compression should progress. 0.1 indicates warmup steps for lagrangian sparsity loss is set to 10% of total steps
- ``save_loc``: directory to save the VIB masks

### Finetuning with [LoRA](https://github.com/microsoft/LoRA)

Speed-up over un-pruned model is seen at the start of finetuning. Our obatined VIB mask stored in /best/ is used below for finetuning. 

Arguments:
- ``lora_r`` and ``lora_alpha`` control the LoRA configuration
- ``distill_ce_loss_alpha``: multplier of logit-based distillation loss
- ``mask_loc``: uploading a saved mask
- ``dataset_name``: we finetune on wikitext2 , but previous techniques have finetuned on c4 have higher train samples. It may be changed to c4 to get better perplexity.

### To evaluate on [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
```
cd lm-evaluation-harness
pip install -e .
```
