# **TVA-Prune: Token Dependency-aware Variational Adapted Pruning**

[![ICML 2024](https://img.shields.io/badge/ICML%20ES--FoMo-2024-blue)](https://openreview.net/forum?id=cqhAzteLzc)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](requirements.txt)

:page_facing_up: [Paper](https://openreview.net/forum?id=cqhAzteLzc) | **Efficient Systems for Foundation Models @ ICML 2024**

![TVA-Prune](https://raw.githubusercontent.com/oshindutta/TVAprune/main/figures/method.svg)

## Introduction

Deploying large language models in production is expensive — serving them at scale adds significant GPU memory pressure and inference latency. Structured pruning offers a path to smaller, faster dense models without architectural changes, but prior methods (LLM-Pruner, FLAP, Bonsai) either sacrifice too much accuracy or require impractically long compression runs.

**TVA-Prune** addresses this through three ideas applied together for the first time:

- **Global token-dependency awareness**: VIB masks are conditioned on all preceding layers' representations — not just local statistics — enabling more principled decisions about which embedding dimensions to remove.
- **GQA-aware head pruning**: Supports modern architectures (**Mistral-7B, LLaMA-3-8B**) where removing a KV head must co-prune all its associated query heads to preserve attention group integrity.
- **Hardware-adapted dimensions**: After pruning, weight matrix dimensions are snapped to GPU Tensor Core block sizes (e.g., **128×256**), converting theoretical FLOPs savings into a **60% inference speedup** over unpruned models.

The result: up to **1.8× overall inference speedup** at **20–50% sparsity**, perplexity competitive with the best finetuned baselines, and a **90% reduction in pruning time** (**2 GPU-hours vs. 40 for Bonsai**) — all while compressing efficiently on a single **NVIDIA A100 (40GB)**.

## Results

### Standard MHA models — ~20% sparsity, Wikitext-2 PPL

| Method | LLaMA-7B PPL ↓ | LLaMA-7B Speedup | LLaMA-2-7B PPL ↓ | LLaMA-2-7B Speedup |
|---|---|---|---|---|
| Unpruned | 5.68 | 1× | 5.11 | 1× |
| Wanda-sp | 366.43 | 1.24× | 97.70 | 1.29× |
| LLM-Pruner | 112.44 | 1.23× | 95.26 | 1.29× |
| FLAP | 35.10 | 1.26× | 25.40 | 1.32× |
| Bonsai | 22.62 | 1.26× | 19.24 | 1.28× |
| **TVA-Prune (ours)** | **18.5** | **1.75×** | **14.15** | **1.8×** |
| **TVA-Prune† (ours)** | **10.58** | **1.75×** | **9.58** | **1.8×** |

†Finetuned with LoRA. All inference on NVIDIA A100 (40GB).

### GQA models — 50% sparsity, Wikitext-2 PPL

| Method | Mistral-7B PPL ↓ | Mistral Speedup ↑ | Mistral Tokens/s ↑ | LLaMA-3-8B PPL ↓ | LLaMA-3 Speedup ↑ | LLaMA-3 Tokens/s ↑ |
|---|---|---|---|---|---|---|
| Unpruned | 4.77 | 1× | 24.78 | 5.57 | 1× | 25.13 |
| Wanda-sp-gq | 116 | 1.1× | 27.26 | 106 | 1.1× | 27.64 |
| FLAP-gq | 34.97 | 1.28× | 31.73 | 34.90 | 1.2× | 30.16 |
| **TVA-Prune (ours)** | **18.37** | **1.67×** | **41.39** | **27.50** | **1.61×** | **40.94** |
| **TVA-Prune† (ours)** | **10.12** | **1.67×** | **41.39** | — | — | — |

### Zero-shot reasoning — 50% pruned LLaMA-7B

| Method | PPL ↓ | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | Avg ↑ | Compression Time |
|---|---|---|---|---|---|---|---|---|---|
| Unpruned | 5.68 | 76.5 | 79.8 | 76.1 | 70.1 | 72.8 | 47.6 | 70.48 | — |
| LLM-Pruner† | 16.41 | 60.28 | 69.31 | 47.06 | 53.43 | 45.96 | 29.18 | 45.95 | 1 hr |
| FLAP | 31.8 | 60.21 | 67.52 | 40.0 | 57.54 | 49.66 | 28.49 | 50.57 | 0.3 hrs |
| LoRAPrune† | 11.60 | 61.88 | 71.53 | 47.86 | 55.01 | 45.13 | 31.62 | 52.17 | >24 hrs |
| Bonsai† | 10.92 | 67.22 | 60.85 | 43.09 | 61.64 | 54.92 | 26.28 | 52.33 | 40 hrs |
| **TVA-Prune† (ours)** | **10.58** | 63.27 | 68.56 | 42.0 | 57.38 | **56.97** | 26.46 | **52.44** | **2 hrs** |

†Finetuned with LoRA. TVA-Prune matches top accuracy at **10–20× lower compression time** than comparable methods.

#### Supported LLMs:
- [Llama-2 Hugging Face](https://huggingface.co/meta-llama)
- [Llama-3 Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Mistral Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1)

## Table of Contents
- [Results](#results)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Evaluation of our pruned models](#evaluation-of-our-pruned-models)
- [Example of Pruning](#example-of-pruning)
- [Finetuning with LoRA](#finetuning-with-lora)
- [Citation](#citation)

## Quick Start

```bash
# 1. Install
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# 2. Prune (example: 20% sparsity on c4)
bash script/tva_prune.sh /path/to/model /path/to/save/masks 0.05 0.2 0.1 256 c4

# 3. (Optional) Fine-tune with LoRA
bash script/tva_fine.sh /path/to/model /path/to/save/finetuned /path/to/mask

# 4. Evaluate
python lora_ft_vib.py --model_name_or_path /path/to/model --mask_loc /path/to/mask \
    --save_loc /path/to/results --output_dir /path/to/output --do_eval --overwrite_output_dir
```

> Pre-computed masks for Mistral-7B and LLaMA-3-8B are in `mistral_saves_tva/` and `llama3_saves_tva/`.

## Installation
```
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
#### Package versions tested on:
- torch 2.2.1
- transformers 4.40.2
- accelerate 0.30.1
- datasets 2.19.1

#### To evaluate on [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
```
cd lm_evaluation_harness_new
pip install -e .
```
## Evaluation of our pruned models
Our pruning masks to prune Mistral-7B and LLaMA-3-7B are in `mistral_saves_tva` and `llama3_saves_tva` respectively. 
The speedup may differ slightly depending on the machine.
```
python lora_ft_vib.py --model_name_or_path [PATH TO UNPRUNED MODEL] \
	--do_eval \
	--overwrite_output_dir \
	--save_loc [PATH TO SAVE RESULTS] \
	--mask_loc [PATH TO SAVED MASK] \
	--output_dir [PATH TO SAVE MODELS] \
	--do_zero_eval True
```
``--write_out True`` can write out into a file the loglikelihood results and examples of zero-shot tasks
``--mask_loc`` can be assigned `'mistral_saves_tva/mask_info_18.891157150268555.pkl'` to denote path to our pruning mask for Mistral-7B

## Example of Pruning
Pruning with TVA-Prune to replicate our model in Table 1
```
UNPRUNED_MODEL=[PATH TO MODEL]
MASK_SAVE=[PATH TO SAVE MASKS]
VIB_LR=0.05 #can be changed to 0.1 for target sparsity>0.5
TARGET_SPARSITY=0.2 
LAGRANGIAN_WARMUP=0.1 #can be changed to 0.2 for target sparsity>0.6
DATASET=c4 #can be changed to wikitext2 for task-specific pruning
ATT_MUL=256 #can be changed to 512 to pruned more attention weights for target sparsity>0.6
bash script/tva_prune.sh $UNPRUNED_MODEL $MASK_SAVE $VIB_LR $TARGET_SPARSITY $LAGRANGIAN_WARMUP $ATT_MUL $DATASET
```

## Finetuning with [LoRA](https://github.com/microsoft/LoRA)
Speed-up over un-pruned model is seen at the start of finetuning.
```
UNPRUNED_MODEL=[PATH TO MODEL]
PATH_MASK=[PATH TO SAVED MASK]
SAVE_MODEL=[PATH TO SAVE MODEL]
bash script/tva_fine.sh $UNPRUNED_MODEL $SAVE_MODEL $PATH_MASK
```

## Acknowledgements
* VIB pruning inspired from https://github.com/zhuchen03/VIBNet/blob/master/ib_layers.py
* Evaluations from [LLM-Pruner](https://github.com/horseee/LLM-Pruner)

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

Please cite our papers if you use TVAprune in your work:

```bibtex
@inproceedings{dutta2024tvaprune,
	title={Efficient LLM Pruning with Global Token-Dependency Awareness and Hardware-Adapted Inference},
	author={Dutta, Oshin and Gupta, Ritvik and Agarwal, Sumeet},
	booktitle={Workshop on Efficient Systems for Foundation Models II @ ICML2024},
	year={2024},
	url={https://openreview.net/forum?id=cqhAzteLzc}
}

@article{dutta2024vtrans,
  title={VTrans: Accelerating Transformer Compression with Variational Information Bottleneck based Pruning},
  author={Dutta, Oshin and Gupta, Ritvik and Agarwal, Sumeet},
  journal={arXiv preprint arXiv:2406.05276},
  year={2024}
}
```
