# Breaking Memory Barriers: Gradient Wavelet Transform Enhances LLMs Training

This repo contains the official pre-release version of GWT algorithm for the paper [Breaking Memory Barriers: Gradient Wavelet Transform Enhances LLMs Training](), and is highly build upon the previous work proposed by [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) and the [Pytorch wavelet toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox). Currently, ptwt does not support fp16 format computations, so we made some adjustments to this box.

<div align="center">
  <img title="Visuliazation the approximate coefficients of 2-level DHT on image (rescaled for visualizability)" img src=".\figures\wavelet_transform_visula_cat.jpg" alt="Image 2" style="width: 800px; margin: 0 auto;">
</div>

Visualization the approximate coefficients of 2-level Haar transform on image (rescaled for visual clarity)

## Reproducibility

### Install
Our experiments were mainly conducted by Python 3.11.9 with a CUDA version [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive). See the [requirements.txt](https://github.com/zqOuO/Score-based-Generative-Models-with-Adaptive-Momentum/blob/main/ImageGeneration/requirements.txt), run Install from pip:
```bash 
pip install -r requirements
```

## Usage

```python
from galore_torch import WaveAdamW
param_groups = [{'params': regular_params}, {'params': wave_params, 'scale': args.scale, 				'wave_type': args.wave_type, 'level': args.level, 'boundary': 			 				args.boundary}]
optimizer = WaveAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
```
## Benchmark 1: Pre-Training LLaMA on C4 dataset
The scripts for pre-training LLaMA models on C4 dataset are in scripts/benchmark_c4 folder. The C4 dataset is available for download from [Hugging Face](https://huggingface.co/datasets/allenai/c4). We present the pre-trained LLaMA models in this [link](https://www.alipan.com/s/DvBSH7TkRBB).

### Script for pre-training LLaMA 1B model
```bash
#!/bin/bash

export type=haar
export level=2
export scale=2.5e-1
export optimizer=wave_adamw
export lr=1e-2

torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr $lr \
    --scale $scale \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 100000 \
    --wave_type $type \
    --level $level \
    --optimizer $optimizer

```

## Benchmark 2: Fine-Tuning RoBERTa on GLUE tasks
`run_glue.py` is the main script for fine-tuning RoBERTa models on GLUE tasks.

```bash
#!/bin/bash

export scale=2
export level=2
export task_name=mnli
export type=haar
export level=2
export lr=5e-6

python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name $task_name \
    --enable_wave \
    --lora_all_modules \
    --max_length 512 \
    --seed 1234 \
    --scale $scale \
    --per_device_train_batch_size 16 \
    --wave_type $type \
    --level $level \
    --learning_rate $lr \
    --num_train_epochs 30 \
    --output_dir $your_direction

```

## Citation