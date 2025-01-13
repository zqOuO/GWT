#!/bin/bash

export type=haar
export level=2
export scale=2.5e-1
export optimizer=wave_adamw
export lr=1e-2

torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr $lr \
    --scale $scale \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 100000 \
    --wave_type $type \
    --level $level \
    --optimizer $optimizer
wait