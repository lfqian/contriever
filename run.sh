#!/bin/bash

# Set CUDA configurations
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# Run training with modified parameters
python train.py \
    --retriever_model_id bert-base-uncased \
    --pooling average \
    --augmentation delete \
    --prob_augmentation 0.1 \
    --train_data "encoded-data/bert-base-uncased/oa_comm_xml" \
    --loading_mode split \
    --ratio_min 0.1 \
    --ratio_max 0.5 \
    --chunk_length 256 \
    --momentum 0.9995 \
    --moco_queue 131072 \
    --temperature 0.05 \
    --warmup_steps 20000 \
    --total_steps 500000 \
    --lr 0.00005 \
    --scheduler linear \
    --optim adamw \
    --per_gpu_batch_size 32 \
    --output_dir qian\
    --save_freq 100 \
    --eval_freq 100 \
    --gradient_accumulation_steps 2