#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=0

python -m ATR.train \
    --enable_intra_modal_alignment \
    --enable_alignment_loss \
    --alignment_loss_weight=0.1 \
    --batch_size=32 \
    --exp_name=alignment_clotho_low_weight \
    --clotho