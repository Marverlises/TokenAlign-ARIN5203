#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=2,3
export MASTER_PORT=29559
echo "Starting experiment: alignment_clotho_low_weight"
python -m ATR.train \
    --enable_intra_modal_alignment \
    --enable_alignment_loss \
    --alignment_loss_weight=0.1 \
    --batch_size=32 \
    --exp_name=alignment_clotho_low_weight_match \
    --clotho \
    --enable_matching_loss

echo "Experiment completed: alignment_clotho_low_weight"