#!/bin/bash
cd ..
#export CUDA_VISIBLE_DEVICES=1
source /share/project/baiyu/interpreter/.virtualenvs/dcase_env/bin/activate

python -m ATR.train \
    --enable_intra_modal_alignment \
    --enable_alignment_loss \
    --alignment_loss_weight=0.1 \
    --batch_size=32 \
    --exp_name=alignment_audiocap_low_weight-reproduce \
    --audiocaps \
    --load_ckpt_path=/share/project/baiyu/project/dcase2025_task6_baseline/checkpoints/alignment_audiocap_low_weight-reproduce/last.ckpt \
    --max_epochs=4