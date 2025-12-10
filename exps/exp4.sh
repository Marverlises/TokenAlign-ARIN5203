cd ..
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29558
echo "Starting experiment: alignment_audiocaps_low_weight"
python -m ATR.train \
    --enable_intra_modal_alignment \
    --enable_alignment_loss \
    --alignment_loss_weight=0.1 \
    --batch_size=32 \
    --exp_name=alignment_audiocaps_low_weight_match \
    --audiocaps \
    --enable_matching_loss

echo "Experiment completed: alignment_audiocaps_low_weight"
