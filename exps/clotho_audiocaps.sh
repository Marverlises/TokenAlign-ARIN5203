#!/bin/bash
cd ..
#export CUDA_VISIBLE_DEVICES=2,3
export MASTER_PORT=29503
export MASTER_ADDR=localhost
python -m ATR.train --compile --data_path=data --seed=13 --ablate_clean_setup --audiocaps --exp_name=dataset_clotho_audiocaps --batch_size=32