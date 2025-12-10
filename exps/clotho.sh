#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29502
export MASTER_ADDR=localhost
python -m ATR.train --compile --data_path=data --seed=13 --ablate_clean_setup --exp_name=dataset_clotho_test --batch_size=32