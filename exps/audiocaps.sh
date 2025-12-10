#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29505
export MASTER_ADDR=localhost
python -m ATR.train --compile --data_path=data --seed=13 --audiocaps --exp_name=dataset_audiocaps --batch_size=32