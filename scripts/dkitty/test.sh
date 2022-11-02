#!/usr/bin/env bash

exptname="sorted_with_32_l_8_h_final"
dataset="dkitty/dkitty_sorted_800x128_64_train.p"
eval_dataset="dkitty/dkitty_sorted_128x128_64_eval.p"
rtgs="0.0 0.01 0.05 0.1"

seeds="123 234 345 456 567"

python scripts/test_desbench_new.py --seeds $seeds --context_length 64 --dataset $dataset --experiment $exptname --dim 56 --task dkitty --cond_rtgs $rtgs --max_timestep 128 --init_len 64 --suffix "test" --layers 32 --heads 8 --no_update_rtg
