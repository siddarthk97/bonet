#!/usr/bin/env bash

exptname="sorted_with_64_ctx_32_l_8_h"
dataset="ant/ant_800x128_64_train.p"
eval_dataset="ant/ant_128x128_64_eval.p"
rtgs="0.0 0.01 0.05 0.1"
seeds=" 123 234 345 456 567"


python scripts/test_desbench.py --seeds $seeds --context_length 64 --dataset $dataset --eval_dataset $eval_dataset --experiment $exptname --dim 60 --task ant --cond_rtgs $rtgs --init_len 64 --heads 8 --layers 32 --suffix "test" --no_update_rtg

