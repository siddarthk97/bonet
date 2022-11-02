#!/usr/bin/env bash

exptname="sorted_64_ctx_32_l_8_h"
dataset="superconductor/superconductor_800x128_sorted_64.p"
evaldataset="superconductor/superconductor_sorted_128x128_64_eval.p"
rtgs="0 0.01 0.05 0.1"
seeds="123 234 345 456 567"

python scripts/test_desbench.py --seeds $seeds --context_length 64 --dataset $dataset --experiment $exptname --dim 86 --task superconductor --cond_rtgs $rtgs --eval_dataset $evaldataset --layers 32 --heads 8 --init_len 64 --suffix "test" --no_update_rtg
