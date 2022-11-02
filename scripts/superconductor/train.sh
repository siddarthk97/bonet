#!/usr/bin/env bash

exptname="sorted_64_ctx_32_l_8_h"
dataset="superconductor/superconductor_800x128_sorted_64.p"
for seed in 123 234 345 456 567
do
  python scripts/train_desbench.py --seed $seed --context_length 64 --epochs 25 --batch_size 128 --dataset $dataset  --experiment $exptname --dim 86 --task superconductor --layers 32 --heads 8
done
