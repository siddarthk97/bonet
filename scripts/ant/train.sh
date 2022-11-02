#!/usr/bin/env bash

exptname="sorted_with_64_context_32_layers_8_heads"
dataset="ant/ant_sorted_800x128_64_train.p"
for seed in 123 234 345 456 567
do
  python scripts/train_desbench.py --seed $seed --context_length 64 --epochs 75 --batch_size 64 --dataset $dataset  --experiment $exptname --dim 60 --task ant --heads 8 --layers 32
done
