#!/usr/bin/env bash

exptname="$1"
dataset="$2"

for seed in 123 234 345 456 567
do
  python scripts/train_desbench_new.py --seed $seed --context_length 64 --epochs $3 --batch_size 128 --dataset $dataset  --experiment $exptname --dim 56 --task dkitty --max_timestep 128 --layers 32 --heads 8
done
