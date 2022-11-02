#!/usr/bin/env bash

exptname="tfbind8_test"
dataset="tf-bind-8/tfbind8_sorted_800x128_64_train.p"
evaldataset="tf-bind-8/tfbind8_sorted_128x128_64_eval.p"
for seed in 123 234 345 456 567
do
  python scripts/train_desbench.py --seed $seed --context_length 64 --epochs 75 --batch_size 128  --dataset $dataset  --experiment $exptname --dim 8 --task tf-bind-8 --discrete --heads 8 --layers 8 --vocab_size 4
done

# python scripts/compute_aggregate_statistics.py --cond_rtg $rtg --experiment $exptname --task dkitty
