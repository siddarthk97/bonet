#!/usr/bin/env bash

exptname="8_heads_8_layers_new"
dataset="tf-bind-8/tfbind8_sorted_800x128_64_train.p"
rtg=10
# for seed in 123 234 345 456 567 678 789 890 901 012
# for seed in 123 234 345 456 567
for seed in 567 
do
  python scripts/train_desbench_new.py --seed $seed --context_length 64 --epochs 75 --batch_size 64 --lr 5e-5  --dataset $dataset  --experiment $exptname --dim 8 --task tf-bind-8 --discrete --heads 8 --layers 8 --vocab_size 4 1>outputs/tf-bind-8/new_${seed}.txt 2>&1 &
done
