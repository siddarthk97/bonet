#!/usr/bin/env bash

exptname="16_heads_32_layers"
dataset="tf-bind-10/tfbind10_sorted_4000x128_64_train.p"
# rtg=10
# for seed in 123 234 345 456 567 678 789 890 901 012
# for seed in 123 234 345 456 567
for seed in 234 345
do
  #python scripts/train_desbench.py --seed $seed --context_length 64 --epochs 100 --batch_size 128 --dataset $dataset  --experiment $exptname --dim 10 --task tf-bind-10 --discrete --vocab_size 4
  python scripts/train_desbench.py --seed $seed --context_length 64 --epochs 75 --batch_size 128 --dataset $dataset  --experiment $exptname --dim 10 --task tf-bind-10 --heads 16 --layers 32 --discrete --vocab_size 4
done

# python scripts/compute_aggregate_statistics.py --cond_rtg $rtg --experiment $exptname --task dkitty
