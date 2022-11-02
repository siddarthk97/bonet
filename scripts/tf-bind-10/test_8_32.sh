#!/usr/bin/env bash

exptname="8_heads_32_layers"
dataset="tf-bind-10/tfbind10_sorted_4000x128_64_train.p"
test_dataset="tf-bind-10/tfbind10_sorted_128x128_64_eval.p"
# rtgs="0.1 0.5 0.8 1 2 3 4 5"
# rtgs="0.1 0.5 0.8 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
rtgs="0.0 0.01 0.05 0.1 0.5"
seeds="123 456"
# for seed in 123 234 345 456 567 678 789 890 901 012

python scripts/test_desbench.py --seeds $seeds --context_length 64 --dataset $dataset --eval_dataset $test_dataset --experiment $exptname --dim 10 --task tf-bind-10 --cond_rtgs $rtgs --discrete --vocab_size 4 --heads 8 --layers 32 --init_len 64 

