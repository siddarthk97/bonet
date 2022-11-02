#!/usr/bin/env bash

exptname="tfbind8_test"
dataset="tf-bind-8/tfbind8_sorted_800x128_64_train.p"
test_dataset="tf-bind-8/tfbind8_sorted_128x128_64_eval.p"
rtgs="0.0 0.01 0.05 0.1 0.5 1 2 3 4"
# rtgs="6 7 8 9 10 11 12 13 14"
# rtgs="0.0 0.01 0.05"
seeds="123 234 345 456 567"
# for seed in 123 234 345 456 567 678 789 890 901 012

python3.9 scripts/test_desbench.py --seeds $seeds --context_length 64 --dataset $dataset --eval_dataset $test_dataset --experiment $exptname --dim 8 --task tf-bind-8 --cond_rtgs $rtgs --discrete --init_len 64 --vocab_size 4

