#!/usr/bin/env bash

exptname="sorted_with_64_context_32_layers_16_heads"
dataset="tf-bind-8/tfbind8_sorted_800x128_64_train.p"
evaldataset="tf-bind-8/tfbind8_sorted_128x128_64_eval.p"
rtgs="0.0 0.01 0.05 0.1"
seeds="123 234 345 456 567"

python scripts/test_desbench.py --seeds $seeds --context_length 64 --dataset $dataset --eval_dataset $evaldataset --experiment $exptname --dim 8 --task tf-bind-8 --cond_rtgs $rtgs --discrete --init_len 64 --vocab_size 4 --suffix "final_no_rtg_update_testing" --layers 32 --heads 16 --no_update_rtg

