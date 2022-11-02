#!/usr/bin/env bash

exptname="8_heads_8_layers_64_batch"
dataset="tf-bind-8/tfbind8_sorted_800x128_64_train.p"
test_dataset="tf-bind-8/tfbind8_sorted_128x128_64_eval.p"
rtgs="0.0 0.01 0.05 0.1 0.5 1.0 2.0"
# rtgs="6 7 8 9 10 11 12 13 14"
# rtgs="0.0 0.01 0.05"
seeds="345"
# for seed in 123 234 345 456 567 678 789 890 901 012

# python scripts/run_dtbo_dkitty.py --seed $seed --context_length 30 --epochs 75 --batch_size 128 --dataset DKitty_800x50_bin.p --experiment dkitty_morphology_continuous --test --cond_rtg 10 # --train
# python scripts/run_dtbo_desbench.py --seed $seed --context_length 30 --epochs 75 --batch_size 128 --dataset DKitty_800x50_new_bin.p --experiment dkitty_morphology_continuous_correct --dim 56 --task dkitty --test --cond_rtg 0.1 --no_update_rtg # --train
# python scripts/run_dtbo_desbench.py --seed $seed --context_length 1 --epochs 75 --batch_size 128 --dataset DKitty_40000x1_new_bin.p --experiment dkitty_morphology_timestep_1 --dim 56 --task dkitty --test --cond_rtg 0.1 # --train
python3.9 scripts/test_desbench.py --seeds $seeds --context_length 64 --dataset $dataset --eval_dataset $test_dataset --experiment $exptname --dim 8 --task tf-bind-8 --cond_rtgs $rtgs --discrete --init_len 64 --vocab_size 4 --heads 8 --layers 8 --no_update_rtg

