#!/usr/bin/env bash

exptname="8_heads_16_layers_new"
dataset="superconductor/new_800x128_64_train.p"
rtg=10
# for seed in 123 234 345 456 567 678 789 890 901 012
for seed in 123 234
# for seed in 123
do
  # python scripts/run_dtbo_dkitty.py --seed $seed --context_length 30 --epochs 75 --batch_size 128 --dataset DKitty_800x50_bin.p --experiment dkitty_morphology_continuous --test --cond_rtg 10 # --train
  # python scripts/run_dtbo_desbench.py --seed $seed --context_length 30 --epochs 75 --batch_size 128 --dataset DKitty_800x128_new_bin.p --experiment $exptname --dim 56 --task dkitty --test --cond_rtg $rtg --train
  # python scripts/run_dtbo_desbench.py --seed $seed --context_length 1 --epochs 75 --batch_size 128 --dataset DKitty_40000x1_new_bin.p --experiment dkitty_morphology_timestep_1 --dim 56 --task dkitty --test --cond_rtg 0.1 # --train
  python scripts/train_desbench.py --seed $seed --context_length 64 --epochs 25 --batch_size 128 --dataset $dataset  --experiment $exptname --dim 86 --task superconductor --heads 8 --layers 16 --resume
done

# python scripts/compute_aggregate_statistics.py --cond_rtg $rtg --experiment $exptname --task dkitty
