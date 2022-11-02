#!/usr/bin/env bash

for n_layer in 2 4 8 16 32; do
  exptname="sorted_with_"$n_layer"_layers"
  dataset="tf-bind-10/tfbind10_sorted_4000x128_64_train.p"
  eval_dataset="tf-bind-10/tfbind10_sorted_128x128_64_eval.p"
  rtgs="0.0 0.01 0.05 0.1 1 2 4 8"
  for seed in 123; do
    #python scripts/train_desbench.py --seed $seed --context_length 64 --epochs 75 --batch_size 128 --dataset $dataset  --experiment $exptname --dim 10 --task tf-bind-10 --layers $n_layer --discrete --vocab_size 4 
    # python scripts/test_desbench.py --seeds $seed --context_length 64 --dataset $dataset --experiment $exptname --dim 8 --task tf-bind-8 --cond_rtgs $rtgs --layers $n_layer --init_len 64 --discrete --vocab_size 4
    python scripts/test_desbench.py --seeds $seed --context_length 64 --dataset $dataset --eval_dataset $eval_dataset --experiment $exptname --dim 10 --task tf-bind-10 --cond_rtgs $rtgs --layers $n_layer --init_len 64 --discrete --vocab_size 4
  done
done
