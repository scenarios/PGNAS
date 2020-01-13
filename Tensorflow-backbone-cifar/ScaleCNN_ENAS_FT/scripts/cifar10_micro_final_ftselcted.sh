#!/bin/bash

export PYTHONPATH="$(pwd)"

fixed_arc="0 2 0 0 0 4 0 1 0 4 1 1 1 0 0 1 0 2 1 1"
fixed_arc="$fixed_arc 1 0 1 0 0 3 0 2 1 1 3 1 1 0 0 4 0 3 1 1"

python src/cifar10/main.py \
  --data_format="NCHW" \
  --search_for="micro" \
  --reset_output_dir=False \
  --data_path="/mnt/data/cifar10/" \
  --output_dir="/mnt/log/NAS/cifar10/enas/FT_SelectedOriginSizeCut16Outfilter72Bs144Ls01Wd1PdropN5e5Wd1e-7/" \
  --batch_size=190 \
  --num_epochs=630 \
  --log_every=50 \
  --eval_every_epochs=5 \
  --child_fixed_arc="${fixed_arc}" \
  --child_use_aux_heads=False \
  --child_num_layers=15 \
  --child_out_filters=72 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.80 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_dec_every=150 \
  --child_l2_reg=1e-7 \
  --child_lr_cosine=True \
  --child_lr_max=0.05 \
  --child_lr_min=0.0001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --nocontroller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.50 \
  --controller_op_tanh_reduce=2.5 \
  --weight_regularizer_ratio=0.1 \
  --dropout_regularizer_ratio=1.0 \
  --num_gpus=2 \
  --is_cdropout=True \
  --child_cutout_size=16 \
  "$@"

