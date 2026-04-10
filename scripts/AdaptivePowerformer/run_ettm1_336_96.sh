#!/bin/bash
# ============================================
# AdaptivePowerformer v2 — ETTm1, pred_len=96, seq_len=336
# ============================================

python3 -u run_longExp.py \
    --attn_decay_type powerLaw \
    --attn_decay_scale 0.5 \
    --is_training 1 \
    --random_seed 2021 \
    --is_sequential 0 \
    --root_path ./datasets/ \
    --data_path ETT-small/ETTm1.csv \
    --model_id ETTm1 \
    --model AdaptivePowerformer \
    --data ETTm1 \
    --features M \
    --seq_len 336 \
    --pred_len 96 \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --des 'Exp_v2' \
    --train_epochs 100 \
    --patience 20 \
    --itr 0 \
    --batch_size 128 \
    --learning_rate 0.0001
