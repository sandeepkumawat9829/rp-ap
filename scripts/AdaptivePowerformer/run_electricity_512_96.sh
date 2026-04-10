#!/bin/bash
# ============================================
# AdaptivePowerformer v2 — Electricity, pred_len=96, seq_len=512
# Paper uses: lradj=TST, pct_start=0.2, patience=10, batch=32
# ============================================

python3 -u run_longExp.py \
    --attn_decay_type powerLaw \
    --attn_decay_scale 0.5 \
    --is_training 1 \
    --random_seed 2021 \
    --is_sequential 0 \
    --root_path ./datasets/ \
    --data_path electricity/electricity.csv \
    --model_id Electricity \
    --model AdaptivePowerformer \
    --data custom \
    --features M \
    --seq_len 512 \
    --pred_len 96 \
    --enc_in 321 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --des 'Exp_v2' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 0 \
    --batch_size 32 \
    --learning_rate 0.0001
