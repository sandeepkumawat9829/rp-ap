#!/bin/bash
# ============================================
# AdaptivePowerformer v2 — Weather, pred_len=336, seq_len=512
# ============================================

python3 -u run_longExp.py \
    --attn_decay_type powerLaw \
    --attn_decay_scale 0.5 \
    --is_training 1 \
    --random_seed 2021 \
    --is_sequential 0 \
    --root_path ./datasets/ \
    --data_path weather/weather.csv \
    --model_id Weather \
    --model AdaptivePowerformer \
    --data custom \
    --features M \
    --seq_len 512 \
    --pred_len 336 \
    --enc_in 21 \
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
    --patience 20 \
    --itr 0 \
    --batch_size 128 \
    --learning_rate 0.0001
