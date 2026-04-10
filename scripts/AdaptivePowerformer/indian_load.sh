#!/bin/bash
# ===========================================================
# Indian Load dataset experiments
# Run AFTER prepare_indian_dataset.py has created india_load.csv
# ===========================================================

set -e
mkdir -p logs/LongForecasting

DECAY_TYPE=powerLaw
DECAY_SCALE=0.5
SEQ_LEN=336
SEED=2021
EPOCHS=100
PATIENCE=20
BATCH=128
LR=0.0001

# The Indian dataset enc_in depends on how many columns exist.
# After preprocessing, check with: head -1 datasets/IndianLoad/india_load.csv
# Count numeric columns (exclude 'date'). Set ENC_IN accordingly.
# For hourly load: typically 5 regions = enc_in 5
ENC_IN=5

# --- Powerformer Baseline on Indian Load ---
for PRED in 96 192 336 720; do
echo "[Baseline] Powerformer IndianLoad pred_len=$PRED"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/IndianLoad/ --data_path india_load.csv \
    --model_id IndianLoad --model Powerformer --data custom \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in $ENC_IN --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Baseline' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/Powerformer_IndianLoad_${SEQ_LEN}_${PRED}.log
done

# --- AdaptivePowerformer on Indian Load ---
for PRED in 96 192 336 720; do
echo "[Adaptive] IndianLoad pred_len=$PRED"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/IndianLoad/ --data_path india_load.csv \
    --model_id IndianLoad --model AdaptivePowerformer --data custom \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in $ENC_IN --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Adaptive_Exp' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_IndianLoad_${SEQ_LEN}_${PRED}.log
done

# --- DLinear Baseline on Indian Load ---
for PRED in 96 192 336 720; do
echo "[Baseline] DLinear IndianLoad pred_len=$PRED"
python3 -u run_longExp.py \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/IndianLoad/ --data_path india_load.csv \
    --model_id IndianLoad --model DLinear --data custom \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in $ENC_IN --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Baseline' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/DLinear_IndianLoad_${SEQ_LEN}_${PRED}.log
done

echo "Indian Load experiments complete! Check result.txt"
