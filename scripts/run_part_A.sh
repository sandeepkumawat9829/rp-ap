#!/bin/bash
# ===========================================================
# Master experiment runner — PART A
# Runs: Weather (Remaining 720), ETTh1 Ablation, Efficiency
# Allows parallel execution on a separate Colab instance
# ===========================================================

set -e  # Exit on error
mkdir -p logs/LongForecasting

# Auto-save function with TabA suffix
autosave() {
    python3 utils/autosave.py TabA 2>/dev/null || echo "[autosave] skipped"
}

echo "============================================"
echo " AdaptivePowerformer — Part A Experiments"
echo " Results auto-saved to Google Drive as *TabA*"
echo "============================================"

# ---- COMMON PARAMS ----
DECAY_TYPE=powerLaw
DECAY_SCALE=0.5
SEQ_LEN=336
SEED=2021
EPOCHS=100
PATIENCE=20
BATCH=128
LR=0.0001

echo ">>> RUNNING REMAINING WEATHER (pred_len=720) <<<"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path weather/weather.csv \
    --model_id Weather --model AdaptivePowerformer --data custom \
    --features M --seq_len $SEQ_LEN --pred_len 720 \
    --enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
    --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Adaptive_Exp' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Weather_${SEQ_LEN}_720.log
autosave

echo ""
echo ">>> PHASE 3: Ablation — Fixed Alpha <<<"
for ALPHA in 0.0 0.3 0.5 0.7 1.0; do
echo "[Ablation] FixedAlpha=$ALPHA ETTh1"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
    --model_id ETTh1_alpha${ALPHA} --model FixedAlphaPowerformer --data ETTh1 \
    --features M --seq_len $SEQ_LEN --pred_len 96 \
    --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Ablation' \
    --fixed_alpha $ALPHA \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/FixedAlpha_${ALPHA}_ETTh1_${SEQ_LEN}_96.log
autosave
done

echo ""
echo ">>> PHASE 4: Efficiency Measurement <<<"
python3 utils/measure_efficiency.py --enc_in 7 --pred_len 96 --d_model 16
python3 utils/measure_efficiency.py --enc_in 7 --pred_len 336 --d_model 16
python3 utils/measure_efficiency.py --enc_in 21 --pred_len 96 --d_model 128
python3 utils/measure_efficiency.py --enc_in 321 --pred_len 96 --d_model 128
autosave

echo "============================================"
echo " PART A COMPLETE!"
echo "============================================"
