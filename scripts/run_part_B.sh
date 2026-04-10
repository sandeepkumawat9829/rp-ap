#!/bin/bash
# ===========================================================
# Master experiment runner — PART B
# Runs: Electricity Dataset (All prediction lengths)
# Allows parallel execution on a separate Colab instance
# ===========================================================

set -e  # Exit on error
mkdir -p logs/LongForecasting

# Auto-save function with TabB suffix
autosave() {
    python3 utils/autosave.py TabB 2>/dev/null || echo "[autosave] skipped"
}

echo "============================================"
echo " AdaptivePowerformer — Part B Experiments"
echo " Results auto-saved to Google Drive as *TabB*"
echo "============================================"

# ---- COMMON PARAMS ----
DECAY_TYPE=powerLaw
DECAY_SCALE=0.5
SEQ_LEN=336
SEED=2021
EPOCHS=100
PATIENCE=20
LR=0.0001

echo ""
echo ">>> PHASE 2: Running AdaptivePowerformer (Part B) <<<"

# --- Electricity (d_model=128, enc_in=321) ---
for PRED in 96 192 336 720; do
echo "[Adaptive] Electricity pred_len=$PRED"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path electricity/electricity.csv \
    --model_id Electricity --model AdaptivePowerformer --data custom \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
    --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Adaptive_Exp' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size 32 --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Electricity_${SEQ_LEN}_${PRED}.log
autosave
done

echo ""
echo "============================================"
echo " PART B COMPLETE!"
echo "============================================"
