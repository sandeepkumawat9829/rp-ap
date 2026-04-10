#!/bin/bash
# ===========================================================
# Master experiment runner for AdaptivePowerformer paper
# Auto-saves results to Google Drive after EVERY experiment
# ===========================================================

set -e  # Exit on error

mkdir -p logs/LongForecasting

# Auto-save function — call after each experiment
autosave() {
    python3 utils/autosave.py 2>/dev/null || echo "[autosave] skipped (Drive not mounted or error)"
}

echo "============================================"
echo " AdaptivePowerformer — Full Experiment Suite"
echo " Results auto-saved to Google Drive"
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

# ==============================
# PHASE 1: BASELINES (Powerformer + DLinear) on ETTh1
# ==============================
echo ""
echo ">>> PHASE 1: Running Baselines <<<"

for PRED in 96 192 336 720; do
echo "[Baseline] Powerformer ETTh1 pred_len=$PRED"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
    --model_id ETTh1 --model Powerformer --data ETTh1 \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Baseline' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/Powerformer_ETTh1_${SEQ_LEN}_${PRED}.log
autosave
done

for PRED in 96 192 336 720; do
echo "[Baseline] DLinear ETTh1 pred_len=$PRED"
python3 -u run_longExp.py \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
    --model_id ETTh1 --model DLinear --data ETTh1 \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Baseline' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/DLinear_ETTh1_${SEQ_LEN}_${PRED}.log
autosave
done

echo ">>> Phase 1 COMPLETE — Baselines done <<<"

# ==============================
# PHASE 2: ADAPTIVE POWERFORMER on all datasets
# ==============================
echo ""
echo ">>> PHASE 2: Running AdaptivePowerformer <<<"

# --- ETTh1 ---
for PRED in 96 192 336 720; do
echo "[Adaptive] ETTh1 pred_len=$PRED"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
    --model_id ETTh1 --model AdaptivePowerformer --data ETTh1 \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Adaptive_Exp' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_ETTh1_${SEQ_LEN}_${PRED}.log
autosave
done

# --- ETTm1 ---
for PRED in 96 192 336 720; do
echo "[Adaptive] ETTm1 pred_len=$PRED"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path ETT-small/ETTm1.csv \
    --model_id ETTm1 --model AdaptivePowerformer --data ETTm1 \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Adaptive_Exp' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_ETTm1_${SEQ_LEN}_${PRED}.log
autosave
done

# --- Weather (d_model=128, enc_in=21) ---
for PRED in 96 192 336 720; do
echo "[Adaptive] Weather pred_len=$PRED"
python3 -u run_longExp.py \
    --attn_decay_type $DECAY_TYPE --attn_decay_scale $DECAY_SCALE \
    --is_training 1 --random_seed $SEED --is_sequential 0 \
    --root_path ./datasets/ --data_path weather/weather.csv \
    --model_id Weather --model AdaptivePowerformer --data custom \
    --features M --seq_len $SEQ_LEN --pred_len $PRED \
    --enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
    --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
    --patch_len 16 --stride 8 --des 'Adaptive_Exp' \
    --train_epochs $EPOCHS --patience $PATIENCE --itr 0 \
    --batch_size $BATCH --learning_rate $LR \
    2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Weather_${SEQ_LEN}_${PRED}.log
autosave
done

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

echo ">>> Phase 2 COMPLETE — AdaptivePowerformer done <<<"

# ==============================
# PHASE 3: ABLATION (Fixed Alpha on ETTh1 pred_len=96)
# ==============================
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

echo ">>> Phase 3 COMPLETE — Ablation done <<<"

# ==============================
# PHASE 4: EFFICIENCY MEASUREMENT
# ==============================
echo ""
echo ">>> PHASE 4: Efficiency Measurement <<<"

python3 utils/measure_efficiency.py --enc_in 7 --pred_len 96 --d_model 16
python3 utils/measure_efficiency.py --enc_in 7 --pred_len 336 --d_model 16
python3 utils/measure_efficiency.py --enc_in 21 --pred_len 96 --d_model 128
python3 utils/measure_efficiency.py --enc_in 321 --pred_len 96 --d_model 128
autosave

echo ""
echo "============================================"
echo " ALL EXPERIMENTS COMPLETE!"
echo " Check result.txt for MSE/MAE results"
echo " Check efficiency_results.txt for latency"
echo " All results auto-saved to Google Drive"
echo "============================================"

# Final save
autosave
