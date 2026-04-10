# Parallel Colab Execution Guide (Cell by Cell)

Since the `run_all.sh` shell scripts take hours, running everything in a single Colab cell can be frustrating. If the connection drops or it crashes, you won't know exactly what finished.

Below are the **exact** commands to copy-paste into **individual Colab cells**. This ensures you can run, track, and monitor them completely one-by-one!

---

## 🟢 TAB A (Google ID 1)

### Cell A1: Setup & Extract
```bash
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/AdaptivePowerformer_Parallel.zip" /content/
!unzip -q /content/AdaptivePowerformer_Parallel.zip -d /content/
%cd /content/AdaptivePowerformer
!pip install -q -r requirements.txt
!mkdir -p logs/LongForecasting
```

### Cell A2: Weather pred_len=720 (~15 mins)
```bash
!python3 -u run_longExp.py --attn_decay_type powerLaw --attn_decay_scale 0.5 --is_training 1 --random_seed 2021 --is_sequential 0 --root_path ./datasets/ --data_path weather/weather.csv --model_id Weather --model AdaptivePowerformer --data custom --features M --seq_len 336 --pred_len 720 --enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Adaptive_Exp' --train_epochs 100 --patience 20 --itr 0 --batch_size 128 --learning_rate 0.0001 2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Weather_336_720.log
!python3 utils/autosave.py TabA
```

### Cell A3: Ablation Testing (Fixed Alphas on ETTh1) (~15 mins)
```bash
# This loop is very fast, so keeping it in one cell is fine!
!for ALPHA in 0.0 0.3 0.5 0.7 1.0; do \
    echo "[Ablation] FixedAlpha=$ALPHA ETTh1"; \
    python3 -u run_longExp.py --attn_decay_type powerLaw --attn_decay_scale 0.5 --is_training 1 --random_seed 2021 --is_sequential 0 --root_path ./datasets/ --data_path ETT-small/ETTh1.csv --model_id ETTh1_alpha${ALPHA} --model FixedAlphaPowerformer --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Ablation' --fixed_alpha $ALPHA --train_epochs 100 --patience 20 --itr 0 --batch_size 128 --learning_rate 0.0001 2>&1 | tee -a logs/LongForecasting/FixedAlpha_${ALPHA}_ETTh1_336_96.log; \
    python3 utils/autosave.py TabA; \
done
```

### Cell A4: Efficiency Measurement (~2 mins)
```bash
!python3 utils/measure_efficiency.py --enc_in 7 --pred_len 96 --d_model 16
!python3 utils/measure_efficiency.py --enc_in 7 --pred_len 336 --d_model 16
!python3 utils/measure_efficiency.py --enc_in 21 --pred_len 96 --d_model 128
!python3 utils/measure_efficiency.py --enc_in 321 --pred_len 96 --d_model 128
!python3 utils/autosave.py TabA
```

---

## 🔵 TAB B (Google ID 2)

### Cell B1: Setup & Extract
```bash
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/AdaptivePowerformer_Parallel.zip" /content/
!unzip -q /content/AdaptivePowerformer_Parallel.zip -d /content/
%cd /content/AdaptivePowerformer
!pip install -q -r requirements.txt
!mkdir -p logs/LongForecasting
```

### Cell B2: Electricity pred_len=96 (~1 hour)
```bash
!python3 -u run_longExp.py --attn_decay_type powerLaw --attn_decay_scale 0.5 --is_training 1 --random_seed 2021 --is_sequential 0 --root_path ./datasets/ --data_path electricity/electricity.csv --model_id Electricity --model AdaptivePowerformer --data custom --features M --seq_len 336 --pred_len 96 --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Adaptive_Exp' --train_epochs 100 --patience 20 --itr 0 --batch_size 32 --learning_rate 0.0001 2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Electricity_336_96.log
!python3 utils/autosave.py TabB
```

### Cell B3: Electricity pred_len=192 (~1 hour)
```bash
!python3 -u run_longExp.py --attn_decay_type powerLaw --attn_decay_scale 0.5 --is_training 1 --random_seed 2021 --is_sequential 0 --root_path ./datasets/ --data_path electricity/electricity.csv --model_id Electricity --model AdaptivePowerformer --data custom --features M --seq_len 336 --pred_len 192 --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Adaptive_Exp' --train_epochs 100 --patience 20 --itr 0 --batch_size 32 --learning_rate 0.0001 2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Electricity_336_192.log
!python3 utils/autosave.py TabB
```

### Cell B4: Electricity pred_len=336 (~1.5 hours)
```bash
!python3 -u run_longExp.py --attn_decay_type powerLaw --attn_decay_scale 0.5 --is_training 1 --random_seed 2021 --is_sequential 0 --root_path ./datasets/ --data_path electricity/electricity.csv --model_id Electricity --model AdaptivePowerformer --data custom --features M --seq_len 336 --pred_len 336 --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Adaptive_Exp' --train_epochs 100 --patience 20 --itr 0 --batch_size 32 --learning_rate 0.0001 2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Electricity_336_336.log
!python3 utils/autosave.py TabB
```

### Cell B5: Electricity pred_len=720 (~1.5 hours)
```bash
!python3 -u run_longExp.py --attn_decay_type powerLaw --attn_decay_scale 0.5 --is_training 1 --random_seed 2021 --is_sequential 0 --root_path ./datasets/ --data_path electricity/electricity.csv --model_id Electricity --model AdaptivePowerformer --data custom --features M --seq_len 336 --pred_len 720 --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Adaptive_Exp' --train_epochs 100 --patience 20 --itr 0 --batch_size 32 --learning_rate 0.0001 2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Electricity_336_720.log
!python3 utils/autosave.py TabB
```
