"""
============================================================
AdaptivePowerformer v2 — Colab/Kaggle Runner
============================================================
HOW TO USE:
  1. Upload AdaptivePowerformer_v2_with_datasets.zip to Colab
  2. Run CELL 1 (Setup) — it unzips and sets up everything
  3. Run experiment cells ONE BY ONE (each is independent)
  4. Results go to result.txt + logs/<Dataset>/<experiment>.log

Datasets are INCLUDED in the zip — no downloading needed!
============================================================
"""

# ============================================================
# CELL 1: SETUP (Run this FIRST — always)
# ============================================================
import os
import subprocess
import time
import shutil

# --- STEP 1: Upload and unzip ---
WORK_DIR = '/content/AdaptivePowerformer'

if not os.path.exists(WORK_DIR):
    # Check if zip was uploaded to Colab
    zip_path = '/content/AdaptivePowerformer_v2_with_datasets.zip'
    
    if os.path.exists(zip_path):
        print("📦 Unzipping uploaded file...")
        os.system(f'unzip -q {zip_path} -d /content/AdaptivePowerformer_unzipped')
        # Move contents to clean path
        # The zip might have a nested folder — handle both cases
        unzipped = '/content/AdaptivePowerformer_unzipped'
        items = os.listdir(unzipped)
        if len(items) == 1 and os.path.isdir(os.path.join(unzipped, items[0])):
            # Nested folder inside zip
            os.rename(os.path.join(unzipped, items[0]), WORK_DIR)
            os.rmdir(unzipped)
        else:
            os.rename(unzipped, WORK_DIR)
        print("✅ Unzipped successfully")
    else:
        # Fallback: clone from git
        print("📥 No zip found — cloning from GitHub...")
        os.system('git clone https://github.com/sandeepkumawat9829/rp-ap.git ' + WORK_DIR)
        print("✅ Cloned from GitHub")

os.chdir(WORK_DIR)

# --- STEP 2: Install requirements ---
os.system('pip install -q -r requirements.txt 2>/dev/null || true')

# --- STEP 3: Create separate log directories ---
for d in ['logs/ETTh1', 'logs/ETTm1', 'logs/Weather', 'logs/Electricity', 'logs/Ablation']:
    os.makedirs(d, exist_ok=True)

# --- STEP 4: Verify datasets ---
datasets_ok = True
for f in ['datasets/ETT-small/ETTh1.csv', 'datasets/weather/weather.csv', 'datasets/electricity/electricity.csv']:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  ✅ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ MISSING: {f}")
        datasets_ok = False

if not datasets_ok:
    print("\n⚠️ Some datasets are missing! Download them manually or use the git clone option.")

# --- STEP 5: Check GPU ---
import torch
if torch.cuda.is_available():
    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("\n⚠️ NO GPU DETECTED — Training will be very slow!")

print(f"📂 Working dir: {os.getcwd()}")
print("=" * 50)
print("✅ SETUP COMPLETE — Run experiment cells below one by one")


# ============================================================
# HELPER FUNCTION (Don't modify)
# ============================================================
def run_experiment(name, cmd, log_dir):
    """Run a single experiment with logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}.log')
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {name}")
    print(f"📝 Log: {log_file}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # Run and capture output
    full_cmd = f"{cmd} 2>&1 | tee {log_file}"
    os.system(full_cmd)
    
    elapsed = time.time() - start
    mins = elapsed / 60
    
    print(f"\n{'='*60}")
    print(f"✅ DONE: {name} ({mins:.1f} min)")
    print(f"{'='*60}")
    
    # Show result from result.txt
    if os.path.exists('result.txt'):
        with open('result.txt', 'r') as f:
            lines = f.readlines()
            # Show last 3 lines (latest result)
            print("\n📊 Latest result:")
            for line in lines[-3:]:
                print(f"   {line.strip()}")


# ============================================================
# COMMON SETTINGS
# ============================================================
BASE_CMD = "python3 -u run_longExp.py --is_training 1 --random_seed 2021 --is_sequential 0"
POWER_FLAGS = "--attn_decay_type powerLaw --attn_decay_scale 0.5"


# ============================================================
# CELL 2: ETTh1 pred_len=96 ⭐ RUN THIS FIRST (fastest ~10 min)
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTh1_pl96",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
        --model_id ETTh1 --model AdaptivePowerformer --data ETTh1 \
        --features M --seq_len 336 --pred_len 96 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTh1"
)


# ============================================================
# CELL 3: ETTh1 pred_len=192
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTh1_pl192",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
        --model_id ETTh1 --model AdaptivePowerformer --data ETTh1 \
        --features M --seq_len 336 --pred_len 192 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTh1"
)


# ============================================================
# CELL 4: ETTh1 pred_len=336
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTh1_pl336",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
        --model_id ETTh1 --model AdaptivePowerformer --data ETTh1 \
        --features M --seq_len 336 --pred_len 336 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTh1"
)


# ============================================================
# CELL 5: ETTh1 pred_len=720
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTh1_pl720",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
        --model_id ETTh1 --model AdaptivePowerformer --data ETTh1 \
        --features M --seq_len 336 --pred_len 720 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTh1"
)


# ============================================================
# CELL 6: Weather pred_len=96, seq_len=512 ⭐ HIGH PRIORITY
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Weather_sl512_pl96",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path weather/weather.csv \
        --model_id Weather --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 96 \
        --enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/Weather"
)


# ============================================================
# CELL 7: Weather pred_len=192, seq_len=512
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Weather_sl512_pl192",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path weather/weather.csv \
        --model_id Weather --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 192 \
        --enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/Weather"
)


# ============================================================
# CELL 8: Weather pred_len=336, seq_len=512
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Weather_sl512_pl336",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path weather/weather.csv \
        --model_id Weather --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 336 \
        --enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/Weather"
)


# ============================================================
# CELL 9: Weather pred_len=720, seq_len=512
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Weather_sl512_pl720",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path weather/weather.csv \
        --model_id Weather --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 720 \
        --enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/Weather"
)


# ============================================================
# CELL 10: Electricity pred_len=96, seq_len=512 ⭐ HIGH PRIORITY
# Paper recipe: lradj=TST, patience=10, batch=32
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Electricity_sl512_pl96",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path electricity/electricity.csv \
        --model_id Electricity --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 96 \
        --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 10 \
        --lradj TST --pct_start 0.2 \
        --itr 0 --batch_size 32 --learning_rate 0.0001""",
    log_dir="logs/Electricity"
)


# ============================================================
# CELL 11: Electricity pred_len=192, seq_len=512
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Electricity_sl512_pl192",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path electricity/electricity.csv \
        --model_id Electricity --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 192 \
        --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 10 \
        --lradj TST --pct_start 0.2 \
        --itr 0 --batch_size 32 --learning_rate 0.0001""",
    log_dir="logs/Electricity"
)


# ============================================================
# CELL 12: Electricity pred_len=336, seq_len=512
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Electricity_sl512_pl336",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path electricity/electricity.csv \
        --model_id Electricity --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 336 \
        --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 10 \
        --lradj TST --pct_start 0.2 \
        --itr 0 --batch_size 32 --learning_rate 0.0001""",
    log_dir="logs/Electricity"
)


# ============================================================
# CELL 13: Electricity pred_len=720, seq_len=512
# ============================================================
run_experiment(
    name="AdaptivePF_v2_Electricity_sl512_pl720",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path electricity/electricity.csv \
        --model_id Electricity --model AdaptivePowerformer --data custom \
        --features M --seq_len 512 --pred_len 720 \
        --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 10 \
        --lradj TST --pct_start 0.2 \
        --itr 0 --batch_size 32 --learning_rate 0.0001""",
    log_dir="logs/Electricity"
)


# ============================================================
# CELL 14: ETTm1 pred_len=96
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTm1_pl96",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTm1.csv \
        --model_id ETTm1 --model AdaptivePowerformer --data ETTm1 \
        --features M --seq_len 336 --pred_len 96 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTm1"
)


# ============================================================
# CELL 15: ETTm1 pred_len=192
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTm1_pl192",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTm1.csv \
        --model_id ETTm1 --model AdaptivePowerformer --data ETTm1 \
        --features M --seq_len 336 --pred_len 192 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTm1"
)


# ============================================================
# CELL 16: ETTm1 pred_len=336
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTm1_pl336",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTm1.csv \
        --model_id ETTm1 --model AdaptivePowerformer --data ETTm1 \
        --features M --seq_len 336 --pred_len 336 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTm1"
)


# ============================================================
# CELL 17: ETTm1 pred_len=720
# ============================================================
run_experiment(
    name="AdaptivePF_v2_ETTm1_pl720",
    cmd=f"""{BASE_CMD} {POWER_FLAGS} \
        --root_path ./datasets/ --data_path ETT-small/ETTm1.csv \
        --model_id ETTm1 --model AdaptivePowerformer --data ETTm1 \
        --features M --seq_len 336 --pred_len 720 \
        --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
        --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
        --patch_len 16 --stride 8 --des Exp_v2 \
        --train_epochs 100 --patience 20 --itr 0 \
        --batch_size 128 --learning_rate 0.0001""",
    log_dir="logs/ETTm1"
)


# ============================================================
# CELL 18: SHOW ALL RESULTS
# ============================================================
print("\n" + "="*60)
print("📊 ALL RESULTS FROM result.txt")
print("="*60)
if os.path.exists('result.txt'):
    with open('result.txt', 'r') as f:
        print(f.read())
else:
    print("No results yet — run experiments first!")

# Show log files created
print("\n📁 LOG FILES:")
for root, dirs, files in os.walk('logs'):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1024
        print(f"   {path} ({size:.1f} KB)")
