# AdaptivePowerformer — Complete Google Colab Notebook
# ====================================================
# Copy each cell into a Colab notebook. Run them in order.
# Runtime > Change Runtime Type > T4 GPU
#
# This notebook runs ALL experiments for the paper:
# 1. Preprocessing Indian dataset
# 2. Baselines (Powerformer, DLinear) on ETTh1
# 3. AdaptivePowerformer on all datasets
# 4. Ablation study (fixed-alpha variants)
# 5. Efficiency measurement
# 6. Alpha analysis & plots

# =============================================================================
# CELL 1: Mount Google Drive
# =============================================================================
# from google.colab import drive
# drive.mount('/content/drive')

# =============================================================================
# CELL 2: Setup — Unzip and enter project directory
# =============================================================================
# !cp "/content/drive/MyDrive/AdaptivePowerformer.zip" /content/
# !unzip -q /content/AdaptivePowerformer.zip -d /content/
# %cd /content/AdaptivePowerformer

# =============================================================================
# CELL 3: Install dependencies
# =============================================================================
# !pip install -q -r requirements.txt

# =============================================================================
# CELL 4: Preprocess Indian Dataset (Excel → CSV)
# =============================================================================
# !python3 datasets/IndianLoad/prepare_indian_dataset.py \
#     --input datasets/IndianLoad/hourlyLoadDataIndia.xlsx \
#     --output datasets/IndianLoad/india_load.csv

# # Check what we got:
# !echo "=== First 3 rows ==="
# !head -3 datasets/IndianLoad/india_load.csv
# !echo "=== Row count ==="
# !wc -l datasets/IndianLoad/india_load.csv
# !echo "=== Column count (set enc_in to this minus 1 for date column) ==="
# !head -1 datasets/IndianLoad/india_load.csv | tr ',' '\n' | wc -l

# =============================================================================
# CELL 5: Quick Sanity Check — Train AdaptivePowerformer on ETTh1 (5 epochs)
# =============================================================================
# !python3 -u run_longExp.py \
#     --attn_decay_type powerLaw --attn_decay_scale 0.5 \
#     --is_training 1 --random_seed 2021 --is_sequential 0 \
#     --root_path ./datasets/ --data_path ETT-small/ETTh1.csv \
#     --model_id ETTh1_sanity --model AdaptivePowerformer --data ETTh1 \
#     --features M --seq_len 336 --pred_len 96 \
#     --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
#     --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
#     --patch_len 16 --stride 8 --des 'SanityCheck' \
#     --train_epochs 5 --patience 5 --itr 0 \
#     --batch_size 128 --learning_rate 0.0001

# =============================================================================
# CELL 6: If sanity check passes, run FULL experiments (takes ~6-8 hours on T4)
# =============================================================================
# !bash scripts/run_all_experiments.sh

# =============================================================================
# CELL 7: Run Indian Load experiments (additional ~2 hours)
# NOTE: First check CELL 4 output to set ENC_IN correctly in the script
# =============================================================================
# !bash scripts/AdaptivePowerformer/indian_load.sh

# =============================================================================
# CELL 8: View all results
# =============================================================================
# !echo "========== ALL RESULTS =========="
# !cat result.txt
# !echo ""
# !echo "========== EFFICIENCY =========="
# !cat efficiency_results.txt

# =============================================================================
# CELL 9: Alpha Analysis on ETTh1
# =============================================================================
# # Find the checkpoint path (it's in checkpoints/ directory)
# !ls checkpoints/ | grep AdaptivePowerformer | grep ETTh1 | head -5
# 
# # Then run (replace <SETTING> with actual folder name from above):
# # !python3 utils/alpha_analysis.py \
# #     --checkpoint ./checkpoints/<SETTING>/checkpoint.pth \
# #     --dataset ETTh1 --root_path ./datasets/ \
# #     --data_path ETT-small/ETTh1.csv --data ETTh1 \
# #     --enc_in 7 --pred_len 96

# =============================================================================
# CELL 10: Alpha Analysis on Indian Load
# =============================================================================
# # !python3 utils/alpha_analysis.py \
# #     --checkpoint ./checkpoints/<SETTING>/checkpoint.pth \
# #     --dataset IndianLoad --root_path ./datasets/IndianLoad/ \
# #     --data_path india_load.csv --data custom \
# #     --enc_in 5 --pred_len 96

# =============================================================================
# CELL 11: View generated plots
# =============================================================================
# from IPython.display import Image, display
# import glob
# for f in sorted(glob.glob('analysis_plots/*.png')):
#     print(f)
#     display(Image(f))

# =============================================================================
# CELL 12: Save results back to Google Drive
# =============================================================================
# !cp result.txt "/content/drive/MyDrive/AdaptivePowerformer_Results/"
# !cp efficiency_results.txt "/content/drive/MyDrive/AdaptivePowerformer_Results/"
# !cp -r analysis_plots/ "/content/drive/MyDrive/AdaptivePowerformer_Results/"
# !cp -r results/ "/content/drive/MyDrive/AdaptivePowerformer_Results/"
# print("All results saved to Google Drive!")
