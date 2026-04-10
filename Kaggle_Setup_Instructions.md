# Kaggle Setup Guide (Circumventing Colab Limits)

Kaggle is an excellent alternative to Google Colab. It gives you 30 hours of free GPU time per week (usually a powerful **P100** or dual **T4** GPUs) and rarely disconnects you prematurely.

Because Kaggle's file system is slightly different from Colab (it doesn't use Google Drive), the setup process is a bit different. Follow these steps exactly:

### Step 1: Create the Notebook and Enable GPU
1. Go to [Kaggle.com](https://www.kaggle.com/) and sign in.
2. Click **"+" Create** -> **New Notebook** (top left).
3. In the right-hand sidebar, look for **Accelerator** (under Notebook Options or Session Options).
4. Change it from *None* to **GPU T4 x2** or **GPU P100**.

### Step 2: Upload Your Code as a "Dataset"
Kaggle requires you to upload files as a "Dataset" so your notebook can access them.
1. In the right-hand sidebar, find the **Input** section and click **Add Data**.
2. Click **Upload New Dataset** (the '+' or upload icon).
3. Give it a simple title (e.g., `AdaptivePowerformerCode`).
4. Drag and drop your `AdaptivePowerformer_Parallel.zip` file from your computer into the upload box.
5. Click **Create** and wait a moment for it to process.

### Step 3: Extract and Setup (Cell 1)
Once the dataset is attached, Kaggle mounts it as Read-Only. We need to copy it to the `working` directory so we can run scripts and save outputs.

Paste this into your first Notebook cell and run it:
```bash
# 1. Unzip the code into the working directory
!unzip -q /kaggle/input/adaptivepowerformercode/AdaptivePowerformer_Parallel.zip -d /kaggle/working/

# 2. Move into the directory and install requirements
%cd /kaggle/working/AdaptivePowerformer
!pip install -q -r requirements.txt
!mkdir -p logs/LongForecasting
```
*(Note: If you named your dataset something other than `adaptivepowerformercode`, replace that part of the path in the unzip command above).*

### Step 4: Run the Electricity Experiments (Cell 2)
Because Kaggle doesn't have Google Drive, we won't use the `autosave.py` script. Instead, Kaggle automatically saves everything left in the output directory when the session ends, and we can just download it manually! 

Paste this into the second cell and run it:
```bash
!for PRED in 96 192 336 720; do \
    echo "======================================================="; \
    echo " Starting [AdaptivePowerformer] Electricity pred_len=$PRED with Patience=5"; \
    echo "======================================================="; \
    python3 -u run_longExp.py --use_amp --attn_decay_type powerLaw --attn_decay_scale 0.5 --is_training 1 --random_seed 2021 --is_sequential 0 --root_path ./datasets/ --data_path electricity/electricity.csv --model_id Electricity --model AdaptivePowerformer --data custom --features M --seq_len 336 --pred_len $PRED --enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Adaptive_Exp' --train_epochs 100 --patience 5 --itr 0 --batch_size 16 --learning_rate 0.0001 2>&1 | tee -a logs/LongForecasting/AdaptivePowerformer_Electricity_336_$PRED.log; \
done
```

### Step 5: How to get your Results
When the cell finishes running (or if you stop it midway), all your results will be directly inside `/kaggle/working/AdaptivePowerformer/`. 
1. In the right sidebar, go to **Output** -> `/kaggle/working`.
2. Expand it to find `AdaptivePowerformer`, then `result.txt`.
3. Hover over `result.txt` and click the **Download symbol** to get your metrics!
4. (Optional) Run `!zip -q -r /kaggle/working/final_results.zip /kaggle/working/AdaptivePowerformer/results/` in a new cell if you want to download all the matrices/folders at once.
