import pandas as pd
import numpy as np
import os
import argparse

def process_indian_load(input_file, output_file):
    print(f"Reading {input_file}...")
    # Read the data, engine explicitly set just in case
    df = pd.read_excel(input_file)
    
    # ETT dataset format expects:
    # 'date' column as datetime
    # Value columns
    
    # We rename columns to match roughly Autoformer/Informer code standards
    # The Indian dataset has 'datetime' or similar for time.
    # Let's inspect columns based on standard generic format.
    # We will rename the first column to 'date' assuming it is the timestamp.
    
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date just to be sure
    df.sort_values('date', inplace=True)
    
    # Handle missing values if any
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Ensure it's hourly - resample and interpolate if needed
    # But Kaggle dataset is already hourly.
    
    # Save to CSV
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Done! Shape: {df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="hourlyLoadDataIndia.xlsx")
    parser.add_argument("--output", type=str, default="india_load.csv")
    args = parser.parse_args()
    
    process_indian_load(args.input, args.output)
