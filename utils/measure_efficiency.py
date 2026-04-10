"""
Efficiency measurement script for all models.
Run this AFTER training is complete to measure inference latency, VRAM, and parameter count.

Usage (from AdaptivePowerformer root):
    python utils/measure_efficiency.py --dataset ETTh1 --pred_len 96
"""

import sys
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Powerformer import Model as PowerformerModel
from models.adaptive_powerformer import Model as AdaptiveModel
from models.fixed_alpha_powerformer import Model as FixedAlphaModel
from models.DLinear import Model as DLinearModel


class SimpleConfig:
    """Minimal config to instantiate models."""
    def __init__(self, enc_in=7, seq_len=336, pred_len=96, d_model=16, 
                 d_ff=128, n_heads=4, e_layers=3, dropout=0.3, 
                 fc_dropout=0.3, head_dropout=0.0, patch_len=16, stride=8,
                 padding_patch='end', revin=1, affine=0, subtract_last=0,
                 decomposition=0, kernel_size=25, individual=0,
                 attn_decay_type='powerLaw', train_attn_decay=False,
                 attn_decay_scale=0.5, is_sequential=0, label_len=48,
                 dec_in=7, c_out=7, d_layers=1, moving_avg=25, factor=1,
                 embed='timeF', activation='gelu', output_attention=False,
                 distil=True, fixed_alpha=0.5):
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.individual = individual
        self.attn_decay_type = attn_decay_type
        self.train_attn_decay = train_attn_decay
        self.attn_decay_scale = attn_decay_scale
        self.is_sequential = is_sequential
        self.dec_in = dec_in
        self.c_out = c_out
        self.moving_avg = moving_avg
        self.factor = factor
        self.embed = embed
        self.activation = activation
        self.output_attention = output_attention
        self.distil = distil
        self.fixed_alpha = fixed_alpha


def measure_efficiency(model, x, n_runs=50, device='cuda'):
    """Measure inference latency and VRAM with proper warmup."""
    model = model.to(device)
    x = x.to(device)
    model.eval()
    
    # Warmup (critical)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    torch.cuda.synchronize()
    
    latency_ms = (time.time() - start) * 1000 / n_runs
    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    
    return latency_ms, vram_mb, params_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_runs', type=int, default=50)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU. Latency numbers will not be representative.")
    
    # Create dummy input
    x = torch.randn(args.batch_size, args.seq_len, args.enc_in).float()
    
    configs = SimpleConfig(
        enc_in=args.enc_in, seq_len=args.seq_len, 
        pred_len=args.pred_len, d_model=args.d_model
    )
    
    results = []
    
    # --- DLinear ---
    try:
        dlinear = DLinearModel(configs).float()
        lat, vram, params = measure_efficiency(dlinear, x, args.n_runs, device)
        results.append(('DLinear', params, lat, vram))
        del dlinear
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"DLinear failed: {e}")
    
    # --- Powerformer ---
    try:
        pf = PowerformerModel(configs).float()
        lat, vram, params = measure_efficiency(pf, x, args.n_runs, device)
        results.append(('Powerformer', params, lat, vram))
        del pf
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Powerformer failed: {e}")
    
    # --- Fixed Alpha 0.3 ---
    try:
        configs.fixed_alpha = 0.3
        fa3 = FixedAlphaModel(configs).float()
        lat, vram, params = measure_efficiency(fa3, x, args.n_runs, device)
        results.append(('FixedAlpha-0.3', params, lat, vram))
        del fa3
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FixedAlpha-0.3 failed: {e}")
    
    # --- Fixed Alpha 0.5 ---
    try:
        configs.fixed_alpha = 0.5
        fa5 = FixedAlphaModel(configs).float()
        lat, vram, params = measure_efficiency(fa5, x, args.n_runs, device)
        results.append(('FixedAlpha-0.5', params, lat, vram))
        del fa5
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FixedAlpha-0.5 failed: {e}")
    
    # --- Fixed Alpha 0.7 ---
    try:
        configs.fixed_alpha = 0.7
        fa7 = FixedAlphaModel(configs).float()
        lat, vram, params = measure_efficiency(fa7, x, args.n_runs, device)
        results.append(('FixedAlpha-0.7', params, lat, vram))
        del fa7
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FixedAlpha-0.7 failed: {e}")
    
    # --- AdaptivePowerformer ---
    try:
        adaptive = AdaptiveModel(configs).float()
        lat, vram, params = measure_efficiency(adaptive, x, args.n_runs, device)
        results.append(('AdaptivePowerformer', params, lat, vram))
        del adaptive
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"AdaptivePowerformer failed: {e}")
    
    # Print results table
    print("\n" + "="*70)
    print(f"Efficiency Results (seq_len={args.seq_len}, pred_len={args.pred_len}, batch={args.batch_size})")
    print("="*70)
    print(f"{'Model':<25} {'Params(M)':>10} {'Latency(ms)':>12} {'VRAM(MB)':>10}")
    print("-"*70)
    for name, params, lat, vram in results:
        print(f"{name:<25} {params:>10.3f} {lat:>12.2f} {vram:>10.1f}")
    print("="*70)
    
    # Save to file
    with open('efficiency_results.txt', 'a') as f:
        f.write(f"\nseq_len={args.seq_len}, pred_len={args.pred_len}, batch={args.batch_size}\n")
        for name, params, lat, vram in results:
            f.write(f"{name}: params={params:.3f}M, latency={lat:.2f}ms, vram={vram:.1f}MB\n")
    print("Results saved to efficiency_results.txt")


if __name__ == '__main__':
    main()
