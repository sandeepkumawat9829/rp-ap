"""
Alpha-value analysis and visualization script.
Run AFTER training AdaptivePowerformer to analyze gate behavior.

Usage:
    python utils/alpha_analysis.py \
        --checkpoint ./checkpoints/<setting>/checkpoint.pth \
        --dataset IndianLoad --root_path ./datasets/IndianLoad/ \
        --data_path india_load.csv --enc_in 7
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab
import matplotlib.pyplot as plt

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptive_powerformer import Model as AdaptiveModel
from data_provider.data_factory import data_provider


class SimpleConfig:
    """Minimal config matching training setup."""
    def __init__(self, **kwargs):
        defaults = dict(
            enc_in=7, seq_len=336, pred_len=96, label_len=48,
            d_model=16, d_ff=128, n_heads=4, e_layers=3, d_layers=1,
            dropout=0.3, fc_dropout=0.3, head_dropout=0.0,
            patch_len=16, stride=8, padding_patch='end',
            revin=1, affine=0, subtract_last=0,
            decomposition=0, kernel_size=25, individual=0,
            attn_decay_type='powerLaw', train_attn_decay=False,
            attn_decay_scale=0.5, is_sequential=0,
            dec_in=7, c_out=7, moving_avg=25, factor=1,
            embed='timeF', activation='gelu', output_attention=False,
            distil=True, features='M', target='OT', freq='h',
            root_path='./datasets/', data_path='ETT-small/ETTh1.csv',
            data='ETTh1', batch_size=32, num_workers=0,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def analyze_alpha(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build config
    configs = SimpleConfig(
        enc_in=args.enc_in, seq_len=args.seq_len, pred_len=args.pred_len,
        root_path=args.root_path, data_path=args.data_path,
        data=args.data, features=args.features, batch_size=args.batch_size,
    )
    
    # Build model
    model = AdaptiveModel(configs).float().to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint loaded, using random weights!")
    
    # Hook to capture alpha values
    alphas = []
    def hook_fn(module, input, output):
        alphas.append(output.detach().cpu())
    
    model.gate.register_forward_hook(hook_fn)
    
    # Get test data
    test_data, test_loader = data_provider(configs, 'test')
    
    # Run inference
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            _ = model(batch_x)
    
    # Concatenate all alpha values
    all_alphas = torch.cat(alphas).squeeze().numpy()
    
    print(f"\n{'='*50}")
    print(f"Alpha Statistics ({args.dataset})")
    print(f"{'='*50}")
    print(f"Mean alpha:   {all_alphas.mean():.4f}")
    print(f"Std alpha:    {all_alphas.std():.4f}")
    print(f"Min alpha:    {all_alphas.min():.4f}")
    print(f"Max alpha:    {all_alphas.max():.4f}")
    print(f"Median alpha: {np.median(all_alphas):.4f}")
    print(f"% alpha > 0.5 (Transformer-dominant): {(all_alphas > 0.5).mean()*100:.1f}%")
    print(f"% alpha < 0.5 (Linear-dominant):      {(all_alphas < 0.5).mean()*100:.1f}%")
    
    # Create output directory
    os.makedirs('analysis_plots', exist_ok=True)
    
    # --- Plot 1: Alpha over time ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(all_alphas, alpha=0.7, linewidth=0.5, color='#2196F3')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='α=0.5 (equal)')
    ax.axhline(y=all_alphas.mean(), color='green', linestyle=':', linewidth=1, 
               label=f'mean α={all_alphas.mean():.3f}')
    ax.set_xlabel('Test Sample Index', fontsize=12)
    ax.set_ylabel('α(x)', fontsize=12)
    ax.set_title(f'Adaptive Gate Values — {args.dataset} (pred_len={args.pred_len})', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname1 = f'analysis_plots/alpha_over_time_{args.dataset}_pl{args.pred_len}.pdf'
    plt.savefig(fname1, dpi=300, bbox_inches='tight')
    plt.savefig(fname1.replace('.pdf', '.png'), dpi=150)
    print(f"Saved: {fname1}")
    plt.close()
    
    # --- Plot 2: Alpha histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_alphas, bins=50, color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='α=0.5')
    ax.axvline(x=all_alphas.mean(), color='blue', linestyle=':', linewidth=1.5, 
               label=f'mean={all_alphas.mean():.3f}')
    ax.set_xlabel('α(x)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Alpha Distribution — {args.dataset}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname2 = f'analysis_plots/alpha_histogram_{args.dataset}_pl{args.pred_len}.pdf'
    plt.savefig(fname2, dpi=300, bbox_inches='tight')
    plt.savefig(fname2.replace('.pdf', '.png'), dpi=150)
    print(f"Saved: {fname2}")
    plt.close()
    
    # Save raw alpha values for paper
    np.save(f'analysis_plots/alphas_{args.dataset}_pl{args.pred_len}.npy', all_alphas)
    print(f"Saved raw alphas to analysis_plots/alphas_{args.dataset}_pl{args.pred_len}.npy")
    
    return all_alphas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./datasets/')
    parser.add_argument('--data_path', type=str, default='ETT-small/ETTh1.csv')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    analyze_alpha(args)
