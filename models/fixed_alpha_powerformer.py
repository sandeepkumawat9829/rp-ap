import torch
import torch.nn as nn
from models.Powerformer import Model as PowerformerModel
from models.adaptive_powerformer import DecomposedLinearTrend, SimpleLinearTrend


class Model(nn.Module):
    """
    Fixed-Alpha Powerformer: uses a CONSTANT alpha (not learned).
    Used as ablation baseline to prove adaptive gating is better.
    
    Variants:
        --fixed_alpha 0.0 => pure Powerformer (no linear branch)
        --fixed_alpha 0.5 => equal blend
        --fixed_alpha 1.0 => pure Linear (no attention)
    
    Uses DecomposedLinearTrend by default (same as AdaptivePowerformer v2)
    for fair ablation comparison.
    """
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Fixed alpha value (default 0.5 = equal weighting)
        self.fixed_alpha = getattr(configs, 'fixed_alpha', 0.5)
        
        # Use simple linear for ablation if flag is set
        self.use_simple_linear = getattr(configs, 'simple_linear', False)
        
        # Base Powerformer
        self.powerformer = PowerformerModel(configs, **kwargs)
        
        # Linear branch (same as AdaptivePowerformer v2 for fair comparison)
        if self.use_simple_linear:
            self.linear_trend = SimpleLinearTrend(self.seq_len, self.pred_len)
        else:
            self.linear_trend = DecomposedLinearTrend(
                self.seq_len, self.pred_len,
                kernel_size=getattr(configs, 'kernel_size', 25)
            )

    def forward(self, x):
        """
        x: [B, seq_len, C]
        Uses fixed alpha instead of learned gate.
        """
        pf_out  = self.powerformer(x)      # [B, pred_len, C]
        lin_out = self.linear_trend(x)     # [B, pred_len, C]
        
        return self.fixed_alpha * pf_out + (1 - self.fixed_alpha) * lin_out
