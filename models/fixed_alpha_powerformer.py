import torch
import torch.nn as nn
from models.Powerformer import Model as PowerformerModel
from models.adaptive_powerformer import SimpleLinearTrend


class Model(nn.Module):
    """
    Fixed-Alpha Powerformer: uses a CONSTANT alpha (not learned).
    Used as ablation baseline to prove adaptive gating is better.
    
    Usage: pass --fixed_alpha 0.3 / 0.5 / 0.7 via configs
    """
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Fixed alpha value (default 0.5 = equal weighting)
        self.fixed_alpha = getattr(configs, 'fixed_alpha', 0.5)
        
        # Base Powerformer
        self.powerformer = PowerformerModel(configs, **kwargs)
        
        # Linear branch (same as AdaptivePowerformer)
        self.linear_trend = SimpleLinearTrend(self.seq_len, self.pred_len)

    def forward(self, x):
        """
        x: [B, seq_len, C]
        Uses fixed alpha instead of learned gate.
        """
        pf_out  = self.powerformer(x)      # [B, pred_len, C]
        lin_out = self.linear_trend(x)     # [B, pred_len, C]
        
        return self.fixed_alpha * pf_out + (1 - self.fixed_alpha) * lin_out
