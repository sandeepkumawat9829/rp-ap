import torch
import torch.nn as nn
from models.Powerformer import Model as PowerformerModel

class SimpleLinearTrend(nn.Module):
    """Channel-independent linear projection: seq_len -> pred_len.
    Matches Powerformer's univariate approach."""
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C] -> permute to apply linear along sequence dimension
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class AdaptiveGate(nn.Module):
    """Input-dependent gate: alpha(x) = sigmoid(MLP([mean(x); std(x)]))
    Outputs alpha in [0,1] per sample.
    
    d_model controls the gate MLP width:
    - Use enc_in (number of input channels) for stats, NOT d_model
    - stats = [mean, std] each of shape [B, C] -> concat -> [B, 2*C]
    """
    def __init__(self, enc_in):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(enc_in * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()   # alpha in [0, 1]
        )

    def forward(self, x):
        # x: [B, L, C]
        mean = x.mean(dim=1)  # [B, C]
        std  = x.std(dim=1)   # [B, C]
        stats = torch.cat([mean, std], dim=-1)  # [B, 2*C]
        return self.mlp(stats)  # [B, 1]


class Model(nn.Module):
    """
    AdaptivePowerformer: alpha(x) * Powerformer(x) + (1 - alpha(x)) * Linear(x)
    
    Key equation (for the paper):
        y_hat = alpha(x) * f_PF(x) + (1 - alpha(x)) * f_L(x)
        where alpha(x) = sigma(MLP([mean(x); std(x)]))
    
    Registered in exp_main.py model_dict as "AdaptivePowerformer".
    exp_main.py dispatches models containing "ower" in their name 
    with forward(batch_x) — a single tensor argument.
    """
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Base Powerformer (its forward also takes just x)
        self.powerformer = PowerformerModel(configs, **kwargs)
        
        # Our additions
        self.linear_trend = SimpleLinearTrend(self.seq_len, self.pred_len)
        self.gate = AdaptiveGate(self.enc_in)

    def forward(self, x):
        """
        x: [B, seq_len, C] — same single-argument signature as Powerformer.
        exp_main.py calls model(batch_x) for any model with 'ower' in its name.
        """
        pf_out  = self.powerformer(x)      # [B, pred_len, C]
        lin_out = self.linear_trend(x)     # [B, pred_len, C]
        alpha   = self.gate(x)             # [B, 1]
        
        # Broadcast alpha: [B, 1] -> [B, 1, 1] -> [B, pred_len, C]
        alpha = alpha.unsqueeze(-1)        # [B, 1, 1]
        
        return alpha * pf_out + (1 - alpha) * lin_out

