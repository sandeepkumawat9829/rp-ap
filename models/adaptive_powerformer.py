import torch
import torch.nn as nn
from models.Powerformer import Model as PowerformerModel
from layers.PatchTST_layers import series_decomp


# ============================================================
# LINEAR BRANCHES
# ============================================================

class SimpleLinearTrend(nn.Module):
    """Original naive linear projection: seq_len -> pred_len.
    Kept for ablation comparison (Task 1.1 baseline)."""
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C] -> permute to apply linear along sequence dimension
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class DecomposedLinearTrend(nn.Module):
    """DLinear-style decomposition + separate linear projections.
    
    Task 1.1: Splits input into trend (moving avg) + seasonal (residual),
    applies separate Linear(seq_len -> pred_len) for each, then sums.
    This matches the proven DLinear architecture that beats vanilla
    Transformers on many benchmarks.
    
    Key insight: By decomposing first, each linear layer only needs to
    learn one pattern type (smooth trend OR periodic seasonal), making
    the linear branch significantly more powerful.
    """
    def __init__(self, seq_len, pred_len, kernel_size=25):
        super().__init__()
        self.decomp = series_decomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        seasonal, trend = self.decomp(x)                                # both [B, L, C]
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1))      # [B, C, pred_len]
        trend = self.linear_trend(trend.permute(0, 2, 1))               # [B, C, pred_len]
        return (seasonal + trend).permute(0, 2, 1)                      # [B, pred_len, C]


# ============================================================
# GATE MECHANISMS
# ============================================================

class AdaptiveGate(nn.Module):
    """Original simple gate using [mean, std] statistics.
    Kept for ablation comparison (Task 1.2 baseline)."""
    def __init__(self, enc_in, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(enc_in * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, L, C]
        mean = x.mean(dim=1)   # [B, C]
        std  = x.std(dim=1)    # [B, C]
        stats = torch.cat([mean, std], dim=-1)  # [B, 2*C]
        return self.mlp(stats)  # [B, 1]


class EnhancedAdaptiveGate(nn.Module):
    """Task 1.2: Richer input-dependent gate using multi-scale statistics.
    
    Uses 6 statistical features per channel:
        1. Global mean         — captures level/offset
        2. Global std          — captures overall volatility
        3. Skewness proxy      — captures asymmetry (3rd moment)
        4. Diff std            — captures stationarity / rate of change
        5. Recent window mean  — captures recent trend direction
        6. Recent window std   — captures recent volatility
    
    Task 1.3 (per-channel mode):
        When channel_gate=True, outputs alpha per channel [B, C] instead
        of scalar [B, 1]. Different channels (e.g., temperature vs wind)
        can have different linear/attention ratios.
    
    Theoretical justification:
        - High diff_std + high skewness => volatile, non-stationary => prefer attention (alpha -> 0)
        - Low diff_std + stable mean => smooth, stationary => prefer linear (alpha -> 1)
        The gate learns this mapping end-to-end via backpropagation.
    """
    def __init__(self, enc_in, seq_len, channel_gate=False):
        super().__init__()
        self.channel_gate = channel_gate
        self.seq_len = seq_len
        
        # 6 features per channel = 6 * enc_in input features
        n_features = enc_in * 6
        out_dim = enc_in if channel_gate else 1
        
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        
        # Feature 1 & 2: Global mean and std
        mean = x.mean(dim=1)                            # [B, C]
        std  = x.std(dim=1) + 1e-8                      # [B, C] (eps for numerical stability)
        
        # Feature 3: Skewness proxy (normalized 3rd central moment)
        skew = ((x - mean.unsqueeze(1)) ** 3).mean(dim=1) / (std ** 3)  # [B, C]
        
        # Feature 4: First-difference std (measures volatility / stationarity)
        diff_std = (x[:, 1:, :] - x[:, :-1, :]).std(dim=1)             # [B, C]
        
        # Feature 5 & 6: Recent window statistics (last 25% of input)
        recent_start = max(1, L - L // 4)
        recent = x[:, recent_start:, :]
        recent_mean = recent.mean(dim=1)                # [B, C]
        recent_std  = recent.std(dim=1) + 1e-8          # [B, C]
        
        # Concatenate all 6 features
        stats = torch.cat([mean, std, skew, diff_std, recent_mean, recent_std], dim=-1)  # [B, 6*C]
        
        alpha = self.mlp(stats)  # [B, 1] or [B, C] depending on channel_gate
        return alpha


# ============================================================
# MAIN MODEL
# ============================================================

class Model(nn.Module):
    """
    AdaptivePowerformer v2: Enhanced hybrid architecture.
    
    Architecture:
        y_hat = alpha(x) * f_PF(x) + (1 - alpha(x)) * f_DL(x)
    
    Where:
        f_PF(x)  = Powerformer backbone (self-attention with power-law decay)
        f_DL(x)  = Decomposed Linear branch (DLinear-style: trend + seasonal)
        alpha(x) = EnhancedAdaptiveGate(multi-scale statistics of x)
    
    Three novel contributions over base Powerformer:
        1. Decomposed Linear branch provides strong inductive bias for trends
        2. Multi-scale statistical gate makes routing decisions based on
           input characteristics (mean, std, skewness, volatility, recent stats)
        3. Per-channel gating (optional) allows different routing per variable
    
    Registered in exp_main.py model_dict as "AdaptivePowerformer".
    exp_main.py dispatches models containing "ower" in their name 
    with forward(batch_x) — a single tensor argument.
    """
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Read optional config flags (with safe defaults)
        self.use_channel_gate = getattr(configs, 'channel_gate', False)
        self.use_simple_gate = getattr(configs, 'simple_gate', False)
        self.use_simple_linear = getattr(configs, 'simple_linear', False)
        
        # ---- Powerformer backbone (attention path) ----
        self.powerformer = PowerformerModel(configs, **kwargs)
        
        # ---- Linear branch ----
        # Task 1.1: Use DecomposedLinearTrend by default, SimpleLinearTrend for ablation
        if self.use_simple_linear:
            self.linear_trend = SimpleLinearTrend(self.seq_len, self.pred_len)
        else:
            self.linear_trend = DecomposedLinearTrend(
                self.seq_len, self.pred_len, 
                kernel_size=getattr(configs, 'kernel_size', 25)
            )
        
        # ---- Adaptive gate ----
        # Task 1.2 + 1.3: Use EnhancedAdaptiveGate by default, simple for ablation
        if self.use_simple_gate:
            self.gate = AdaptiveGate(self.enc_in)
        else:
            self.gate = EnhancedAdaptiveGate(
                self.enc_in, 
                self.seq_len,
                channel_gate=self.use_channel_gate
            )

    def forward(self, x):
        """
        x: [B, seq_len, C] — same single-argument signature as Powerformer.
        exp_main.py calls model(batch_x) for any model with 'ower' in its name.
        """
        pf_out  = self.powerformer(x)      # [B, pred_len, C]
        lin_out = self.linear_trend(x)     # [B, pred_len, C]
        alpha   = self.gate(x)             # [B, 1] or [B, C]
        
        if self.use_channel_gate:
            # Per-channel gating: alpha is [B, C] -> [B, 1, C] for broadcasting
            alpha = alpha.unsqueeze(1)     # [B, 1, C] broadcasts over pred_len
        else:
            # Scalar gating: alpha is [B, 1] -> [B, 1, 1] for broadcasting
            alpha = alpha.unsqueeze(-1)    # [B, 1, 1] broadcasts over pred_len and C
        
        return alpha * pf_out + (1 - alpha) * lin_out
