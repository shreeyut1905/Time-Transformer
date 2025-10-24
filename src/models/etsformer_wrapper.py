"""Wrapper around ETSFormer to unify interface with existing models.

The ETSFormer library forecasts multiple steps and (optionally) multiple time features.
We adapt it to our single-target next-step (or limited horizon) regression by:
- Passing the feature window as (B, T, F)
- Projecting features down to the model's expected `time_features` count via a linear layer if needed
- Requesting `num_steps_forecast=1` and returning the last-step prediction for a single feature
"""

import torch
import torch.nn as nn
from etsformer_pytorch import ETSFormer

class ETSFormerWrapper(nn.Module):
    def __init__(self, input_size: int, time_features: int = 4, model_dim: int = 128, layers: int = 2,
                 heads: int = 4, K: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_features = time_features
        if input_size != time_features:
            self.adapt = nn.Linear(input_size, time_features)
        else:
            self.adapt = nn.Identity()
        self.core = ETSFormer(
            time_features=time_features,
            model_dim=model_dim,
            embed_kernel_size=3,
            layers=layers,
            heads=heads,
            K=K,
            dropout=dropout
        )
        self.head = nn.Linear(time_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.adapt(x)
        pred = self.core(z, num_steps_forecast=1)  
        out = self.head(pred.squeeze(1))  
        return out.squeeze(-1)
