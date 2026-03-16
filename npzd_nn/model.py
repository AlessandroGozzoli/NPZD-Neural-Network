# =============================================================================
# model.py
# Feedforward MLP that predicts state increments  Δs = s_{t+1} − s_t.
#
# Input:  [N, P, Z, D, I, T]          (6 features, normalised)
# Output: [ΔN, ΔP, ΔZ, ΔD]           (4 targets, normalised)
#
# Rollout:  s_{t+1} = max(0, s_t + inverse_normalise(NN(normalise(s_t, I_t, T_t))))
# =============================================================================

import torch
import torch.nn as nn
from config import MODEL


class NPZDMLP(nn.Module):

    def __init__(self, cfg: dict = None):
        super().__init__()
        cfg = cfg or MODEL

        in_dim   = cfg["input_dim"]
        out_dim  = cfg["output_dim"]
        hiddens  = cfg["hidden_dims"]
        drop_p   = cfg.get("dropout_p", 0.0)

        layers = []
        prev   = in_dim
        for h in hiddens:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if drop_p > 0.0:
                layers.append(nn.Dropout(p=drop_p))
            prev = h
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict = None, device: str = "cpu") -> NPZDMLP:
    return NPZDMLP(cfg=cfg).to(device)


if __name__ == "__main__":
    m = build_model()
    print(m)
    print(f"\nParameters: {m.count_parameters():,}")
    x = torch.randn(4, MODEL["input_dim"])
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {m(x).shape}")
