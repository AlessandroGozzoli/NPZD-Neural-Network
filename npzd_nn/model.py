# =============================================================================
# model.py
# Defines the feedforward Multi-Layer Perceptron (MLP) that learns to
# emulate the one-step NPZD state transition.
#
# Architecture:
#   Input  (6)  ->  Hidden (128) ->  Hidden (128) -> Hidden (64) -> Output (4)
#   Activations: ReLU on hidden layers, linear on output.
#   Output is optionally clamped to [0, ∞) to enforce non-negativity.
# =============================================================================

import torch
import torch.nn as nn
from config import MODEL


# =============================================================================
# MLP definition
# =============================================================================

class NPZDMLP(nn.Module):
    """
    Feedforward neural network for NPZD one-step state prediction.

    Input:  [N_t, P_t, Z_t, D_t, I_t, T_t]            (6 features, normalised)
    Output: [N_{t+1}, P_{t+1}, Z_{t+1}, D_{t+1}]       (4 targets, normalised)

    The network predicts normalised outputs; the caller must apply the
    inverse normalisation to recover physical units.
    """

    def __init__(self, cfg: dict = None):
        super().__init__()

        cfg = cfg or MODEL

        input_dim   = cfg["input_dim"]
        output_dim  = cfg["output_dim"]
        hidden_dims = cfg["hidden_dims"]
        self.output_clamp = cfg["output_clamp"]
        self.clamp_min    = cfg["clamp_min"]

        # Build hidden layers dynamically
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Output layer (linear — regression)
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Weight initialisation (Kaiming for ReLU networks)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch_size, 6)  — normalised inputs

        Returns
        -------
        out : Tensor of shape (batch_size, 4) — normalised predictions
        NOTE: clamping is applied in physical space by the caller after
              inverse-normalisation, NOT here, because clamping normalised
              values would break the symmetry around zero.
        """
        return self.net(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Model factory
# =============================================================================

def build_model(cfg: dict = None, device: str = "cpu") -> NPZDMLP:
    """Instantiate the model and move it to the target device."""
    model = NPZDMLP(cfg=cfg)
    model = model.to(device)
    return model


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    model = build_model()
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # Dummy forward pass
    dummy_input = torch.randn(8, MODEL["input_dim"])
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")