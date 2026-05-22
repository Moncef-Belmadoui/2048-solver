"""
network.py
----------
Dueling Double-DQN network for 2048.

Input:  (batch, 1, 4, 4)  log2-encoded board
Output: (batch, 4)         Q-values for each action
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Convolutional dueling network.

    Architecture
    ------------
    Input (1, 4, 4)
      → Conv 64 filters 2×2, ReLU
      → Conv 128 filters 2×2, ReLU
      → Flatten
      → FC 512, ReLU
      → Split into value and advantage streams
          Value:     FC 256 → FC 1
          Advantage: FC 256 → FC n_actions
      → Q = V + (A - mean(A))
    """

    def __init__(self, n_actions: int = 4, in_channels: int = 1):
        super().__init__()

        # --- Convolutional backbone ---
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64,  kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64,          128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )
        # After 2 conv layers on 4×4: (4-1)=3 → (3-1)=2  =>  2×2 feature maps
        conv_out_size = 128 * 2 * 2  # = 512

        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        # --- Value stream ---
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # --- Advantage stream ---
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, 4, 4)

        Returns
        -------
        q : (batch, n_actions)
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)   # flatten
        h = self.fc_shared(h)

        value     = self.value_stream(h)        # (B, 1)
        advantage = self.advantage_stream(h)    # (B, A)

        # Combine: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def get_action(self, state: torch.Tensor) -> int:
        """Greedy action for a single state (no grad)."""
        with torch.no_grad():
            q = self.forward(state)
        return int(q.argmax(dim=1).item())