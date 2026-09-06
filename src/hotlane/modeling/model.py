"""The speed-prediction network."""
from __future__ import annotations

import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HOTSpeedLSTM(nn.Module):
    """LSTM over a short window of gantry observations, predicting one speed.

    Only the final time step of the LSTM output is projected, so the model maps
    ``(batch, lookback, n_features) -> (batch, num_output)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_output: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out)[:, -1, :]
