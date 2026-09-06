"""Training loop for :class:`hotlane.modeling.model.HOTSpeedLSTM`."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

from hotlane.modeling.model import HOTSpeedLSTM, get_device


@dataclass
class TrainConfig:
    hidden_size: int = 32
    num_layers: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 5
    n_epochs: int = 150
    #: How often (in epochs) to record train/test RMSE.
    eval_every: int = 5


@dataclass
class TrainHistory:
    epochs: list[int] = field(default_factory=list)
    train_rmse: list[float] = field(default_factory=list)
    test_rmse: list[float] = field(default_factory=list)
    train_mse: list[float] = field(default_factory=list)
    test_mse: list[float] = field(default_factory=list)


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    config: TrainConfig | None = None,
) -> tuple[HOTSpeedLSTM, TrainHistory]:
    """Fit the model and return it alongside the RMSE curves.

    RMSE is reported in mph: targets are kept unscaled, so the loss is directly
    interpretable.
    """
    config = config or TrainConfig()
    device = get_device()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model = HOTSpeedLSTM(
        input_size=X_train.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_output=1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    loader = torch_data.DataLoader(
        torch_data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=config.batch_size,
    )

    history = TrainHistory()
    for epoch in range(config.n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            loss = loss_fn(model(X_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config.eval_every != 0:
            continue

        model.eval()
        with torch.no_grad():
            train_mse = float(loss_fn(model(X_train).cpu(), y_train.cpu()))
            test_mse = float(loss_fn(model(X_test).cpu(), y_test.cpu()))
        train_rmse, test_rmse = np.sqrt(train_mse), np.sqrt(test_mse)

        history.epochs.append(epoch)
        history.train_mse.append(train_mse)
        history.test_mse.append(test_mse)
        history.train_rmse.append(train_rmse)
        history.test_rmse.append(test_rmse)
        print(f"Epoch {epoch:4d}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")

    return model, history
