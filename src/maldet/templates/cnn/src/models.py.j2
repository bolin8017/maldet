"""CNN model factory."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn


class ByteCNN(pl.LightningModule):
    def __init__(self, embedding_dim: int = 32, conv1_out: int = 64, conv2_out: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, conv1_out, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv2_out, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        h = emb.transpose(1, 2)
        h = torch.relu(self.conv1(h))
        h = self.pool(h)
        h = torch.relu(self.conv2(h))
        h = self.gap(h).squeeze(-1)
        return self.fc(h)

    def training_step(self, batch, batch_idx):  # type: ignore[no-untyped-def]
        x, y = batch
        loss = self.loss(self.forward(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def make_cnn(**kwargs) -> ByteCNN:  # type: ignore[no-untyped-def]
    return ByteCNN(**kwargs)
