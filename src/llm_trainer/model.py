from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = input_ids.shape
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Input seq_length {seq_length} exceeds max_seq_length {self.max_seq_length}"
            )

        positions = torch.arange(0, seq_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=mask)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return logits, loss

