from typing import Optional

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_idx: Optional[int],
    ) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_batch)
