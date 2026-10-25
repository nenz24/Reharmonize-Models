from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChordCandidateRanker(nn.Module):
    """Score arbitrary chord candidates against audio, harmony, genre, and structure context."""

    def __init__(self, context_size: int, candidate_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, context: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        context_embedding = self.context_encoder(context)
        candidate_embedding = self.candidate_encoder(candidate)
        interaction = context_embedding * candidate_embedding
        return self.score(torch.cat([context_embedding, candidate_embedding, interaction], dim=-1)).squeeze(-1)


def pairwise_ranking_loss(preferred_score: torch.Tensor, rejected_score: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(preferred_score - rejected_score).mean()
