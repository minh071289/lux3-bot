from __future__ import annotations

import torch
from torch import nn

from .features import GLOBAL_DIM, NUM_SHIP_BINS, PAIR_FEAT_DIM, PLANET_FEAT_DIM


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class OrbitWarsGraphPolicy(nn.Module):
    def __init__(
        self,
        global_dim: int = GLOBAL_DIM,
        planet_dim: int = PLANET_FEAT_DIM,
        pair_dim: int = PAIR_FEAT_DIM,
        hidden_dim: int = 128,
        ship_bins: int = NUM_SHIP_BINS,
    ):
        super().__init__()
        self.global_encoder = MLP(global_dim, hidden_dim, hidden_dim)
        self.planet_encoder = MLP(planet_dim, hidden_dim, hidden_dim)
        self.pair_encoder = MLP(pair_dim, hidden_dim, hidden_dim)

        self.launch_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ship_bins),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_features, planet_features, pair_features, planet_mask):
        global_hidden = self.global_encoder(global_features)
        planet_hidden = self.planet_encoder(planet_features)
        pair_hidden = self.pair_encoder(pair_features)

        global_expanded = global_hidden.unsqueeze(1).expand(-1, planet_hidden.size(1), -1)
        planet_global = torch.cat([planet_hidden, global_expanded], dim=-1)

        launch_logits = self.launch_head(planet_global).squeeze(-1)
        ship_logits = self.ship_head(planet_global)

        source_hidden = planet_hidden.unsqueeze(2).expand(-1, -1, planet_hidden.size(1), -1)
        target_hidden = planet_hidden.unsqueeze(1).expand(-1, planet_hidden.size(1), -1, -1)
        global_pair = global_hidden.unsqueeze(1).unsqueeze(2).expand(
            -1, planet_hidden.size(1), planet_hidden.size(1), -1
        )
        target_inputs = torch.cat([source_hidden, target_hidden, pair_hidden, global_pair], dim=-1)
        target_logits = self.target_head(target_inputs).squeeze(-1)

        large_neg = torch.tensor(-1e9, dtype=launch_logits.dtype, device=launch_logits.device)
        valid_planets = planet_mask > 0
        launch_logits = torch.where(valid_planets, launch_logits, large_neg)
        ship_logits = torch.where(valid_planets.unsqueeze(-1), ship_logits, large_neg)

        target_valid = valid_planets.unsqueeze(1) & valid_planets.unsqueeze(2)
        diagonal = torch.eye(planet_hidden.size(1), dtype=torch.bool, device=planet_hidden.device).unsqueeze(0)
        target_valid = target_valid & (~diagonal)
        target_logits = torch.where(target_valid, target_logits, large_neg)

        return launch_logits, target_logits, ship_logits
