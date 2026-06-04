from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .features import NUM_SHIP_BINS, featurize_observation
from .geometry import bin_to_ship_count
from .model import OrbitWarsGraphPolicy
from .types import normalize_observation


class OrbitWarsAgent:
    def __init__(self, weights_path: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = OrbitWarsGraphPolicy().to(self.device)
        self.model.eval()
        self.ready = False

        if weights_path is not None:
            weights_path = Path(weights_path)
            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state_dict)
                self.ready = True

    def act(self, observation: Any) -> list[list[float]]:
        obs = normalize_observation(observation)
        if not self.ready:
            return []

        encoded = featurize_observation(obs)
        with torch.no_grad():
            launch_logits, target_logits, ship_logits = self.model(
                torch.from_numpy(encoded.global_features).unsqueeze(0).to(self.device),
                torch.from_numpy(encoded.planet_features).unsqueeze(0).to(self.device),
                torch.from_numpy(encoded.pair_features).unsqueeze(0).to(self.device),
                torch.from_numpy(encoded.planet_mask).unsqueeze(0).to(self.device),
            )
        return decode_actions_from_outputs(
            obs,
            encoded.planet_ids,
            encoded.planet_mask,
            encoded.owned_planet_mask,
            launch_logits.squeeze(0).cpu(),
            target_logits.squeeze(0).cpu(),
            ship_logits.squeeze(0).cpu(),
        )


def decode_actions_from_outputs(
    obs,
    planet_ids,
    planet_mask,
    owned_planet_mask,
    launch_logits,
    target_logits,
    ship_logits,
    launch_threshold: float = 0.5,
) -> list[list[float]]:
    planets = {planet.id: planet for planet in obs.planets}
    actions = []
    launch_scores = torch.sigmoid(launch_logits)
    for idx, source_id in enumerate(planet_ids.tolist()):
        if source_id < 0 or planet_mask[idx] <= 0 or owned_planet_mask[idx] <= 0:
            continue
        if float(launch_scores[idx]) < launch_threshold:
            continue
        source = planets.get(source_id)
        if source is None or source.owner != obs.player or source.ships <= 0:
            continue

        target_idx = int(torch.argmax(target_logits[idx]).item())
        if target_idx < 0 or target_idx >= len(planet_ids):
            continue
        target_id = int(planet_ids[target_idx])
        if target_id < 0 or target_id == source_id:
            continue
        target = planets.get(target_id)
        if target is None:
            continue

        ship_bin = int(torch.argmax(ship_logits[idx]).item())
        ship_bin = max(0, min(ship_bin, NUM_SHIP_BINS - 1))
        ships = bin_to_ship_count(source.ships, ship_bin)
        if ships <= 0:
            continue

        angle = math.atan2(target.y - source.y, target.x - source.x)
        if not math.isfinite(angle):
            continue
        actions.append([int(source.id), float(angle), int(ships)])
    return actions


_AGENT: OrbitWarsAgent | None = None


def agent(obs, config=None):
    global _AGENT
    if _AGENT is None:
        if config is not None and isinstance(config, dict):
            weights_path = config.get("weights_path")
        else:
            weights_path = None
        _AGENT = OrbitWarsAgent(weights_path=weights_path)
    return _AGENT.act(obs)
