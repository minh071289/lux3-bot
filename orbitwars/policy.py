from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch

from .features import NUM_SHIP_BINS, featurize_observation
from .geometry import bin_to_ship_count, distance
from .model import OrbitWarsGraphPolicy
from .types import Planet, normalize_observation


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_weights_path(weights_path: str | Path | None) -> Path | None:
    if weights_path is None:
        candidates = []
    else:
        candidates = [Path(weights_path)]

    candidates.extend(
        [
            Path("/kaggle_simulations/agent/imitation_learning/weights/orbitwars_graph_policy.pth"),
            Path(__file__).resolve().parents[1] / "imitation_learning" / "weights" / "orbitwars_graph_policy.pth",
            Path.cwd() / "imitation_learning" / "weights" / "orbitwars_graph_policy.pth",
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.is_absolute() else candidate
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def choose_heuristic_actions(obs) -> list[list[float]]:
    my_planets = [planet for planet in obs.planets if planet.owner == obs.player and planet.ships > 1]
    targets = [planet for planet in obs.planets if planet.owner != obs.player]
    if not my_planets or not targets:
        return []

    actions = []
    targeted_ids = set()

    # Prefer targets that are productive, weakly defended, and nearby.
    def target_score(source: Planet, target: Planet) -> tuple[float, float]:
        dist = distance(source.x, source.y, target.x, target.y)
        value = target.production * 12.0 - target.ships - dist * 0.35
        if target.owner < 0:
            value += 6.0
        return (-value, dist)

    for source in sorted(my_planets, key=lambda planet: (-planet.ships, -planet.production)):
        available = source.ships
        if available <= 1:
            continue

        remaining_targets = [target for target in targets if target.id not in targeted_ids]
        if not remaining_targets:
            remaining_targets = targets

        ordered_targets = sorted(remaining_targets, key=lambda candidate: target_score(source, candidate))
        target = ordered_targets[0]
        capturable = [candidate for candidate in ordered_targets if available > candidate.ships]
        if capturable:
            target = capturable[0]

        required = max(1, target.ships + 1)
        if available > required:
            send = required
            if target.owner >= 0:
                send = max(send, int(round(available * 0.35)))
            else:
                send = max(send, int(round(available * 0.2)))
        else:
            # Stay active even before we can cleanly capture a strong target.
            if target.owner >= 0:
                send = max(1, int(round(available * 0.6)))
            else:
                send = max(1, int(round(available * 0.5)))

        send = min(available - 1, send)
        if send <= 0:
            continue

        angle = math.atan2(target.y - source.y, target.x - source.x)
        if not math.isfinite(angle):
            continue

        actions.append([int(source.id), float(angle), int(send)])
        targeted_ids.add(target.id)

    return actions


class OrbitWarsAgent:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        use_heuristic_fallback: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = OrbitWarsGraphPolicy().to(self.device)
        self.model.eval()
        self.ready = False
        self.weights_path = resolve_weights_path(weights_path)
        self.use_heuristic_fallback = use_heuristic_fallback

        if self.weights_path is not None:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.ready = True

    def set_use_heuristic_fallback(self, value: bool) -> None:
        self.use_heuristic_fallback = bool(value)

    def act(self, observation: Any) -> list[list[float]]:
        obs = normalize_observation(observation)
        if not self.ready:
            if self.use_heuristic_fallback:
                return choose_heuristic_actions(obs)
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
        if not actions and self.use_heuristic_fallback:
            return choose_heuristic_actions(obs)
        return actions


def decode_actions_from_outputs(
    obs,
    planet_ids,
    planet_mask,
    owned_planet_mask,
    launch_logits,
    target_logits,
    ship_logits,
    launch_threshold: float = 0.35,
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
    use_heuristic_fallback = True
    weights_path = None
    if config is not None and isinstance(config, dict):
        weights_path = config.get("weights_path")
        use_heuristic_fallback = _coerce_bool(
            config.get("orbitwars_use_heuristic_fallback"),
            True,
        )
    if _AGENT is None:
        _AGENT = OrbitWarsAgent(
            weights_path=weights_path,
            use_heuristic_fallback=use_heuristic_fallback,
        )
    else:
        _AGENT.set_use_heuristic_fallback(use_heuristic_fallback)
    return _AGENT.act(obs)


def debug_policy_outputs(
    observation: Any,
    weights_path: str | Path | None = None,
    use_heuristic_fallback: bool = False,
) -> dict[str, Any]:
    obs = normalize_observation(observation)
    agent = OrbitWarsAgent(
        weights_path=weights_path,
        use_heuristic_fallback=use_heuristic_fallback,
    )

    info: dict[str, Any] = {
        "ready": agent.ready,
        "resolved_weights_path": str(agent.weights_path) if agent.weights_path is not None else None,
        "player": obs.player,
        "step": obs.step,
        "num_planets": len(obs.planets),
        "num_fleets": len(obs.fleets),
        "num_owned_planets": sum(1 for planet in obs.planets if planet.owner == obs.player),
        "heuristic_fallback_enabled": use_heuristic_fallback,
        "heuristic_actions": choose_heuristic_actions(obs),
    }

    if not agent.ready:
        info["decoded_actions"] = []
        info["owned_planet_debug"] = []
        return info

    encoded = featurize_observation(obs)
    with torch.no_grad():
        launch_logits, target_logits, ship_logits = agent.model(
            torch.from_numpy(encoded.global_features).unsqueeze(0).to(agent.device),
            torch.from_numpy(encoded.planet_features).unsqueeze(0).to(agent.device),
            torch.from_numpy(encoded.pair_features).unsqueeze(0).to(agent.device),
            torch.from_numpy(encoded.planet_mask).unsqueeze(0).to(agent.device),
        )

    launch_logits = launch_logits.squeeze(0).cpu()
    target_logits = target_logits.squeeze(0).cpu()
    ship_logits = ship_logits.squeeze(0).cpu()
    launch_scores = torch.sigmoid(launch_logits)

    decoded_actions = decode_actions_from_outputs(
        obs,
        encoded.planet_ids,
        encoded.planet_mask,
        encoded.owned_planet_mask,
        launch_logits,
        target_logits,
        ship_logits,
    )
    info["decoded_actions"] = decoded_actions

    planets = {planet.id: planet for planet in obs.planets}
    owned_planet_debug = []
    for idx, source_id in enumerate(encoded.planet_ids.tolist()):
        if source_id < 0 or encoded.planet_mask[idx] <= 0 or encoded.owned_planet_mask[idx] <= 0:
            continue

        source = planets.get(source_id)
        if source is None:
            continue

        launch_prob = float(launch_scores[idx].item())
        best_target_idx = int(torch.argmax(target_logits[idx]).item())
        best_target_id = None
        best_target_logit = None
        if 0 <= best_target_idx < len(encoded.planet_ids):
            best_target_id = int(encoded.planet_ids[best_target_idx])
            best_target_logit = float(target_logits[idx, best_target_idx].item())

        best_ship_bin = int(torch.argmax(ship_logits[idx]).item())
        best_ship_bin = max(0, min(best_ship_bin, NUM_SHIP_BINS - 1))
        best_ship_count = bin_to_ship_count(source.ships, best_ship_bin)

        decode_blockers = []
        if launch_prob < 0.35:
            decode_blockers.append("launch_below_threshold")
        if best_target_id is None or best_target_id < 0:
            decode_blockers.append("invalid_target")
        elif best_target_id == source_id:
            decode_blockers.append("self_target")
        if best_ship_count <= 0:
            decode_blockers.append("ship_count_zero")

        owned_planet_debug.append(
            {
                "planet_id": int(source.id),
                "ships": int(source.ships),
                "production": int(source.production),
                "launch_prob": launch_prob,
                "launch_logit": float(launch_logits[idx].item()),
                "best_target_idx": best_target_idx,
                "best_target_id": best_target_id,
                "best_target_logit": best_target_logit,
                "best_ship_bin": best_ship_bin,
                "best_ship_fraction": SHIP_FRACTIONS[best_ship_bin],
                "best_ship_count": best_ship_count,
                "decode_blockers": decode_blockers,
            }
        )

    owned_planet_debug.sort(key=lambda row: (-row["launch_prob"], -row["ships"], row["planet_id"]))
    info["owned_planet_debug"] = owned_planet_debug
    return info
