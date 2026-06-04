from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import (
    BOARD_SIZE,
    CENTER,
    MAX_PLANETS,
    MAX_PLAYERS,
    SHIP_FRACTIONS,
    aggregate_fleet_features,
    angle_between,
    angular_phase,
    distance,
    estimate_travel_time,
    fraction_to_bin,
    infer_target_planet_id,
    initial_planet_map,
    is_orbiting_planet,
    path_intersects_sun,
    reconstructed_comet_positions,
)
from .types import OrbitWarsObservation, Planet

GLOBAL_DIM = 4 + MAX_PLAYERS * 4
PLANET_FEAT_DIM = 29
PAIR_FEAT_DIM = 5 + len(SHIP_FRACTIONS)
NUM_SHIP_BINS = len(SHIP_FRACTIONS)


@dataclass(frozen=True)
class OrbitWarsFeatures:
    global_features: np.ndarray
    planet_features: np.ndarray
    pair_features: np.ndarray
    planet_mask: np.ndarray
    owned_planet_mask: np.ndarray
    planet_ids: np.ndarray


def _planet_owner_one_hot(owner: int) -> list[float]:
    values = [0.0 for _ in range(MAX_PLAYERS + 1)]
    if owner < 0:
        values[-1] = 1.0
    else:
        values[min(owner, MAX_PLAYERS - 1)] = 1.0
    return values


def _build_global_features(obs: OrbitWarsObservation, fleets_summary: np.ndarray) -> np.ndarray:
    ownership_counts = np.zeros(MAX_PLAYERS, dtype=np.float32)
    total_ships = np.zeros(MAX_PLAYERS, dtype=np.float32)
    for planet in obs.planets:
        if 0 <= planet.owner < MAX_PLAYERS:
            ownership_counts[planet.owner] += 1.0
            total_ships[planet.owner] += float(planet.ships)
    for fleet in obs.fleets:
        if 0 <= fleet.owner < MAX_PLAYERS:
            total_ships[fleet.owner] += float(fleet.ships)

    active_comets = len(obs.comet_planet_ids)
    return np.concatenate(
        [
            np.array(
                [
                    obs.step / 500.0,
                    obs.player_count / 4.0,
                    obs.angular_velocity,
                    active_comets / 20.0,
                ],
                dtype=np.float32,
            ),
            ownership_counts / MAX_PLANETS,
            total_ships / 2000.0,
            fleets_summary[:MAX_PLAYERS] / 50.0,
            fleets_summary[MAX_PLAYERS:] / 2000.0,
        ]
    ).astype(np.float32)


def featurize_observation(obs: OrbitWarsObservation) -> OrbitWarsFeatures:
    planets = sorted(obs.planets, key=lambda planet: planet.id)
    num_planets = min(len(planets), MAX_PLANETS)
    planets = planets[:num_planets]
    planet_mask = np.zeros(MAX_PLANETS, dtype=np.float32)
    owned_planet_mask = np.zeros(MAX_PLANETS, dtype=np.float32)
    planet_ids = np.full(MAX_PLANETS, -1, dtype=np.int16)
    planet_features = np.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    pair_features = np.zeros((MAX_PLANETS, MAX_PLANETS, PAIR_FEAT_DIM), dtype=np.float32)

    initial_map = initial_planet_map(obs)
    comet_positions = reconstructed_comet_positions(obs)
    fleet_by_planet, fleet_globals = aggregate_fleet_features(planets, obs.fleets, obs.player)

    for idx, planet in enumerate(planets):
        planet_mask[idx] = 1.0
        owned_planet_mask[idx] = 1.0 if planet.owner == obs.player else 0.0
        planet_ids[idx] = planet.id
        initial_planet = initial_map.get(planet.id, planet)
        orbiting = is_orbiting_planet(initial_planet)
        comet = planet.id in obs.comet_planet_ids
        incoming = fleet_by_planet.get(planet.id, np.zeros(8, dtype=np.float32))

        enemy_distances = [
            distance(planet.x, planet.y, other.x, other.y)
            for other in planets
            if other.owner >= 0 and other.owner != obs.player and other.id != planet.id
        ]
        neutral_distances = [
            distance(planet.x, planet.y, other.x, other.y)
            for other in planets
            if other.owner < 0 and other.id != planet.id
        ]
        center_distance = distance(planet.x, planet.y, CENTER[0], CENTER[1]) / 50.0
        phase = angular_phase(initial_planet) / (2.0 * math.pi)
        comet_pos = comet_positions.get(planet.id, (planet.x, planet.y))
        feature_row = [
            planet.x / BOARD_SIZE,
            planet.y / BOARD_SIZE,
            initial_planet.x / BOARD_SIZE,
            initial_planet.y / BOARD_SIZE,
            comet_pos[0] / BOARD_SIZE,
            comet_pos[1] / BOARD_SIZE,
            planet.radius / 8.0,
            planet.ships / 1000.0,
            planet.production / 5.0,
            center_distance,
            phase,
            1.0 if orbiting else 0.0,
            1.0 if comet else 0.0,
            obs.step / 500.0,
            * _planet_owner_one_hot(planet.owner),
            incoming[0] / 10.0,
            incoming[1] / 10.0,
            incoming[2] / 1000.0,
            incoming[3] / 1000.0,
            incoming[4] / BOARD_SIZE,
            incoming[5] / BOARD_SIZE,
            incoming[6] / 10.0,
            incoming[7] / 1000.0,
            (min(enemy_distances) if enemy_distances else BOARD_SIZE) / BOARD_SIZE,
            (min(neutral_distances) if neutral_distances else BOARD_SIZE) / BOARD_SIZE,
        ]
        planet_features[idx, : len(feature_row)] = np.asarray(feature_row, dtype=np.float32)

    for source_idx, source in enumerate(planets):
        for target_idx, target in enumerate(planets):
            if source_idx == target_idx:
                continue
            base = [
                distance(source.x, source.y, target.x, target.y) / BOARD_SIZE,
                angle_between(source.x, source.y, target.x, target.y) / math.pi,
                (target.x - source.x) / BOARD_SIZE,
                (target.y - source.y) / BOARD_SIZE,
                1.0 if path_intersects_sun(source, target) else 0.0,
            ]
            travel_times = [
                estimate_travel_time(source, target, max(1, int(round(source.ships * fraction))))
                / BOARD_SIZE
                for fraction in SHIP_FRACTIONS
            ]
            pair_features[source_idx, target_idx] = np.asarray(base + travel_times, dtype=np.float32)

    return OrbitWarsFeatures(
        global_features=_build_global_features(obs, fleet_globals),
        planet_features=planet_features,
        pair_features=pair_features,
        planet_mask=planet_mask,
        owned_planet_mask=owned_planet_mask,
        planet_ids=planet_ids,
    )


def encode_labels(
    obs: OrbitWarsObservation,
    moves: list[list[float]],
    planet_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    launch = np.full(MAX_PLANETS, -100, dtype=np.int16)
    target = np.full(MAX_PLANETS, -100, dtype=np.int16)
    ship_bin = np.full(MAX_PLANETS, -100, dtype=np.int16)
    planets = sorted(obs.planets, key=lambda planet: planet.id)[:MAX_PLANETS]
    planet_by_id = {planet.id: planet for planet in planets}
    id_to_index = {int(planet_ids[idx]): idx for idx in range(MAX_PLANETS) if planet_ids[idx] >= 0}
    collapsed = {}
    duplicate_moves = 0
    for move in moves or []:
        if len(move) != 3:
            continue
        source_id = int(move[0])
        ships = int(move[2])
        prev = collapsed.get(source_id)
        if prev is not None:
            duplicate_moves += 1
            if ships <= int(prev[2]):
                continue
        collapsed[source_id] = move

    failed_target_inference = 0
    for planet in planets:
        idx = id_to_index[planet.id]
        if planet.owner != obs.player:
            continue
        launch[idx] = 0
        source_move = collapsed.get(planet.id)
        if source_move is None:
            continue
        action_angle = float(source_move[1])
        launched_ships = int(source_move[2])
        inferred_target_id = infer_target_planet_id(planets, planet, action_angle)
        if inferred_target_id is None or inferred_target_id not in id_to_index:
            failed_target_inference += 1
            continue
        launch[idx] = 1
        target[idx] = id_to_index[inferred_target_id]
        ship_bin[idx] = fraction_to_bin(planet.ships, launched_ships)

    stats = {
        "duplicate_moves": duplicate_moves,
        "collapsed_sources": len(collapsed),
        "failed_target_inference": failed_target_inference,
    }
    return launch, target, ship_bin, stats
