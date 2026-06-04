from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .types import OrbitWarsObservation, Planet

BOARD_SIZE = 100.0
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_PLAYERS = 4
MAX_PLANETS = 64
SHIP_FRACTIONS = (0.10, 0.20, 0.33, 0.50, 0.66, 0.80, 1.00)


def normalize_angle(angle: float) -> float:
    two_pi = 2.0 * math.pi
    angle = angle % two_pi
    if angle < 0:
        angle += two_pi
    return angle


def angle_difference(a: float, b: float) -> float:
    delta = abs(normalize_angle(a) - normalize_angle(b))
    return min(delta, 2.0 * math.pi - delta)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def angle_between(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.atan2(y2 - y1, x2 - x1)


def fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    ships = max(1, int(ships))
    if ships == 1:
        return 1.0
    scale = math.log(ships) / math.log(1000.0)
    scale = max(0.0, min(1.0, scale))
    return 1.0 + (max_speed - 1.0) * (scale**1.5)


def segment_distance_to_point(
    x1: float, y1: float, x2: float, y2: float, px: float, py: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return distance(x1, y1, px, py)
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return distance(proj_x, proj_y, px, py)


def path_intersects_circle(
    x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, radius: float
) -> bool:
    return segment_distance_to_point(x1, y1, x2, y2, cx, cy) <= radius


def path_intersects_sun(source: Planet, target: Planet) -> bool:
    return path_intersects_circle(source.x, source.y, target.x, target.y, CENTER[0], CENTER[1], SUN_RADIUS)


def is_orbiting_planet(planet: Planet) -> bool:
    orbital_radius = distance(planet.x, planet.y, CENTER[0], CENTER[1])
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def angular_phase(planet: Planet) -> float:
    return normalize_angle(angle_between(CENTER[0], CENTER[1], planet.x, planet.y))


def initial_planet_map(obs: OrbitWarsObservation) -> dict[int, Planet]:
    return {planet.id: planet for planet in obs.initial_planets}


def reconstructed_planet_positions(obs: OrbitWarsObservation, step: int | None = None) -> dict[int, tuple[float, float]]:
    if step is None:
        step = obs.step
    positions: dict[int, tuple[float, float]] = {}
    for planet in obs.initial_planets:
        if not is_orbiting_planet(planet):
            positions[planet.id] = (planet.x, planet.y)
            continue
        dx = planet.x - CENTER[0]
        dy = planet.y - CENTER[1]
        theta = obs.angular_velocity * step
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        positions[planet.id] = (CENTER[0] + rx, CENTER[1] + ry)
    return positions


def reconstructed_comet_positions(obs: OrbitWarsObservation) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for group in obs.comets:
        for planet_id, path in zip(group.planet_ids, group.paths):
            if not path:
                continue
            idx = max(0, min(group.path_index, len(path) - 1))
            positions[planet_id] = path[idx]
    return positions


def estimate_travel_time(source: Planet, target: Planet, ships: int) -> float:
    travel_distance = max(0.0, distance(source.x, source.y, target.x, target.y) - source.radius - target.radius)
    return travel_distance / fleet_speed(ships)


def infer_target_planet_id(planets: list[Planet], source_planet: Planet, action_angle: float) -> int | None:
    candidates = []
    for planet in planets:
        if planet.id == source_planet.id:
            continue
        theta = angle_between(source_planet.x, source_planet.y, planet.x, planet.y)
        diff = angle_difference(theta, action_angle)
        dist = distance(source_planet.x, source_planet.y, planet.x, planet.y)
        penalty = 0.1 if path_intersects_sun(source_planet, planet) else 0.0
        candidates.append((diff + penalty, dist, planet.id))
    if not candidates:
        return None
    candidates.sort()
    best_score, _, best_id = candidates[0]
    if best_score > math.pi / 8:
        return None
    return best_id


def fraction_to_bin(source_ships: int, launched_ships: int) -> int:
    if source_ships <= 0 or launched_ships <= 0:
        return 0
    ratio = launched_ships / max(1, source_ships)
    return min(range(len(SHIP_FRACTIONS)), key=lambda idx: abs(SHIP_FRACTIONS[idx] - ratio))


def bin_to_ship_count(source_ships: int, bin_index: int) -> int:
    if source_ships <= 0:
        return 0
    bin_index = max(0, min(bin_index, len(SHIP_FRACTIONS) - 1))
    ships = int(round(source_ships * SHIP_FRACTIONS[bin_index]))
    return max(1, min(source_ships, ships))


def aggregate_fleet_features(
    planets: list[Planet], fleets, player_id: int
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    by_planet = defaultdict(lambda: np.zeros(8, dtype=np.float32))
    global_features = np.zeros(MAX_PLAYERS * 2, dtype=np.float32)
    if not planets:
        return by_planet, global_features
    for fleet in fleets:
        owner_slot = min(MAX_PLAYERS - 1, max(0, fleet.owner))
        global_features[owner_slot] += 1.0
        global_features[MAX_PLAYERS + owner_slot] += float(fleet.ships)

        nearest_planet = min(
            planets,
            key=lambda planet: distance(fleet.x, fleet.y, planet.x, planet.y),
        )
        values = by_planet[nearest_planet.id]
        friendly = 1.0 if fleet.owner == player_id else 0.0
        enemy = 1.0 - friendly
        values[0] += friendly
        values[1] += enemy
        values[2] += friendly * fleet.ships
        values[3] += enemy * fleet.ships
        values[4] = min(values[4] if values[4] > 0 else 1e9, distance(fleet.x, fleet.y, nearest_planet.x, nearest_planet.y))
        values[5] = max(values[5], distance(fleet.x, fleet.y, nearest_planet.x, nearest_planet.y))

        if fleet.from_planet_id == nearest_planet.id:
            values[6] += 1.0
            values[7] += float(fleet.ships)

    for planet_id, values in by_planet.items():
        if values[4] == 1e9:
            values[4] = 0.0
    return by_planet, global_features.astype(np.float32)
