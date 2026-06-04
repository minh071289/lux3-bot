from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(frozen=True)
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(frozen=True)
class CometGroup:
    planet_ids: tuple[int, ...]
    paths: tuple[tuple[tuple[float, float], ...], ...]
    path_index: int


@dataclass(frozen=True)
class OrbitWarsObservation:
    planets: tuple[Planet, ...]
    fleets: tuple[Fleet, ...]
    player: int
    angular_velocity: float
    initial_planets: tuple[Planet, ...]
    comets: tuple[CometGroup, ...]
    comet_planet_ids: tuple[int, ...]
    remaining_overage_time: float
    step: int = 0

    @property
    def player_count(self) -> int:
        owners = {planet.owner for planet in self.planets if planet.owner >= 0}
        owners.update(fleet.owner for fleet in self.fleets if fleet.owner >= 0)
        owners.add(self.player)
        return max(2, len(owners))


def _get_value(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _parse_planet(item: Any) -> Planet:
    if isinstance(item, Planet):
        return item
    return Planet(
        id=int(item[0]),
        owner=int(item[1]),
        x=float(item[2]),
        y=float(item[3]),
        radius=float(item[4]),
        ships=int(item[5]),
        production=int(item[6]),
    )


def _parse_fleet(item: Any) -> Fleet:
    if isinstance(item, Fleet):
        return item
    return Fleet(
        id=int(item[0]),
        owner=int(item[1]),
        x=float(item[2]),
        y=float(item[3]),
        angle=float(item[4]),
        from_planet_id=int(item[5]),
        ships=int(item[6]),
    )


def _parse_comet_group(item: Any) -> CometGroup:
    if isinstance(item, CometGroup):
        return item
    return CometGroup(
        planet_ids=tuple(int(x) for x in item.get("planet_ids", [])),
        paths=tuple(
            tuple((float(p[0]), float(p[1])) for p in path) for path in item.get("paths", [])
        ),
        path_index=int(item.get("path_index", 0)),
    )


def normalize_observation(raw: Any) -> OrbitWarsObservation:
    planets = tuple(_parse_planet(x) for x in _get_value(raw, "planets", []) or [])
    fleets = tuple(_parse_fleet(x) for x in _get_value(raw, "fleets", []) or [])
    initial_planets_raw = _get_value(raw, "initial_planets", None)
    if initial_planets_raw is None:
        initial_planets = planets
    else:
        initial_planets = tuple(_parse_planet(x) for x in initial_planets_raw or [])
    return OrbitWarsObservation(
        planets=planets,
        fleets=fleets,
        player=int(_get_value(raw, "player", 0)),
        angular_velocity=float(_get_value(raw, "angular_velocity", 0.0)),
        initial_planets=initial_planets,
        comets=tuple(_parse_comet_group(x) for x in _get_value(raw, "comets", []) or []),
        comet_planet_ids=tuple(int(x) for x in _get_value(raw, "comet_planet_ids", []) or []),
        remaining_overage_time=float(_get_value(raw, "remainingOverageTime", _get_value(raw, "remaining_overage_time", 0.0))),
        step=int(_get_value(raw, "step", 0)),
    )
