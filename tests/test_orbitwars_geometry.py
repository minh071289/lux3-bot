import math

from orbitwars.geometry import (
    CENTER,
    SHIP_FRACTIONS,
    bin_to_ship_count,
    fleet_speed,
    infer_target_planet_id,
    is_orbiting_planet,
    reconstructed_planet_positions,
)
from orbitwars.types import normalize_observation


def test_reconstructed_planet_positions_rotate_inner_planets():
    obs = normalize_observation(
        {
            "player": 0,
            "step": 1,
            "angular_velocity": math.pi / 2,
            "planets": [[1, 0, 50.0, 30.0, 2.0, 10, 2]],
            "initial_planets": [[1, 0, 50.0, 30.0, 2.0, 10, 2]],
            "fleets": [],
        }
    )
    assert is_orbiting_planet(obs.initial_planets[0])
    positions = reconstructed_planet_positions(obs)
    x, y = positions[1]
    assert round(x, 4) == 70.0
    assert round(y, 4) == 50.0


def test_infer_target_planet_id_prefers_matching_angle():
    obs = normalize_observation(
        {
            "player": 0,
            "planets": [
                [1, 0, 10.0, 10.0, 2.0, 40, 3],
                [2, -1, 30.0, 10.0, 2.0, 20, 2],
                [3, -1, 10.0, 40.0, 2.0, 20, 2],
            ],
            "fleets": [],
        }
    )
    target_id = infer_target_planet_id(list(obs.planets), obs.planets[0], 0.0)
    assert target_id == 2


def test_bin_to_ship_count_clamps_to_available_ships():
    source_ships = 7
    max_bin = len(SHIP_FRACTIONS) - 1
    assert bin_to_ship_count(source_ships, max_bin) == 7
    assert bin_to_ship_count(source_ships, 0) >= 1
    assert fleet_speed(1) == 1.0
