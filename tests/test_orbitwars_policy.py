import torch

from orbitwars.features import NUM_SHIP_BINS, featurize_observation
from orbitwars.model import OrbitWarsGraphPolicy
from orbitwars.policy import decode_actions_from_outputs
from orbitwars.types import normalize_observation


def test_model_forward_shapes():
    obs = normalize_observation(
        {
            "player": 0,
            "planets": [
                [1, 0, 10.0, 10.0, 2.0, 50, 3],
                [2, -1, 30.0, 10.0, 2.0, 20, 2],
            ],
            "fleets": [],
        }
    )
    encoded = featurize_observation(obs)
    model = OrbitWarsGraphPolicy()
    launch, target, ship = model(
        torch.from_numpy(encoded.global_features).unsqueeze(0),
        torch.from_numpy(encoded.planet_features).unsqueeze(0),
        torch.from_numpy(encoded.pair_features).unsqueeze(0),
        torch.from_numpy(encoded.planet_mask).unsqueeze(0),
    )
    assert launch.shape[1] == encoded.planet_features.shape[0]
    assert target.shape[1] == encoded.planet_features.shape[0]
    assert ship.shape[-1] == NUM_SHIP_BINS


def test_decode_actions_emits_legal_launch():
    obs = normalize_observation(
        {
            "player": 0,
            "planets": [
                [1, 0, 10.0, 10.0, 2.0, 40, 3],
                [2, -1, 30.0, 10.0, 2.0, 20, 2],
            ],
            "fleets": [],
        }
    )
    encoded = featurize_observation(obs)
    launch_logits = torch.full((encoded.planet_features.shape[0],), -10.0)
    launch_logits[0] = 10.0
    target_logits = torch.full((encoded.planet_features.shape[0], encoded.planet_features.shape[0]), -1e9)
    target_logits[0, 1] = 1.0
    ship_logits = torch.zeros((encoded.planet_features.shape[0], NUM_SHIP_BINS))
    ship_logits[0, -1] = 5.0
    actions = decode_actions_from_outputs(
        obs,
        encoded.planet_ids,
        encoded.planet_mask,
        encoded.owned_planet_mask,
        launch_logits,
        target_logits,
        ship_logits,
    )
    assert actions == [[1, 0.0, 40]]
