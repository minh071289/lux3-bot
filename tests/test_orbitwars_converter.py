from orbitwars.features import encode_labels, featurize_observation
from orbitwars.replay import normalize_kaggle_replay
from orbitwars.types import normalize_observation


def test_normalize_kaggle_replay_extracts_observations_and_actions():
    replay = {
        "info": {"EpisodeId": 123, "TeamNames": ["A", "B"], "SubmissionIds": [11, 22]},
        "steps": [
            [
                {"observation": {"player": 0, "planets": [], "fleets": []}},
                {"observation": {"player": 1, "planets": [], "fleets": []}},
            ],
            [
                {
                    "observation": {"player": 0, "planets": [], "fleets": []},
                    "action": [[1, 0.0, 5]],
                    "reward": 1,
                },
                {
                    "observation": {"player": 1, "planets": [], "fleets": []},
                    "action": [],
                    "reward": 0,
                },
            ],
        ],
    }
    normalized = normalize_kaggle_replay(replay)
    assert normalized["metadata"]["episode_id"] == 123
    assert normalized["actions"][1]["player_0"] == [[1, 0.0, 5]]
    assert normalized["observations"][0]["player_1"]["player"] == 1


def test_encode_labels_collapses_duplicate_source_moves():
    obs = normalize_observation(
        {
            "player": 0,
            "step": 5,
            "planets": [
                [1, 0, 10.0, 10.0, 2.0, 50, 3],
                [2, -1, 25.0, 10.0, 2.0, 10, 1],
                [3, -1, 10.0, 25.0, 2.0, 10, 1],
            ],
            "initial_planets": [
                [1, 0, 10.0, 10.0, 2.0, 50, 3],
                [2, -1, 25.0, 10.0, 2.0, 10, 1],
                [3, -1, 10.0, 25.0, 2.0, 10, 1],
            ],
            "fleets": [],
        }
    )
    encoded = featurize_observation(obs)
    launch, target, ship_bin, stats = encode_labels(
        obs,
        [[1, 0.0, 10], [1, 0.0, 20]],
        encoded.planet_ids,
    )
    assert stats["duplicate_moves"] == 1
    assert launch[0] == 1
    assert target[0] == 1
    assert ship_bin[0] >= 0
