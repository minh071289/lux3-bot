import json
import os
import sys
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import tyro
from tqdm.contrib.concurrent import process_map

WORKING_FOLDER = Path(__file__).parent
BOT_DIR = WORKING_FOLDER.parent.parent
sys.path.append(str(BOT_DIR))

from orbitwars.features import (
    GLOBAL_DIM,
    NUM_SHIP_BINS,
    PAIR_FEAT_DIM,
    PLANET_FEAT_DIM,
    encode_labels,
    featurize_observation,
)
from orbitwars.types import normalize_observation

EPISODES_DIR = f"{WORKING_FOLDER}/episodes"
OUTPUT_DIR = f"{WORKING_FOLDER}/agent_episodes"
SUBMISSIONS_PATH = f"{WORKING_FOLDER}/submissions.csv"
GAMES_PATH = f"{WORKING_FOLDER}/games.csv"


def is_winning_agent(agent_reward, max_reward, include_draws):
    if agent_reward > max_reward:
        return True
    if include_draws and agent_reward == max_reward:
        return True
    return False


def convert_episode(episode_data, team_id, include_draws=False):
    agents = episode_data["metadata"]["agents"]
    rewards = [agent.get("reward", 0) for agent in agents]
    max_reward = max(rewards) if rewards else 0
    if not is_winning_agent(rewards[team_id], max_reward, include_draws):
        return None

    global_features = []
    planet_features = []
    pair_features = []
    planet_masks = []
    owned_planet_masks = []
    launch_labels = []
    target_labels = []
    ship_bin_labels = []
    steps = []
    planet_ids = []
    episode_ids = []
    stats = {"duplicate_moves": 0, "failed_target_inference": 0}
    episode_id = episode_data["metadata"].get("episode_id", -1)

    observations = episode_data.get("observations", [])
    actions = episode_data.get("actions", [])
    num_steps = min(len(observations), len(actions))
    for step_idx in range(num_steps):
        player_key = f"player_{team_id}"
        raw_obs = observations[step_idx].get(player_key)
        if raw_obs is None:
            continue
        raw_obs["step"] = step_idx
        obs = normalize_observation(raw_obs)
        if not obs.planets:
            continue

        encoded = featurize_observation(obs)
        moves = actions[step_idx].get(player_key, [])
        launch, target, ship_bins, step_stats = encode_labels(obs, moves, encoded.planet_ids)
        if encoded.owned_planet_mask.sum() <= 0:
            continue

        stats["duplicate_moves"] += step_stats["duplicate_moves"]
        stats["failed_target_inference"] += step_stats["failed_target_inference"]

        global_features.append(encoded.global_features)
        planet_features.append(encoded.planet_features)
        pair_features.append(encoded.pair_features)
        planet_masks.append(encoded.planet_mask)
        owned_planet_masks.append(encoded.owned_planet_mask)
        launch_labels.append(launch)
        target_labels.append(target)
        ship_bin_labels.append(ship_bins)
        steps.append(step_idx)
        planet_ids.append(encoded.planet_ids)
        episode_ids.append(episode_id)

    if not steps:
        return None

    return {
        "global_features": np.asarray(global_features, dtype=np.float32).reshape(-1, GLOBAL_DIM),
        "planet_features": np.asarray(planet_features, dtype=np.float32).reshape(-1, encoded.planet_features.shape[0], PLANET_FEAT_DIM),
        "pair_features": np.asarray(pair_features, dtype=np.float32).reshape(-1, encoded.pair_features.shape[0], encoded.pair_features.shape[1], PAIR_FEAT_DIM),
        "planet_mask": np.asarray(planet_masks, dtype=np.float32),
        "owned_planet_mask": np.asarray(owned_planet_masks, dtype=np.float32),
        "launch_labels": np.asarray(launch_labels, dtype=np.int16),
        "target_labels": np.asarray(target_labels, dtype=np.int16),
        "ship_bin_labels": np.asarray(ship_bin_labels, dtype=np.int16),
        "planet_ids": np.asarray(planet_ids, dtype=np.int16),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "steps": np.asarray(steps, dtype=np.int16),
        "stats_duplicate_moves": np.asarray([stats["duplicate_moves"]], dtype=np.int32),
        "stats_failed_target_inference": np.asarray([stats["failed_target_inference"]], dtype=np.int32),
        "player_id": np.asarray([team_id], dtype=np.int8),
    }


def convert_episodes(submission_id, num_episodes=None, min_opp_score=None, num_workers=1, include_draws=False):
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    games = pd.read_csv(GAMES_PATH, usecols=["SubmissionId", "EpisodeId", "OppSubmissionIds"])
    games = games[games["SubmissionId"] == submission_id]
    if min_opp_score is not None:
        submissions_df = pd.read_csv(SUBMISSIONS_PATH, usecols=["submission_id", "score"])
        sid_to_score = dict(zip(submissions_df["submission_id"], submissions_df["score"]))
        opp_scores = []
        for value in games["OppSubmissionIds"]:
            scores = [sid_to_score.get(int(x), 0) for x in str(value).split("|") if x]
            opp_scores.append(max(scores) if scores else 0)
        games["opp_score"] = opp_scores
        games = games[games["opp_score"] >= min_opp_score]

    episodes = sorted(int(x) for x in games["EpisodeId"].unique())
    episodes_to_convert = []
    for episode_id in episodes:
        episode_path = f"{EPISODES_DIR}/{episode_id}.json"
        agent_episode_path = f"{OUTPUT_DIR}/{submission_id}_{episode_id}.npz"
        if os.path.exists(episode_path) and not os.path.exists(agent_episode_path):
            episodes_to_convert.append(episode_id)
    if num_episodes is not None:
        episodes_to_convert = episodes_to_convert[:num_episodes]

    if num_workers <= 1:
        for idx, episode_id in enumerate(episodes_to_convert, start=1):
            print(f"converting {episode_id}: {idx}/{len(episodes_to_convert)}")
            convert_and_save(submission_id, episode_id, include_draws)
    else:
        process_map(
            convert_and_save,
            [submission_id for _ in episodes_to_convert],
            episodes_to_convert,
            [include_draws for _ in episodes_to_convert],
            max_workers=num_workers,
            chunksize=4,
        )


def convert_and_save(submission_id, episode_id, include_draws=False):
    episode_path = f"{EPISODES_DIR}/{episode_id}.json"
    agent_episode_path = f"{OUTPUT_DIR}/{submission_id}_{episode_id}.npz"
    episode_data = json.load(open(episode_path, "r"))
    agents = episode_data["metadata"]["agents"]
    for team_id, agent in enumerate(agents):
        if agent.get("submission_id") != submission_id:
            continue
        converted = convert_episode(episode_data, team_id, include_draws=include_draws)
        if converted is not None:
            np.savez(agent_episode_path, **converted)
        break


@dataclass
class Args:
    submission_id: int
    num_episodes: Optional[int] = None
    min_opp_score: Optional[int] = None
    num_workers: int = 1
    include_draws: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    convert_episodes(
        args.submission_id,
        num_episodes=args.num_episodes,
        min_opp_score=args.min_opp_score,
        num_workers=args.num_workers,
        include_draws=args.include_draws,
    )
