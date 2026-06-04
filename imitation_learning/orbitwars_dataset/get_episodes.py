import os
import json
import glob
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
import tyro

from orbitwars.replay import normalize_kaggle_replay

BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
GET_URL = BASE_URL + "GetEpisodeReplay"
WORKING_FOLDER = Path(__file__).parent
OUTPUT_DIR = f"{WORKING_FOLDER}/episodes"
SUBMISSIONS_PATH = f"{WORKING_FOLDER}/submissions.csv"
GAMES_PATH = f"{WORKING_FOLDER}/games.csv"


def _parse_opponents(value) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [int(x) for x in str(value).split("|") if x]


def get_submission_ids(episode_id, games_df=None):
    if games_df is None:
        games_df = pd.read_csv(GAMES_PATH)
    df = games_df[games_df["EpisodeId"] == episode_id].sort_values("Index", ascending=True)
    return [int(x) for x in df["SubmissionId"]]


def update_submission_names(submission_to_name):
    if not os.path.exists(SUBMISSIONS_PATH):
        return
    df = pd.read_csv(SUBMISSIONS_PATH)
    df["name"] = [submission_to_name.get(sid, name) for sid, name in zip(df["submission_id"], df["name"])]
    df.to_csv(SUBMISSIONS_PATH, index=False)


def _download_replay_via_cli(episode_id: int) -> dict:
    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI is not available for replay fallback")

    before = set(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    cmd = ["kaggle", "competitions", "replay", str(int(episode_id)), "-p", OUTPUT_DIR]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    after = set(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))

    candidates = sorted(after - before)
    if not candidates:
        direct_path = os.path.join(OUTPUT_DIR, f"{episode_id}.json")
        if os.path.exists(direct_path):
            candidates = [direct_path]
        else:
            candidates = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"*{episode_id}*.json")))

    if not candidates:
        raise FileNotFoundError(
            f"Kaggle CLI reported success but no replay JSON was found for episode {episode_id}. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    replay_path = max(candidates, key=os.path.getmtime)
    with open(replay_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_replay_json(episode_id: int) -> dict:
    try:
        response = requests.post(GET_URL, json={"episodeId": int(episode_id)}, timeout=60)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, requests.exceptions.JSONDecodeError):
        return _download_replay_via_cli(episode_id)


def get_episode(episode_id, games_df=None):
    replay = _fetch_replay_json(episode_id)
    assert episode_id == replay["info"]["EpisodeId"]
    normalized = normalize_kaggle_replay(replay)
    submissions = get_submission_ids(episode_id, games_df)
    team_names = normalized["metadata"]["team_names"]
    update_submission_names(dict(zip(submissions, team_names)))
    for idx, agent in enumerate(normalized["metadata"]["agents"]):
        if idx < len(submissions):
            agent["submission_id"] = submissions[idx]
    path = os.path.join(OUTPUT_DIR, f"{episode_id}.json")
    json.dump(normalized, open(path, "w"))


def get_episodes(submission_id, num_episodes=1000, min_score=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    games = pd.read_csv(GAMES_PATH)
    if min_score is not None:
        submissions = pd.read_csv(SUBMISSIONS_PATH)
        sid_to_score = dict(zip(submissions["submission_id"], submissions["score"]))
        opp_scores = []
        for value in games["OppSubmissionIds"]:
            scores = [sid_to_score.get(opp_id, 0) for opp_id in _parse_opponents(value)]
            opp_scores.append(max(scores) if scores else 0)
        games["opp_score"] = opp_scores
        games = games[games["opp_score"] >= min_score]

    episodes = set(games[games["SubmissionId"] == submission_id]["EpisodeId"])
    episodes_to_download = []
    for episode_id in episodes:
        path = os.path.join(OUTPUT_DIR, f"{episode_id}.json")
        if not os.path.exists(path):
            episodes_to_download.append(episode_id)

    for i, episode_id in enumerate(episodes_to_download[:num_episodes], start=1):
        print(f"request episode {episode_id}: {i}/{len(episodes_to_download)}")
        get_episode(episode_id, games_df=games)


@dataclass
class Args:
    submission_id: int
    num_episodes: Optional[int] = 1000
    min_score: Optional[int] = None


if __name__ == "__main__":
    args = tyro.cli(Args)
    get_episodes(args.submission_id, args.num_episodes, args.min_score)
