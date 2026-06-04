import os
from pathlib import Path
import datetime
import collections

import polars as pl
import tyro
from dataclasses import dataclass


def get_meta_dir() -> Path:
    candidates = []

    env_path = os.environ.get("META_KAGGLE_DIR")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path("/kaggle/input/meta-kaggle"),
            Path(__file__).resolve().parents[3] / "input" / "meta-kaggle",
            Path("../input/meta-kaggle"),
        ]
    )

    for candidate in candidates:
        if (candidate / "Episodes.csv").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Meta Kaggle dataset. Expected Episodes.csv in one of: "
        + ", ".join(str(path) for path in candidates)
    )


@dataclass
class Args:
    competition_id: int
    min_create_time: str = "2026-01-01"
    output_path: str = "games.csv"


def main(args: Args):
    meta_dir = get_meta_dir()

    episodes_df = pl.scan_csv(meta_dir / "Episodes.csv")
    episodes_df = (
        episodes_df.filter(pl.col("CompetitionId") == args.competition_id)
        .with_columns(
            pl.col("CreateTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
            pl.col("EndTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
        )
        .sort("Id")
        .collect()
    )

    min_dt = datetime.datetime.fromisoformat(args.min_create_time)
    episodes_df = episodes_df.filter(pl.col("CreateTime") > min_dt)

    agents_df = pl.scan_csv(
        meta_dir / "EpisodeAgents.csv",
        schema_overrides={
            "Reward": pl.Float32,
            "UpdatedConfidence": pl.Float32,
            "UpdatedScore": pl.Float32,
        },
    )
    agents_df = (
        agents_df.filter(pl.col("EpisodeId").is_in(episodes_df["Id"].to_list()))
        .with_columns(
            [
                pl.when(pl.col("InitialConfidence") == "")
                .then(None)
                .otherwise(pl.col("InitialConfidence"))
                .cast(pl.Float64)
                .alias("InitialConfidence"),
                pl.when(pl.col("InitialScore") == "")
                .then(None)
                .otherwise(pl.col("InitialScore"))
                .cast(pl.Float64)
                .alias("InitialScore"),
            ]
        )
        .collect()
    )

    games_df = agents_df.join(episodes_df, left_on="EpisodeId", right_on="Id").select(
        ["EpisodeId", "Index", "Reward", "SubmissionId", "UpdatedScore", "CreateTime"]
    )

    episode_id_to_sids = collections.defaultdict(list)
    for episode_id, index, sid in zip(
        games_df["EpisodeId"], games_df["Index"], games_df["SubmissionId"]
    ):
        episode_id_to_sids[episode_id].append((index, sid))

    opponent_lists = []
    for episode_id, index in zip(games_df["EpisodeId"], games_df["Index"]):
        entries = sorted(episode_id_to_sids[episode_id])
        opponent_lists.append(
            "|".join(str(sid) for other_index, sid in entries if other_index != index)
        )

    games_df = games_df.with_columns(
        pl.Series(name="OppSubmissionIds", values=opponent_lists)
    )
    games_df.write_csv(args.output_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
