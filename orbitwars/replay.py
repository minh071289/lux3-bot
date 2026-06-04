from __future__ import annotations

from typing import Any


def _safe_get(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return default


def _extract_actions(step: Any, player_count: int) -> dict[str, list[list[float]]]:
    actions: dict[str, list[list[float]]] = {}
    for player_idx in range(player_count):
        player_state = step[player_idx] if isinstance(step, list) and player_idx < len(step) else {}
        action = _safe_get(player_state, "action")
        if action is None:
            replay_info = _safe_get(player_state, "info", {})
            replay_actions = _safe_get(_safe_get(replay_info, "replay", {}), "actions")
            if isinstance(replay_actions, list) and player_idx < len(replay_actions):
                action = replay_actions[player_idx]
        actions[f"player_{player_idx}"] = action or []
    return actions


def _extract_observations(step: Any, player_count: int) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for player_idx in range(player_count):
        player_state = step[player_idx] if isinstance(step, list) and player_idx < len(step) else {}
        obs = _safe_get(player_state, "observation")
        if obs is None:
            replay_info = _safe_get(player_state, "info", {})
            replay_obs = _safe_get(_safe_get(replay_info, "replay", {}), "observations")
            if isinstance(replay_obs, list) and player_idx < len(replay_obs):
                obs = replay_obs[player_idx]
        if obs is None and isinstance(step, list) and step:
            first_info = _safe_get(step[0], "info", {})
            replay_obs = _safe_get(_safe_get(first_info, "replay", {}), "observations")
            if isinstance(replay_obs, list) and player_idx < len(replay_obs):
                obs = replay_obs[player_idx]
        obs = dict(obs or {})
        obs["player"] = obs.get("player", player_idx)
        observations[f"player_{player_idx}"] = obs
    return observations


def normalize_kaggle_replay(kaggle_replay: dict[str, Any]) -> dict[str, Any]:
    info = kaggle_replay.get("info", {})
    steps = kaggle_replay.get("steps", [])
    team_names = info.get("TeamNames", [])
    player_count = max(2, len(team_names))
    if steps and isinstance(steps[0], list):
        player_count = max(player_count, len(steps[0]))

    observations = []
    actions = []
    for step in steps:
        observations.append(_extract_observations(step, player_count))
        actions.append(_extract_actions(step, player_count))

    final_rewards = []
    if steps and isinstance(steps[-1], list):
        for player_idx in range(player_count):
            state = steps[-1][player_idx]
            reward = state.get("reward")
            if reward is None:
                reward = state.get("observation", {}).get("reward", 0)
            final_rewards.append(reward if reward is not None else 0)
    else:
        final_rewards = [0 for _ in range(player_count)]

    agents = []
    submissions = info.get("SubmissionIds", [])
    for player_idx in range(player_count):
        agents.append(
            {
                "name": team_names[player_idx] if player_idx < len(team_names) else f"player_{player_idx}",
                "reward": final_rewards[player_idx] if player_idx < len(final_rewards) else 0,
                "submission_id": submissions[player_idx] if player_idx < len(submissions) else None,
            }
        )

    return {
        "metadata": {
            "episode_id": info.get("EpisodeId"),
            "agents": agents,
            "team_names": team_names,
        },
        "observations": observations,
        "actions": actions,
    }
