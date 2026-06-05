import json
import os
from pathlib import Path
from argparse import Namespace
from typing import Any

from orbitwars import OrbitWarsAgent
from orbitwars.types import normalize_observation

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict()
agent_prev_obs = dict()
orbitwars_agent = None


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _is_orbitwars_observation(obs):
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return False
    if isinstance(obs, dict):
        return "planets" in obs and "fleets" in obs
    return hasattr(obs, "planets") and hasattr(obs, "fleets")


def _get_working_folder():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle_simulations"):
        return "/kaggle_simulations/agent/"
    return Path(__file__).parent


def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    from agent import Agent
    from agent.kit import from_json

    working_folder = _get_working_folder()

    weights_dir = f"{working_folder}/imitation_learning/weights/"

    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = Agent(player, configurations["env_cfg"], weights_dir)
    agent = agent_dict[player]
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())


def agent(observation, configuration=None):
    global orbitwars_agent
    use_heuristic_fallback = True
    launch_threshold = 0.35
    if isinstance(configuration, dict):
        use_heuristic_fallback = _coerce_bool(
            configuration.get("orbitwars_use_heuristic_fallback"),
            True,
        )
        if configuration.get("orbitwars_launch_threshold") is not None:
            launch_threshold = float(configuration.get("orbitwars_launch_threshold"))

    if orbitwars_agent is None:
        working_folder = _get_working_folder()
        weights_path = f"{working_folder}/imitation_learning/weights/orbitwars_graph_policy.pth"
        orbitwars_agent = OrbitWarsAgent(
            weights_path=weights_path,
            use_heuristic_fallback=use_heuristic_fallback,
            launch_threshold=launch_threshold,
        )
    else:
        orbitwars_agent.set_use_heuristic_fallback(use_heuristic_fallback)
        orbitwars_agent.set_launch_threshold(launch_threshold)

    obs = normalize_observation(observation)
    return orbitwars_agent.act(obs)


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    while True:
        inputs = read_input()
        raw_input = json.loads(inputs)
        observation = Namespace(
            **dict(
                step=raw_input["step"],
                obs=raw_input["obs"],
                remainingOverageTime=raw_input["remainingOverageTime"],
                player=raw_input["player"],
                info=raw_input["info"],
            )
        )
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))
