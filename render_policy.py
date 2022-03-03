from TD3.TD3 import TD3
from typing import Union
import gym
import numpy as np
from typing import Any
import argparse
import os.path
from src.sac_eval import SACEvaluator


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model_file",
        help="Which saved model should be run (pass a file in the models folder). Filename needs to be of the correct format (include env name, etc.).",
    )
    args = vars(parser.parse_args())

    # extracting the model and env name from the filename:
    filepath = args["model_file"]
    filename: str = os.path.basename(filepath)
    file_info = filename.split("_")

    args["alg_name"] = file_info[0]

    allowed_envs = ["HalfCheetah-v3", "Hopper-v3"]
    for allowed_env in allowed_envs:
        if allowed_env in filename:
            args["env"] = allowed_env
            break

    assert args["alg_name"] in ["SAC", "TD3"]
    assert "env" in args.keys()
    return args


def main():
    """
    Renders an agent specified by command line parameter on the corresponding environment for 10000 steps
    """
    args = parse_args()

    env = gym.make(args["env"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    alg: Union[SACEvaluator, TD3]
    if args["alg_name"] == "SAC":
        alg = SACEvaluator(args["model_file"], max_action=max_action)
    elif args["alg_name"] == "TD3":
        alg = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
        alg.load(args["model_file"])

    observation = env.reset()
    for _ in range(10000):
        env.render()
        action = alg.select_action(np.array(observation))
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
    env.close()


if __name__ == "__main__":
    main()
