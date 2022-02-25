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
    args["env"] = file_info[1]
    assert args["alg_name"] in ["SAC", "TD3"]
    assert args["env"] in ["Hopper-v3", "Cheetah-v3"]

    return args


def main():
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
        alg.load("models/TD3_Hopper-v3_1337")

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
