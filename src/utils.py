import torch
import numpy as np
import time


def set_seeds(seed, env):
    """
    Seed all relevant seeds (torch, environment and numpy) to ensure deterministic behaviour
    """
    env.seed(seed)
    env.action_space.seed(seed)
    # set another seed for action space for deterministic sampling behaviour
    # https://github.com/openai/gym/issues/681
    env.action_space.np_random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_timestamp() -> str:
    return time.strftime("%Y_%m_%d-%H-%M-%S")
