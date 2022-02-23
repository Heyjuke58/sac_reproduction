import torch
import numpy as np

def set_seeds(seed, env):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)