import unittest
import os
from torch.nn import Module
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tester(unittest.TestCase):
    def setUp(self) -> None:
        # clear models and results folders before tests:
        for folder_name in ["test/models", "test/results"]:
            file_names = os.listdir(folder_name)

            for file_name in file_names:
                os.remove(os.path.join(folder_name, file_name))
        return super().setUp()


def models_equal(model_a: Module, model_b: Module):
    """
    Check that all parameters in a and b are equal.
    """
    modules_equal = [
        torch.equal(par_a, par_b)
        for par_a, par_b in zip(model_a.parameters(), model_b.parameters())
    ]
    return all(modules_equal)


def sample_tensors(env):
    """
    Samples state, action, next_state, reward and done by resetting the env and performing one step with a randomly sampled action
    """
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    tensors = map(
        lambda x: torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0),
        [state, action, next_state, reward, done],
    )
    return (x for x in tensors)
