import torch
from torch import Tensor


# for sampling random actions to fill up the replay buffer at the start:
class UniformPolicy:
    def __init__(self, action_dim: int, max_action: float) -> None:
        self.action_dim = action_dim
        self.max_action = max_action

    def get_random_action(self) -> Tensor:
        """
        Return a uniformly sampled random action of shape (self.action_dim,).
        Each action scalar is in [-self.max_action, self.max_action]
        """
        return torch.rand((self.action_dim,)) * 2 * self.max_action - self.max_action
