import numpy as np
import torch


class ReplayBuffer(object):
    """
    Ring buffer for (state, action, next_state, reward, done)-tuples from stepping through gym envs.

    Taken from the TD3 implementation with some small changes.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        """
        Add one step of an environment to the replay buffer. Oldest sample is forgotten if buffer
        is full.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        Sample a batch of size batch_size from the replay buffer (with replacement).
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.state[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.action[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_state[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.reward[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.done[ind], dtype=torch.float32, device=self.device),
        )
