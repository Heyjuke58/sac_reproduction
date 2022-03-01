import gym
from gym.spaces import Box, Discrete

# from gym.utils import seeding
import gym.utils.seeding as seeding
import numpy as np
from typing import Optional
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FloatDiscrete:
    """
    Like gym.spaces.Discrete, but converts the integer output into floats
    """

    def __init__(self, n, start: int = 0):
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, (int, np.integer))
        self.n = int(n)
        self.start = int(start)
        self.shape = (1,)
        self.np_random, _ = seeding.np_random(None)

    def sample(self) -> float:
        return float(self.start + np.random.randint(self.n))

    def seed(self, _) -> None:
        return


class MinusPlusDiscrete:
    """
    returns values of -1 or +1
    """

    def __init__(self) -> None:
        self.np_random, _ = seeding.np_random(None)
        self.shape = (1,)
        return

    def sample(self) -> float:
        return float(np.random.randint(1)) * 2 - 1

    def seed(self, _):
        return


class Probe1(gym.Env):
    """
    One state, one action, always get +1 reward and done instantly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = FloatDiscrete(1)
        self.observation_space = FloatDiscrete(1)

    def step(self, action):
        return np.array([0.0], dtype=np.float32), 1, 1, {}

    def reset(self):
        return np.array([0.0], dtype=np.float32)


class Probe2(gym.Env):
    """
    Two states, one action, get reward (+1, -1) based on state you get thrown in at the start. Done instantly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = FloatDiscrete(1)
        self.observation_space = FloatDiscrete(2)
        # self.last_state = self.observation_space.sample()
        # self.last_state = 0.0

    def step(self, action):
        # returns observation, reward, done, info
        observation = self.observation_space.sample()
        r = np.array(observation, dtype=np.float32), self.last_state, 1, {}
        self.last_state = observation
        return r

    def reset(self):
        obs = self.observation_space.sample()
        self.last_state = obs
        return np.array([obs], dtype=np.float32)


class Probe3(gym.Env):
    """
    Two states, one action. Start in s_0, goto s_1, done. s_0 gives 0 reward, s_1 gives +1 reward.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = FloatDiscrete(1)
        self.observation_space = FloatDiscrete(2)
        self.state = 0.0

    def step(self, action):
        if self.state == 0.0:
            self.state = 1.0
            return np.array([self.state]), 0, 0, {}
        elif self.state == 1.0:
            # should never end up in state 2 because done:
            self.state = 2.0
            return np.array([self.state]), 1, 1, {}

    def reset(self):
        self.state = 0.0
        return np.array([self.state])


class Probe4(gym.Env):
    """
    One state, two actions. Reward (+1, -1) based on action. Done instantly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = MinusPlusDiscrete()
        self.observation_space = FloatDiscrete(1)

    def step(self, action):
        action = -1.0 if action <= 0 else 1.0
        return np.array([0.0]), action, 1, {}

    def reset(self):
        return np.array([0.0])


class Probe5(gym.Env):
    """
    Two states, two actions. Have to match action to state that you start in to get reward of +1,
    otherwise get -1 reward. Done instantly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = MinusPlusDiscrete()
        # self.observation_space = FloatDiscrete(3)
        self.observation_space = FloatDiscrete(2)

    def step(self, action):
        action = 0.0 if action <= 0 else 1.0
        if self.state == 2.0:
            reward = 0.0
        else:
            reward = float(action == self.state) * 2 - 1
            # reward = action
        r = np.array([0.0]), reward, 1, {}
        self.state = 100.0  # shouldn't matter
        return r

    def reset(self):
        obs = self.observation_space.sample()
        self.state = obs
        return np.array([obs], dtype=np.float32)
