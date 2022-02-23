from typing import Optional
from time import perf_counter

import gym
import torch
import torch.nn as nn
from numpy import ndarray
from TD3.utils import ReplayBuffer
from torch import Tensor, tanh
from torch.distributions.normal import Normal
from torch.nn.functional import relu

from src.uniform_policy import UniformPolicy
from src.utils import set_seeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(
        self,
        env: str,
        seed: int,
        hidden_dim: int,
        max_action: int,
        grad_steps: int,
        batch_size: int,
        replay_buffer_size: int,
        n_initial_exploration_steps: int,
        min_replay_buffer_size: int,
        max_env_steps: int,
        adam_kwargs: dict,
    ) -> None:
        # Make env
        self.env = gym.make(env)

        # neural net functions:
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.v = Value(state_dim, hidden_dim, adam_kwargs)
        self.qf1 = Q(state_dim, action_dim, hidden_dim, adam_kwargs)
        self.qf2 = Q(action_dim, action_dim, hidden_dim, adam_kwargs)
        self.policy = Policy(action_dim, state_dim, hidden_dim, max_action, adam_kwargs)

        # Other hyperparameters
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)
        self.grad_steps = grad_steps
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.initial_exploration_policy = UniformPolicy(action_dim, max_action)
        self.min_replay_buffer_size = min_replay_buffer_size
        self.max_env_steps = max_env_steps

        # Set seeds
        set_seeds(seed, self.env)

        # TODO init weights?

    def train(self):
        """
        Train the model for a number of iterations.
        In every iteration, one environment step is taken
        and self.grad_step gradient steps are done.
        """
        self.start_time = perf_counter()
        state = self.env.reset()
        
        for i in range(self.max_env_steps):
            state = self._train_iteration(state, i)


    def _train_iteration(self, state, iteration: int) -> ndarray:
        """
        Do one iteration of training. Returns the state that we end up in after
        the environment step phase.
        """

        # Decide whether action is sampled from initial exploration or actual policy
        if iteration <= self.n_initial_exploration_steps:
            action = self.initial_exploration_policy.get_random_action()
        else:
            action = self.policy(state)

        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.add(state, action, next_state, reward, done)
        if done:
            state = self.env.reset()
        else:
            state = next_state

        for _ in range(self.grad_steps):
            if self.replay_buffer.size >= self.min_replay_buffer_size:
                batch = self.replay_buffer.sample(self.batch_size)

        # logging
        elapsed_time = perf_counter() - self.start_time


        return state


class Policy(nn.Module):
    # Gaussian policy
    # For each action in the action space, output a scalar in [-1, 1]
    def __init__(
        self, action_dim: int, state_dim: int, hidden_dim: int, max_action: float, adam_kwargs: dict
    ):
        super(Policy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        # hidden_dim -> mu0, mu1, ..., sig0, sig1, ...
        self.l3 = nn.Linear(hidden_dim, action_dim * 2)

        self.action_dim = action_dim
        self.max_action = max_action

        self.optimizer = torch.optim.Adam(self.parameters(), **adam_kwargs)

    def forward(
        self, state: Tensor, deterministic: bool
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor, Tensor]]]:
        # returns (action, (log_prob(action), mus, log_sigmas))
        h = relu(self.l1(state))
        h = relu(self.l2(h))
        h = self.l3(h)

        mus = h[: self.action_dim]
        log_sigmas = h[self.action_dim :]

        if deterministic:
            return self.max_action * tanh(mus), None
        else:
            normal = Normal(mus, torch.exp(log_sigmas))
            actions = normal.rsample()
            log_probs = normal.log_prob(actions)
            log_probs -= self._correction(actions)
            return self.max_action * tanh(actions), (log_probs, mus, log_sigmas)

    def _correction(self, actions):
        # apply a squash correction to the actions for calculating the log_probs correctly (?) (sac code gaussian_policy line 74):
        # https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/policies/gaussian_policy.py#L74
        return torch.sum(torch.log(1 - tanh(actions) ** 2 + 1e-6), dim=1)

    # def _loss(self) -> Tensor:
    #     pass
    #
    # def step(self):
    #     self.optimizer.zero_grad()
    #     loss = self._loss()
    #     loss.backward()
    #     self.optimizer.step()


class Value(nn.Module):
    # V gets the state, outputs a single scalar (soft value)
    def __init__(self, state_dim: int, hidden_dim: int):
        super(Value, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor) -> Tensor:
        h = relu(self.l1(state))
        h = relu(self.l2(h))
        return self.l3(h)


class Q(nn.Module):
    # Q gets the state and action, outputs a single scalar (state-action value)
    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int):
        super(Q, self).__init__()

        self.l1 = nn.Linear(action_dim + state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, action: Tensor, state: Tensor) -> Tensor:
        h = relu(self.l1(torch.cat((action, state), dim=0)))
        h = relu(self.l2(h))
        return self.l3(h)
