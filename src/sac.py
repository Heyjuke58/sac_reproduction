import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch import Tensor, tanh
from torch.distributions.normal import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, max_action: int) -> None:
        self.v = Value(state_dim, hidden_dim)
        self.qf1 = Q(state_dim, action_dim, hidden_dim)
        self.qf2 = Q(action_dim, action_dim, hidden_dim)
        self.policy = Policy(action_dim, state_dim, hidden_dim, max_action)
        self.replay_buffer = []


class Policy(nn.Module):
    # Gaussian policy
    # For each action in the action space, output a scalar in [-1, 1]
    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, max_action: float):
        super(Policy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        # hidden_dim -> mu0, mu1, ..., sig0, sig1, ...
        self.l3 = nn.Linear(hidden_dim, action_dim * 2)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, state: Tensor, action: Tensor, deterministic: bool) -> Tensor:
        h = relu(self.l1(state))
        h = relu(self.l2(h))
        h = self.l3(h)

        mus = h[:self.action_dim]
        sigmas = h[self.action_dim:]

        if deterministic:
            return self.max_action * tanh(mus)
        else:
            action = Normal(mus, sigmas).rsample()
            return self.max_action * tanh(action)


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
