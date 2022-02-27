from torch import Tensor, relu, tanh
import torch.nn as nn
from typing import Any, Optional
import torch
from torch.distributions.normal import Normal
from numpy import ndarray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    # Gaussian policy
    # For each action in the action space, output a scalar in [-1, 1]
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float,
        adam_kwargs: dict[str, Any],
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
        """
        Returns (action, (log_prob(action), mus, log_sigmas)).
        mus and log_sigmas are passed along for regularization.
        """
        h = relu(self.l1(state))
        h = relu(self.l2(h))
        h = self.l3(h)

        mus = h[:, : self.action_dim]
        log_sigmas = h[:, self.action_dim :]

        if deterministic:
            return self.max_action * tanh(mus), None
        else:
            # reparametrization trick, sampling closely following the original implementation
            normal = Normal(mus, torch.exp(log_sigmas))
            actions = normal.rsample()  # (b, action space dim)
            log_probs = normal.log_prob(actions)  # (b, action space dim)
            log_probs -= self._correction(actions).unsqueeze(0).T
            if self.max_action == 0.0:  # only for debugging
                log_probs *= 0.0
            return self.max_action * tanh(actions), (log_probs, mus, log_sigmas)

    def _correction(self, actions):
        # apply a squash correction to the actions for calculating the log_probs correctly (?) (sac code gaussian_policy line 74):
        # https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/policies/gaussian_policy.py#L74
        return torch.sum(torch.log(1 - tanh(actions) ** 2 + 1e-6), dim=1)

    def get_action(self, state: ndarray):
        """Get action for evaluation of policy"""
        with torch.no_grad():
            action, _ = self.forward(
                torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                deterministic=True,
            )
        return action.cpu().numpy()

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
    def __init__(self, state_dim: int, hidden_dim: int, adam_kwargs: dict[str, Any]):
        super(Value, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), **adam_kwargs)

    def forward(self, state: Tensor) -> Tensor:
        h = relu(self.l1(state))
        h = relu(self.l2(h))
        return self.l3(h)


class Q(nn.Module):
    # Q gets the state and action, outputs a single scalar (state-action value)
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, adam_kwargs: dict[str, Any]
    ):
        super(Q, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), **adam_kwargs)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        h = relu(self.l1(torch.cat((state, action), dim=1)))
        h = relu(self.l2(h))
        return self.l3(h)
