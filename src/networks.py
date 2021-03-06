from torch import Tensor, relu, tanh
import torch.nn as nn
from typing import Any, Optional
from torch.nn.functional import softplus
import torch
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from numpy import ndarray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Policy(nn.Module):
    """
    Gaussian policy
    For each action in the action space, output a scalar in [-1, 1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float,
        adam_kwargs: dict[str, Any],
        log_sigmas_post: str,
    ):
        """
        :param state_dim: Dimension of the observation space
        :param action_dim: Dimension of the action space
        :param hidden_dim: Hidden dimension size in 2-layer NN
        :param max_action: Max action scaling (after tanh)
        :param adam_kwargs: keyword arguments for adam optimizer
        :param log_sigmas_post: post-processing of log_sigmas (clipping/clamping or softplus)
        """
        super(Policy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        # hidden_dim -> mu0, mu1, ..., sig0, sig1, ...
        self.l3 = nn.Linear(hidden_dim, action_dim * 2)

        self.action_dim = action_dim
        self.max_action = max_action

        self.optimizer = torch.optim.Adam(self.parameters(), **adam_kwargs)

        assert log_sigmas_post in ["softplus", "clamp"]
        self.log_sigmas_post = log_sigmas_post

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

        # apply softplus and add epsilon
        # https://github.com/rail-berkeley/softlearning/blob/master/softlearning/policies/gaussian_policy.py line 276
        if self.log_sigmas_post == "softplus":
            log_sigmas = softplus(log_sigmas) + 1e-5
        elif self.log_sigmas_post == "clamp":
            log_sigmas = torch.clamp(log_sigmas, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            raise Exception(f"bad option for log sigma post processing: {self.log_sigmas_post=}")

        normal = TransformedDistribution(
            Independent(Normal(mus, torch.exp(log_sigmas)), 1), [TanhTransform(cache_size=1)]
        )
        if deterministic:
            return self.max_action * tanh(normal.base_dist.mean), None
        else:
            # reparametrization trick, sampling closely following the original implementation
            actions = normal.rsample()  # (b, action space dim)
            log_probs = normal.log_prob(actions).unsqueeze(-1)  # (b, 1)

            if self.max_action == 0.0:  # only for debugging
                log_probs *= 0.0
            return self.max_action * actions, (log_probs, mus, log_sigmas)

    def get_action(self, state: ndarray):
        """
        Get action for evaluation of policy
        """
        with torch.no_grad():
            action, _ = self.forward(
                torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                deterministic=True,
            )
        return action.cpu().numpy()


class Value(nn.Module):
    """
    Value funtion network gets the state, outputs a single scalar (soft value)
    """

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
    """
    Q function network gets the state and action, outputs a single scalar (state-action value)
    """

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
