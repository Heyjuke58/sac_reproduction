from typing import Optional
from collections import OrderedDict
from src.sac_trainer import SACTrainer
from src.networks import Q, PolicyV2
import torch
import gym
from typing import Union
from copy import deepcopy
from src.replay_buffer import ReplayBuffer
from src.utils import set_seeds
import os
from torch import Tensor
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACTrainerV2(SACTrainer):
    """
    Version 2 of SACTrainer which incorporates changes mentioned in second SAC paper (https://arxiv.org/abs/1812.05905)
    Changes:
        - Removal of value function approximators
        - Automatic tuning of temperature hyperparameter alpha
    """

    def __init__(
        self,
        seed: int,
        hidden_dim: int,
        grad_steps: int,
        batch_size: int,
        replay_buffer_size: int,
        min_replay_buffer_size: int,  # check usage
        target_smoothing: float,
        target_update_freq: int,
        env: Union[str, gym.Env],
        n_initial_exploration_steps: int,  # check usage
        discount: float,
        max_env_steps: int,
        eval_freq: int,
        eval_episodes: int,
        file_name: str,
        adam_kwargs: dict,
        dest_model_path: str = "./models",
        dest_res_path: str = "./results",
        max_action: float = 1.0,
        fixed_alpha: Optional[float] = None,
    ) -> None:
        # Make env
        if isinstance(env, str):
            self.env = gym.make(env)
            self.eval_env = gym.make(env)
            self.env_str = env
        else:
            self.env = env
            self.eval_env = deepcopy(env)
            self.env_str = env.__class__.__name__

        # set seeds
        set_seeds(seed, self.env)
        self.seed = seed

        # neural net functions:
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.qf1 = Q(state_dim, action_dim, hidden_dim, adam_kwargs).to(device)
        self.qf2 = Q(state_dim, action_dim, hidden_dim, adam_kwargs).to(device)
        self.policy = PolicyV2(state_dim, action_dim, hidden_dim, max_action, adam_kwargs).to(
            device
        )
        self.target_qf1 = deepcopy(self.qf1)
        self.target_qf2 = deepcopy(self.qf2)

        # alpha temperature
        if fixed_alpha is None:  # learn alpha
            self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
            self.log_alpha_optim = torch.optim.Adam([self.log_alpha], **adam_kwargs)
            self.target_entropy = -action_dim
            self.fixed_alpha = False
        else:  # alpha is a fixed hyperparameter
            self.log_alpha = torch.log(
                torch.tensor(fixed_alpha, device=device, requires_grad=False)
            )
            self.log_alpha_optim = None
            self.target_entropy = None
            self.fixed_alpha = True

        # Other hyperparameters
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.min_replay_buffer_size = min_replay_buffer_size
        self.max_env_steps = max_env_steps
        self.discount = discount
        self.grad_steps = grad_steps
        self.target_smoothing = target_smoothing
        self.target_update_freq = target_update_freq

        # Evaluation
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.file_name = file_name
        self.dest_model_path = f"{dest_model_path}/{self.file_name}"
        self.res_file = f"{dest_res_path}/{self.file_name}.csv"
        self.avg_log_probs = 0.0
        self.log_probs_num = 0

        # Create log file if it does not exist already
        if not os.path.exists(self.res_file):
            with open(self.res_file, "x") as csv_f:
                hyperpars_str = (
                    "Hyperparameters\n"
                    f"Env: {self.env_str}\n"
                    f"Seed: {self.seed}\n"
                    f"Eval frequency: {self.eval_freq}\n"
                    f"Number of initial exploration steps: {self.n_initial_exploration_steps}\n"
                    f"Replay buffer size: {replay_buffer_size}\n"
                    f"Minimum replay buffer size: {self.min_replay_buffer_size}\n"
                    f"Max env steps: {self.max_env_steps}\n"
                    f"Batch size: {self.batch_size}\n"
                    f"Discount factor: {self.discount}\n"
                    f"Target entropy: {self.target_entropy}\n"
                    f"Target network update smoothing (τ): {self.target_smoothing}\n"
                    f"Fixed alpha: {fixed_alpha}\n"
                    f"Frequency of target updates: {self.target_update_freq}\n\n"
                )
                csv_f.write(hyperpars_str)
                csv_f.write("avg_reward,log_probs,alpha,time,env_steps,grad_steps,seed\n")

        # logged things
        self.elapsed_grad_steps = 0

    def _q_update(self, states, actions, next_states, rewards, dones):
        with torch.no_grad():
            sampled_actions, (log_probs, mus, log_sigmas) = self.policy(
                next_states, deterministic=False
            )
            next_q1 = self.target_qf1(next_states, sampled_actions)
            next_q2 = self.target_qf2(next_states, sampled_actions)
            next_q = torch.minimum(next_q1, next_q2)

            next_values = next_q - torch.exp(self.log_alpha) * log_probs

            q_targets = rewards + self.discount * (1.0 - dones) * next_values

        q1_values = self.qf1(states, actions)
        q1_loss = torch.mean(0.5 * (q_targets - q1_values) ** 2)
        q2_values = self.qf2(states, actions)
        q2_loss = torch.mean(0.5 * (q_targets - q2_values) ** 2)

        for q, loss in zip([self.qf1, self.qf2], [q1_loss, q2_loss]):
            q.optimizer.zero_grad()
            loss.backward()
            q.optimizer.step()

    def _policy_update(self, states: Tensor):
        # all: (b, |action_space|)
        sampled_actions, (log_probs, mus, log_sigmas) = self.policy(states, deterministic=False)

        # with torch.no_grad():
        q1s = self.qf1(states, sampled_actions)
        q2s = self.qf2(states, sampled_actions)
        q_mean = torch.mean(torch.stack([q1s, q2s]), dim=0)
        policy_loss = torch.mean(torch.exp(self.log_alpha).detach() * log_probs - q_mean)

        # gradient update:
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

    def _alpha_update(self, states: Tensor):
        with torch.no_grad():
            sampled_actions, (log_probs, mus, log_sigmas) = self.policy(states, deterministic=False)
        alpha_loss = torch.mean(
            -1.0 * (torch.exp(self.log_alpha) * (log_probs + self.target_entropy))
        )
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

    def _target_q_update(self):
        # update exponentially moving average of both Q functions (or hard update):
        if self.elapsed_grad_steps % self.target_update_freq == 0:
            for q, t_q in zip([self.qf1, self.qf2], [self.target_qf1, self.target_qf2]):

                new_target_q = OrderedDict()
                for (q_param_name, q_param), (_, q_t_param) in zip(
                    q.state_dict().items(), t_q.state_dict().items()
                ):
                    new_target_q[q_param_name] = (
                        self.target_smoothing * q_param + (1 - self.target_smoothing) * q_t_param
                    )

                t_q.load_state_dict(new_target_q)

    def _do_updates(self, states, actions, next_states, rewards, dones) -> None:
        self._q_update(states, actions, next_states, rewards, dones)
        self._policy_update(states)
        if not self.fixed_alpha:
            self._alpha_update(states)
        self._target_q_update()

    def _log_updates(self):
        pass

    def write_eval_to_csv(self, avg_reward, time, env_steps):
        with open(self.res_file, "a") as csv_f:
            writer = csv.writer(csv_f, delimiter=",")
            writer.writerow(
                [
                    avg_reward,
                    "nan" if self.log_probs_num == 0 else self.avg_log_probs / self.log_probs_num,
                    torch.exp(self.log_alpha).item(),
                    time,
                    env_steps,
                    self.elapsed_grad_steps,
                    self.seed,
                ]
            )
        # reset log probs metrics
        self.avg_log_probs = 0.0
        self.log_probs_num = 0
