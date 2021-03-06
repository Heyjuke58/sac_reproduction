from collections import OrderedDict
from time import perf_counter
import csv
import os
from typing import Union
from copy import deepcopy

import gym
import torch
from numpy import ndarray
from src.replay_buffer import ReplayBuffer
from torch import Tensor

from src.utils import set_seeds
from src.networks import Policy, Value, Q

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACTrainer:
    """
    SAC Trainer class which enables training of the SAC model structure proposed in https://arxiv.org/pdf/1801.01290.pdf
    """

    def __init__(
        self,
        seed: int,
        hidden_dim: int,
        max_action: int,
        grad_steps: int,
        batch_size: int,
        replay_buffer_size: int,
        min_replay_buffer_size: int,
        target_smoothing: float,
        target_update_freq: int,
        policy_reg: float,
        env: Union[str, gym.Env],
        n_initial_exploration_steps: int,
        scale_reward: int,
        discount: float,
        max_env_steps: int,
        eval_freq: int,
        eval_episodes: int,
        file_name: str,
        adam_kwargs: dict,
        dest_model_path: str = "./models",
        dest_res_path: str = "./results",
    ) -> None:
        """
        :param seed: The seed for deterministic behaviour
        :param hidden_dim: Hidden dimension size for all the used NN
        :param max_action: Max action scaling
        :param grad_steps: Number of gradient steps in each iteration (default=1)
        :param batch_size: Batch size
        :param replay_buffer_size: Maximum size of the replay buffer
        :param min_replay_buffer_size: Minimum size of the replay buffer before batches will be sampled from it
        :param target_smoothing: τ from the paper, for exponential moving average of Value. Set to 1 for hard update (with higher interval).
        :param target_update_freq: For hard update of Value. Set to 1 for exponentially moving average or Value.
        :param env: The environment to train on
        :param n_initial_exploration_steps: Number of initial exploration steps with a uniform policy
        :param scale_reward: Reward scaling (basically this corresponds to the inverse of the temperature hyperparameter)
        :param discount: Discount
        :param max_env_steps: Max steps for one episode
        :param eval_freq: Frequency of evaluation of the policy (per training iteration)
        :param eval_episodes: Number of episodes for evaluation
        :param file_name: Name of the log and model export file
        :param adam_kwargs: keyword arguments for adam optimizer (same for all nets)
        :param dest_model_path: Path destination of the exported models (different path needed for testing)
        :param dest_res_path: Path destination of the logged results (different path needed for testing)
        """
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
        self.value = Value(state_dim, hidden_dim, adam_kwargs).to(device)
        # self.target_value = Value(state_dim, hidden_dim, adam_kwargs).to(device)
        self.target_value = deepcopy(self.value)
        self.qf1 = Q(state_dim, action_dim, hidden_dim, adam_kwargs).to(device)
        self.qf2 = Q(state_dim, action_dim, hidden_dim, adam_kwargs).to(device)
        self.policy = Policy(
            state_dim, action_dim, hidden_dim, max_action, adam_kwargs, "clamp"
        ).to(device)

        # Other hyperparameters
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)
        self.grad_steps = grad_steps
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.min_replay_buffer_size = min_replay_buffer_size
        self.max_env_steps = max_env_steps
        self.scale_reward = scale_reward
        self.discount = discount
        self.policy_reg = policy_reg
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
                    f"Policy regularization factor: {self.policy_reg}\n"
                    f"Reward scaling: {self.scale_reward}\n"
                    f"Target network update smoothing (τ): {self.target_smoothing}\n"
                    f"Frequency of target updates: {self.target_update_freq}\n\n"
                )
                csv_f.write(hyperpars_str)
                csv_f.write("avg_reward,avg_log_probs,time,env_steps,grad_steps,seed\n")

        # logged things
        self.elapsed_grad_steps = 0

        # TODO init weights?

    def train(self) -> None:
        """
        Train the model for a number of iterations.
        In every iteration, one environment step is taken
        and self.grad_step gradient steps are done.
        """
        state = self.env.reset()

        # initial evaluation
        avg_return = self.eval_policy()
        self.write_eval_to_csv(avg_return, 0, 0)

        self.start_time = perf_counter()
        self.episode_num = 0
        self.episode_timesteps = 0
        self.episode_reward = 0
        for i in range(self.max_env_steps):
            state = self._train_iteration(state, i)

    def _value_and_policy_update(self, states: Tensor) -> None:
        """
        Parameter updates for the policy and the value network.
        They are done together since the same sampled actions and log probs from the policy are used for both updates.
        """
        # all: (b, |action_space|)
        sampled_actions, (log_probs, mus, log_sigmas) = self.policy(states, deterministic=False)
        q1s = self.qf1(states, sampled_actions)
        policy_kl_loss = torch.mean(log_probs - q1s)
        policy_reg_loss = (
            self.policy_reg * 0.5 * (torch.mean(log_sigmas**2) + torch.mean(mus**2))
        )
        policy_loss = policy_kl_loss + policy_reg_loss

        # gradient update:
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        values = self.value(states)  # (b, 1)

        q2s = self.qf2(states, sampled_actions)
        q_mins = torch.minimum(q1s, q2s)  # (b, 1)
        log_probs = log_probs.detach()
        value_loss = torch.mean(0.5 * (values - (q_mins.detach() - log_probs)) ** 2)

        # gradient update:
        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

    def _value_update(self, states: Tensor) -> None:
        """
        Parameter update for the value network
        Only used when order of algorithm is done as stated in the paper (commented lines TODO: line numbers)
        """
        values = self.value(states)  # (b, 1)
        with torch.no_grad():
            sampled_actions, (log_probs, _, _) = self.policy(states, deterministic=False)
            q1s = self.qf1(states, sampled_actions)
            q2s = self.qf2(states, sampled_actions)
            q_mins = torch.minimum(q1s, q2s)  # (b, 1)
        value_loss = torch.mean(0.5 * (values - (q_mins - log_probs)) ** 2)

        # gradient update:
        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

    def _q_update(self, states, actions, next_states, rewards, dones) -> None:
        """
        Parameter updates for both Q networks
        """
        with torch.no_grad():
            value_targets_next = self.target_value(next_states)  # (b, 1)
        q_hat = self.scale_reward * rewards + (1 - dones) * self.discount * value_targets_next
        for qf in [self.qf1, self.qf2]:
            qf_loss = torch.mean(0.5 * (qf(states, actions) - q_hat) ** 2)

            # gradient update:
            qf.optimizer.zero_grad()
            qf_loss.backward()
            qf.optimizer.step()

    def _policy_update(self, states: Tensor) -> None:
        """
        Parameter update for the policy network
        Only used when order of algorithm is done as stated in the paper (commented lines TODO: line numbers)
        """
        # all: (b, |action_space|)
        sampled_actions, (log_probs, mus, log_sigmas) = self.policy(states, deterministic=False)
        q1s = self.qf1(states, sampled_actions)
        policy_kl_loss = torch.mean(log_probs - q1s)
        policy_reg_loss = (
            self.policy_reg * 0.5 * (torch.mean(log_sigmas**2) + torch.mean(mus**2))
        )
        policy_loss = policy_kl_loss + policy_reg_loss

        # gradient update:
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

    def _target_value_update(self) -> None:
        """
        Parameter update of the target value network with exponentially moving average of Value
        """
        if self.elapsed_grad_steps % self.target_update_freq == 0:
            new_target_value = OrderedDict()
            for (v_param_name, v_param), (_, v_t_param) in zip(
                self.value.state_dict().items(), self.target_value.state_dict().items()
            ):
                new_target_value[v_param_name] = (
                    self.target_smoothing * v_param + (1 - self.target_smoothing) * v_t_param
                )

            self.target_value.load_state_dict(new_target_value)

    def _train_iteration(self, state, iteration: int) -> ndarray:
        """
        Do one iteration of training. Returns the state that we end up in after
        the environment step phase.
        """

        ### ENV STEP ###
        # Decide whether action is sampled from initial exploration or actual policy
        if iteration <= self.n_initial_exploration_steps and self.n_initial_exploration_steps != 0:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():  # TODO maybe needs to be detach()?
                action, (log_probs, _, _) = self.policy(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                    deterministic=False,
                )
                action = action.detach().cpu().numpy()
                self.log_probs_num += 1
                self.avg_log_probs += log_probs.item()

        next_state, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_timesteps += 1
        self.replay_buffer.add(state, action, next_state, reward, done)
        if done:
            state = self.env.reset()
            print(
                f"Total T: {iteration+1} Episode Num: {self.episode_num+1} Episode T: {self.episode_timesteps} Reward: {self.episode_reward:.3f}"
            )
            self.episode_num += 1
            self.episode_timesteps = 0
            self.episode_reward = 0
        else:
            state = next_state

        ### GRAD STEP(S) ###
        # multiple grad steps per iteration. Only after replay buffer has enough samples:
        for _ in range(self.grad_steps):
            if self.replay_buffer.size > self.min_replay_buffer_size:
                states, actions, next_states, rewards, dones = self.replay_buffer.sample(
                    self.batch_size
                )

                self._do_updates(states, actions, next_states, rewards, dones)
                self.elapsed_grad_steps += 1

        if (iteration + 1) % self.eval_freq == 0:
            elapsed_time = perf_counter() - self.start_time
            start_time_eval = perf_counter()
            avg_return = self.eval_policy()
            self.write_eval_to_csv(avg_return, elapsed_time, iteration + 1)

            # ignore time for evaluation by adding it to start time
            self.start_time += perf_counter() - start_time_eval

        return state

    def _do_updates(self, states, actions, next_states, rewards, dones) -> None:
        """
        Update all the parameters of the networks
        We find, that the order from the paper differs to the actual order in the code.
        We will replicate the order like it is in the code.
        """
        # order as in code
        self._value_and_policy_update(states)
        self._q_update(states, actions, next_states, rewards, dones)
        self._target_value_update()

        # order as in paper
        # self._value_update(states)
        # self._q_update(states, actions, next_states, rewards, dones)
        # self._policy_update(states)
        # self._target_value_update()

    def eval_policy(self) -> float:
        """
        Evaluate the policy for #self.eval_episodes episodes.
        Returns the average return over the episodes
        """
        avg_return = 0.0
        for i in range(self.eval_episodes):
            # several seeds for evaluation
            self.eval_env.seed(self.seed + 100 + i)
            state, done = self.eval_env.reset(), False
            while not done:
                action = self.policy.get_action(state)
                state, reward, done, _ = self.eval_env.step(action)
                avg_return += reward

        avg_return /= self.eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {self.eval_episodes} episodes: {avg_return:.3f}")
        print("---------------------------------------")
        self.save(self.dest_model_path)
        return avg_return

    def write_eval_to_csv(self, avg_return, time, env_steps):
        """
        Logs evaluation and training results to csv.
        """
        with open(self.res_file, "a") as csv_f:
            writer = csv.writer(csv_f, delimiter=",")
            writer.writerow(
                [
                    avg_return,
                    "nan" if self.log_probs_num == 0 else self.avg_log_probs / self.log_probs_num,
                    time,
                    env_steps,
                    self.elapsed_grad_steps,
                    self.seed,
                ]
            )
        # reset log probs metrics
        self.avg_log_probs = 0.0
        self.log_probs_num = 0

    def save(self, filename: str):
        torch.save(self.policy, filename + "_policy")

    def load(self, filename: str):
        self.policy.load_state_dict(torch.load(filename))
