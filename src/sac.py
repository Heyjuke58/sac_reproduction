from time import perf_counter
import csv
import os

import gym
import torch
from numpy import ndarray
from src.replay_buffer import ReplayBuffer
from torch import Tensor

from src.utils import set_seeds
from src.networks import Policy, Value, Q
from src.utils import get_timestamp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(
        self,
        seed: int,
        hidden_dim: int,
        max_action: int,
        grad_steps: int,
        batch_size: int,
        replay_buffer_size: int,
        min_replay_buffer_size: int,
        target_smoothing: int,
        target_update_freq: int,
        policy_reg: float,
        env: str,
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
        :param seed:
        :param hidden_dim:
        :param max_action:
        :param grad_steps:
        :param batch_size:
        :param replay_buffer_size:
        :param min_replay_buffer_size:
        :param target_smoothing: τ from the paper, for exponential moving average of Value. Set to 1 for hard update (with higher interval).
        :param target_update_freq: For hard update of Value. Set to 1 for exponentially moving average or Value.
        :param env:
        :param n_initial_exploration_steps:
        :param scale_reward:
        :param max_env_steps:
        :param adam_kwargs:
        """
        # Make env
        self.env = gym.make(env)
        self.env_str = env

        # set seeds
        set_seeds(seed, self.env)
        self.seed = seed

        # neural net functions:
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.value = Value(state_dim, hidden_dim, adam_kwargs).to(device)
        self.target_value = Value(state_dim, hidden_dim, adam_kwargs).to(device)
        self.qf1 = Q(state_dim, action_dim, hidden_dim, adam_kwargs).to(device)
        self.qf2 = Q(state_dim, action_dim, hidden_dim, adam_kwargs).to(device)
        self.policy = Policy(state_dim, action_dim, hidden_dim, max_action, adam_kwargs).to(device)

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
                csv_f.write("avg_reward,time,env_steps,grad_steps,seed\n")

        # logged things
        self.elapsed_grad_steps = 0

        # TODO init weights?

    def train(self):
        """
        Train the model for a number of iterations.
        In every iteration, one environment step is taken
        and self.grad_step gradient steps are done.
        """
        state = self.env.reset()

        # initial evaluation
        avg_reward = self.eval_policy()
        self.write_eval_to_csv(avg_reward, 0, 0)

        self.start_time = perf_counter()
        self.episode_num = 0
        self.episode_timesteps = 0
        self.episode_reward = 0
        for i in range(self.max_env_steps):
            state = self._train_iteration(state, i)

    def _value_update(self, states: Tensor):
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

    def _q_update(self, states, actions, next_states, rewards, dones):
        with torch.no_grad():
            value_targets_next = self.target_value(next_states)  # (b, 1)
        q_hat = self.scale_reward * rewards + (1 - dones) * self.discount * value_targets_next
        for qf in [self.qf1, self.qf2]:
            qf_loss = torch.mean(0.5 * (qf(states, actions) - q_hat) ** 2)

            # gradient update:
            qf.optimizer.zero_grad()
            qf_loss.backward()
            qf.optimizer.step()

    def _policy_update(self, states):
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

    def _target_value_update(self):
        # update exponentially moving average of Value / hard update:
        if self.elapsed_grad_steps % self.target_update_freq == 0:
            for v_param, v_t_param in zip(self.value.parameters(), self.target_value.parameters()):
                v_t_param = (
                    self.target_smoothing * v_param + (1 - self.target_smoothing) * v_t_param
                )

    def _train_iteration(self, state, iteration: int) -> ndarray:
        """
        Do one iteration of training. Returns the state that we end up in after
        the environment step phase.
        """

        # Decide whether action is sampled from initial exploration or actual policy
        if iteration <= self.n_initial_exploration_steps:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():  # TODO maybe needs to be detach()?
                action, _ = self.policy(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                    deterministic=False,
                )
                action = action.detach().cpu().numpy()

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

        # multiple grad steps per iteration. Only after replay buffer has enough samples:
        for _ in range(self.grad_steps):
            if self.replay_buffer.size > self.min_replay_buffer_size:
                states, actions, next_states, rewards, dones = self.replay_buffer.sample(
                    self.batch_size
                )

                self._value_update(states)
                self._q_update(states, actions, next_states, rewards, dones)
                self._policy_update(states)
                self._target_value_update()

                self.elapsed_grad_steps += 1

        if (iteration + 1) % self.eval_freq == 0:
            elapsed_time = perf_counter() - self.start_time
            start_time_eval = perf_counter()
            avg_reward = self.eval_policy()
            self.write_eval_to_csv(avg_reward, elapsed_time, iteration + 1)

            # TODO: save model

            # ignore time for evaluation by adding it to start time
            self.start_time += perf_counter() - start_time_eval

        return state

    def eval_policy(self):
        eval_env = gym.make(self.env_str)
        eval_env.seed(self.seed + 100)

        avg_reward = 0.0
        for _ in range(self.eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.policy.get_action(state)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= self.eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {self.eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        self.save(self.dest_model_path)
        return avg_reward

    def write_eval_to_csv(self, avg_reward, time, env_steps):
        with open(self.res_file, "a") as csv_f:
            writer = csv.writer(csv_f, delimiter=",")
            writer.writerow([avg_reward, time, env_steps, self.elapsed_grad_steps, self.seed])

    def save(self, filename: str):
        torch.save(self.policy.state_dict(), filename + "_policy")
