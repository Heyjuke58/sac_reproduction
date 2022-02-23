from time import perf_counter

import gym
import torch
from numpy import ndarray
from TD3.utils import ReplayBuffer

from src.utils import set_seeds
from src.networks import Policy, Value, Q

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
        target_update_interval: int,
        env: str,
        n_initial_exploration_steps: int,
        scale_reward: int,
        discount: float,
        max_env_steps: int,
        adam_kwargs: dict,
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
        :param target_update_interval: For hard update of Value. Set to 1 for exponentially moving average or Value.
        :param env: 
        :param n_initial_exploration_steps: 
        :param scale_reward: 
        :param max_env_steps: 
        :param adam_kwargs:
        """
        # Make env
        self.env = gym.make(env)

        # neural net functions:
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.value = Value(state_dim, hidden_dim, adam_kwargs)
        self.qf1 = Q(state_dim, action_dim, hidden_dim, adam_kwargs)
        self.qf2 = Q(state_dim, action_dim, hidden_dim, adam_kwargs)
        self.policy = Policy(state_dim, action_dim, hidden_dim, max_action, adam_kwargs)

        # Other hyperparameters
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)
        self.grad_steps = grad_steps
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.min_replay_buffer_size = min_replay_buffer_size
        self.max_env_steps = max_env_steps
        self.scale_reward = scale_reward
        self.discount = discount

        # Set seeds
        set_seeds(seed, self.env)

        # TODO init weights?

    def train(self):
        """
        Train the model for a number of iterations.
        In every iteration, one environment step is taken
        and self.grad_step gradient steps are done.
        """
        state = self.env.reset()

        self.start_time = perf_counter()
        for i in range(self.max_env_steps):
            state = self._train_iteration(state, i)

    def _train_iteration(self, state, iteration: int) -> ndarray:
        """
        Do one iteration of training. Returns the state that we end up in after
        the environment step phase.
        """

        # Decide whether action is sampled from initial exploration or actual policy
        if iteration <= self.n_initial_exploration_steps:
            action = self.env.action_space.sample()
        else:
            action = self.policy(state)

        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.add(state, action, next_state, reward, done)
        if done:
            state = self.env.reset()
        else:
            state = next_state

        # multiple grad steps per iteration. Only after replay buffer has enough samples:
        for _ in range(self.grad_steps):
            if self.replay_buffer.size >= self.min_replay_buffer_size:
                states, actions, next_states, rewards, dones = self.replay_buffer.sample(
                    self.batch_size
                )

                # update Value:
                sampled_actions, (log_probs, _, _) = self.policy(states)
                values = self.value(states)
                q1s = self.qf1(states, sampled_actions)
                q2s = self.qf2(states, sampled_actions)
                q_mins = torch.minimum(q1s, q2s)
                value_loss = torch.mean(0.5 * (values - (q_mins - log_probs)) ** 2)
                # TODO gradient descent

                # update both Q's:
                values_next = self.value(next_states)
                q_hat = self.scale_reward * rewards + (1 - dones) * self.discount * values_next
                for qf in [self.qf1, self.qf2]:
                    qf_loss = torch.mean(0.5 * (qf(states, actions) - q_hat) ** 2)
                    # TODO grad desc

                # update Policy:

                # update exponentially moving average of Value / hard update:

        # logging (after evaluation)
        elapsed_time = perf_counter() - self.start_time

        return state

