import unittest
from argparse import Namespace
from TD3.TD3 import TD3
import gym
from copy import deepcopy
import torch
from TD3.main import main as td3_main


class ReproducibilityTester(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_td3_same_seed(self):
        # test whether multiple runs with the same seed leads to exact same model weights
        SEED = 12
        ENV = "Hopper-v3"
        env = gym.make(ENV)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        td3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)

        td3_params = Namespace(
            policy="TD3",
            env=ENV,
            seed=SEED,
            start_timesteps=1000,
            eval_freq=5e3,
            max_timesteps=2000,
            expl_noise=0.1,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            save_model=True,
            load_model="",
            dest_model_path="./test/models",
            dest_res_path="./test/results",
        )
        # run experiment for the first time
        td3_main(td3_params)
        td3.load(f"./test/models/TD3_{ENV}_{SEED}")

        # copy models to later compare them
        actor = deepcopy(td3.actor)
        critic = deepcopy(td3.critic)

        # run experiment for the second time
        td3_main(td3_params)
        td3.load(f"./test/models/TD3_{ENV}_{SEED}")

        for x, y in zip(actor.parameters(), td3.actor.parameters()):
            self.assertTrue(torch.equal(x, y))
        for x, y in zip(critic.parameters(), td3.critic.parameters()):
            self.assertTrue(torch.equal(x, y))
