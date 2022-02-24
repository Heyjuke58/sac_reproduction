import unittest
from argparse import Namespace
from TD3.TD3 import TD3
import gym
from copy import deepcopy
import torch
from TD3.main import main as td3_main
from src.sac import SAC
from src.hyperparameters import SAC_HOPPER
from test import Tester


class ReproducibilityTester(Tester):
    def setUp(self) -> None:
        self.seed = 12
        self.env_str = "Hopper-v3"
        self.env = gym.make(self.env_str)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        return super().setUp()

    def test_td3_same_seed(self):
        # test whether multiple runs with the same seed leads to exact same model weights for their TD3

        td3 = TD3(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action)
        file_name = f"TD3_{self.env_str}_{self.seed}"
        td3_params = Namespace(
            policy="TD3",
            env=self.env_str,
            seed=self.seed,
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
            file_name=file_name,
            dest_model_path="./test/models",
            dest_res_path="./test/results",
        )
        # run experiment for the first time
        td3_main(vars(td3_params))
        td3.load(f"./test/models/{file_name}")

        # copy models to later compare them
        actor = deepcopy(td3.actor)
        critic = deepcopy(td3.critic)

        # run experiment for the second time
        td3_main(vars(td3_params))
        td3.load(f"./test/models/{file_name}")

        for x, y in zip(actor.parameters(), td3.actor.parameters()):
            self.assertTrue(torch.equal(x, y))
        for x, y in zip(critic.parameters(), td3.critic.parameters()):
            self.assertTrue(torch.equal(x, y))

    def test_sac_same_seed(self):
        # test whether multiple runs with the same seed leads to exact same model weights for our SAC
        sac_hpars = SAC_HOPPER.copy()
        sac_hpars.update(
            {
                "seed": 12,
                "max_env_steps": 2000,
                "file_name": "test",
                "dest_model_path": "./test/models",
                "dest_res_path": "./test/results",
            }
        )

        sac1 = SAC(**sac_hpars)
        sac1.train()

        sac2 = SAC(**sac_hpars)
        sac2.train()

        for x, y in zip(sac1.policy.parameters(), sac2.policy.parameters()):
            self.assertTrue(torch.equal(x, y))
        for x, y in zip(sac1.qf1.parameters(), sac2.qf1.parameters()):
            self.assertTrue(torch.equal(x, y))
        for x, y in zip(sac1.qf2.parameters(), sac2.qf2.parameters()):
            self.assertTrue(torch.equal(x, y))
        for x, y in zip(sac1.value.parameters(), sac2.value.parameters()):
            self.assertTrue(torch.equal(x, y))
        for x, y in zip(sac1.target_value.parameters(), sac2.target_value.parameters()):
            self.assertTrue(torch.equal(x, y))
