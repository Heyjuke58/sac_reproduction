import unittest
from test import Tester
from src.hyperparameters import SAC_HOPPER
from src.sac import SAC
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightUpdateTester(Tester):
    def setUp(self) -> None:
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

        self.sac = SAC(**sac_hpars)
        return super().setUp()

    def test_value_update(self):
        value_before = copy.deepcopy(self.sac.value)
        state = torch.tensor(self.sac.env.reset(), device=device, dtype=torch.float32).unsqueeze(0)
        self.sac._value_update(state)

        for x, y in zip(value_before.parameters(), self.sac.value.parameters()):
            self.assertFalse(torch.equal(x, y))

    def test_q_update(self):
        q_before = copy.deepcopy(self.sac.qf1)
        state = torch.tensor(self.sac.env.reset(), device=device, dtype=torch.float32).unsqueeze(0)
        # self.sac._q_update(state)

        for x, y in zip(q_before.parameters(), self.sac.qf1.parameters()):
            self.assertFalse(torch.equal(x, y))

    def test_policy_update(self):
        policy_before = copy.deepcopy(self.sac.policy)
        state = torch.tensor(self.sac.env.reset(), device=device, dtype=torch.float32).unsqueeze(0)
        self.sac._policy_update(state)

        for x, y in zip(policy_before.parameters(), self.sac.policy.parameters()):
            self.assertFalse(torch.equal(x, y))

    def test_target_value_update(self):
        target_value_before = copy.deepcopy(self.sac.target_value)
        self.sac._target_value_update()

        for x, y in zip(target_value_before.parameters(), self.sac.target_value.parameters()):
            self.assertFalse(torch.equal(x, y))
