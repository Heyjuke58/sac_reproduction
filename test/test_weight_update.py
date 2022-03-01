from test import Tester, models_equal, sample_tensors
from src.hyperparameters import SAC_HOPPER, SAC_V2_HOPPER
from src.sac_trainer import SACTrainer
from src.sac_trainer_v2 import SACTrainerV2
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightUpdateTester(Tester):
    """
    Tests whether weight update functions actually update the correct parameters for SACTrainer and SACTrainerV2
    """
    def setUp(self) -> None:
        sac_hpars = SAC_HOPPER.copy()
        sac_hpars.update(
            {
                "seed": 12,
                "max_env_steps": 2000,
                "file_name": "sac",
                "dest_model_path": "./test/models",
                "dest_res_path": "./test/results",
            }
        )
        self.sac = SACTrainer(**sac_hpars)
        self.sac.env.reset()

        sac_v2_hpars = SAC_V2_HOPPER.copy()
        sac_v2_hpars.update(
            {
                "seed": 12,
                "max_env_steps": 2000,
                "file_name": "sac_v2",
                "dest_model_path": "./test/models",
                "dest_res_path": "./test/results",
            }
        )
        self.sac_v2 = SACTrainerV2(**sac_v2_hpars)
        self.sac_v2.env.reset()
        return super().setUp()

    def test_value_update(self):
        """
        For these four tests, check if the own network updates and that the other networks don't update.
        """
        sac_before = copy.deepcopy(self.sac)
        state = torch.tensor(self.sac.env.reset(), device=device, dtype=torch.float32).unsqueeze(0)

        self.sac._value_update(state)

        self.assertFalse(models_equal(sac_before.value, self.sac.value))
        self.assertTrue(models_equal(sac_before.policy, self.sac.policy))
        self.assertTrue(models_equal(sac_before.target_value, self.sac.target_value))
        self.assertTrue(models_equal(sac_before.qf1, self.sac.qf1))
        self.assertTrue(models_equal(sac_before.qf2, self.sac.qf2))

    def test_q_update(self):
        sac_before = copy.deepcopy(self.sac)
        state, action, next_state, reward, done = sample_tensors(self.sac.env)

        self.sac._q_update(state, action, next_state, reward, done)

        self.assertTrue(models_equal(sac_before.value, self.sac.value))
        self.assertTrue(models_equal(sac_before.policy, self.sac.policy))
        self.assertTrue(models_equal(sac_before.target_value, self.sac.target_value))
        self.assertFalse(models_equal(sac_before.qf1, self.sac.qf1))
        self.assertFalse(models_equal(sac_before.qf2, self.sac.qf2))

    def test_policy_update(self):
        sac_before = copy.deepcopy(self.sac)
        state, action, next_state, reward, done = sample_tensors(self.sac.env)

        self.sac._policy_update(state)

        self.assertTrue(models_equal(sac_before.value, self.sac.value))
        self.assertFalse(models_equal(sac_before.policy, self.sac.policy))
        self.assertTrue(models_equal(sac_before.target_value, self.sac.target_value))
        self.assertTrue(models_equal(sac_before.qf1, self.sac.qf1))
        self.assertTrue(models_equal(sac_before.qf2, self.sac.qf2))

    def test_target_value_update(self):
        sac_before = copy.deepcopy(self.sac)

        self.sac._target_value_update()

        self.assertTrue(models_equal(sac_before.value, self.sac.value))
        self.assertTrue(models_equal(sac_before.policy, self.sac.policy))
        self.assertFalse(models_equal(sac_before.target_value, self.sac.target_value))
        self.assertTrue(models_equal(sac_before.qf1, self.sac.qf1))
        self.assertTrue(models_equal(sac_before.qf2, self.sac.qf2))

    def test_value_and_policy_update(self):
        sac_before = copy.deepcopy(self.sac)
        state, action, next_state, reward, done = sample_tensors(self.sac.env)

        self.sac._value_and_policy_update(state)

        self.assertFalse(models_equal(sac_before.value, self.sac.value))
        self.assertFalse(models_equal(sac_before.policy, self.sac.policy))
        self.assertTrue(models_equal(sac_before.target_value, self.sac.target_value))
        self.assertTrue(models_equal(sac_before.qf1, self.sac.qf1))
        self.assertTrue(models_equal(sac_before.qf2, self.sac.qf2))

    def test_policy_update_v2(self):
        sac_v2_before = copy.deepcopy(self.sac_v2)
        state, action, next_state, reward, done = sample_tensors(self.sac_v2.env)

        self.sac_v2._policy_update(state)

        self.assertFalse(models_equal(sac_v2_before.policy, self.sac_v2.policy))
        self.assertTrue(models_equal(sac_v2_before.qf1, self.sac_v2.qf1))
        self.assertTrue(models_equal(sac_v2_before.qf2, self.sac_v2.qf2))
        self.assertTrue(models_equal(sac_v2_before.target_qf1, self.sac_v2.target_qf1))
        self.assertTrue(models_equal(sac_v2_before.target_qf2, self.sac_v2.target_qf2))
        self.assertTrue(torch.equal(sac_v2_before.log_alpha, self.sac_v2.log_alpha))

    def test_q_update_v2(self):
        sac_v2_before = copy.deepcopy(self.sac_v2)
        state, action, next_state, reward, done = sample_tensors(self.sac_v2.env)

        self.sac_v2._q_update(state, action, next_state, reward, done)

        self.assertTrue(models_equal(sac_v2_before.policy, self.sac_v2.policy))
        self.assertFalse(models_equal(sac_v2_before.qf1, self.sac_v2.qf1))
        self.assertFalse(models_equal(sac_v2_before.qf2, self.sac_v2.qf2))
        self.assertTrue(models_equal(sac_v2_before.target_qf1, self.sac_v2.target_qf1))
        self.assertTrue(models_equal(sac_v2_before.target_qf2, self.sac_v2.target_qf2))
        self.assertTrue(torch.equal(sac_v2_before.log_alpha, self.sac_v2.log_alpha))

    def test_target_q_update_v2(self):
        sac_v2_before = copy.deepcopy(self.sac_v2)

        self.sac_v2._target_q_update()

        self.assertTrue(models_equal(sac_v2_before.policy, self.sac_v2.policy))
        self.assertTrue(models_equal(sac_v2_before.qf1, self.sac_v2.qf1))
        self.assertTrue(models_equal(sac_v2_before.qf2, self.sac_v2.qf2))

        self.assertFalse(models_equal(sac_v2_before.qf1, self.sac_v2.target_qf1))
        self.assertFalse(models_equal(sac_v2_before.qf2, self.sac_v2.target_qf2))

        self.assertFalse(models_equal(sac_v2_before.target_qf1, self.sac_v2.target_qf1))
        self.assertFalse(models_equal(sac_v2_before.target_qf1, self.sac_v2.target_qf2))
        self.assertFalse(models_equal(sac_v2_before.target_qf2, self.sac_v2.target_qf2))
        self.assertFalse(models_equal(sac_v2_before.target_qf2, self.sac_v2.target_qf1))
        self.assertTrue(torch.equal(sac_v2_before.log_alpha, self.sac_v2.log_alpha))

    def test_alpha_update(self):
        sac_v2_before = copy.deepcopy(self.sac_v2)
        state, action, next_state, reward, done = sample_tensors(self.sac_v2.env)

        self.sac_v2._alpha_update(state)

        self.assertTrue(models_equal(sac_v2_before.policy, self.sac_v2.policy))
        self.assertTrue(models_equal(sac_v2_before.qf1, self.sac_v2.qf1))
        self.assertTrue(models_equal(sac_v2_before.qf2, self.sac_v2.qf2))
        self.assertTrue(models_equal(sac_v2_before.target_qf1, self.sac_v2.target_qf1))
        self.assertTrue(models_equal(sac_v2_before.target_qf2, self.sac_v2.target_qf2))
        self.assertFalse(torch.equal(sac_v2_before.log_alpha, self.sac_v2.log_alpha))
