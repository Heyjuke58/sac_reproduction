from test import Tester
from src.hyperparameters import sac_base, sac_base_v2
from src.sac_trainer import SACTrainer
from src.sac_trainer_v2 import SACTrainerV2
import torch
from test.probing_envs import Probe1, Probe2, Probe3, Probe4, Probe5
from parameterized import parameterized_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@parameterized_class([{"sac_version": "v1"}, {"sac_version": "v2"}])
class ProbingEnvs(Tester):
    def setUp(self) -> None:
        if self.sac_version == "v1":
            self.sac_trainer = SACTrainer
            self.sac_hpars = sac_base.copy()
            self.sac_hpars.update(
                {
                    "seed": 12,
                    "batch_size": 10,
                    "replay_buffer_size": 50,
                    "min_replay_buffer_size": 10,
                    "eval_freq": 100,
                    "n_initial_exploration_steps": 0,
                    "scale_reward": 1,
                    "file_name": "sac",
                    "dest_model_path": "./test/models",
                    "dest_res_path": "./test/results",
                }
            )
        else:
            self.sac_trainer = SACTrainerV2
            self.sac_hpars = sac_base_v2.copy()
            self.sac_hpars.update(
                {
                    "seed": 12,
                    "batch_size": 10,
                    "replay_buffer_size": 50,
                    "min_replay_buffer_size": 10,
                    "eval_freq": 100,
                    "n_initial_exploration_steps": 0,
                    "file_name": "sac_v2",
                    "dest_model_path": "./test/models",
                    "dest_res_path": "./test/results",
                }
            )

        self.s_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        self.s_1 = torch.tensor([1.0], device=device).unsqueeze(0)
        self.a_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        self.a_1 = torch.tensor([1.0], device=device).unsqueeze(0)
        self.a_n1 = torch.tensor([-1.0], device=device).unsqueeze(0)

        return super().setUp()

    def test_probe_1(self):
        """
        Test probing env 1. This isolates the value loss calculation and the optimizer of the value network.
        One action, one state, always get +1 reward and done instantly.
        """
        env = Probe1()
        sac_hpars_1 = self.sac_hpars.copy()
        sac_hpars_1.update({"env": env, "max_env_steps": 200, "max_action": 0})
        sac = self.sac_trainer(**sac_hpars_1)
        sac.train()

        self.assertAlmostEqual(sac.qf1(self.s_0, self.a_0).item(), 1.0, delta=0.1)
        self.assertAlmostEqual(sac.qf2(self.s_0, self.a_0).item(), 1.0, delta=0.1)

        if self.sac_version == "v1":
            self.assertAlmostEqual(sac.value(self.s_0).item(), 1.0, delta=0.1)

    def test_probe_2(self):
        """
        Test probing env 2. Tests whether the backpropagation of the value function is working correctly.
        """
        env = Probe2()
        sac_hpars_2 = self.sac_hpars.copy()
        sac_hpars_2.update({"env": env, "max_env_steps": 500, "max_action": 0})
        sac = self.sac_trainer(**sac_hpars_2)
        sac.train()

        self.assertAlmostEqual(sac.qf1(self.s_0, self.a_0).item(), 0.0, delta=0.1)
        self.assertAlmostEqual(sac.qf1(self.s_1, self.a_0).item(), 1.0, delta=0.1)
        self.assertAlmostEqual(sac.qf2(self.s_0, self.a_0).item(), 0.0, delta=0.1)
        self.assertAlmostEqual(sac.qf2(self.s_1, self.a_0).item(), 1.0, delta=0.1)

        if self.sac_version == "v1":
            self.assertAlmostEqual(sac.value(self.s_0).item(), 0.0, delta=0.1)
            self.assertAlmostEqual(sac.value(self.s_1).item(), 1.0, delta=0.1)

    def test_probe_3(self):
        """
        Test probing env 3. Test whether reward discounting works. One action, two states.
        """
        for seed in range(5):
            env = Probe3()
            sac_hpars_3 = self.sac_hpars.copy()
            sac_hpars_3.update(
                {"env": env, "discount": 0.5, "max_env_steps": 2000, "max_action": 0, "seed": seed}
            )
            sac = self.sac_trainer(**sac_hpars_3)
            sac.train()

            for qf in [sac.qf1, sac.qf2]:
                self.assertAlmostEqual(qf(self.s_0, self.a_0).item(), 0.5, delta=0.1)
                self.assertAlmostEqual(qf(self.s_1, self.a_0).item(), 1.0, delta=0.1)

            if self.sac_version == "v1":
                self.assertAlmostEqual(sac.value(self.s_0).item(), 0.5, delta=0.1)
                self.assertAlmostEqual(sac.value(self.s_1).item(), 1.0, delta=0.1)

    def test_probe_4(self):
        """
        Test probing env 4. Test whether policy can learn to pick a better action.
        """
        env = Probe4()
        sac_hpars_4 = self.sac_hpars.copy()
        sac_hpars_4.update({"env": env, "max_env_steps": 2000, "max_action": 1})
        sac = self.sac_trainer(**sac_hpars_4)
        sac.train()

        action, _ = sac.policy(self.s_0, deterministic=True)
        action = 0.0 if action.item() <= 0 else 1.0
        self.assertEqual(action, 1.0)

        for qf in [sac.qf1, sac.qf2]:
            self.assertAlmostEqual(qf(self.s_0, self.a_1).item(), 1.0, delta=0.1)
            self.assertGreater(qf(self.s_0, self.a_1).item(), qf(self.s_0, self.a_0))

        if self.sac_version == "v1":
            self.assertAlmostEqual(sac.value(self.s_0).item(), 1.0, delta=0.1)

    def test_probe_5(self):
        """
        State is either 0 or 1, need to take action -1 in state 0 and 1 in state 1.
        """
        env = Probe5()
        sac_hpars_5 = self.sac_hpars.copy()
        sac_hpars_5.update({"env": env, "max_env_steps": 2000, "max_action": 1})
        sac = self.sac_trainer(**sac_hpars_5)
        sac.train()

        p_0, _ = sac.policy(self.s_0, deterministic=True)
        p_1, _ = sac.policy(self.s_1, deterministic=True)
        p_0 = -1.0 if p_0.item() <= 0 else 1.0
        p_1 = -1.0 if p_1.item() <= 0 else 1.0

        # check Policy
        self.assertEqual(p_0, -1.0)
        self.assertEqual(p_1, 1.0)

        # check in-distribution Q
        for qf in [sac.qf1, sac.qf2]:
            self.assertAlmostEqual(qf(self.s_0, self.a_n1).item(), 1.0, delta=0.1)
            self.assertAlmostEqual(qf(self.s_1, self.a_1).item(), 1.0, delta=0.1)
            self.assertGreater(qf(self.s_0, self.a_n1).item(), qf(self.s_0, self.a_1))
            self.assertGreater(qf(self.s_1, self.a_1).item(), qf(self.s_1, self.a_n1))

        if self.sac_version == "v1":
            # check V
            self.assertAlmostEqual(sac.value(self.s_0).item(), 1.0, delta=0.1)
            self.assertAlmostEqual(sac.value(self.s_1).item(), 1.0, delta=0.1)
