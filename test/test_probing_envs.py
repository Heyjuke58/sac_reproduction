from test import Tester, sample_tensors
from src.hyperparameters import sac_base, sac_base_v2
from src.sac_trainer import SACTrainer
import torch
from test.probing_envs import Probe1, Probe2, Probe3, Probe4, Probe5
from parameterized import parameterized_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@parameterized_class([{"sac_version": "v1"}, {"sac_version": "v2"}])
class ProbingEnvs(Tester):
    def setUp(self) -> None:
        if self.sac_version == "v1":
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
            self.sac_hpars = sac_base_v2.copy()
            self.sac_hpars.update({
                    "seed": 12,
                    "batch_size": 10,
                    "replay_buffer_size": 50,
                    "min_replay_buffer_size": 10,
                    "eval_freq": 100,
                    "n_initial_exploration_steps": 0,
                    "file_name": "sac_v2",
                    "dest_model_path": "./test/models",
                    "dest_res_path": "./test/results",
            })
            


        return super().setUp()

    def test_probe_1(self):
        """
        Test probing env 1. This isolates the value loss calculation and the optimizer of the value network.
        """
        env = Probe1()
        sac_hpars_1 = self.sac_hpars.copy()
        sac_hpars_1.update({"env": env, "max_env_steps": 200, "max_action": 0})
        sac = SACTrainer(**sac_hpars_1)
        sac.train()

        state, action, next_state, reward, done = sample_tensors(env)
        self.assertAlmostEqual(sac.value(state).detach().cpu().item(), 1.0, places=3)
        self.assertAlmostEqual(
            sac.qf1(state, action.unsqueeze(0)).detach().cpu().item(), 1.0, places=3
        )
        self.assertAlmostEqual(
            sac.qf2(state, action.unsqueeze(0)).detach().cpu().item(), 1.0, places=3
        )

    def test_probe_2(self):
        """
        Test probing env 2. Tests whether the backpropagation of the value function is working correctly.
        """
        env = Probe2()
        sac_hpars_2 = self.sac_hpars.copy()
        sac_hpars_2.update({"env": env, "max_env_steps": 500, "max_action": 0})
        sac = SACTrainer(**sac_hpars_2)
        sac.train()

        s_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        s_1 = torch.tensor([1.0], device=device).unsqueeze(0)
        a_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        self.assertAlmostEqual(sac.value(s_0).detach().cpu().item(), 0.0, places=3)
        self.assertAlmostEqual(sac.value(s_1).detach().cpu().item(), 1.0, places=3)
        self.assertAlmostEqual(sac.qf1(s_0, a_0).detach().cpu().item(), 0.0, places=3)
        self.assertAlmostEqual(sac.qf1(s_1, a_0).detach().cpu().item(), 1.0, places=3)

    def test_probe_3(self):
        """
        Test probing env 3. Test whether reward discounting works.
        """
        env = Probe3()
        sac_hpars_3 = self.sac_hpars.copy()
        sac_hpars_3.update({"env": env, "discount": 0.5, "max_env_steps": 2000, "max_action": 0})
        sac = SACTrainer(**sac_hpars_3)
        sac.train()

        s_0 = torch.tensor([0.0], device=device)
        s_1 = torch.tensor([1.0], device=device)
        self.assertAlmostEqual(sac.value(s_0).detach().cpu().item(), 0.5, places=3)
        self.assertAlmostEqual(sac.value(s_1).detach().cpu().item(), 1.0, places=3)

    def test_probe_4(self):
        """
        Test probing env 4. Test whether policy can learn to pick a better action.
        """
        env = Probe4()
        sac_hpars_4 = self.sac_hpars.copy()
        sac_hpars_4.update({"env": env, "max_env_steps": 2000, "max_action": 1})
        sac = SACTrainer(**sac_hpars_4)
        sac.train()

        s_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        action, _ = sac.policy(s_0, deterministic=True)
        action = 0.0 if action.detach().cpu().item() <= 0 else 1.0
        self.assertAlmostEqual(sac.value(s_0).detach().cpu().item(), 1.0, delta=0.1)
        self.assertEqual(action, 1.0)

        a_1 = torch.tensor([1.0], device=device).unsqueeze(0)
        self.assertAlmostEqual(sac.qf1(s_0, a_1).detach().cpu().item(), 1.0, delta=0.1)
        self.assertAlmostEqual(sac.qf2(s_0, a_1).detach().cpu().item(), 1.0, delta=0.1)
        # cannot test OOD actions:
        # self.assertAlmostEqual(sac.qf1(s_0, a_0).detach().cpu().item(), 0.0, delta=0.1)
        # self.assertAlmostEqual(sac.qf2(s_0, a_0).detach().cpu().item(), 0.0, delta=0.1)

    def test_probe_5(self):
        """
        State is either 0 or 1, need to take action -1 in state 0 and 1 in state 1.
        """
        env = Probe5()
        sac_hpars_5 = self.sac_hpars.copy()
        sac_hpars_5.update({"env": env, "max_env_steps": 2000, "max_action": 1})
        sac = SACTrainer(**sac_hpars_5)
        sac.train()

        s_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        s_1 = torch.tensor([1.0], device=device).unsqueeze(0)
        a_0, _ = sac.policy(s_0, deterministic=True)
        a_1, _ = sac.policy(s_1, deterministic=True)
        a_0 = -1.0 if a_0.detach().cpu().item() <= 0 else 1.0
        a_1 = -1.0 if a_1.detach().cpu().item() <= 0 else 1.0

        # check V
        self.assertAlmostEqual(sac.value(s_0).detach().cpu().item(), 1.0, delta=0.1)
        self.assertAlmostEqual(sac.value(s_1).detach().cpu().item(), 1.0, delta=0.1)
        # check Policy
        self.assertEqual(a_0, -1.0)
        self.assertEqual(a_1, 1.0)
        # check in-distribution Q
        a_0 = torch.tensor([-1.0], device=device).unsqueeze(0)
        a_1 = torch.tensor([1.0], device=device).unsqueeze(0)
        self.assertAlmostEqual(sac.qf1(s_0, a_0).detach().cpu().item(), 1.0, delta=0.1)
        self.assertAlmostEqual(sac.qf1(s_1, a_1).detach().cpu().item(), 1.0, delta=0.1)
        self.assertAlmostEqual(sac.qf2(s_0, a_0).detach().cpu().item(), 1.0, delta=0.1)
        self.assertAlmostEqual(sac.qf2(s_1, a_1).detach().cpu().item(), 1.0, delta=0.1)
