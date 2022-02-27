from test import Tester, sample_tensors
from src.hyperparameters import sac_base
from src.sac_trainer import SACTrainer
import torch
from test.probing_envs import Probe1, Probe2, Probe3, Probe4, Probe5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbingEnvs(Tester):
    def setUp(self) -> None:
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

    def test_probe_2(self):
        """
        Test probing env 2. Tests whether the backpropagation of the value function is working correctly.
        """
        env = Probe2()
        sac_hpars_2 = self.sac_hpars.copy()
        sac_hpars_2.update({"env": env, "max_env_steps": 500, "max_action": 0})
        sac = SACTrainer(**sac_hpars_2)
        sac.train()

        s_0 = torch.tensor([0.0], device=device)
        s_1 = torch.tensor([1.0], device=device)
        self.assertAlmostEqual(sac.value(s_0).detach().cpu().item(), 0.0, places=3)
        self.assertAlmostEqual(sac.value(s_1).detach().cpu().item(), 1.0, places=3)

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
        sac_hpars_4.update({"env": env, "max_env_steps": 5000, "max_action": 1})
        sac = SACTrainer(**sac_hpars_4)
        sac.train()

        s_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        action, _ = sac.policy(s_0, deterministic=True)
        action = 0.0 if action.detach().cpu().item() <= 0 else 1.0
        self.assertAlmostEqual(sac.value(s_0).detach().cpu().item(), 1.0, places=3)
        self.assertEqual(action, 1.0)

    def test_probe_5(self):
        """
        State is either 0 or 1, need to take action equal to state.
        """
        env = Probe5()
        sac_hpars_5 = self.sac_hpars.copy()
        sac_hpars_5.update({"env": env, "max_env_steps": 2000, "max_action": 1, "discount": 0})
        sac = SACTrainer(**sac_hpars_5)
        sac.train()

        s_0 = torch.tensor([0.0], device=device).unsqueeze(0)
        s_1 = torch.tensor([1.0], device=device).unsqueeze(0)
        s_2 = torch.tensor([2.0], device=device).unsqueeze(0)
        a_0, _ = sac.policy(s_0, deterministic=True)
        a_1, _ = sac.policy(s_1, deterministic=True)
        a_0 = 0.0 if a_0.detach().cpu().item() <= 0 else 1.0
        a_1 = 0.0 if a_1.detach().cpu().item() <= 0 else 1.0

        self.assertAlmostEqual(sac.value(s_0).detach().cpu().item(), 1.0, places=3)
        self.assertAlmostEqual(sac.value(s_1).detach().cpu().item(), 1.0, places=3)
        self.assertAlmostEqual(sac.value(s_2).detach().cpu().item(), 0.0, places=3)
        self.assertEqual(a_0, 0.0)
        self.assertEqual(a_1, 1.0)
