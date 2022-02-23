from src.parse_arguments import parse_arguments
from src.sac import SAC
import torch
import gym
from typing import Optional

from TD3.main import main as td3_main
from argparse import Namespace


class Experiment:
    def __init__(
        self,
        runs: int,
        sac_hpars: Optional[dict] = None,
        td3_hpars: Optional[dict] = None,
        dest_model_path: str = "./models",
        dest_res_path: str = "./results",
    ) -> None:

        self.runs = runs  # runs per algorithms
        self.dest_model_path = dest_model_path
        self.dest_res_path = dest_res_path
        self.sac_hpars = sac_hpars
        self.td3_hpars = td3_hpars

    def run(self):
        self._run_sac()
        self._run_td3()

    def _run_sac(self):
        if self.sac_hpars is None:
            return

        start_seed = self.sac_hpars["seed"]
        for run in range(self.runs):
            seed = start_seed + run
            self.sac_hpars["seed"] = seed

            sac_experiment = SAC(**self.sac_hpars)
            sac_experiment.train()

    def _run_td3(self):
        if self.td3_hpars is None:
            return

        self.td3_hpars.update({
            "dest_model_path": self.dest_model_path,
            "dest_res_path": self.dest_res_path
        })

        start_seed = self.td3_hpars["seed"]
        for run in range(self.runs):
            seed = start_seed + run
            self.td3_hpars["seed"] = seed

            td3_main(self.td3_hpars)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    args = parse_arguments()
    
    exp = Experiment(**args)
    exp.run()

    # if "td3" in args["algs"]:
    #     td3_hpars = Namespace(
    #         policy="TD3",
    #         env=exp.env,
    #         seed=exp.start_seed,
    #         start_timesteps=25e3,
    #         eval_freq=5e3,
    #         max_timesteps=30e3,
    #         expl_noise=0.1,
    #         batch_size=256,
    #         discount=0.99,
    #         tau=0.005,
    #         policy_noise=0.2,
    #         noise_clip=0.5,
    #         policy_freq=2,
    #         save_model=True,
    #         load_model="",
    #         dest_model_path=exp.dest_model_path,
    #         dest_res_path=exp.dest_res_path,
    #     )
    #     exp.run_td3(td3_hpars)
    # if "sac" in args["algs"]:
    #     adam_kwargs = {
    #         "lr": 3e-4,
    #         "betas": (0.9, 0.999),
    #         "eps": 1e-08,
    #         "weight_decay": 0,
    #         "amsgrad": False,
    #     }
    #     exp.run_sac()
