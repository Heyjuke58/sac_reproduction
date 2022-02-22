from src.parse_arguments import parse_arguments
import torch
from pathlib import Path

from TD3.main import main as td3_main
from argparse import Namespace


class Experiment:
    def __init__(
        self,
        algs: list[str],
        env: str,
        seed: int,
        runs: int,
        dest_model_path: str = "./models",
        dest_res_path: str = "./results",
    ) -> None:

        self.algs = algs  # which algorithms to run
        self.start_seed = seed  # subsequent runs use seed + n
        self.runs = runs  # runs per algorithms
        self.env = env  # gym environment string
        self.dest_model_path = dest_model_path
        self.dest_res_path = dest_res_path

    def run_sac(self):
        for run in range(self.runs):
            seed = self.start_seed + run
            pass

    def run_td3(self, td3_params: Namespace):
        for run in range(self.runs):
            seed = self.start_seed + run
            td3_params.seed = seed
            td3_main(td3_params)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    args = parse_arguments()
    exp = Experiment(**args)
    if "td3" in args["algs"]:
        td3_params = Namespace(
            policy="TD3",
            env=exp.env,
            seed=exp.start_seed,
            start_timesteps=25e3,
            eval_freq=5e3,
            max_timesteps=30e3,
            expl_noise=0.1,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            save_model=True,
            load_model="",
            dest_model_path=exp.dest_model_path,
            dest_res_path=exp.dest_res_path,
        )
        exp.run_td3(td3_params)
    if "sac" in args["algs"]:
        adam_kwargs = {
            "lr": 3e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        exp.run_sac()
