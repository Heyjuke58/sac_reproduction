from src.parse_arguments import parse_arguments
from src.sac_trainer import SACTrainer
from src.sac_trainer_v2 import SACTrainerV2
import torch
from typing import Optional

from TD3.main import main as td3_main
from src.utils import get_timestamp


class Experiment:
    def __init__(
        self,
        runs: int,
        sac_hpars: Optional[dict] = None,
        sac_v2_hpars: Optional[dict] = None,
        td3_hpars: Optional[dict] = None,
        dest_model_path: str = "./models",
        dest_res_path: str = "./results",
    ) -> None:

        self.runs = runs  # runs per algorithms
        self.dest_model_path = dest_model_path
        self.dest_res_path = dest_res_path
        self.sac_hpars = sac_hpars
        self.sac_v2_hpars = sac_v2_hpars
        self.td3_hpars = td3_hpars

    def run(self):
        self._run_sac()
        self._run_sac_v2()
        self._run_td3()

    def _run_sac(self):
        if self.sac_hpars is None:
            return

        timestamp = get_timestamp()
        start_seed = self.sac_hpars["seed"]
        for run in range(self.runs):
            seed = start_seed + run
            self.sac_hpars["seed"] = seed

            sac_experiment = SACTrainer(
                **self.sac_hpars, file_name=f"SAC_{self.sac_hpars['env']}_{timestamp}"
            )
            sac_experiment.train()

    def _run_sac_v2(self):
        if self.sac_v2_hpars is None:
            return

        timestamp = get_timestamp()
        start_seed = self.sac_v2_hpars["seed"]
        for run in range(self.runs):
            seed = start_seed + run
            self.sac_v2_hpars["seed"] = seed

            sac_v2_experiment = SACTrainerV2(
                **self.sac_v2_hpars, file_name=f"SAC_V2_{self.sac_v2_hpars['env']}_{timestamp}"
            )
            sac_v2_experiment.train()

    def _run_td3(self):
        if self.td3_hpars is None:
            return

        self.td3_hpars.update(
            {
                "dest_model_path": self.dest_model_path,
                "dest_res_path": self.dest_res_path,
                "file_name": f"TD3_{self.td3_hpars['env']}_{get_timestamp()}",
            }
        )

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
