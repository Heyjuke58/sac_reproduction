from src.parse_arguments import parse_arguments
from ray.rllib.agents.ddpg import TD3Trainer
import torch



class Experiment:
    def __init__(self, algs: list[str], env: str, seed: int) -> None:

        if 'td3' in algs:
            self.td3_config = {
                "env": env,
                # Use 2 environment workers (aka "rollout workers") that parallelly
                # collect samples from their own environment clone(s).
                "num_workers": 1,
                "framework": "torch",
                # Tweak the default model provided automatically by RLlib,
                # given the environment's observation- and action spaces.
                "model": {
                    "fcnet_hiddens": [64, 64],
                    "fcnet_activation": "relu",
                },
                # Set up a separate evaluation worker set for the
                # `trainer.evaluate()` call after training (see below).
                "evaluation_num_workers": 1,
                # Only for evaluation runs, render the env.
                "evaluation_config": {
                    "render_env": True,
                },
                "buffer_size": 12345,
                "replay_buffer_config":{
                    "capacity": 1000000,
                },
                "num_gpus": 1,
            }
        self.algs = algs
        self.seed = seed

    def run_sac(self):
        pass

    def run_td3(self):
        trainer = TD3Trainer(config=self.td3_config)
        for _ in range(100):
            out = trainer.train()

        trainer.evaluate()

if __name__ == "__main__":
    print(torch.cuda.is_available())
    args = parse_arguments()
    exp = Experiment(**args)
    if "td3" in args["algs"]:
        exp.run_td3()
    if "sac" in args["algs"]:
        exp.run_sac()