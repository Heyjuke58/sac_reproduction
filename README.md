# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor -  A reproduction
Reproduction of SAC from scratch [[Haarnoja et al.]](https://arxiv.org/pdf/1801.01290.pdf)

# Setup

- Install MuJoCo ([Instructions](https://github.com/openai/mujoco-py)) 

Specifically for Ubuntu:
1. Download the MuJoCo version 2.1.0 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Set LD_LIBRARY_PATH environment variable (for instance in .bashrc, etc.): `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin`

- Create and activate conda environment from `environment.yml`

```sh
conda env create -f environment.yml
conda activate sac
```

- To test your setup you can run `python try_mujoco.py`

# Run Experiments

Set wanted hyperparameters in `src/hyperparameters.py`

### SAC
```sh
python main.py --sac=<sac_env> --seed=<seed>
```

### TD3
```sh
python main.py --td3=<td3_env> --seed=<seed>
```

Seed is optional, default seed is given in `src/hyperparameters.py`.
Possible SAC envs: `{SAC_HOPPER, SAC_CHEETAH}`.
Possible TD3 envs: `{TD3_HOPPER, TD3_CHEETAH}`.

# Render a Learned Policy

To check out what a learned policy actually does in the environment, you can evaluate a leared policy (from the `models` folder) yourself by running:

```sh
python render_policy.py --model=<path-to-model-file>
```

Note that the file name needs to adhere to the used format as the algorithm and environment types are extracted from it.

# Compute Resources
TODO

# Results, Random Seeds, etc.

...can all be found in their respective files in the `results` folder.