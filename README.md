# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor -  A reproduction
Reproduction of SAC from scratch [[Haarnoja et al. 2018]](https://arxiv.org/pdf/1801.01290.pdf) and [Haarnoja et al. 2019](https://arxiv.org/pdf/1812.05905.pdf)


# Setup


- Install MuJoCo ([Instructions (See "Install MuJoCo")](https://github.com/openai/mujoco-py#install-mujoco))

Specifically for Ubuntu:
1. Download the MuJoCo version 2.1.0 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Set LD_LIBRARY_PATH environment variable (for instance in .bashrc, etc.): `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin`

- Clone repository including submodule for TD3:

```sh
git clone --recurse-submodules git@github.com:Heyjuke58/sac_reproduction.git
```
- Create and activate conda environment from `environment.yml`

```sh
conda env create -f environment.yml
conda activate sac2
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
Possible SAC envs: `{SAC_HOPPER, SAC_CHEETAH, SAC_HOPPER_FIXED_ALPHA, SAC_CHEETAH_FIXED_ALPHA}`.
Possible TD3 envs: `{TD3_HOPPER, TD3_CHEETAH}`.

# Render a Learned Policy

To check out what a learned policy actually does in the environment, you can evaluate a leared policy (from the `models` folder) yourself by running:

```sh
python render_policy.py --model=<path-to-model-file>
```

Note that the file name needs to adhere to the used format as the algorithm and environment types are extracted from it.

# Plot results

To plot results from runs, you have to move one or multiple result csv files into the folder `results/plot` and run the following command:

```sh
python plot.py --x=<x> -y=<y> -e=<env>
```
Choices:
- `x`: `["time", "env_steps", "grad_steps"]`
- `y`: `["avg_return", "log_probs_alpha"]`
- `e`: `["Hopper", "HalfCheetah"]`
- `b`: Bin size (to reduce noise in plot)

Note that the data from every file is splitted by the column `seed` s.t. the return is averaged over multiple runs.

# Results, Random Seeds, etc.

...can all be found in their respective files in the `results` folder.

# Exact Software Versions

... can be found in `environment-exact.yml`

# Compute Resources

Runs have been splitted to 2 GPUs:
- Nvidia Geforce RTX 3060 Laptop GPU
    - 23.2 h with 17 W ≅ 0.3944 kWh ≘ 12.68 cent (€) (assuming 32.16 cent/kWh)
- Nvidia Geforce GTX 1070Ti
    - 19.56 h with 70 W ≅ 1.3692 kWh ≘ 44.03 cent
Total: 56.71 cent (Only final evaluation and only GPU costs)
