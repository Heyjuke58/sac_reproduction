# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor -  A reproduction
Reproduction of SAC from scratch [[Haarnoja et al.]](https://arxiv.org/pdf/1801.01290.pdf)

# Setup

- Install MuJoCo
1. Download the MuJoCo version 2.1.0 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

- Create and activate conda environment from `environment.yml`

```sh
conda env create -f environment.yml
conda activate sac
```

- To test your setup you can run `python test_mujoco.py`