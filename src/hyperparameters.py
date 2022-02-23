adam_kwargs = {
    "lr": 3e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}

sac_base = {
    "seed": 33,
    "hidden_dim": 256,
    "max_action": 1.0,
    "grad_steps": 1,
    "batch_size": 256,
    "replay_buffer_size": int(1e6),
    "min_replay_buffer_size": 1000,
    "adam_kwargs": adam_kwargs,
}

SAC_HOPPER = sac_base.copy()
SAC_HOPPER.update({
    "env": "Hopper-v3",
    "n_initial_exploration_steps": 1000,
    "scale_reward": 5,
    "max_env_steps": int(1e6),
})

SAC_CHEETAH = sac_base.copy()
SAC_CHEETAH.update({
    "env": "HalfCheetah-v3",
    "n_initial_exploration_steps": 10000,
    "scale_reward": 5,
    "max_env_steps": int(3e6),
})

SAC_V2 = {}

