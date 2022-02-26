adam_kwargs = {
    "lr": 3e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}

sac_base = {
    "seed": 42,
    "hidden_dim": 256,
    "max_action": 1.0,
    "grad_steps": 1,
    "batch_size": 256,
    "replay_buffer_size": int(1e6),
    "min_replay_buffer_size": 1000,
    # target Value update (exp moving avg. or hard update):
    "target_smoothing": 0.005,
    "target_update_freq": 1,
    "policy_reg": 1e-3,
    "discount": 0.99,
    "eval_freq": int(5e3),
    "eval_episodes": 10,
    "adam_kwargs": adam_kwargs,
}

SAC_HOPPER = sac_base.copy()
SAC_HOPPER.update(
    {
        "env": "Hopper-v3",
        "n_initial_exploration_steps": 1000,
        "scale_reward": 5,
        "max_env_steps": int(1e6),
        # "max_env_steps": int(5e3),
    }
)

SAC_CHEETAH = sac_base.copy()
SAC_CHEETAH.update(
    {
        "env": "HalfCheetah-v3",
        "n_initial_exploration_steps": 10000,
        "scale_reward": 5,
        "max_env_steps": int(3e6),
    }
)

SAC_V2 = {}

td3_base = {
    "policy": "TD3",
    "seed": 33,
    "start_timesteps": 25e3,
    "eval_freq": 5e3,
    "max_timesteps": 30e3,
    "expl_noise": 0.1,
    "batch_size": 256,
    "discount": 0.99,
    "tau": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_freq": 2,
    "save_model": True,
    "load_model": "",
}

TD3_HOPPER = td3_base.copy()
TD3_HOPPER.update(
    {
        "env": "Hopper-v3",
    }
)

TD3_CHEETAH = td3_base.copy()
TD3_CHEETAH.update(
    {
        "env": "HalfCheetah-v3",
        "max_timesteps": 3e6,
    }
)
