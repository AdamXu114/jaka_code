import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##



gym.register(
    id="Tracking-Flat-jaka-ProjGravObs-MNMLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
        flat_env_cfg.JAKAFlatProjGravObsEnvCfg,
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:JAKAFlatPPOModNormRunnerCfg",
    },
)



gym.register(
    id="Tracking-Flat-jakamini-ProjGravObs-MNMLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
        flat_env_cfg.JAKAminiFlatProjGravObsEnvCfg,
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:JAKAFlatPPOModNormRunnerCfg",
    },
)