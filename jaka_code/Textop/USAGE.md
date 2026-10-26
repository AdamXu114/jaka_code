# 使用说明

## TextOpTracker

### 安装

- 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab v2.1.0。假设使用 conda 环境 `env_isaaclab`。

- 然后安装 tracker：

```bash
cd TextOpTracker
python -m pip install -e source/textop_tracker
```

- 替换 `rsl_rl_lib` 为修改后的版本  
  > 此修改版包含 [modular norm](https://arxiv.org/abs/2405.14813) 的简化实现，可加快网络优化速度约 15%。如果你不想使用它，请在以下脚本中移除所有 `-MNMLP`。预训练检查点在 IsaacLab 中评估时需要该模块。

```bash
cd ..
pip uninstall rsl_rl_lib -y
pip install -e deps/rsl_rl-modular-normed/

python -c "import rsl_rl;print(rsl_rl)" # 验证安装
```

### 数据集转换
将dataset_pkl文件夹下的3个pkl数据集转换为npz格式
```bash
cd TextOpTracker
python scripts/pklpack_to_npz.py --input_file /path/to/***_new_wrist.pkl  --output_dir ./artifacts/jaka_motions --input_fps 50 --output_fps 50
```

### 训练
从头开始训练
- 修改 `--num_envs=16384`。`16384` 个环境大约需要 40GB GPU 显存。
```bash
python scripts/rsl_rl/train.py --headless --log_project_name TextOpTracr --task=Tracking-Flat-jakamini-ProjGravObs-MNMLP-v0 --motion_file=jaka_motions/* --run_name base agent.experiment_name=ExampleRun agent.max_iterations=40000 --num_envs=16384 env.commands.motion.anchor_body_name="base_link" env.commands.motion.future_steps=5 env.commands.motion.random_static_prob=-1.0 env.rewards.feet_slide.params.pfail_threshold=1.0 env.rewards.soft_landing.params.pfail_threshold=1.0 env.rewards.overspeed.params.pfail_threshold=1.0 env.rewards.overeffort.params.pfail_threshold=1.0 env.rewards.feet_slide.weight=-0.3 env.rewards.soft_landing.weight=-0.0003 env.rewards.overspeed.weight=-1.0 env.rewards.overeffort.weight=-1.0 env.commands.motion.enable_adaptive_sampling=True env.commands.motion.ads_type=v2 env.commands.motion.adaptive_beta=0.5 env.commands.motion.adaptive_alpha=0.1 env.commands.motion.adaptive_uniform_ratio=0.1 agent.policy.actor_hidden_dims=[2048,1024,512] agent.policy.critic_hidden_dims=[2048,1024,512] --seed=1 --device=cuda:0
```
