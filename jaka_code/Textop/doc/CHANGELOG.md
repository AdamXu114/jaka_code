# 变更记录（CHANGELOG）

> 记录 Textop / TextOpTracker 代码库每次功能更新。**最新变更在最上方。**
> 文件路径均相对于 `TextOpTracker/` 项目根目录。

---

## 2026-10-26 — 执行器域随机化：动作延迟 / PD 增益 / armature

### 1. 本次新增或修改的功能

| # | 功能 | 实现方式 | 涉及文件 |
|---|---|---|---|
| 1 | **动作延迟随机化** | 把 `Khan_mini_CFG`（当前训练的 jakamini 机器人）的 4 组执行器由 `ImplicitActuatorCfg` 换成仓库内**已有**的 `DelayedImplicitActuatorCfg`，每个 episode 在 reset 时重采样 `[0, 4]` 物理步的指令延迟（`sim.dt = 0.005 s` → **0–20 ms**），作用于位置/速度/力矩目标 | `source/textop_tracker/textop_tracker/robots/K1L.py` |
| 2 | **PD 增益随机化** | `EventCfg` 新增两个 `mdp.randomize_actuator_gains` 事件（`mode="reset"`）：上肢（shoulder/elbow/wrist）刚度与阻尼 `[0.9, 1.1]`；下肢+腰+颈 `[0.8, 1.2]`，均为相对 nominal 的缩放 | `source/textop_tracker/textop_tracker/tasks/tracking/tracking_env_cfg_jaka.py` |
| 3 | **armature 随机化** | `EventCfg` 新增 `mdp.randomize_joint_parameters` 事件（`mode="reset"`），全部关节 armature 缩放 `[0.75, 1.25]` | 同上 |

### 2. 关键参数与依据

**动作延迟**（`K1L.py` 顶部常量 `ACTUATOR_DELAY_MIN_STEPS` / `ACTUATOR_DELAY_MAX_STEPS`）

| 来源 | 取值 | 换算（physics dt = 5 ms） |
|---|---|---|
| HEFT（`motion_tracking/cfg/task/profile/G1_tracking.yaml`，单位为物理步，见 `active_adaptation/envs/mdp/action/core.py:40` 注释） | `max_delay: 2` | 0–10 ms |
| MimicLite（`mimic-lite/cfg/task/tracking-base.yaml:44-46`） | `min_delay: 1` / `max_delay: 3` | 5–15 ms |
| **本仓库本次采用** | `min_delay: 0` / `max_delay: 4` | **0–20 ms** |

> ⚠️ 单位说明：上述 `delay` 均为**物理步**（非控制步）。若需覆盖实测 20–40 ms 的链路延迟，把 `ACTUATOR_DELAY_MAX_STEPS` 调到 `8` 即可（DelayBuffer 的 `max_time_lag ≤ history_length`，由 `DelayedImplicitActuator` 自动按 `max_delay` 建缓冲区）。

**PD 增益 / armature**（照搬 MimicLite）

| 关节组 | 刚度 & 阻尼缩放 | MimicLite 出处 |
|---|---|---|
| shoulder / elbow / wrist | `[0.9, 1.1]` | `mimic-lite/cfg/task/robot/g1.yaml` `stiffness_range` / `damping_range` 上肢档 |
| hip / knee / ankle / waist / neck | `[0.8, 1.2]` | 同上，腿/腰/踝档（G1 无颈部，本仓库把 Neck 归入该档） |
| armature（全部关节） | `[0.75, 1.25]` | 同上 `armature_range` |

- 实现采用 **IsaacLab v2.1.0 原生事件函数**（`isaaclab.envs.mdp.events.randomize_actuator_gains` / `randomize_joint_parameters`），它们内部调用的 `write_joint_stiffness_to_sim` / `write_joint_damping_to_sim` / `write_joint_armature_to_sim` 与 **MimicLite 的 isaaclab 后端分支**（`active-adaptation/.../randomizations/actuator.py` 的 `actuator_pd_gains._init_isaac/ reset`）是同一套底层 API，参数语义一致（相对 nominal 缩放、逐 env 独立采样）。
- 选择 `mode="reset"`（每个 episode 重采样）而非 `"startup"`，是为了对齐 MimicLite 的 `Randomization.reset()` 语义；如需降低成本，把两处 `mode="reset"` 改成 `"startup"` 即可（每个环境只在初始化时采样一次）。
- 关节名正则同时兼容两套命名（`Khan_mini_CFG` 的 `Left_shoulder_pitch_joint` 与 `K1L_CYLINDER_CFG` 的 `left_wrist_joint`），因为两个已注册任务（`Tracking-Flat-jakamini-*` 与 `Tracking-Flat-jaka-*`）**共用同一个 `EventCfg`**；正则解析失败会直接抛错，故必须两边都能匹配。

### 3. 本次未做的部分

| 未做项 | 原因 |
|---|---|
| 动作平滑 / 限幅（EMA `alpha`、slew-rate、action clip） | 按要求本次不增加该功能；后续若要加，需自定义 `ActionTerm`（IsaacLab 侧无内置等价项） |
| 关节摩擦随机化（MimicLite `friction_range: [0.05, 0.25]`） | 本次范围只包含 PD 增益与 armature；该项可直接复用同一个 `randomize_joint_parameters`（加 `friction_distribution_params=(0.05, 0.25)`、`operation="abs"`） |
| `K1L_CYLINDER_CFG`（`jaka` 任务机器人）的动作延迟 | 本次只对当前训练的 `Khan_mini_CFG` 启用延迟执行器；如 `jaka` 任务也要用，按同样方式替换其 4 处 `ImplicitActuatorCfg` |
| PD 增益的**观测**（把缩放量喂给策略/特权 critic） | 属于观测空间改动，不在本次范围；MimicLite/HEFT 会把它作为特权信息 |

### 4. 回退方式

- **整体回退**：直接使用备份 `backups/jaka_code_2026-10-26_pre_actuator_dr/`（相对工作区根目录 `/home/xu/IRMV/jaka`）。
- **只回退某一项**：
  - 动作延迟 → `K1L.py` 中 `Khan_mini_CFG` 的 4 个 `DelayedImplicitActuatorCfg` 改回 `ImplicitActuatorCfg`，并删除 `min_delay/max_delay` 两行；
  - PD 增益 → 删除 `EventCfg` 中的 `actuator_gains_upper` / `actuator_gains_lower` 两个 `EventTerm`；
  - armature → 删除 `EventCfg` 中的 `joint_armature`。

### 5. 验证状态

- [x] 改动文件通过 `python -m py_compile`（本机 Python 3.12，未安装 IsaacLab，仅语法检查）；
- [x] 对照 IsaacLab v2.1.0 官方源码核验所用 API 均存在且签名匹配（`randomize_actuator_gains` / `randomize_joint_parameters`、`DelayBuffer`）；
- [ ] **待运行时验证**（需在配置好的 IsaacLab 环境中）：确认 `SceneEntityCfg` 关节正则对两套机器人均解析成功、`mode="reset"` 下写入 PhysX 增益无异常、训练初期不因延迟/随机化出现明显不稳定。

### 6. 备注

- 若训练出现“前期站不稳”，优先把 `ACTUATOR_DELAY_MAX_STEPS` 从 4 降到 2，或将延迟事件改为按训练进度逐步放开（HEFT 的做法：前 20% 进度无延迟，之后线性到全量）。
- 后续可对照的实验顺序（来自审查报告的建议）：先只加“动作延迟 + PD 增益 + armature”（本次改动），观察实机遥操作跟踪指标，再决定是否需要补参考噪声/漂移注入与部署端延迟补偿。