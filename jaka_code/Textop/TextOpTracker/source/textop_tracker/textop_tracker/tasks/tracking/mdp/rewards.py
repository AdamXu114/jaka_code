from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from typing import Callable

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.envs.mdp.rewards import contact_forces
from textop_tracker.tasks.tracking.mdp import MotionCommand
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:

    @torch.jit.script
    def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Apply an inverse quaternion rotation to a vector.

        Args:
            quat: The quaternion in (w, x, y, z). Shape is (..., 4).
            vec: The vector in (x, y, z). Shape is (..., 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (..., 3).
        """
        # store shape
        shape = vec.shape
        # reshape to (N, 3) for multiplication
        quat = quat.reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        # extract components from quaternions
        xyz = quat[:, 1:]
        t = xyz.cross(vec, dim=-1) * 2
        return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

def _get_body_indexes_wowrist(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names_wo_ankle) if (body_names is None) or (name in body_names)]


def reward_cond_on_pfail(rew_fn: Callable) -> Callable:
    def wrapper(env: ManagerBasedRLEnv, pfail_threshold: float, *args) -> torch.Tensor:
        reward = rew_fn(env, *args).float()
        pfail_total = env.command_manager.get_term("motion").metrics["pfail_total"]
        reward = reward * (pfail_total < pfail_threshold).float()
        # print(f"pfail_total: {pfail_total}, reward: {reward}")
        return reward

    return wrapper


def contact_forces_cond_on_pfail(
    env: ManagerBasedRLEnv, pfail_threshold: float, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    return reward_cond_on_pfail(contact_forces)(env, pfail_threshold, threshold, sensor_cfg)


#####################


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)**2
    return torch.exp(-error / std**2)


def motion_anchor_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_vel = quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)

    robot_root_vel = quat_apply_inverse(command.robot_anchor_quat_w, command.robot_anchor_lin_vel_w)
    error = torch.sum(torch.square(ref_root_vel-robot_root_vel),dim=-1)
    return torch.exp(-error / std**2)


def motion_anchor_ang_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_ang_vel = quat_apply_inverse(command.anchor_quat_w, command.anchor_ang_vel_w)

    robot_root_ang_vel = quat_apply_inverse(command.robot_anchor_quat_w, command.robot_anchor_ang_vel_w)
    error = torch.sum(torch.square(ref_root_ang_vel-robot_root_ang_vel),dim=-1)
    return torch.exp(-error / std**2)

def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # body_indexes = _get_body_indexes(command, body_names)
    body_indexes = _get_body_indexes_wowrist(command, body_names)
    
    #body_indexes = torch.tensor([0,1,2,4,5,7,8,9,10,11,12,13],device=command.body_quat_relative_w.device)#_get_body_indexes_wo_ankle(command, body_names)
    error = quat_error_magnitude(
        command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]
    )**2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_pos_error_exp(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
   # body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.joint_pos - command.robot_joint_pos), dim=-1
    )
    return torch.exp(-error*0.15)
#self.joint_pos -self.robot_joint_pos,
#####################


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    env.extras["log"]["Metrics/reward/feet_slide_velocity_sum"] = reward
    return reward


def feet_slide_cond_on_pfail(
    env: ManagerBasedRLEnv, pfail_threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    return reward_cond_on_pfail(feet_slide)(env, pfail_threshold, sensor_cfg, asset_cfg)


def soft_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]

    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]

    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/reward/landing_force_mean"] = mean_landing_force
    return cost


# soft_landing_cond_on_pfail = reward_cond_on_pfail(soft_landing)
def soft_landing_cond_on_pfail(
    env: ManagerBasedRLEnv, pfail_threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    return reward_cond_on_pfail(soft_landing)(env, pfail_threshold, sensor_cfg)


def joint_vel_out_of_manual_limit_reward(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    rew = torch.sum((torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity).float(), dim=1)

    env.extras["log"]["Metrics/reward/overspeed"] = rew
    return rew


def joint_vel_out_of_manual_limit_cond_on_pfail_reward(
    env: ManagerBasedRLEnv,
    pfail_threshold: float,
    max_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return reward_cond_on_pfail(joint_vel_out_of_manual_limit_reward)(env, pfail_threshold, max_velocity, asset_cfg)


def joint_effort_out_of_limit_fixed_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.logical_not(
        torch.isclose(
            asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
        )
    ).float()
    rew = torch.sum(out_of_limits, dim=1)
    env.extras["log"]["Metrics/reward/overeffort"] = rew
    return rew


def joint_effort_out_of_limit_fixed_cond_on_pfail_reward(
    env: ManagerBasedRLEnv, pfail_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return reward_cond_on_pfail(joint_effort_out_of_limit_fixed_reward)(env, pfail_threshold, asset_cfg)

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        print(env.step_dt,"step dt !!!!!!!!!!!!!!")
        self.prev_prev_action = None
        self.prev_action = None
        self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty