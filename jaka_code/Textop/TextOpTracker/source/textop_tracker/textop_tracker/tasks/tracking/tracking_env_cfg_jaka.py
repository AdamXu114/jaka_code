from __future__ import annotations

from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise, GaussianNoiseCfg

import textop_tracker.tasks.tracking.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}
cmd_min=torch.tensor([-0.15,-0.15,-0.01,-0.1,-0.1,-0.2])
cmd_max=torch.tensor([0.15,0.15,0.01,0.1,0.1,0.2])
# dof_pos_noise=torch.ones(25)*0.1
dof_pos_noise=torch.ones(27)*0.1*0.5
#dof_pos_noise[-6:]=0.01

# cmd_noise_min=torch.cat([cmd_min,-dof_pos_noise]).numpy().tolist()#-0.1#torch.tensor([-0.1])#torch.cat([cmd_min,-dof_pos_noise])
# cmd_noise_max=torch.cat([cmd_max,dof_pos_noise]).numpy().tolist()

# ###
# n_future = 5  # 假设 n_future 为 5，你可以替换成你实际的数值
# block = torch.zeros(31)
# block[3:4] = 0.01       # 对应 i*31+3 : i*31+4   root_z
# block[4:31] = 0.05      # 对应 i*31+4 : i*31+31  joint_pos


# n_timesteps = 10

# # Define noise for each command component separately to match feature-by-feature concatenation in command_multi.py
# # ref_root_vel_all: shape (num_envs, n_timesteps, 2)
# vel_noise = torch.tensor([0.15, 0.15]).repeat(n_timesteps)

# # height_all: shape (num_envs, n_timesteps, 1)
# height_noise = torch.tensor([0.01]).repeat(n_timesteps)

# # ref_projected_gravity: shape (num_envs, n_timesteps, 3)
# gravity_noise = torch.tensor([0.05, 0.05, 0.05]).repeat(n_timesteps)

# # ref_root_ang_vel_all[..., 2:3]: shape (num_envs, n_timesteps, 1)
# ang_vel_noise = torch.tensor([0.2]).repeat(n_timesteps)

# # motion_joint_pos: shape (num_envs, n_timesteps, 27)
# joint_noise = torch.tensor([0.05]).repeat(27 * n_timesteps)

# # Concatenate all component noises in the exact order they are stacked in command_multi.py
# noise_std = torch.cat([
#     vel_noise,      # n_timesteps * 2
#     height_noise,   # n_timesteps * 1
#     gravity_noise,  # n_timesteps * 3
#     ang_vel_noise,  # n_timesteps * 1
#     joint_noise     # n_timesteps * 27
# ], dim=-1)

n_timesteps = 5

# Define noise for each command component separately to match feature-by-feature concatenation in command_multi.py
# ref_root_vel_all: shape (num_envs, n_timesteps, 2)
pos_diff_noise = torch.tensor([0., 0.,0.]).repeat(n_timesteps)

# height_all: shape (num_envs, n_timesteps, 1)
height_noise = torch.tensor([0.01]).repeat(n_timesteps)

# motion_joint_pos: shape (num_envs, n_timesteps, 27)
joint_noise = torch.tensor([0.05]).repeat(27 * n_timesteps)

# Concatenate all component noises in the exact order they are stacked in command_multi.py
noise_std = torch.cat([
    pos_diff_noise,      # n_timesteps * 2
    height_noise,   # n_timesteps * 1
 
    joint_noise     # n_timesteps * 27
], dim=-1)

cmd_noise_min = (-1 * noise_std).numpy().tolist()
cmd_noise_max = noise_std.numpy().tolist()
# print(cmd_noise_max,cmd_noise_min)
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        future_steps=5,  # Future N-step lookahead
        # ref_history_steps=5,  # History N-step lookback
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",scale=0.5, joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel_no_ankle,
        #     noise=Unoise(n_min=-0.5, n_max=0.5),
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"])}
        # )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b_future, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b_future, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

##this used!!!!!
@configclass
class ProjGravObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
            noise=Unoise(n_min=cmd_noise_min, n_max=cmd_noise_max)#GaussianNoiseCfg(operation="add", mean=0.0, std=0.0)
        )
        # motion_anchor_pos_b = ObsTerm(
        #     func=mdp.motion_anchor_pos_b_future,
        #     params={"command_name": "motion"},
        #     noise=Unoise(n_min=-0.25, n_max=0.25)
        # )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.07, n_max=0.07),history_length = 5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25,history_length = 5)
        # projected_gravity = ObsTerm(func=mdp.torso_projected_gravity_b, noise=Unoise(n_min=-0.07, n_max=0.07),params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_Link"])},history_length = 5)
        # base_ang_vel = ObsTerm(func=mdp.torso_ang_vel_b, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25,params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_Link"])},history_length = 5)
         
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),history_length = 5)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5),scale=0.05,history_length = 5)
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel_no_ankle,
        #     noise=Unoise(n_min=-0.5, n_max=0.5),
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"])}
        # )
        actions = ObsTerm(func=mdp.last_action,history_length = 5)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
           # self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b_future, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b_future, params={"command_name": "motion"})
        
        projected_gravity = ObsTerm(func=mdp.projected_gravity)#, noise=Unoise(n_min=-0.07, n_max=0.07))
        # projected_gravity = ObsTerm(func=mdp.torso_projected_gravity_b,params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_Link"])})

        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
      #  motion_joint_vel = ObsTerm(func=mdp.motion_joint_vel, params={"command_name": "motion"},scale=0.05)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,scale=0.25)
        # base_ang_vel = ObsTerm(func=mdp.torso_ang_vel_b,scale=0.25,params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_Link"])})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class PropPropObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class PrivPrivObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b_future, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b_future, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PrivilegedCfg = PrivilegedCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class NoisePrivObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PrivilegedCfg = PrivilegedCfg(enable_corruption=True)
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_Link"), "mass_distribution_params": (-2.0, 3.0), "operation": "add"},#-2 4
    )
    
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_Link"), #"waist_yaw_link"),
            "com_range": {
                # "x": (-0.05, 0.05),
                # "y": (-0.075, 0.075),
                # "z": (-0.075, 0.075)
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05)
            },
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""



    motion_root_vel = RewTerm(
        func=mdp.motion_anchor_vel_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 1.
        },
    )

    motion_root_ang_vel = RewTerm(
        func=mdp.motion_anchor_ang_vel_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 3.14
        },
    )

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 0.3
        },
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 0.4
        },
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.3
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.4
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 1.0
        },
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 3.14
        },
    )
    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_pos_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            #"std": 3.14
        },
    )


    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # feet_acc =  RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"])})
    
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg":
    #             SceneEntityCfg(
    #                 "contact_forces",
    #                 body_names=[
    #                     r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_link$)(?!right_wrist_link$)(?!left_knee_link$)(?!right_knee_link$).+$"
    #                 ],
    #             ),
    #         "threshold":
    #             1.0,
    #     },
    # )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg":
                SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!Left_ankle_roll_Link$)(?!Right_ankle_roll_Link$)(?!Left_wrist_yaw_Link$)(?!Right_wrist_yaw__Link$)(?!Left_knee_Link$)(?!Right_knee_Link$)(?!waist_yaw_Link$).+$"#waist_yaw_Link
                    ],
                ),
            "threshold":
                1.0,
        },
    )
    feet_force = RewTerm(
        func=mdp.contact_forces_cond_on_pfail,
        weight=-0.0,  # -0.05/100
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 600,
            "pfail_threshold": 1.0,  # for backward compatibility
        }
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_cond_on_pfail,
        weight=-0.1*5,  # -0.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "pfail_threshold": 0.2,
        }
    )
    soft_landing = RewTerm(
        func=mdp.soft_landing_cond_on_pfail,
        weight=-1e-5,  # -0.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "pfail_threshold": 0.2,
        }
    )

    overspeed = RewTerm(
        func=mdp.joint_vel_out_of_manual_limit_cond_on_pfail_reward,
        weight=-0.1,
        params={
            "max_velocity": 20.0,
            "pfail_threshold": 0.15,
        },
    )

    overeffort = RewTerm(
        func=mdp.joint_effort_out_of_limit_fixed_cond_on_pfail_reward,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pfail_threshold": 0.15,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8
        },
    )
    ee_body_pos_ankle = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name":
                "motion",
            "threshold":
                0.25,
            "body_names":
                [
                    "Left_ankle_roll_Link",
                    "Right_ankle_roll_Link",
                    # "left_wrist_link",
                    # "right_wrist_link",
                ],
        },
    )
    ee_body_pos_wrist = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name":
                "motion",
            "threshold":
                0.4,
            "body_names":
                [
                    # "left_ankle_roll_link",
                    # "right_ankle_roll_link",
                    "Left_wrist_yaw_Link",
                    "Right_wrist_yaw__Link",
                ],
        },
    )
    # overboundary = DoneTerm(
    #     func=mdp.joint_pos_out_of_limit_cond_on_pfail,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "pfail_threshold": 0.15,
    #     }
    # )
    # overspeed = DoneTerm(
    #     func=mdp.joint_vel_out_of_manual_limit_cond_on_pfail, params={
    #         "max_velocity": 20.0,
    #         "pfail_threshold": 0.15,
    #     }
    # )
    # overeffort = DoneTerm(
    #     func=mdp.joint_effort_out_of_limit_fixed_cond_on_pfail,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "pfail_threshold": 0.15,
    #     }
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.viewer.origin_type = "env"
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
